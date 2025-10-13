import os
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Literal

from .generate import generate
import asyncio
from podcast_llm.streamer import streamer  # import reusable module
import shutil
import tempfile
from fastapi import File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Podcast LLM API",
    description="An API for generating podcast episodes from topics or source material.",
    version="0.1.0",
)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PodcastRequest(BaseModel):
    topic: str
    mode: Literal["research", "context"]
    sources: Optional[List[str]] = None
    qa_rounds: int = 2
    audio_output: Optional[str] = "podcast.mp3"
    text_output: Optional[str] = "podcast.md"
    user_id: str = "mock_user"


@app.post("/generate/upload")
async def upload_and_generate(
    background_tasks: BackgroundTasks,
    topic: str = Form(...),
    mode: Literal["context", "research"] = Form(...),
    main_user_id: str = Form(...),
    sources: Optional[List[str]] = Form(None),  # For URLs
    qa_rounds: int = Form(2),
    audio_output: Optional[str] = Form("podcast.mp3"),
    text_output: Optional[str] = Form("podcast.md"),
    user_id: str = Form("mock_user"),
    files: Optional[List[UploadFile]] = File(None),  # For files
):
    """
    Accepts uploaded files and URLs as sources for podcast generation.
    """
    temp_dir = tempfile.mkdtemp(prefix="podcast_llm_uploads_")
    all_sources = []
    if sources:
        all_sources.extend(sources)

    if files:
        for file in files:
            # NOTE: You might want to sanitize the filename in a production environment
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            all_sources.append(temp_path)

    async def generate_and_cleanup():
        """
        A wrapper task to run generation and then clean up the temporary directory.
        """
        try:
            await generate(
                topic=topic,
                main_user_id=main_user_id,
                mode=mode,
                sources=all_sources,
                qa_rounds=qa_rounds,
                use_checkpoints=True,
                audio_output=audio_output,
                text_output=text_output,
                user_id=user_id,
            )
        finally:
            shutil.rmtree(temp_dir)

    background_tasks.add_task(generate_and_cleanup)

    return {"message": "Podcast generation started with uploaded files and URLs."}


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await streamer.connect(user_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        streamer.disconnect(user_id)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Podcast LLM API"}


@app.post("/generate/start")
async def start_generation(request: PodcastRequest, background_tasks: BackgroundTasks):
    """
    Starts a new podcast generation job in the background.
    """
    background_tasks.add_task(
        generate,
        topic=request.topic,
        mode=request.mode,
        sources=request.sources,
        qa_rounds=request.qa_rounds,
        use_checkpoints=True,
        audio_output=request.audio_output,
        text_output=request.text_output,
        user_id=request.user_id,
    )
    return {"message": "Podcast generation started in the background."}


@app.post("/generate/episode")
async def generate_episode(request: PodcastRequest):
    """
    Generates a new podcast episode and returns the audio file.
    """
    await generate(
        topic=request.topic,
        mode=request.mode,
        sources=request.sources,
        qa_rounds=request.qa_rounds,
        use_checkpoints=True,
        audio_output=request.audio_output,
        text_output=request.text_output,
    )
    return FileResponse(request.audio_output)
