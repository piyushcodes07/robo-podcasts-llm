from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Literal

from .generate import generate
import asyncio
from podcast_llm.streamer import streamer  # import reusable module

app = FastAPI(
    title="Podcast LLM API",
    description="An API for generating podcast episodes from topics or source material.",
    version="0.1.0",
)


class PodcastRequest(BaseModel):
    topic: str
    mode: Literal["research", "context"]
    sources: Optional[List[str]] = None
    qa_rounds: int = 2
    audio_output: Optional[str] = "podcast.mp3"
    text_output: Optional[str] = "podcast.md"
    user_id: str = "mock_user"


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await streamer.connect(user_id, websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        streamer.disconnect(user_id)


@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
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
