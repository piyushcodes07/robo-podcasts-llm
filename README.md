# Podcast LLM: Automated Podcast Generation Engine

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-username/podcast-llm)

An end-to-end, event-driven system that automates the creation of entire podcast episodes from a single topic. This project demonstrates a robust, scalable backend architecture for orchestrating long-running AI and media generation pipelines.

## Overview

`podcast-llm` is not just a script. It's a complete backend service that takes a user-provided topic and generates a full-length audio episode featuring an interview between two AI hosts. It handles everything from autonomous online research and scriptwriting to voice generation and audio production.

The core of this project is its resilient, asynchronous architecture, designed to handle complex, multi-stage workflows that can take several minutes to complete without compromising API responsiveness or reliability.

## Features & Architectural Highlights

This project was engineered with scalability and resilience as first-class citizens.

*   **Asynchronous Job Processing**: The FastAPI backend uses background tasks to offload long-running generation jobs. The API remains non-blocking and highly responsive, immediately returning a job confirmation.
*   **Real-time Progress Updates**: A WebSocket interface streams live progress updates to the client, providing visibility into the generation pipeline (e.g., `Researching`, `Writing Script`, `Generating Audio`).
*   **Resilient & Resumable Pipelines**:
    *   **Checkpointing**: The system saves the output of each major stage. If a job fails midway (e.g., during audio synthesis), it can be resumed from the last successful checkpoint, saving significant time and compute.
    *   **Exponential Backoff & Retries**: All external API calls (to LLMs, TTS services, etc.) are wrapped in a resilient client that automatically handles transient errors and rate limits.
*   **Scalable Map-Reduce Research Engine**: To gather context, the system performs parallel research using a `ThreadPoolExecutor` to fetch and summarize dozens of articles concurrently before synthesizing them into a structured outline.
*   **Modular & Extensible**: Built with a clear separation of concerns. Each component (`research`, `outline`, `writer`, `text_to_speech`) is a distinct module, making it easy to maintain and extend (e.g., adding a new TTS provider).

## Architecture Diagram

The system follows an event-driven, pipeline architecture:

```mermaid
graph TD
    A[Client Request: /generate] --> B{API Server (FastAPI)};
    B --> C[1. Start Background Job];
    C --> D{Research Agent};
    D --> E{Outline Generator};
    E --> F{Script Writer};
    F --> G{Text-to-Speech Engine};
    G --> H[2. Merge Audio & Finalize];
    H --> I[3. Upload to Storage];

    subgraph Real-time Feedback
        B -- Job Started --> J(WebSocket Streamer);
        D -- Researching... --> J;
        F -- Writing Script... --> J;
        G -- Generating Audio... --> J;
        I -- Complete --> J;
    end

    J --> K[Client];
```

## Technology Stack

*   **Backend**: FastAPI, Pydantic, Uvicorn
*   **AI & Orchestration**: LangChain, OpenAI GPT-4 & GPT-4o-mini-tts
*   **Research**: Tavily API, Wikipedia
*   **Text-to-Speech**: ElevenLabs, Google Cloud TTS
*   **Audio Processing**: pydub
*   **Real-time Communication**: WebSockets
*   **Testing**: Pytest

## Getting Started

### Prerequisites

*   Python 3.9+
*   An `.env` file with your API keys.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/podcast-llm.git
    cd podcast-llm
    ```

2.  **Create a virtual environment and install dependencies:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure your environment:**

    Create a `.env` file in the project root by copying the example. Then, fill in your API keys.
    ```sh
    cp .env.example .env
    ```
    ```.env
    # .env
    OPENAI_API_KEY="sk-..."
    ELEVENLABS_API_KEY="..."
    GOOGLE_API_KEY="..."
    TAVILY_API_KEY="..."
    ANTHROPIC_API_KEY="..."
    ```

### Running the Server

Launch the FastAPI application with Uvicorn:
```sh
uvicorn podcast_llm.api:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

