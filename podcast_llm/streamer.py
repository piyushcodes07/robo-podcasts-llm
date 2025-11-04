import json
from typing import Any, Dict, Optional
from fastapi import WebSocket
import asyncio


class Streamer:
    def __init__(self):
        # Keeps track of active WebSocket connections: {user_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        """Register a new user connection"""
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        """Remove user connection"""
        self.active_connections.pop(user_id, None)

    async def send(
        self,
        user_id: str,
        stage: str,
        progress: int,
        message: str,
        *,
        update_payload: Optional[Any] = None,
        url: Optional[str] = None,
    ):
        print("log from streamer$$$$$$$$$$$$$$", update_payload)
        """Send structured update to the correct client"""
        websocket = self.active_connections.get(user_id)
        if websocket:
            payload = {
                "stage": stage,
                "progress": progress,
                "message": message,
                "payload": update_payload,
            }
            if url:
                payload["url"] = url
            try:
                await websocket.send_json(payload)
            except Exception:
                # If sending fails (client gone), cleanup
                self.disconnect(user_id)


# Create a global streamer instance
streamer = Streamer()
