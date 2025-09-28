from typing import Dict, Optional
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
        url: Optional[str] = None,
    ):
        """Send structured update to the correct client"""
        websocket = self.active_connections.get(user_id)
        if websocket:
            payload = {"stage": stage, "progress": progress, "message": message}
            if url:
                payload["url"] = url
            try:
                await websocket.send_json(payload)
            except Exception:
                # If sending fails (client gone), cleanup
                self.disconnect(user_id)


# Create a global streamer instance
streamer = Streamer()
