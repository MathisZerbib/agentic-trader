import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import json
import asyncio
from typing import List, Optional
from sqlalchemy.orm import Session
from database import SessionLocal
import models
from services.state import get_current_state

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send initial data
        db = SessionLocal()
        try:
            # Send initial state
            state = get_current_state(db)
            await websocket.send_text(json.dumps(state))
            
            # Send initial logs
            logs = db.query(models.AgentLog).order_by(models.AgentLog.timestamp.desc()).limit(50).all()
            logs_msg = {
                "type": "logs",
                "data": [{"timestamp": str(log.timestamp), "title": log.title, "content": log.content} for log in logs]
            }
            await websocket.send_text(json.dumps(logs_msg))
        except Exception as e:
            print(f"Error sending initial WebSocket data: {e}")
            self.disconnect(websocket)
            try:
                await websocket.close()
            except Exception:
                pass
        finally:
            db.close()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        data = json.dumps(message)
        connections = list(self.active_connections)

        async def _send_one(connection: WebSocket):
            if connection.application_state != WebSocketState.CONNECTED:
                raise RuntimeError("WebSocket not connected")
            await asyncio.wait_for(connection.send_text(data), timeout=1.0)

        results = await asyncio.gather(
            *(_send_one(connection) for connection in connections),
            return_exceptions=True,
        )

        for connection, result in zip(connections, results):
            if result is not None:
                self.disconnect(connection)
                try:
                    await connection.close()
                except Exception:
                    pass

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except RuntimeError as e:
        if "WebSocket is not connected" in str(e):
            manager.disconnect(websocket)
        else:
            raise

async def broadcast_ws_message(message: dict):
    await manager.broadcast(message)
