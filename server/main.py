import json
from typing import List

from fastapi import FastAPI
from starlette.websockets import WebSocket, WebSocketDisconnect

from HealthIndexes import HealthIndexes

app = FastAPI()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.get("/health-check")
def read_root():
    return {"Hello": "World"}


@app.post("/api/health-indexes")
async def push_health_indexes(healths: List[HealthIndexes]):
    message = {'data': [health.dict() for health in healths]}
    await manager.broadcast(message=json.dumps(message))
    return message


@app.websocket("/ws/health-indexes")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
