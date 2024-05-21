import json
from typing import List

from fastapi import FastAPI
from starlette.websockets import WebSocket, WebSocketDisconnect

from GeminiHelper import GeminiHelper
from HealthIndexes import HealthIndexes

app = FastAPI()

API_KEY = "your-api-key-here"

gemini_helper = GeminiHelper(api_key=API_KEY)


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
    result_message = {'data': [health.dict() for health in healths]}

    message_1 = f"""
            My heart rate = 121, spo2 = 91, and my body temperature = 36 degrees celsius. 
            Is my health okay? If not, could you give me some suggestions about my health? 
            Limit the response in 100 words
        """

    message_2 = f"""
                Here are my heart rate, spo2, and my temperature in celsius (displayed as json format). 
                Is my health okay? If not, could you give me some suggestions about my health? 
                Limit the response in 100 words, short but informative.
                
                {healths}
            """

    response = gemini_helper.request(
        message_2
    )

    print(response.text)

    indexes = list(map(lambda x: x.to_dict(), healths))

    await manager.broadcast(message=json.dumps(
        {
            'data': indexes,
            'message': response.text
        }
    ))

    return True


@app.websocket("/ws/health-indexes")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            response = gemini_helper.request(data).text

            await websocket.send_text(response)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
