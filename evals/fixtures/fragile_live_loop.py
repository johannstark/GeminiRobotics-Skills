import asyncio
from google.genai import types

async def run_live(client):
    async with client.aio.live.connect(
        model="gemini-robotics-er-2-streaming-preview",
        config=types.LiveConnectConfig(response_modalities=["TEXT"])
    ) as session:
        async for response in session.receive():
            print("Received response:", response)
