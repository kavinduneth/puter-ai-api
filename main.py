
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from putergenai import PuterClient
import os
import base64
import requests
from dotenv import load_dotenv
from typing import Optional, List, AsyncGenerator
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
load_dotenv()

app = FastAPI(title="Puter AI API Service", description="API for text chat, image recognition, and streaming using Puter AI", version="1.0.0")

# Rate limiting to manage 100 users
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Load token

client = PuterClient(token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0IjoiYXUiLCJ2IjoiMC4wLjAiLCJ1dSI6Ik5wNTF4YzQwUXZhVS91N2NZVFRVTWc9PSIsImF1IjoiaWRnL2ZEMDdVTkdhSk5sNXpXUGZhUT09IiwicyI6Inp4YTVmYmhaNXYxc0ZSUWpXT2Ftenc9PSIsImlhdCI6MTc1ODgxOTI2NH0.SW93dRLbHsVg9meoOE8iBrU2HCzmmkCP_gIEzjD4WRU')

class ChatRequest(BaseModel):
    question: str
    model: Optional[str] = "gpt-4.1-nano"
    stream: Optional[bool] = False

class ImageRequest(BaseModel):
    image_path_or_url: str
    prompt: Optional[str] = "Describe this image in detail."
    model: Optional[str] = "gpt-4o"

class MultiTurnRequest(BaseModel):
    messages: List[dict]
    model: Optional[str] = "gpt-4.1-nano"
    stream: Optional[bool] = False

# Helper function for AI chat with timeout
async def _ai_chat_with_timeout(messages: List[dict], options: dict, timeout: int = 30):
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(client.ai_chat, messages=messages, options=options),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI chat error: {str(e)}")

# Streaming response generator
async def stream_chat_response(messages: List[dict], options: dict) -> AsyncGenerator[str, None]:
    try:
        response = await _ai_chat_with_timeout(messages, options)
        for chunk, _ in response:
            yield chunk  # Yield each token for real-time streaming
    except Exception as e:
        yield f"Error streaming response: {str(e)}"

# Text chat endpoint (GET for simple URL calls)
@app.get("/chat")
@limiter.limit("10/minute")
async def chat_get(
    request: Request,  # Added for slowapi
    question: str = Query(..., description="The question to ask"),
    model: str = Query("gpt-4.1-nano", description="AI model to use"),
    stream: bool = Query(False, description="Enable streaming response")
):
    messages = [{"role": "user", "content": question}]
    options = {
        "model": model,
        "strict_model": True,
        "temperature": 1,
        "stream": stream
    }
    
    if stream:
        return StreamingResponse(
            stream_chat_response(messages, options),
            media_type="text/plain"
        )
    
    try:
        response = await _ai_chat_with_timeout(messages, options)
        content = response["response"]["result"]["message"]["content"]
        used_model = response["used_model"]
        return {"answer": content, "model_used": used_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI chat error: {str(e)}")

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_post(request: Request, chat_request: ChatRequest):
    return await chat_get(request, chat_request.question, chat_request.model, chat_request.stream)

# Image recognition endpoint
@app.post("/recognize-image")
@limiter.limit("5/minute")
async def recognize_image(request: Request, image_request: ImageRequest):
    if image_request.image_path_or_url.startswith(("http://", "https://")):
        try:
            response = requests.head(image_request.image_path_or_url, timeout=5)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Image URL inaccessible: HTTP {response.status_code}")
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error accessing image URL: {str(e)}")
    else:
        try:
            with open(image_request.image_path_or_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_request.image_path_or_url = f"data:image/jpeg;base64,{image_data}"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image read error: {str(e)}")
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": image_request.prompt},
            {"type": "image_url", "image_url": {"url": image_request.image_path_or_url}}
        ]
    }]
    
    options = {"model": image_request.model, "strict_model": True, "temperature": 1}
    
    try:
        response = await _ai_chat_with_timeout(messages, options)
        content = response["response"]["result"]["message"]["content"]
        used_model = response["used_model"]
        return {"description": content, "model_used": used_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image recognition error: {str(e)}")

# Streaming chat endpoint for real-time token output
@app.get("/stream-chat")
@limiter.limit("10/minute")
async def stream_chat(
    request: Request,  # Added for slowapi
    question: str = Query(..., description="The question to ask"),
    model: str = Query("gpt-4.1-nano", description="AI model to use")
):
    messages = [{"role": "user", "content": question}]
    options = {
        "model": model,
        "strict_model": True,
        "temperature": 1,
        "stream": True
    }
    
    return StreamingResponse(
        stream_chat_response(messages, options),
        media_type="text/plain"
    )

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
