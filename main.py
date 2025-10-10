from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
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
from supabase import create_client, Client  # ✅ Add this
import urllib.parse
import json
from pathlib import Path
import pdfplumber
from io import BytesIO
import time

# --------------------------------------------------------

# Load environment variables
# Load environment variables
load_dotenv()

# -------------------- Supabase Setup --------------------
SUPABASE_URL = "https://ebtqetaxmpfbarkcturc.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVidHFldGF4bXBmYmFya2N0dXJjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk5MzczNjcsImV4cCI6MjA3NTUxMzM2N30.Ekg8ylQ4E84bCilGCDQLM6GO9ju7lv8nbFNc2zJ1lz0"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# --------------------------------------------------------

app = FastAPI(title="Puter AI API Service", description="API with Supabase API-key check, TTS, and File Handling", version="2.0.0")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Load Puter token
client = PuterClient(token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0IjoiYXUiLCJ2IjoiMC4wLjAiLCJ1dSI6Ik5wNTF4YzQwUXZhVS91N2NZVFRVTWc9PSIsImF1IjoiaWRnL2ZEMDdVTkdhSk5sNXpXUGZhUT09IiwicyI6Inp4YTVmYmhaNXYxc0ZSUWpXT2Ftenc9PSIsImlhdCI6MTc2MDA4MDg2M30.FtwaKqFabzLsRJVDd_FbQKzHP2ZMhwM-FMoMb14E9J8')

class ChatRequest(BaseModel):
    question: str
    model: Optional[str] = "grok-beta"
    stream: Optional[bool] = False
    apikey: str
    email: str

class ImageRequest(BaseModel):
    image_path_or_url: str
    prompt: Optional[str] = "Describe this image in detail."
    model: Optional[str] = "gpt-4o"
    apikey: str
    email: str

class TTSRequest(BaseModel):
    text: str
    use_gpt: Optional[bool] = False
    model: Optional[str] = "gpt-4.1-nano"
    language: Optional[str] = "en"
    apikey: str
    email: str

class FileRequest(BaseModel):
    prompt: Optional[str] = "Analyze this file and provide insights."
    model: Optional[str] = "gpt-4.1-nano"
    apikey: str
    email: str

# ---------------- Helper: Verify API key and increment usage ----------------
# ---------------- Helper: Verify API key and increment usage ----------------
async def verify_apikey(email: str, apikey: str) -> bool:
    """
    Check if apikey exists for the given email in Supabase.
    If valid, check API call limit and increment `apicalls` for that specific API key.
    """
    try:
        # Search for the API key with matching email
        response = supabase.table('users')\
            .select('*')\
            .eq('email', email)\
            .eq('apikey', apikey)\
            .execute()
        
        # Check if any matching record found
        if not response.data or len(response.data) == 0:
            print(f"❌ No matching API key found for email: {email}")
            return False
        
        # Get the first matching record
        record = response.data[0]
        current_calls = record.get('apicalls', 0)
        
        # Check if user has exceeded the limit
        if current_calls >= 500:
            print(f"⚠️ API call limit exceeded for {email}. Current calls: {current_calls}")
            raise HTTPException(
                status_code=429, 
                detail="API call limit exceeded (500 calls). Please contact the web admin at support@example.com for more API calls."
            )
        
        # Increment API calls using email and apikey as identifiers
        update_response = supabase.table('users')\
            .update({'apicalls': current_calls + 1})\
            .eq('email', email)\
            .eq('apikey', apikey)\
            .execute()
        
        if update_response.data:
            print(f"✅ API call incremented for {email}. Total calls: {current_calls + 1}/500")
            return True
        else:
            print(f"⚠️ Failed to increment API calls for {email}")
            return False
            
    except HTTPException:
        # Re-raise HTTP exceptions (like the 429 limit exceeded)
        raise
    except Exception as e:
        print(f"❌ Error verifying API key: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
# ---------------- AI Chat Helpers ----------------
async def _ai_chat_with_timeout(messages: List[dict], options: dict, timeout: int = 200):
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

async def stream_chat_response(messages: List[dict], options: dict) -> AsyncGenerator[str, None]:
    try:
        response = await _ai_chat_with_timeout(messages, options)
        async for chunk, _ in response:
            if chunk.strip():
                yield chunk
    except Exception as e:
        yield f"Error streaming response: {str(e)}"

# ---------------- TTS Helper ----------------
def generate_tts(text: str, language: str = "en") -> bytes:
    """
    Generate TTS audio using Google Translate TTS API.
    """
    try:
        encoded_text = urllib.parse.quote(text)
        tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_text}&tl={language}&client=tw-ob"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(tts_url, headers=headers, timeout=10)
        response.raise_for_status()
        if len(response.content) < 100:
            raise HTTPException(status_code=500, detail="Invalid TTS audio data")
        return response.content
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")

# ---------------- File Handling Helper ----------------
async def process_file_content(file: UploadFile, max_size_mb: int = 5) -> str:
    """
    Read and encode file content. Supports text, JSON, and PDF; rejects oversized files.
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    content_type = file.content_type

    try:
        # Read file content asynchronously
        content = await file.read()
        file_size = len(content)
        if file_size > max_size_bytes:
            raise HTTPException(status_code=400, detail=f"File size exceeds {max_size_mb}MB limit")

        # Handle based on content type
        if content_type in ["text/plain", "application/json"]:
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
        elif content_type == "application/pdf":
            try:
                with pdfplumber.open(BytesIO(content)) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    if not text.strip():
                        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
                    return text
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"PDF processing error: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use text, JSON, or PDF")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")
    finally:
        await file.close()

# ---------------- Endpoints ----------------

@app.get("/chat")
@limiter.limit("10/minute")
async def chat_get(
    request: Request,
    question: str = Query(..., description="The question to ask"),
    model: str = Query("gpt-4.1-nano", description="AI model to use"),
    stream: bool = Query(False, description="Enable streaming response"),
    apikey: str = Query(..., description="API key of the user"),
    email: str = Query(..., description="Email of the user")
):
    if not await verify_apikey(email, apikey):
        raise HTTPException(status_code=401, detail="Invalid API key or email")

    messages = [{"role": "user", "content": question}]
    options = {"model": model, "strict_model": True, "temperature": 1, "stream": stream}
    
    if stream:
        return StreamingResponse(
            stream_chat_response(messages, options),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    response = await _ai_chat_with_timeout(messages, options)
    content = response["response"]["result"]["message"]["content"]
    used_model = response["used_model"]
    return {"answer": content, "model_used": used_model}

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_post(request: Request, chat_request: ChatRequest):
    if not await verify_apikey(chat_request.email, chat_request.apikey):
        raise HTTPException(status_code=401, detail="Invalid API key or email")

    return await chat_get(
        request,
        chat_request.question,
        chat_request.model,
        chat_request.stream,
        chat_request.apikey,
        chat_request.email
    )

@app.post("/recognize-image")
@limiter.limit("5/minute")
async def recognize_image(request: Request, image_request: ImageRequest):
    if not await verify_apikey(image_request.email, image_request.apikey):
        raise HTTPException(status_code=401, detail="Invalid API key or email")

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

@app.post("/tts")
@limiter.limit("10/minute")
async def tts(request: Request, tts_request: TTSRequest):
    if not await verify_apikey(tts_request.email, tts_request.apikey):
        raise HTTPException(status_code=401, detail="Invalid API key or email")

    try:
        text_to_speak = tts_request.text
        if tts_request.use_gpt:
            messages = [{"role": "user", "content": tts_request.text}]
            options = {"model": tts_request.model, "strict_model": True, "temperature": 1}
            response = await _ai_chat_with_timeout(messages, options)
            text_to_speak = response["response"]["result"]["message"]["content"]

        audio_bytes = generate_tts(text_to_speak, tts_request.language)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.post("/file-analyze")
@limiter.limit("5/minute")
async def file_analyze(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form("Analyze this file and provide insights."),
    model: str = Form("gpt-4.1-nano"),
    apikey: str = Form(...),
    email: str = Form(...)
):
    if not await verify_apikey(email, apikey):
        raise HTTPException(status_code=401, detail="Invalid API key or email")

    try:
        file_content = await process_file_content(file)
        messages = [{"role": "user", "content": f"{prompt}\n\nFile content:\n{file_content}"}]
        options = {"model": model, "strict_model": True, "temperature": 1}

        response = await _ai_chat_with_timeout(messages, options)
        content = response["response"]["result"]["message"]["content"]
        used_model = response["used_model"]
        return {"analysis": content, "model_used": used_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File analysis error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
