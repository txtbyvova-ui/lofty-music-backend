import io
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Lofty Music API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/musicgen-small"
HF_TOKEN = os.environ.get("HF_TOKEN")

GENRE_PROMPTS = {
    "rock": "energetic rock song with electric guitars, heavy drums and bass",
    "rap": "hip hop beat with deep bass, trap drums and dark synth",
    "pop": "catchy pop melody with upbeat synth, claps and bright chords",
    "electronic": "electronic dance music with pulsing bass, arpeggiated synth and four on the floor beat",
    "jazz": "smooth jazz with piano, double bass, brushed drums and saxophone",
    "lofi": "lo-fi hip hop beat with mellow piano, vinyl crackle and slow drums",
    "classical": "orchestral classical music with strings, piano and dramatic dynamics",
}


class GenerateRequest(BaseModel):
    genre: str
    description: str = ""
    duration: float = 10.0


@app.post("/api/generate")
def generate(req: GenerateRequest):
    genre_key = req.genre.lower()
    base_prompt = GENRE_PROMPTS.get(genre_key)
    if not base_prompt:
        raise HTTPException(status_code=400, detail=f"Unknown genre: {req.genre}. Available: {list(GENRE_PROMPTS.keys())}")

    prompt = base_prompt
    if req.description.strip():
        prompt = f"{base_prompt}, {req.description.strip()}"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Model inference timed out (120s)")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace API error: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"HuggingFace API returned {response.status_code}: {response.text[:500]}")

    audio_bytes = response.content
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=lofty-music-generated.wav"},
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/genres")
def genres():
    return {"genres": list(GENRE_PROMPTS.keys())}
