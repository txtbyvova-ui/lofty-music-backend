import io
import os
import time
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

RUPLICATE_TOKEN = "r8_eff5ac19c92c359556bc1120d159a7a4e15e6f9ad2eb0a89517f3c8e787b"
RUPLICATE_URL = "https://ruplicate.ru"
MODEL = "minimax/music-1.5"

HEADERS = {
    "Authorization": f"Token {RUPLICATE_TOKEN}",
    "Content-Type": "application/json",
}

GENRE_CONFIG = {
    "rock": {
        "prompt": "energetic rock song with electric guitars, heavy drums, distorted bass, powerful rhythm, male vocal",
        "lyrics": "[verse]\nГром гремит в моих венах\nОгонь горит сквозь ночь\nНичто меня не держит\nЯ рождён чтобы сражаться\n[chorus]\nРок-н-ролл не умрёт\nКричим в небеса\nКаждый удар каждый крик\nМы будем жить вечно\n",
    },
    "rap": {
        "prompt": "hip hop beat with deep 808 bass, trap hi-hats, dark synth pads, bouncy rhythm, male rap vocal",
        "lyrics": "[verse]\nСтроки летят до потолка\nДрайв и сила на века\nКаждое слово как лекарство\nДержу ритм и не сдаюсь\n[chorus]\nМы поднимаемся со дна\nНе остановимся никогда\nКаждый барьер мы сломаем\nЛегенды не забывают\n",
    },
    "pop": {
        "prompt": "catchy pop melody with bright synth, upbeat claps, cheerful chords, danceable groove, female vocal, 120 bpm",
        "lyrics": "[verse]\nПросыпаюсь с солнцем в окнах\nВсё вокруг как будто в сказке\nТанцую по улицам города\nПод самые сладкие звуки\n[chorus]\nЛа ла ла мы сияем\nКаждая звезда горит\nЛа ла ла мы летаем\nМузыка в сердце звучит\n",
    },
    "electronic": {
        "prompt": "electronic dance music with pulsing bassline, arpeggiated synth, four on the floor kick, energetic drop",
        "lyrics": "[verse]\nПульс нарастает волной\nЛазеры режут ночь\nБас пробивает пол\nОткрой цифровую дверь\n[chorus]\nМы электричество мы одно\nТанцуем под неоновым солнцем\nЧувствуй поток чувствуй ритм\nМузыка забирает нас\n",
    },
    "jazz": {
        "prompt": "smooth jazz with piano improvisation, walking double bass, brushed drums, tenor saxophone, warm atmosphere",
        "lyrics": "[verse]\nДым кружится в воздухе\nРояль играет сам\nПолночный блюз и золото\nВ этом ритме мой покой\n[chorus]\nДжаз течёт как река\nВ бархатном сне растворяюсь\nКаждая нота как поцелуй\nВ ночном блаженстве качаюсь\n",
    },
    "lofi": {
        "prompt": "lo-fi hip hop with mellow piano chords, vinyl crackle, tape hiss, slow lazy drums, relaxing mood",
        "lyrics": "[verse]\nДождь на окне туман в стекле\nВремя бежит слишком быстро\nКофе и старые книги\nТихий уголок покоя\n[chorus]\nОтпусти просто дыши\nПусть тишина кружится\nНичего не потеряно\nЛоу-фай звучит в душе\n",
    },
    "classical": {
        "prompt": "orchestral classical music with sweeping strings, grand piano, French horns, dramatic dynamics, epic feel",
        "lyrics": "[verse]\nСквозь мраморные залы\nЭхо забытых миров\nСтруны взлетают к свету\nТени уходят во тьму\n[chorus]\nВзлети симфония веков\nИстории и судьбы\nКаждый аккорд каждый звук\nКрасота без конца\n",
    },
    "metal": {
        "prompt": "heavy metal with blast beats, aggressive distorted guitars, double bass pedal, dark atmosphere, screaming vocal",
        "lyrics": "[verse]\nОгонь бушует в бездне\nКости трещат в ночи\nТьма поглощает свет\nМы падаем во мрак\n[chorus]\nСожги всё дотла\nПочувствуй ярость и боль\nМеталл в крови и венах\nРазрывая все цепи\n",
    },
    "ambient": {
        "prompt": "ambient atmospheric soundscape with reverb pads, gentle drones, ethereal textures, slow evolution, dreamy",
        "lyrics": "[verse]\nПлыву сквозь бесконечность\nВремя растворяется\nТихий гул под звёздами\nИсцеляя все раны\n[chorus]\nДыши тишиной\nНичто не давит\nДрейфую в космосе\nНаконец я свободен\n",
    },
    "indie": {
        "prompt": "indie folk with acoustic guitar fingerpicking, soft vocals, warm piano, gentle percussion, autumn feel",
        "lyrics": "[verse]\nИду по осенней дороге\nЛистья падают тихо\nНесу лёгкую ношу\nОтпускаю тревоги\n[chorus]\nМы молоды и свободны\nБежим среди деревьев\nКаждая песня как память\nИнди сердца и мелодии\n",
    },
}


class GenerateRequest(BaseModel):
    genre: str
    description: str = ""
    duration: float = 10.0


def create_prediction(prompt: str, lyrics: str) -> str:
    r = requests.post(
        f"{RUPLICATE_URL}/v1/models/{MODEL}/predictions",
        headers=HEADERS,
        json={
            "input": {
                "prompt": prompt,
                "lyrics": lyrics,
                "audio_format": "mp3",
            }
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ruplicate error: {r.text}")
    return r.json()["id"]


def poll_prediction(prediction_id: str, max_wait: int = 600) -> str:
    for _ in range(max_wait // 5):
        time.sleep(5)
        r = requests.get(
            f"{RUPLICATE_URL}/v1/predictions/{prediction_id}",
            headers=HEADERS,
            timeout=15,
        )
        data = r.json()
        status = data.get("status")

        if status == "succeeded":
            output = data.get("output")
            if output:
                return output
            raise HTTPException(status_code=500, detail="No output URL")

        if status in ("failed", "canceled"):
            error = data.get("error", "Unknown error")
            raise HTTPException(status_code=500, detail=f"Generation failed: {error}")

    raise HTTPException(status_code=504, detail="Generation timed out")


@app.post("/api/generate")
def generate(req: GenerateRequest):
    genre_key = req.genre.lower()
    config = GENRE_CONFIG.get(genre_key)
    if not config:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown genre: {req.genre}. Available: {list(GENRE_CONFIG.keys())}",
        )

    prompt = config["prompt"]
    lyrics = config["lyrics"]

    if req.description.strip():
        user_text = req.description.strip()
        # Если пользователь не указал теги структуры — оборачиваем
        if "[verse]" not in user_text.lower() and "[chorus]" not in user_text.lower():
            lyrics = f"[verse]\n{user_text}\n"
        else:
            lyrics = user_text

    try:
        prediction_id = create_prediction(prompt=prompt, lyrics=lyrics)

        audio_url = poll_prediction(prediction_id)

        audio_response = requests.get(audio_url, timeout=120)
        audio_response.raise_for_status()

        return StreamingResponse(
            io.BytesIO(audio_response.content),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=lofty-music.mp3"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "ruplicate-minimax-music"}


@app.get("/api/genres")
def genres():
    return {"genres": list(GENRE_CONFIG.keys())}
