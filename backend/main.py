# main.py
# A simple FastAPI backend that exposes a single POST endpoint (/analyze)
# which calls the Hugging Face Inference API to classify a piece of text
# as POSITIVE or NEGATIVE, along with a confidence score.
#
# NOTE: We deliberately call the Hugging Face *hosted* Inference API over
# HTTP instead of loading the model locally with torch/transformers. This
# keeps the backend lightweight (fast builds, low memory) so it comfortably
# fits within Render's free-tier 512MB RAM limit.

import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load variables from a local .env file (if present) into the environment.
# This has no effect on Render/production, where environment variables are
# set directly in the dashboard instead of via a .env file.
load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Sentiment Analyzer API")

# CORS configuration
# ------------------
# The frontend (deployed on Vercel) and the backend (deployed on Render)
# live on different domains, so we need CORS enabled.
#
# You can set the ALLOWED_ORIGINS environment variable on Render to a
# comma-separated list of allowed origins, e.g.:
#   ALLOWED_ORIGINS=https://your-frontend.vercel.app,http://localhost:5173
#
# If ALLOWED_ORIGINS is not set, we fall back to allowing all origins ("*"),
# which is fine for local development and quick testing.
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")

if allowed_origins_env.strip() == "*":
    allow_origins = ["*"]
else:
    # Split on commas and strip whitespace around each origin.
    allow_origins = [origin.strip() for origin in allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Hugging Face Inference API configuration
# ---------------------------------------------------------------------------
# We call Hugging Face's hosted model instead of loading it locally.
# You need a free Hugging Face account and an Access Token:
#   1. Sign up at https://huggingface.co/join
#   2. Create a token at https://huggingface.co/settings/tokens (Read access is enough)
#   3. Set it as the HF_API_TOKEN environment variable (locally in .env,
#      and on Render under Environment Variables)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    text: str


class AnalyzeResponse(BaseModel):
    """Response body for the /analyze endpoint."""
    sentiment: str
    confidence: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    """Simple health-check endpoint.

    Useful for confirming the backend is alive (e.g. after deploying to
    Render, or when the frontend shows a "backend unavailable" error and
    you want to check things manually in the browser).
    """
    return {"status": "ok", "message": "Sentiment Analyzer API is running."}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_sentiment(request: AnalyzeRequest):
    """Analyze the sentiment of the given text.

    Accepts a JSON body like: {"text": "I love this!"}
    Returns a JSON body like: {"sentiment": "Positive", "confidence": 0.999}

    Internally, this calls the Hugging Face Inference API rather than
    running the model locally.
    """
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    if not HF_API_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Server is missing the HF_API_TOKEN environment variable.",
        )

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=30,
        )
    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the Hugging Face Inference API. Please try again.",
        )

    # The model may still be "warming up" on Hugging Face's servers the
    # first time it's called (cold start). It responds with a 503 and an
    # "estimated_time" while it loads.
    if response.status_code == 503:
        raise HTTPException(
            status_code=503,
            detail="The model is warming up on Hugging Face's servers. Please try again in a few seconds.",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Hugging Face API returned an error (status {response.status_code}).",
        )

    data = response.json()

    # The API returns a nested list, e.g.:
    # [[{"label": "POSITIVE", "score": 0.9998}, {"label": "NEGATIVE", "score": 0.0002}]]
    try:
        scores = data[0]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Unexpected response format from Hugging Face API.",
        )

    # Pick the label with the highest confidence score.
    best = max(scores, key=lambda item: item["score"])

    label = best["label"]  # "POSITIVE" or "NEGATIVE"
    score = float(best["score"])

    # Convert Hugging Face's "POSITIVE"/"NEGATIVE" into a nicer display format.
    sentiment = "Positive" if label == "POSITIVE" else "Negative"

    return AnalyzeResponse(sentiment=sentiment, confidence=round(score, 4))
