# main.py
# A simple FastAPI backend that exposes a single POST endpoint (/analyze)
# which uses Hugging Face's pre-trained sentiment-analysis pipeline to
# classify a piece of text as POSITIVE or NEGATIVE, along with a confidence
# score.

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

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
# Load the Hugging Face sentiment-analysis pipeline once, at startup.
# ---------------------------------------------------------------------------
# We explicitly pick a small, well-known model so that:
#   1. The download size is manageable on Render's free tier.
#   2. The behavior is predictable (Hugging Face may change their "default"
#      model for the pipeline over time, so we pin it explicitly).
#
# This model outputs two possible labels: "POSITIVE" or "NEGATIVE".
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
)


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
    """
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    # Run the Hugging Face pipeline. It returns a list of dicts, e.g.:
    # [{"label": "POSITIVE", "score": 0.9998}]
    result = sentiment_pipeline(text)[0]

    label = result["label"]  # "POSITIVE" or "NEGATIVE"
    score = float(result["score"])  # confidence score between 0 and 1

    # Convert Hugging Face's "POSITIVE"/"NEGATIVE" into a nicer display format.
    sentiment = "Positive" if label == "POSITIVE" else "Negative"

    return AnalyzeResponse(sentiment=sentiment, confidence=round(score, 4))
