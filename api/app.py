"""FastAPI application entry point for the Voice-Based Regional Accent Recommender."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from models.accent_classifier import get_accent_classifier, resolve_torch_device
from recommender.vector_store import get_vector_store

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown procedures."""
    logger.info("Starting up Voice-Based Regional Accent Recommender MVP...")
    device = resolve_torch_device()
    logger.info("Torch compute device: %s", device)

    # Initialize acoustic classifier with fine-tuned checkpoint if present
    default_checkpoint = Path("checkpoints/wav2vec2_accent_finetuned.pt")
    default_hf_dir = Path("checkpoints/wav2vec2_accent_hf")
    active_ckpt = str(default_checkpoint) if default_checkpoint.is_file() else (str(default_hf_dir) if default_hf_dir.is_dir() else None)
    classifier = get_accent_classifier(checkpoint_path=active_ckpt)
    logger.info("Acoustic classifier loaded: %s (checkpoint: %s)", classifier.is_checkpoint_loaded, classifier.checkpoint_path)

    # Initialize and seed ChromaDB vector store
    try:
        store = get_vector_store()
        loaded = store.load_catalog(catalog_path="data/regional_catalog.json", force_reload=False)
        logger.info("ChromaDB vector store ready with %d items.", loaded)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to seed ChromaDB on startup: %s", exc)

    yield

    logger.info("Shutting down Voice-Based Regional Accent Recommender MVP.")


app = FastAPI(
    title="Voice-Based Regional Accent Recommender MVP",
    description=(
        "Ingests user voice audio, infers Indian regional accents (Wav2Vec2), "
        "transcribes user intent (Whisper), and delivers hyper-localized "
        "recommendations across Food, Fashion, Travel, Diaspora (via ChromaDB), "
        "and News (via Live RSS)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)


@app.get("/", tags=["Root"])
async def root() -> JSONResponse:
    """Root metadata endpoint with system information and links."""
    return JSONResponse(
        content={
            "app": "Voice-Based Regional Accent Recommender API",
            "version": "1.0.0",
            "frontend": "http://localhost:3000",
            "docs": "/docs",
            "health": "/api/v1/health",
            "endpoints": {
                "voice_recommendation": "POST /api/v1/recommend-by-voice",
                "text_recommendation": "POST /api/v1/recommend-by-text",
            },
        }
    )


@app.get("/api/info", tags=["Root"])
async def api_info() -> JSONResponse:
    """Root metadata endpoint with system information and links."""
    return JSONResponse(
        content={
            "app": "Voice-Based Regional Accent Recommender MVP",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
            "endpoints": {
                "voice_recommendation": "POST /api/v1/recommend-by-voice",
                "text_recommendation": "POST /api/v1/recommend-by-text",
            },
        }
    )


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
