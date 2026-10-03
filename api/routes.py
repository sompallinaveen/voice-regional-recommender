"""FastAPI route handlers for voice and text regional recommendations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from api.schemas import (
    AccentInfo,
    HealthCheckResponse,
    IntentInfo,
    RecommendationItem,
    RecommendationResponse,
    TextRecommendationRequest,
    TranscriptionInfo,
)
from models.accent_classifier import get_accent_classifier, resolve_torch_device
from models.nlp_intent import classify_user_intent
from models.transcription import transcribe_audio
from recommender.news_feed import fetch_top_headlines
from recommender.vector_store import get_vector_store, retrieve_recommendations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/api/v1", tags=["Voice Recommender"])


@router.post(
    "/recommend-by-voice",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate regional recommendations from uploaded speech audio",
)
async def recommend_by_voice(
    file: Annotated[
        UploadFile,
        File(description="Audio file payload (WAV, MP3, FLAC, OGG)."),
    ],
    query_hint: Annotated[
        str | None,
        Form(description="Optional text override or prompt hint for testing or silent audio."),
    ] = None,
    region_hint: Annotated[
        str | None,
        Form(description="Optional regional hint/target (e.g. 'tamil_nadu', 'karnataka', 'punjab', 'bengal')."),
    ] = None,
    top_k: Annotated[
        int,
        Form(ge=1, le=10, description="Max recommendations to retrieve."),
    ] = 3,
) -> RecommendationResponse:
    """Ingest speech audio, detect regional accent, transcribe intent, and route recommendations.

    Args:
        file: Uploaded audio file.
        query_hint: Optional text override for synthetic audio testing.
        region_hint: Optional regional hint for multi-region testing.
        top_k: Number of recommendations to return.

    Returns:
        RecommendationResponse: Localized items from ChromaDB or live RSS news.

    Raises:
        HTTPException: On corrupted audio, unhandled file types, or pipeline failure.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing or empty.",
        )

    try:
        audio_bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read audio file upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded audio: {exc}",
        ) from exc

    if not audio_bytes or len(audio_bytes) < 44:  # Minimum valid audio header
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty or contains insufficient data.",
        )

    # 1. Acoustic Accent Classification
    accent_classifier = get_accent_classifier()
    try:
        accent_result = accent_classifier.predict_from_audio_bytes(
            audio_bytes=audio_bytes,
            target_region=region_hint,
        )
        detected_region: str = str(accent_result.get("region", "telangana")).lower()
    except ValueError as exc:
        logger.warning("Acoustic accent inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or unsupported audio format: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected accent classifier failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Accent classifier processing error: {exc}",
        ) from exc

    # 2. Speech-to-Text Transcription
    try:
        transcription_result = transcribe_audio(
            audio_bytes=audio_bytes,
            query_hint=query_hint,
        )
        query_text: str = str(transcription_result.get("text", "")).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Transcription failure (%s); falling back to default inquiry.", exc)
        query_text = query_hint or "Where can I find authentic local food and handloom clothing?"
        transcription_result = {
            "text": query_text,
            "model": "error_fallback",
            "device": resolve_torch_device(),
            "status": "fallback",
        }

    # 3. NLP Zero-Shot Intent Classification
    try:
        intent_result = classify_user_intent(query_text)
        top_intent: str = str(intent_result.get("top_intent", "food")).lower()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Intent classification failed (%s); defaulting to 'food'.", exc)
        top_intent = "food"
        intent_result = {
            "top_intent": "food",
            "confidence": 0.5,
            "scores": {"food": 1.0},
            "engine": "fallback",
        }

    # 4. Routing Decision Engine
    recommendation_items: list[RecommendationItem] = []
    routing_destination: str

    if top_intent == "news":
        routing_destination = "rss_news_feed"
        try:
            articles = fetch_top_headlines(region=detected_region, top_k=top_k)
            for art in articles:
                recommendation_items.append(
                    RecommendationItem(
                        id=str(art.get("id", "")),
                        title=str(art.get("title", "")),
                        region=str(art.get("region", detected_region)),
                        category="news",
                        description=str(art.get("summary", "")),
                        source=str(art.get("source", "Regional RSS")),
                        link=str(art.get("link", "")),
                        published=str(art.get("published", "")),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed fetching news in route handler: %s", exc)
    else:
        routing_destination = "chromadb_vector_store"
        try:
            items = retrieve_recommendations(
                query=query_text,
                region=detected_region,
                category=top_intent,
                top_k=top_k,
            )
            for item in items:
                recommendation_items.append(
                    RecommendationItem(
                        id=str(item.get("id", "")),
                        title=str(item.get("title", "")),
                        region=str(item.get("region", detected_region)),
                        category=str(item.get("category", top_intent)),
                        description=str(item.get("description", "")),
                        location=str(item.get("location", "")),
                        tags=item.get("tags") or [],
                        highlights=str(item.get("highlights", "")),
                        confidence=item.get("confidence"),
                        source="ChromaDB Regional Catalog",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed querying ChromaDB in route handler: %s", exc)

    return RecommendationResponse(
        status="success",
        detected_accent=AccentInfo(
            region=detected_region,
            confidence=float(accent_result.get("confidence", 0.88)),
            detected_accent=str(accent_result.get("detected_accent", "Telangana Regional Accent")),
            sample_rate=accent_result.get("sample_rate"),
            duration_seconds=accent_result.get("duration_seconds"),
            device=str(accent_result.get("device", "cuda")),
            checkpoint_loaded=bool(accent_result.get("checkpoint_loaded", False)),
            checkpoint_path=accent_result.get("checkpoint_path"),
            distribution=accent_result.get("distribution"),
        ),
        transcription=TranscriptionInfo(
            text=query_text,
            model=str(transcription_result.get("model", "whisper")),
            device=str(transcription_result.get("device", "cuda")),
            status=str(transcription_result.get("status", "success")),
        ),
        intent_analysis=IntentInfo(
            top_intent=top_intent,
            confidence=float(intent_result.get("confidence", 0.9)),
            scores=intent_result.get("scores", {}),
            engine=str(intent_result.get("engine", "zero_shot")),
        ),
        routing_destination=routing_destination,
        recommendations_count=len(recommendation_items),
        recommendations=recommendation_items,
    )


@router.post(
    "/recommend-by-text",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate localized recommendations from direct text queries",
)
async def recommend_by_text(
    payload: TextRecommendationRequest,
) -> RecommendationResponse:
    """Analyze query text, classify intent, and route recommendations.

    Args:
        payload: Text request parameters including query and optional region.

    Returns:
        RecommendationResponse: Structured recommendation result.
    """
    region = (payload.region or "telangana").strip().lower()
    intent_result = classify_user_intent(payload.query)
    top_intent = (payload.category or intent_result.get("top_intent", "food")).lower()

    recommendation_items: list[RecommendationItem] = []
    routing_destination: str

    if top_intent == "news":
        routing_destination = "rss_news_feed"
        articles = fetch_top_headlines(region=region, top_k=payload.top_k)
        for art in articles:
            recommendation_items.append(
                RecommendationItem(
                    id=str(art.get("id", "")),
                    title=str(art.get("title", "")),
                    region=region,
                    category="news",
                    description=str(art.get("summary", "")),
                    source=str(art.get("source", "Regional RSS")),
                    link=str(art.get("link", "")),
                    published=str(art.get("published", "")),
                )
            )
    else:
        routing_destination = "chromadb_vector_store"
        items = retrieve_recommendations(
            query=payload.query,
            region=region,
            category=top_intent,
            top_k=payload.top_k,
        )
        for item in items:
            recommendation_items.append(
                RecommendationItem(
                    id=str(item.get("id", "")),
                    title=str(item.get("title", "")),
                    region=region,
                    category=str(item.get("category", top_intent)),
                    description=str(item.get("description", "")),
                    location=str(item.get("location", "")),
                    tags=item.get("tags") or [],
                    highlights=str(item.get("highlights", "")),
                    confidence=item.get("confidence"),
                    source="ChromaDB Regional Catalog",
                )
            )

    return RecommendationResponse(
        status="success",
        detected_accent=AccentInfo(
            region=region,
            confidence=1.0,
            detected_accent="Manual Text Query",
            sample_rate=None,
            duration_seconds=None,
            device="cpu",
        ),
        transcription=TranscriptionInfo(
            text=payload.query,
            model="text_input",
            device="cpu",
            status="success",
        ),
        intent_analysis=IntentInfo(
            top_intent=top_intent,
            confidence=float(intent_result.get("confidence", 0.9)),
            scores=intent_result.get("scores", {}),
            engine=str(intent_result.get("engine", "zero_shot")),
        ),
        routing_destination=routing_destination,
        recommendations_count=len(recommendation_items),
        recommendations=recommendation_items,
    )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check verifying model devices and database status",
)
async def health_check() -> HealthCheckResponse:
    """Verify runtime health, CUDA availability, and vector store document count.

    Returns:
        HealthCheckResponse: Detailed system readiness status.
    """
    store = get_vector_store()
    count = store.collection.count()
    device = resolve_torch_device()
    accent_clf = get_accent_classifier()
    accent_status = (
        f"Wav2Vec2 Fine-Tuned Checkpoint ({Path(accent_clf.checkpoint_path).name})"
        if (accent_clf.is_checkpoint_loaded and accent_clf.checkpoint_path)
        else "Wav2Vec2 Simulation / Resilient Mode"
    )

    return HealthCheckResponse(
        status="healthy",
        cuda_available=device == "cuda",
        torch_device=device,
        catalog_items_indexed=count,
        modules={
            "accent_classifier": accent_status,
            "intent_classifier": "ZeroShot (valhalla/distilbart-mnli-12-3)",
            "transcription": "Whisper (openai/whisper)",
            "vector_store": "ChromaDB Persistent (all-MiniLM-L6-v2)",
            "news_aggregator": "Feedparser Regional RSS",
        },
    )
