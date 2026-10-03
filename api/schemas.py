"""Pydantic schemas for the Voice-Based Regional Recommender API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AccentInfo(BaseModel):
    """Acoustic regional accent classification details."""

    region: str = Field(..., description="Predicted geographic region (e.g. 'telangana').")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")
    detected_accent: str = Field(..., description="Descriptive accent label.")
    sample_rate: int | None = Field(None, description="Audio sampling rate in Hz.")
    duration_seconds: float | None = Field(None, description="Duration of processed audio.")
    device: str | None = Field(None, description="Hardware compute device used.")
    checkpoint_loaded: bool | None = Field(None, description="Whether fine-tuned acoustic weights were used.")
    checkpoint_path: str | None = Field(None, description="Path to active checkpoint weights.")
    distribution: dict[str, float] | None = Field(None, description="Confidence distribution across regional accents.")


class TranscriptionInfo(BaseModel):
    """Speech-to-text ASR transcription details."""

    text: str = Field(..., description="Transcribed query text.")
    model: str = Field(..., description="ASR model used for transcription.")
    device: str | None = Field(None, description="Compute device used.")
    status: str = Field(..., description="Transcription execution status.")


class IntentInfo(BaseModel):
    """Zero-shot NLP intent analysis details."""

    top_intent: str = Field(..., description="Predicted category (food, fashion, travel, news, diaspora).")
    confidence: float = Field(..., description="Prediction confidence score.")
    scores: dict[str, float] = Field(default_factory=dict, description="Probabilities across all candidate categories.")
    engine: str = Field(..., description="Inference engine or model identifier.")


class RecommendationItem(BaseModel):
    """Individual localized item or news recommendation."""

    id: str = Field(..., description="Unique item identifier.")
    title: str = Field(..., description="Title of the recommendation or headline.")
    region: str = Field(..., description="Associated geographic region.")
    category: str = Field(..., description="Item category (food, fashion, travel, news, diaspora).")
    description: str | None = Field(None, description="Item description or article summary.")
    location: str | None = Field(None, description="Physical location or city.")
    tags: list[str] = Field(default_factory=list, description="Descriptive thematic tags.")
    highlights: str | None = Field(None, description="Key features or cultural significance.")
    confidence: float | None = Field(None, description="Relevance or retrieval confidence score.")
    source: str | None = Field(None, description="Source publisher or catalog provider.")
    link: str | None = Field(None, description="External URL if news item.")
    published: str | None = Field(None, description="Publication timestamp for news.")

    @field_validator("tags", mode="before")
    @classmethod
    def sanitize_tags(cls, val: list[str] | None) -> list[str]:
        """Ensure tags is always a list even if None is supplied."""
        if val is None:
            return []
        return val


class RecommendationResponse(BaseModel):
    """Structured response payload for voice-driven and text-driven recommendations."""

    status: str = Field("success", description="API response status ('success' or 'error').")
    detected_accent: AccentInfo
    transcription: TranscriptionInfo
    intent_analysis: IntentInfo
    routing_destination: str = Field(
        ...,
        description="Destination subsystem ('chromadb_vector_store' or 'rss_news_feed').",
    )
    recommendations_count: int = Field(..., description="Number of items returned.")
    recommendations: list[RecommendationItem] = Field(
        default_factory=list,
        description="Ranked localized recommendations.",
    )


# Alias for backwards compatibility
VoiceRecommendationResponse = RecommendationResponse


class TextRecommendationRequest(BaseModel):
    """Request payload for direct text queries."""

    query: str = Field(..., min_length=2, description="User search text.")
    region: str | None = Field("telangana", description="Target region.")
    category: str | None = Field(None, description="Optional category override.")
    top_k: int = Field(3, ge=1, le=10, description="Max recommendations to return.")


class HealthCheckResponse(BaseModel):
    """Health check status payload."""

    status: str = Field(..., description="System health status ('healthy' or 'degraded').")
    cuda_available: bool
    torch_device: str
    catalog_items_indexed: int
    modules: dict[str, str]
