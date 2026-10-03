"""NLP zero-shot intent classifier for regional recommendations.

Classifies user utterance/query text into 5 core recommendation categories:
['food', 'fashion', 'travel', 'news', 'diaspora'].
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def resolve_torch_device() -> str:
    """Resolve the optimal compute device across CUDA, MPS, and CPU.

    Returns:
        str: Target device identifier string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ZeroShotIntentClassifier:
    """Zero-shot intent classifier powered by Hugging Face transformers."""

    DEFAULT_CANDIDATES: list[str] = [
        "food",
        "fashion",
        "travel",
        "news",
        "diaspora",
    ]

    # Keyword patterns for graceful fallback if model pipeline is unavailable
    FALLBACK_KEYWORDS: dict[str, list[str]] = {
        "food": ["food", "eat", "restaurant", "biryani", "breakfast", "curry", "snack", "sweet", "dinner", "lunch"],
        "fashion": ["fashion", "saree", "clothes", "silk", "handloom", "outfit", "wear", "dress", "textile", "ikat"],
        "travel": ["travel", "visit", "trip", "temple", "tour", "heritage", "monument", "hotel", "sightseeing"],
        "news": ["news", "headline", "current events", "politics", "breaking", "update", "paper", "feed"],
        "diaspora": ["diaspora", "community", "expat", "association", "telugu network", "nri", "cultural club", "group"],
    }

    def __init__(
        self,
        model_name: str = "valhalla/distilbart-mnli-12-3",
        candidate_labels: Optional[list[str]] = None,
        device: Optional[str] = None,
        lazy_load: bool = True,
    ) -> None:
        """Initialize the zero-shot classifier.

        Args:
            model_name: Hugging Face model repository ID. Default uses lightweight DistilBART MNLI.
            candidate_labels: List of intent labels to classify against.
            device: Explicit device string. Defaults to auto-resolved device.
            lazy_load: If True, delays pipeline loading until first inference call.
        """
        self.model_name = model_name
        self.candidate_labels = candidate_labels or list(self.DEFAULT_CANDIDATES)
        self.device = device or resolve_torch_device()
        self._pipeline: Optional[Any] = None

        if not lazy_load:
            self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Instantiate the Hugging Face zero-shot classification pipeline."""
        if self._pipeline is not None:
            return

        logger.info(
            "Loading zero-shot classification pipeline '%s' on %s...",
            self.model_name,
            self.device,
        )
        try:
            self._pipeline = pipeline(
                task="zero-shot-classification",
                model=self.model_name,
                device=self.device,
            )
            logger.info("Successfully loaded zero-shot pipeline on %s.", self.device)
        except Exception as exc:
            logger.error(
                "Failed to initialize Hugging Face pipeline (%s). Falling back to rule-based parser.",
                exc,
            )
            self._pipeline = None

    def _fallback_classification(self, text: str) -> dict[str, Any]:
        """Rule-based heuristic fallback if neural model is inaccessible.

        Args:
            text: Input text string.

        Returns:
            dict[str, Any]: Classification prediction with scored categories.
        """
        lowered = text.lower()
        scored: dict[str, float] = {}

        for category, keywords in self.FALLBACK_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in lowered)
            scored[category] = float(matches)

        total_matches = sum(scored.values())
        if total_matches > 0:
            normalized_scores = {k: round(v / total_matches, 4) for k, v in scored.items()}
            top_category = max(normalized_scores, key=lambda k: normalized_scores[k])
            confidence = normalized_scores[top_category]
        else:
            # Default uniform baseline
            baseline = round(1.0 / len(self.candidate_labels), 4)
            normalized_scores = {label: baseline for label in self.candidate_labels}
            top_category = self.candidate_labels[0]
            confidence = baseline

        return {
            "top_intent": top_category,
            "confidence": confidence,
            "scores": normalized_scores,
            "raw_text": text,
            "engine": "keyword_fallback",
        }

    def classify_intent(self, text: str) -> dict[str, Any]:
        """Classify user text into one of the target intent categories.

        Args:
            text: Transcribed speech or user prompt text.

        Returns:
            dict[str, Any]: Top intent label, confidence, and distribution scores.

        Raises:
            ValueError: If input text is empty.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Input text for intent classification cannot be empty.")

        self._load_pipeline()

        if self._pipeline is None:
            return self._fallback_classification(clean_text)

        try:
            result = self._pipeline(
                clean_text,
                candidate_labels=self.candidate_labels,
                hypothesis_template="This user request is about {}.",
            )
            labels: list[str] = result.get("labels", [])
            scores: list[float] = result.get("scores", [])

            scores_map = {label: round(float(score), 4) for label, score in zip(labels, scores)}
            top_intent = labels[0] if labels else "food"
            top_confidence = round(scores[0], 4) if scores else 0.5

            return {
                "top_intent": top_intent,
                "confidence": top_confidence,
                "scores": scores_map,
                "raw_text": clean_text,
                "engine": f"zero_shot ({self.model_name})",
                "device": self.device,
            }
        except Exception as exc:
            logger.warning("Pipeline inference failed with %s; invoking fallback.", exc)
            return self._fallback_classification(clean_text)


# Module-level singleton
_intent_classifier_instance: Optional[ZeroShotIntentClassifier] = None


def get_intent_classifier() -> ZeroShotIntentClassifier:
    """Singleton getter for the intent classifier.

    Returns:
        ZeroShotIntentClassifier: Initialized classifier instance.
    """
    global _intent_classifier_instance
    if _intent_classifier_instance is None:
        _intent_classifier_instance = ZeroShotIntentClassifier()
    return _intent_classifier_instance


def classify_user_intent(text: str) -> dict[str, Any]:
    """Classify user query into food, fashion, travel, news, or diaspora.

    Args:
        text: Query text string.

    Returns:
        dict[str, Any]: Structured intent prediction with scores.
    """
    classifier = get_intent_classifier()
    return classifier.classify_intent(text)


if __name__ == "__main__":
    print(f"Device: {resolve_torch_device()}")
    test_queries = [
        "Where can I eat authentic spicy Hyderabadi biryani?",
        "I am looking for traditional Pochampally silk sarees for a wedding.",
        "Tell me about historical Kakatiya temples to visit in Warangal.",
        "What are the latest political and economic headlines today?",
        "Are there any Telangana Telugu community associations in Dallas or California?",
    ]

    for q in test_queries:
        res = classify_user_intent(q)
        print(f"Query: '{q}' => Top Intent: {res['top_intent'].upper()} (Confidence: {res['confidence']}) [Engine: {res['engine']}]")
