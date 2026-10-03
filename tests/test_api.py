"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
import torch
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def generate_test_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory WAV audio bytes for testing.

    Args:
        duration_seconds: Audio length in seconds.
        sample_rate: Sampling frequency.

    Returns:
        bytes: Binary WAV data.
    """
    num_samples = int(duration_seconds * sample_rate)
    samples = torch.randn(num_samples).numpy()
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


def test_root_info() -> None:
    """Test that the API root endpoint returns metadata and frontend link."""
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "Voice-Based Regional Accent Recommender" in data["app"]
    assert "frontend" in data
    print(" PASS: test_root_info passed! Root API endpoint verified.")


def test_health_check() -> None:
    """Test the /api/v1/health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["catalog_items_indexed"] >= 5
    print(" PASS: test_health_check passed:", data)


def test_voice_recommendation_food() -> None:
    """Test voice endpoint routing to ChromaDB for food intent."""
    wav_bytes = generate_test_wav()
    files = {"file": ("test_speech.wav", wav_bytes, "audio/wav")}
    data = {
        "query_hint": "I want to eat spicy Hyderabadi biryani and traditional snacks.",
        "region_hint": "telangana",
        "top_k": "2",
    }

    response = client.post("/api/v1/recommend-by-voice", files=files, data=data)
    assert response.status_code == 200, f"Expected 200, got {response.text}"
    body = response.json()

    assert body["status"] == "success"
    assert body["detected_accent"]["region"] == "telangana"
    assert body["intent_analysis"]["top_intent"] == "food"
    assert body["routing_destination"] == "chromadb_vector_store"
    assert body["recommendations_count"] > 0
    assert any("Biryani" in r["title"] or "Sarva Pindi" in r["title"] for r in body["recommendations"])
    print(" PASS: test_voice_recommendation_food passed!")
    print(f"   Destination: {body['routing_destination']} | Intent: {body['intent_analysis']['top_intent']}")
    for r in body["recommendations"]:
        print(f"   -> [{r['category'].upper()}] {r['title']} (Location: {r['location']})")


def test_voice_recommendation_news() -> None:
    """Test voice endpoint routing to Live RSS for news intent."""
    wav_bytes = generate_test_wav()
    files = {"file": ("test_speech.wav", wav_bytes, "audio/wav")}
    data = {
        "query_hint": "What are the latest news headlines and politics today?",
        "region_hint": "telangana",
        "top_k": "2",
    }

    response = client.post("/api/v1/recommend-by-voice", files=files, data=data)
    assert response.status_code == 200, f"Expected 200, got {response.text}"
    body = response.json()

    assert body["status"] == "success"
    assert body["detected_accent"]["region"] == "telangana"
    assert body["intent_analysis"]["top_intent"] == "news"
    assert body["routing_destination"] == "rss_news_feed"
    assert body["recommendations_count"] > 0
    print(" PASS: test_voice_recommendation_news passed!")
    print(f"   Destination: {body['routing_destination']} | Intent: {body['intent_analysis']['top_intent']}")
    for r in body["recommendations"]:
        print(f"   -> [NEWS] {r['title']} (Source: {r['source']})")


def test_voice_recommendation_fashion() -> None:
    """Test voice endpoint routing to ChromaDB for fashion intent."""
    wav_bytes = generate_test_wav()
    files = {"file": ("test_speech.wav", wav_bytes, "audio/wav")}
    data = {
        "query_hint": "I want to buy Pochampally silk and handloom sarees.",
        "region_hint": "telangana",
        "top_k": "1",
    }

    response = client.post("/api/v1/recommend-by-voice", files=files, data=data)
    assert response.status_code == 200, f"Expected 200, got {response.text}"
    body = response.json()

    assert body["status"] == "success"
    assert body["intent_analysis"]["top_intent"] == "fashion"
    assert body["routing_destination"] == "chromadb_vector_store"
    assert len(body["recommendations"]) == 1
    assert "Pochampally" in body["recommendations"][0]["title"]
    print(" PASS: test_voice_recommendation_fashion passed!")
    print(f"   -> [FASHION] {body['recommendations'][0]['title']}")


def test_corrupted_audio_error_handling() -> None:
    """Test that uploading corrupted audio bytes returns 400 Bad Request."""
    corrupted_bytes = b"CORRUPTED_NON_AUDIO_DATA_TESTING_ERROR_HANDLING"
    files = {"file": ("corrupt.wav", corrupted_bytes, "audio/wav")}

    response = client.post("/api/v1/recommend-by-voice", files=files)
    assert response.status_code == 400, f"Expected 400 for corrupted audio, got {response.status_code}"
    print(" PASS: test_corrupted_audio_error_handling passed! Correctly returned HTTP 400.")


def test_multi_region_expansion() -> None:
    """Test recommendations across Tamil Nadu, Karnataka, Punjab, and Bengal."""
    regions_to_test = [
        {
            "region": "tamil_nadu",
            "query": "I am looking for traditional Kanchipuram mulberry silk wedding sarees.",
            "expected_keyword": "Kanchipuram",
            "expected_category": "fashion",
        },
        {
            "region": "karnataka",
            "query": "Where can I eat crisp Davangere benne dosa with potato palya?",
            "expected_keyword": "Benne Dosa",
            "expected_category": "food",
        },
        {
            "region": "punjab",
            "query": "Tell me about visiting the Golden Temple and Harmandir Sahib in Amritsar.",
            "expected_keyword": "Golden Temple",
            "expected_category": "travel",
        },
        {
            "region": "bengal",
            "query": "Where can I find tender Shorshe Ilish fish and authentic sweets in Kolkata?",
            "expected_keyword": "Shorshe Ilish",
            "expected_category": "food",
        },
    ]

    for item in regions_to_test:
        payload = {
            "query": item["query"],
            "region": item["region"],
            "top_k": 2,
        }
        response = client.post("/api/v1/recommend-by-text", json=payload)
        assert response.status_code == 200, f"Failed for {item['region']}: {response.text}"
        data = response.json()

        assert data["status"] == "success"
        assert data["detected_accent"]["region"] == item["region"]
        assert len(data["recommendations"]) > 0
        match_found = any(item["expected_keyword"].lower() in rec["title"].lower() for rec in data["recommendations"])
        assert match_found, f"Expected '{item['expected_keyword']}' in recommendations for {item['region']}, got {[r['title'] for r in data['recommendations']]}"
        print(f" PASS: Multi-Region [{item['region'].upper()}] retrieved {data['recommendations'][0]['title']}")


def test_acoustic_checkpoint_loading_and_inference() -> None:
    """Test Wav2Vec2 checkpoint integration, serialization, and neural inference."""
    from models.accent_classifier import get_accent_classifier

    classifier = get_accent_classifier()
    checkpoint_test_path = "checkpoints/wav2vec2_accent_finetuned.pt"

    # Ensure checkpoint file exists or create it
    classifier.save_checkpoint(checkpoint_test_path)
    loaded = classifier.load_checkpoint(checkpoint_test_path)
    assert loaded is True, "Failed to load fine-tuned checkpoint"
    assert classifier.is_checkpoint_loaded is True, "Expected is_checkpoint_loaded to be True"

    # Test tensor inference
    test_waveform = torch.randn(1, 16000)
    result = classifier.predict(test_waveform, sample_rate=16000)
    assert result["checkpoint_loaded"] is True
    assert "distribution" in result
    assert result["region"] in classifier.SUPPORTED_REGIONS
    print(f" PASS: test_acoustic_checkpoint_loading_and_inference passed! Detected: {result['region']} (Device: {result['device']})")

    # Test through voice endpoint
    wav_bytes = generate_test_wav()
    files = {"file": ("test_speech.wav", wav_bytes, "audio/wav")}
    data = {
        "query_hint": "Authentic regional food options",
        "top_k": "2",
    }
    response = client.post("/api/v1/recommend-by-voice", files=files, data=data)
    assert response.status_code == 200
    body = response.json()
    assert body["detected_accent"]["checkpoint_loaded"] is True
    print(" PASS: Voice endpoint successfully reported active acoustic checkpoint!")


if __name__ == "__main__":
    print("\n================== RUNNING INTEGRATION TESTS ==================")
    test_root_info()
    test_health_check()
    test_acoustic_checkpoint_loading_and_inference()
    test_voice_recommendation_food()
    test_voice_recommendation_news()
    test_voice_recommendation_fashion()
    test_corrupted_audio_error_handling()
    test_multi_region_expansion()
    print("================== ALL TESTS PASSED SUCCESSFULLY! ==================\n")
