# SwaraRegional Frontend (Next.js & Material UI)

Modern, interactive Web UI for the **Voice-Based Regional Accent Recommender MVP**, built with **Next.js 16**, **React 19**, and **Material UI (MUI)**.

## Features
- **Voice Studio:** Live microphone capture with in-browser 16kHz PCM WAV encoder, pulsating FAB, audio playback, file upload, and query guidance hint.
- **Text Studio:** Manual query input with regional targeting (Telangana, Tamil Nadu, Karnataka, Punjab, West Bengal), domain category filters, and Top-K result selector.
- **Processing Pipeline Stepper:** 5-stage architectural stepper tracking execution from Audio Ingestion through Wav2Vec2, Whisper, DistilBART, and ChromaDB/RSS routing.
- **Diagnostic Inference Inspector:** Real-time breakdown of detected regional accent, zero-shot intent, Whisper speech transcript, and active routing subsystem.
- **Responsive Recommendations Grid:** Dynamic cards displaying cultural tags, location badges, match confidence percentages, and direct news links.

## Quickstart

```bash
# 1. Install dependencies
npm install

# 2. Run development server (runs on port 3000)
npm run dev -- -p 3000

# 3. Build for production
npm run build
npm start
```

Ensure the FastAPI backend is running on `http://127.0.0.1:8000` (or configure `NEXT_PUBLIC_API_URL`).
