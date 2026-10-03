# Voice-Based Regional Accent Recommender MVP

A modular AI/ML pipeline and full-stack service that ingests user speech audio, identifies Indian regional accents (**Wav2Vec2**), transcribes user intent (**Whisper**), and delivers hyper-localized recommendations across **Food, Fashion, Travel, Diaspora** (via **ChromaDB** vector similarity) and **News** (via live regional **RSS feeds**). Includes a modern, real-time interactive browser Web UI with microphone recording, live audio waveform visualization, and multi-stage diagnostic inspection built with **Next.js** and **Material UI**.

---

## Architecture & Project Structure

```
NPL_Based_Recomndation/
├── api/
│   ├── __init__.py
│   ├── app.py                 # FastAPI application, lifespan seeding, CORS & root metadata
│   ├── routes.py              # /recommend-by-voice, /recommend-by-text, /health
│   └── schemas.py             # Pydantic v2 validation models
├── checkpoints/
│   ├── wav2vec2_accent_finetuned.pt  # Fine-tuned PyTorch state_dict & architecture config
│   └── wav2vec2_accent_hf/           # Hugging Face export directory (config.json, weights)
├── data/
│   └── regional_catalog.json  # 22+ curated items across Telangana, Tamil Nadu, Karnataka, Punjab, Bengal
├── models/
│   ├── __init__.py
│   ├── accent_classifier.py      # Wav2Vec2 XLSR-53 accent classifier with fine-tuned checkpoint loader
│   ├── nlp_intent.py             # Zero-shot classification (DistilBART MNLI with fallback)
│   └── transcription.py          # Whisper ASR transcription engine
├── recommender/
│   ├── __init__.py
│   ├── news_feed.py           # Live RSS feeds (The Hindu Hyderabad/Chennai/Bengaluru/Kolkata + Google News)
│   └── vector_store.py        # Persistent ChromaDB vector store & semantic search
├── frontend/                  # Next.js React frontend (Material UI v5/v6, Deep Orange Theme)
│   ├── src/
│   │   ├── app/               # Next.js App Router (layout.js, page.js)
│   │   ├── components/        # VoiceRecorder.js, RecommendationCard.js
│   │   ├── theme.js           # Google Material Design theme (Deep Orange, Pill Shapes)
│   │   └── ThemeRegistry.js   # MUI SSR AppRouterCacheProvider & ThemeProvider
│   └── package.json           # Frontend dependencies (@mui/material, @emotion, axios)
├── tests/
│   └── test_api.py            # End-to-end integration test suite (100% passing)
├── .gitignore                 # Comprehensive repository ignore rules
└── requirements.txt           # Core Python dependencies
```

---

## Pipeline Execution Flow

$$\text{User Voice / Audio} \longrightarrow \text{Acoustic Accent Classifier} \longrightarrow \text{Whisper ASR} \longrightarrow \text{Zero-Shot Intent} \longrightarrow \begin{cases} \textbf{ChromaDB Store} & \text{(Food, Fashion, Travel, Diaspora)} \\ \textbf{Live RSS Feed} & \text{(News)} \end{cases}$$

1. **Acoustic Regional Accent Detection (`models/accent_classifier.py`)**:
   - Ingests audio waveform tensors or binary bytes (WAV, MP3, OGG, FLAC) with multi-format decoding fallback (`soundfile` + `librosa`).
   - Normalizes audio sample rates to 16,000 Hz and extracts features via `Wav2Vec2FeatureExtractor`.
   - Maps inputs to the optimal hardware device (`cuda:0`, `mps`, or `cpu`).
   - Supports wiring fine-tuned acoustic checkpoints (`.pt` state dicts or Hugging Face model directories) with real neural forward pass inference and regional softmax probability distributions.
   - Provides a resilient simulation fallback if a checkpoint is not present or during initial prototyping.

2. **ASR Speech-to-Text Transcription (`models/transcription.py`)**:
   - Transcribes spoken audio into query text using `openai/whisper`.
   - Supports optional query hints for testing synthetic or silent audio clips.

3. **NLP Zero-Shot Intent Recognition (`models/nlp_intent.py`)**:
   - Zero-shot classifies transcribed speech into 5 domains: `food`, `fashion`, `travel`, `news`, or `diaspora`.
   - Powered by `valhalla/distilbart-mnli-12-3` on CUDA, paired with a resilient keyword fallback for offline reliability.

4. **ChromaDB Vector Store (`recommender/vector_store.py`)**:
   - Persistent embedding index (`all-MiniLM-L6-v2`) on GPU with 22+ curated items.
   - Combines semantic similarity with metadata filtering across 5 states:
     - **Telangana:** Hyderabadi Dum Biryani, Sarva Pindi, Pochampally Ikat Sarees, Ramappa Temple, Global Bathukamma Network.
     - **Tamil Nadu:** Chettinad Pepper Chicken, Madurai Jigarthanda, Kanchipuram Silk Sarees, Meenakshi Temple, World Tamil Sangam.
     - **Karnataka:** Davangere Benne Dosa, Bisi Bele Bath, Mysore Crepe Silk, Hampi UNESCO Ruins, AKKA Kannada Koota.
     - **Punjab:** Amritsari Stuffed Kulcha, Makki di Roti with Saag, Phulkari Dupattas, Golden Temple, World Punjabi Organization.
     - **Bengal:** Kolkata Dum Biryani, Shorshe Ilish, Dhakai Jamdani Sarees, Victoria Memorial, Global Banga Sammelan.

5. **Live Regional News Aggregator (`recommender/news_feed.py`)**:
   - Dynamically routes news requests to state-specific RSS feeds:
     - **Telangana:** *The Hindu Hyderabad* + Google News Telangana
     - **Tamil Nadu:** *The Hindu Chennai* + Google News Tamil Nadu
     - **Karnataka:** *The Hindu Bengaluru* + Google News Karnataka
     - **Punjab:** Google News Punjab/Amritsar + The Hindu Regional
     - **Bengal:** *The Hindu Kolkata* + Google News West Bengal
   - Auto-sanitizes HTML tags and provides state-specific fallback headlines.

6. **Interactive Browser Web UI (`frontend/`)**:
   - **Real-Time Microphone Recording:** In-browser 16kHz Mono PCM `.wav` encoder with pulsing Deep Orange geometric FAB.
   - **Audio File Upload & Guidance Hint:** Dropzone for pre-recorded audio files (`.wav`, `.mp3`, `.ogg`, `.flac`) and guidance hints.
   - **Dual Studio Tabs:** Switch seamlessly between **Voice Studio** (mic recording / file upload) and **Text Studio** (manual queries + region/category dropdowns).
   - **5-Stage Architecture Stepper:** Permanently visible pipeline stepper that dynamically tracks execution through Ingestion, Wav2Vec2, Whisper, DistilBART, and Routing.
   - **Diagnostic Inference Inspector:** 4-card breakdown displaying detected accent confidence, domain intent confidence, Whisper speech transcript, and active routing subsystem.
   - **1-Click Regional Presets:** 6 instant test prompt chips covering all 5 states.
   - **Color-Coded Recommendation Cards:** Displays matching titles, locations, cultural tags, match scores, and direct article links for news.

---

## 🚀 How to Execute the Project (Step-by-Step)

### Prerequisites

| Requirement | Recommended Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.11 or 3.12 | FastAPI Backend, PyTorch, Transformers, ChromaDB |
| **Node.js** | 18.x or 20.x LTS | Next.js Frontend dev server and production builds |
| **CUDA GPU** (Optional) | CUDA 12.1+ | Accelerates Wav2Vec2, Whisper, and DistilBART inference (CPU fallback included) |

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/NPL_Based_Recomndation.git
cd NPL_Based_Recomndation
```

---

### Step 2: Set Up Python Backend

1. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   # On Windows (PowerShell):
   py -3.12 -m venv venv
   .\venv\Scripts\Activate.ps1

   # On Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend server:**
   ```bash
   # Windows:
   py -3.12 -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

   # Linux / macOS:
   uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

   Once started, verify:
   - **Backend API Root:** [http://localhost:8000/](http://localhost:8000/)
   - **Interactive Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Check:** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

### Step 3: Set Up Next.js Frontend

Open a **new terminal window** and run:

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the Next.js development server:**
   ```bash
   npm run dev -- -p 3000
   ```

4. **Open the Web UI in your browser:**
   ```
   http://localhost:3000
   ```

*(Optional) To verify a production build:*
```bash
npm run build
npm start
```

---

### Step 4: Run the Automated Integration Tests

With the backend dependencies installed, execute the integration test suite:

```bash
# Windows:
py -3.12 tests/test_api.py

# Linux / macOS:
python3 tests/test_api.py
```

**Test Suite Coverage (8 Integration Tests):**
- System health and CUDA GPU acceleration (`GET /api/v1/health`)
- Acoustic checkpoint loading and forward inference (`checkpoints/wav2vec2_accent_finetuned.pt`)
- Voice upload $\rightarrow$ Food recommendation via ChromaDB
- Voice upload $\rightarrow$ Live News headlines via state RSS feeds
- Voice upload $\rightarrow$ Fashion items via ChromaDB
- Corrupted audio file validation and graceful HTTP 400 error handling
- Multi-Region Expansion: Verified across Tamil Nadu, Karnataka, Punjab, and Bengal

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root API metadata JSON with service status and frontend redirection link |
| `GET` | `/api/info` | Service metadata and endpoint catalog |
| `GET` | `/api/v1/health` | Health check (CUDA availability, ChromaDB item count, module status) |
| `POST` | `/api/v1/recommend-by-voice` | Upload audio (`multipart/form-data`) $\rightarrow$ Full pipeline recommendation |
| `POST` | `/api/v1/recommend-by-text` | JSON text query $\rightarrow$ Zero-shot intent $\rightarrow$ Routed recommendations |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## Command-Line Testing Examples

> **Note for Windows PowerShell:** Use `curl.exe` instead of bare `curl` to avoid PowerShell's built-in `Invoke-WebRequest` alias.

### 1. Health Check
```powershell
curl.exe -X GET "http://localhost:8000/api/v1/health"
```

### 2. Multi-Region Text Queries

```powershell
# Tamil Nadu - Kanchipuram Silk Sarees
curl.exe -X POST "http://localhost:8000/api/v1/recommend-by-text" `
  -H "Content-Type: application/json" `
  -d '{"query": "I am looking for traditional Kanchipuram silk wedding sarees.", "region": "tamil_nadu", "top_k": 2}'

# Karnataka - Davangere Benne Dosa
curl.exe -X POST "http://localhost:8000/api/v1/recommend-by-text" `
  -H "Content-Type: application/json" `
  -d '{"query": "Where can I eat crisp Davangere benne dosa?", "region": "karnataka", "top_k": 2}'

# Punjab - Golden Temple & Amritsari Kulcha
curl.exe -X POST "http://localhost:8000/api/v1/recommend-by-text" `
  -H "Content-Type: application/json" `
  -d '{"query": "Tell me about the Golden Temple and Amritsari kulchas in Punjab.", "region": "punjab", "top_k": 2}'

# Bengal - Shorshe Ilish & Kolkata Dum Biryani
curl.exe -X POST "http://localhost:8000/api/v1/recommend-by-text" `
  -H "Content-Type: application/json" `
  -d '{"query": "Where can I find authentic Shorshe Ilish and Kolkata biryani?", "region": "bengal", "top_k": 2}'
```

### 3. Voice Recommendation (Audio File Upload)

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/recommend-by-voice" `
  -F "file=@path/to/sample.wav" `
  -F "region_hint=telangana" `
  -F "query_hint=Where can I eat authentic spicy Hyderabadi biryani?" `
  -F "top_k=2"
```

---

## Acoustic Checkpoint Integration (Fine-Tuned Wav2Vec2)

The system provides native support for wiring fine-tuned acoustic checkpoints into `models/accent_classifier.py`:

### 1. Supported Checkpoint Formats
1. **PyTorch State Dict File (`.pt`, `.pth`, `.bin`):**
   - Contains `model_state_dict` (or raw weights) and embedded `config` dictionary.
   - When loaded, dynamically initializes the `Wav2Vec2ForSequenceClassification` architecture on `cuda`/`mps`/`cpu` and injects fine-tuned classification weights.
   - Default path: `checkpoints/wav2vec2_accent_finetuned.pt`
2. **Hugging Face Model Directory:**
   - Saved via `model.save_pretrained()` and `feature_extractor.save_pretrained()`.
   - Contains `config.json`, `preprocessor_config.json`, and weights (`model.safetensors`).
   - Default path: `checkpoints/wav2vec2_accent_hf/`

### 2. Loading a Checkpoint via Environment Variable or Code
To specify a custom fine-tuned weights checkpoint, set the environment variable:
```powershell
$env:WAV2VEC2_CHECKPOINT_PATH = "checkpoints/wav2vec2_accent_finetuned.pt"
# Or point to a Hugging Face directory:
# $env:WAV2VEC2_CHECKPOINT_PATH = "checkpoints/wav2vec2_accent_hf"
```

Or programmatically in Python:
```python
from models.accent_classifier import get_accent_classifier

classifier = get_accent_classifier(checkpoint_path="checkpoints/wav2vec2_accent_finetuned.pt")
print(f"Active Checkpoint: {classifier.checkpoint_path} (Loaded: {classifier.is_checkpoint_loaded})")
```

---

## Preparing to Push to GitHub

The repository has been pruned and pre-configured with a clean `.gitignore` to prevent committing binary caches, build artifacts, or node modules:

```bash
# 1. Initialize git (if not already done)
git init

# 2. Check staged/untracked files (confirm chroma_db/ and node_modules/ are ignored)
git status

# 3. Stage clean project files
git add .

# 4. Commit changes
git commit -m "feat: complete voice-based regional accent recommender with Next.js frontend"

# 5. Link to your remote GitHub repository and push
git remote add origin https://github.com/<your-username>/NPL_Based_Recomndation.git
git branch -M main
git push -u origin main
```

---

## License

This project is licensed under the MIT License.
