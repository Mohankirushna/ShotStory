# ShotStory — Explainable Image-to-Story Generation

ShotStory is a full-stack application that generates coherent narratives from images **and** provides human-understandable visual explanations showing exactly which image features influenced each part of the story.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  Frontend  (React + Vite + Tailwind CSS)                  │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │ Image Upload │→│ Results View │→│ Attribution Panels │  │
│  └─────────────┘ └──────────────┘ └────────────────────┘  │
└────────────────────────┬──────────────────────────────────┘
                         │  /api/analyze  (POST)
┌────────────────────────▼──────────────────────────────────┐
│  Backend  (FastAPI)                                        │
│  ┌──────────────────┐ ┌────────────────┐ ┌─────────────┐  │
│  │  Image Analyzer   │ │ Story Generator│ │  Explainer  │  │
│  │  BLIP + DETR      │ │ GPT-2 / Templ  │ │ Heatmaps   │  │
│  └──────────────────┘ └────────────────┘ └─────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### ML Pipeline

| Step | Model | Purpose |
|------|-------|---------|
| 1. Captioning | BLIP (`Salesforce/blip-image-captioning-base`) | Generate natural-language caption |
| 2. VQA | BLIP (`Salesforce/blip-vqa-base`) | Extract mood, setting, time, emotions, etc. |
| 3. Object Detection | DETR (`facebook/detr-resnet-50`) | Find objects with bounding boxes |
| 4. Attention Rollout | ViT (inside BLIP) | Produce spatial saliency map |
| 5. Story Generation | GPT-2 (with template fallback) | Create narrative from visual features |
| 6. Attribution | Custom scoring | Map visual features → story elements |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- ~3 GB disk space for model weights (downloaded on first run)

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py                   # → http://localhost:8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                     # → http://localhost:5173
```

Open **http://localhost:5173** in your browser, upload an image, and see:

- **Generated Story** — a narrative inspired by your photo
- **Attention Heatmap** — warm regions = where the model focused
- **Object Detection** — bounding boxes around recognised objects
- **Scene Attributes** — mood, setting, time, weather, emotions, colors
- **Feature → Story Attributions** — which visual feature drove which part of the story

## API

### `POST /api/analyze`
Upload an image (`multipart/form-data`, field `file`) → returns JSON with story, features, heatmaps, and attributions.

### `GET /api/health`
Health check; reports whether models are loaded.

### `POST /api/preload`
Proactively download and load all models.

## Project Structure

```
xai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Settings
│   │   ├── routes/api.py        # Endpoints
│   │   ├── services/
│   │   │   ├── image_analyzer.py   # BLIP + DETR
│   │   │   ├── story_generator.py  # GPT-2 / templates
│   │   │   └── explainer.py        # Heatmaps & attributions
│   │   └── utils/image_utils.py
│   ├── requirements.txt
│   └── run.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api/client.js
│   │   └── components/
│   │       ├── Header.jsx
│   │       ├── ImageUploader.jsx
│   │       ├── LoadingOverlay.jsx
│   │       ├── ResultsView.jsx
│   │       ├── StoryDisplay.jsx
│   │       ├── AttributionMap.jsx
│   │       └── FeaturePanel.jsx
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## License

MIT
