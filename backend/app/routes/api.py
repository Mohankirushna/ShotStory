"""API route definitions."""
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from app.services.image_analyzer import ImageAnalyzer
from app.services.story_generator import StoryGenerator
from app.services.explainer import Explainer
from app.services.xai_reasoner import XAIReasoner
from app.utils.image_utils import process_uploaded_image, image_to_base64

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Upload an image → receive a story, visual features, attribution maps."""

    content_type = file.content_type or ""
    logger.info("Received file: name=%s, type=%s", file.filename, content_type)

    # Accept common image content-types and also octet-stream (some browsers)
    allowed = ("image/", "application/octet-stream")
    if not any(content_type.startswith(a) for a in allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file must be an image (got content-type: {content_type}).",
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        image = process_uploaded_image(contents)
        original_b64 = image_to_base64(image)

        # ---------- 1. Analyse image ----------
        analyzer = ImageAnalyzer.get_instance()
        analysis = analyzer.analyze(image)

        # ---------- 2. Generate story ----------
        generator = StoryGenerator.get_instance()
        story = generator.generate(analysis)

        # ---------- 3. Build explanations ----------
        explainer = Explainer()

        attention_overlay = explainer.create_heatmap_overlay(
            image, analysis["attention_map"]
        )
        raw_heatmap = explainer.create_raw_heatmap(analysis["attention_map"])
        object_overlay = explainer.create_object_overlay(image, analysis["objects"])

        region_heatmaps = explainer.create_region_heatmaps(
            image, analysis["attention_map"], analysis["objects"]
        )

        attributions = explainer.generate_attributions(analysis, story)

        # ---------- 4. XAI Reasoning ----------
        reasoner = XAIReasoner()
        xai_reasoning = reasoner.generate_reasoning(analysis, story, attributions)

        return {
            "success": True,
            "original_image": original_b64,
            "story": story,
            "caption": analysis["caption"],
            "scene_attributes": analysis["scene_attributes"],
            "objects": [
                {
                    "label": o["label"],
                    "confidence": round(o["confidence"], 3),
                    "bbox": [round(v, 1) for v in o["bbox"]],
                }
                for o in analysis["objects"]
            ],
            "visualizations": {
                "attention_overlay": attention_overlay,
                "raw_heatmap": raw_heatmap,
                "object_overlay": object_overlay,
                "region_heatmaps": region_heatmaps,
            },
            "attributions": attributions,
            "xai_reasoning": xai_reasoning,
            "image_size": {
                "width": analysis["image_size"][0],
                "height": analysis["image_size"][1],
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": ImageAnalyzer.get_instance()._loaded,
    }


@router.post("/preload")
async def preload_models():
    """Proactively download & load models so the first /analyze is faster."""
    try:
        ImageAnalyzer.get_instance().load_models()
        StoryGenerator.get_instance().load_model()
        return {"status": "models_loaded"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
