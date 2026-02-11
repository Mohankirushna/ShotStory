"""
Image analysis service using BLIP (captioning + VQA) and DETR (object detection).
Extracts visual features, scene attributes, and attention maps for explainability.
"""
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Analyzes images using BLIP and DETR models."""

    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.caption_model = None
        self.caption_processor = None
        self.vqa_model = None
        self.vqa_processor = None
        self.detection_model = None
        self.detection_processor = None
        self._loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _load(cls_or_fn, name, **kwargs):
        """Load from local cache first; fall back to download if missing."""
        try:
            return cls_or_fn.from_pretrained(name, local_files_only=True, **kwargs)
        except OSError:
            logger.info("Cache miss for %s – downloading...", name)
            return cls_or_fn.from_pretrained(name, **kwargs)

    def load_models(self):
        """Lazy-load all models on first use."""
        if self._loaded:
            return

        import os
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        from transformers import (
            BlipProcessor,
            BlipForConditionalGeneration,
            BlipForQuestionAnswering,
            DetrImageProcessor,
            DetrForObjectDetection,
        )

        logger.info("Loading BLIP captioning model (%s)...", settings.CAPTION_MODEL)
        self.caption_processor = self._load(BlipProcessor, settings.CAPTION_MODEL)
        self.caption_model = self._load(
            BlipForConditionalGeneration, settings.CAPTION_MODEL
        ).to(self.device)
        self.caption_model.eval()

        logger.info("Loading BLIP VQA model (%s)...", settings.VQA_MODEL)
        self.vqa_processor = self._load(BlipProcessor, settings.VQA_MODEL)
        self.vqa_model = self._load(
            BlipForQuestionAnswering, settings.VQA_MODEL
        ).to(self.device)
        self.vqa_model.eval()

        logger.info("Loading DETR detection model (%s)...", settings.DETECTION_MODEL)
        self.detection_processor = self._load(
            DetrImageProcessor, settings.DETECTION_MODEL, revision="no_timm"
        )
        self.detection_model = self._load(
            DetrForObjectDetection, settings.DETECTION_MODEL, revision="no_timm"
        ).to(self.device)
        self.detection_model.eval()

        self._loaded = True
        logger.info("All models loaded successfully on %s!", self.device)

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------
    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.caption_processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            ids = self.caption_model.generate(**inputs, max_new_tokens=60)
        return self.caption_processor.decode(ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Visual Question Answering
    # ------------------------------------------------------------------
    def ask_question(self, image: Image.Image, question: str) -> str:
        inputs = self.vqa_processor(
            images=image, text=question, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            ids = self.vqa_model.generate(**inputs, max_new_tokens=30)
        return self.vqa_processor.decode(ids[0], skip_special_tokens=True)

    def extract_scene_attributes(self, image: Image.Image) -> Dict[str, str]:
        questions = {
            "mood": "What is the mood or atmosphere of this image?",
            "setting": "What is the setting or location in this image?",
            "time_of_day": "What time of day is it in this image?",
            "weather": "What is the weather like in this image?",
            "main_subject": "Who or what is the main subject of this image?",
            "activity": "What activity is happening in this image?",
            "emotions": "What emotions are expressed in this image?",
            "colors": "What are the dominant colors in this image?",
        }
        attrs: Dict[str, str] = {}
        for key, q in questions.items():
            try:
                attrs[key] = self.ask_question(image, q)
            except Exception as exc:
                logger.warning("VQA failed for '%s': %s", key, exc)
                attrs[key] = "unknown"
        return attrs

    # ------------------------------------------------------------------
    # Object Detection
    # ------------------------------------------------------------------
    def detect_objects(
        self, image: Image.Image, threshold: float | None = None
    ) -> List[Dict[str, Any]]:
        threshold = threshold or settings.DETECTION_THRESHOLD
        inputs = self.detection_processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.detection_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        objects: List[Dict[str, Any]] = []
        for score, label_id, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy(),
        ):
            objects.append(
                {
                    "label": self.detection_model.config.id2label[int(label_id)],
                    "confidence": float(score),
                    "bbox": box.tolist(),
                }
            )
        return objects

    # ------------------------------------------------------------------
    # Attention Map via ViT Attention Rollout
    # ------------------------------------------------------------------
    def extract_attention_map(self, image: Image.Image) -> np.ndarray:
        """ViT attention-rollout to produce a spatial saliency map."""
        inputs = self.caption_processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            vision_out = self.caption_model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_attentions=True,
            )

        attentions = vision_out.attentions  # tuple of [1, heads, seq, seq]
        result = torch.eye(attentions[0].shape[-1]).to(self.device)

        for attn in attentions:
            attn_fused = attn.mean(dim=1)[0]  # average across heads
            identity = torch.eye(attn_fused.shape[0]).to(self.device)
            a = (attn_fused + identity) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)

        # CLS-token → patch-tokens attention
        mask = result[0, 1:].cpu().numpy()
        grid = int(np.sqrt(len(mask)))
        mask = mask.reshape(grid, grid)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        self.load_models()

        if image.mode != "RGB":
            image = image.convert("RGB")

        logger.info("Generating caption...")
        caption = self.generate_caption(image)

        logger.info("Extracting scene attributes via VQA...")
        scene_attributes = self.extract_scene_attributes(image)

        logger.info("Detecting objects...")
        objects = self.detect_objects(image)

        logger.info("Computing attention map...")
        attention_map = self.extract_attention_map(image)

        return {
            "caption": caption,
            "scene_attributes": scene_attributes,
            "objects": objects,
            "attention_map": attention_map,
            "image_size": image.size,  # (width, height)
        }
