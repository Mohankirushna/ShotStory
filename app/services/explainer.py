"""
Explainability service.
Produces visual attribution overlays (heatmaps, bounding-box diagrams)
and structured attribution mappings that link visual features to story elements.
"""
import io
import base64
import numpy as np
from PIL import Image
from typing import Dict, List, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import zoom


class Explainer:
    """Creates visual explanations and feature-to-story attribution mappings."""

    # ------------------------------------------------------------------
    # Heatmap overlay on original image
    # ------------------------------------------------------------------
    @staticmethod
    def create_heatmap_overlay(
        image: Image.Image,
        attention_map: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> str:
        img = np.array(image)
        h, w = img.shape[:2]

        scale_h = h / attention_map.shape[0]
        scale_w = w / attention_map.shape[1]
        att = zoom(attention_map, (scale_h, scale_w), order=3)
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)

        cmap = cm.get_cmap(colormap)
        heatmap = (cmap(att)[:, :, :3] * 255).astype(np.uint8)

        blended = (img * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        return _pil_to_b64(Image.fromarray(blended))

    # ------------------------------------------------------------------
    # Standalone heatmap
    # ------------------------------------------------------------------
    @staticmethod
    def create_raw_heatmap(attention_map: np.ndarray, colormap: str = "jet") -> str:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(attention_map, cmap=colormap, interpolation="bilinear")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------
    # Object detection overlay
    # ------------------------------------------------------------------
    @staticmethod
    def create_object_overlay(
        image: Image.Image, objects: List[Dict[str, Any]]
    ) -> str:
        img = np.array(image)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img)

        palette = plt.cm.Set2(np.linspace(0, 1, max(len(objects), 1)))

        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj["bbox"]
            c = palette[i % len(palette)]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=c, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                f"{obj['label']} ({obj['confidence']:.0%})",
                color="white",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=c, alpha=0.85),
            )
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------
    # Region-level heatmaps for individual objects
    # ------------------------------------------------------------------
    @staticmethod
    def create_region_heatmaps(
        image: Image.Image,
        attention_map: np.ndarray,
        objects: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Per-object region heatmaps showing attention within each bounding box."""
        img = np.array(image)
        h, w = img.shape[:2]

        scale_h = h / attention_map.shape[0]
        scale_w = w / attention_map.shape[1]
        att = zoom(attention_map, (scale_h, scale_w), order=3)
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)

        region_maps: Dict[str, str] = {}
        for obj in objects:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            region_att = np.zeros_like(att)
            region_att[y1:y2, x1:x2] = att[y1:y2, x1:x2]

            cmap = cm.get_cmap("hot")
            hm = (cmap(region_att)[:, :, :3] * 255).astype(np.uint8)
            blended = (img * 0.5 + hm * 0.5).astype(np.uint8)
            region_maps[obj["label"]] = _pil_to_b64(Image.fromarray(blended))

        return region_maps

    # ------------------------------------------------------------------
    # Structured attribution mappings
    # ------------------------------------------------------------------
    @staticmethod
    def generate_attributions(
        analysis: Dict[str, Any], story: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        attributions: List[Dict[str, Any]] = []
        story_text = story["text"]
        attrs = analysis["scene_attributes"]
        objects = analysis["objects"]

        influence_templates = {
            "object": "The detected '{label}' provided a concrete visual element that anchored the narrative.",
            "mood": "The perceived mood '{value}' shaped the emotional tone of the story.",
            "setting": "The identified setting '{value}' established the story's backdrop and spatial context.",
            "time_of_day": "The inferred time of day '{value}' influenced the lighting and temporal references.",
            "weather": "The weather condition '{value}' contributed to the atmosphere described in the narrative.",
            "emotions": "The perceived emotions '{value}' guided the characters' feelings and interactions.",
            "colors": "The dominant colors '{value}' influenced the descriptive palette of the story.",
            "activity": "The observed activity '{value}' drove the action and plot of the narrative.",
            "main_subject": "The main subject '{value}' became the focal point around which the story revolves.",
        }

        # Objects
        for obj in objects:
            rel = _story_relevance(obj["label"], story_text)
            attributions.append(
                {
                    "visual_feature": obj["label"],
                    "feature_type": "object",
                    "confidence": round(obj["confidence"], 3),
                    "bbox": [round(v, 1) for v in obj["bbox"]],
                    "story_influence": influence_templates["object"].format(
                        label=obj["label"]
                    ),
                    "importance_score": round(rel * obj["confidence"], 3),
                }
            )

        # Scene attributes
        for key, value in attrs.items():
            if value and value.lower() != "unknown":
                rel = _story_relevance(value, story_text)
                tmpl = influence_templates.get(key, "'{value}' influenced the story.")
                attributions.append(
                    {
                        "visual_feature": value,
                        "feature_type": key,
                        "confidence": 1.0,
                        "story_influence": tmpl.format(label=value, value=value),
                        "importance_score": round(max(0.3, rel), 3),
                    }
                )

        attributions.sort(key=lambda x: x["importance_score"], reverse=True)
        return attributions


# ======================================================================
# Helpers
# ======================================================================
def _story_relevance(feature: str, story_text: str) -> float:
    """Quick text-overlap relevance score."""
    story_lower = story_text.lower()
    feat_lower = feature.lower()
    if feat_lower in story_lower:
        return 0.9
    words = feat_lower.split()
    matches = sum(1 for w in words if w in story_lower and len(w) > 2)
    if matches:
        return 0.5 + 0.3 * (matches / len(words))
    return 0.2


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
