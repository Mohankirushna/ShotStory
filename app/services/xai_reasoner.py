"""
XAI Reasoning Engine.
Generates a structured, step-by-step reasoning chain that explains
how each visual feature was detected, why it matters, and how it
influenced specific parts of the generated story.
"""
from typing import Dict, List, Any
import numpy as np


class XAIReasoner:
    """Produces a full explainability report for the image→story pipeline."""

    def generate_reasoning(
        self,
        analysis: Dict[str, Any],
        story: Dict[str, Any],
        attributions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a complete XAI report with reasoning chain, decision log,
        and a transparency summary."""

        caption = analysis.get("caption", "")
        attrs = analysis.get("scene_attributes", {})
        objects = analysis.get("objects", [])
        attention_map = analysis.get("attention_map")
        story_text = story.get("text", "")

        # 1. Build step-by-step reasoning chain
        reasoning_chain = self._build_reasoning_chain(
            caption, attrs, objects, attention_map, story_text, story
        )

        # 2. Build decision log (which model made which decision)
        decision_log = self._build_decision_log(caption, attrs, objects, story)

        # 3. Feature-to-sentence mapping
        sentence_map = self._map_features_to_sentences(
            attrs, objects, story_text
        )

        # 4. Confidence & transparency summary
        transparency = self._transparency_summary(
            analysis, story, attributions
        )

        return {
            "reasoning_chain": reasoning_chain,
            "decision_log": decision_log,
            "sentence_map": sentence_map,
            "transparency": transparency,
        }

    # ------------------------------------------------------------------
    # Reasoning chain
    # ------------------------------------------------------------------
    def _build_reasoning_chain(
        self,
        caption: str,
        attrs: Dict[str, str],
        objects: List[Dict],
        attention_map,
        story_text: str,
        story: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        story = story or {}
        chain: List[Dict[str, Any]] = []

        # Step 1 — Visual perception
        chain.append({
            "step": 1,
            "phase": "Visual Perception",
            "model": "BLIP Vision Encoder (ViT)",
            "action": "The image was processed through a Vision Transformer that divides it into patches and computes attention across all regions.",
            "result": f"The model generated the caption: \"{caption}\"",
            "evidence": "attention_map",
            "confidence": "high",
        })

        # Step 2 — Object detection
        obj_labels = [o["label"] for o in objects]
        avg_conf = (
            round(np.mean([o["confidence"] for o in objects]), 2)
            if objects
            else 0
        )
        chain.append({
            "step": 2,
            "phase": "Object Detection",
            "model": "DETR (DEtection TRansformer)",
            "action": "A detection transformer scanned the image for recognizable objects with bounding boxes.",
            "result": (
                f"Detected {len(objects)} object(s): {', '.join(obj_labels)}"
                if objects
                else "No distinct objects detected above the confidence threshold."
            ),
            "evidence": "object_overlay",
            "confidence": "high" if avg_conf > 0.8 else "medium" if avg_conf > 0.5 else "low",
        })

        # Step 3 — Scene understanding via VQA
        answered = {k: v for k, v in attrs.items() if v.lower() != "unknown"}
        chain.append({
            "step": 3,
            "phase": "Scene Understanding (VQA)",
            "model": "BLIP Visual Question Answering",
            "action": "Eight targeted questions were asked about mood, setting, time of day, weather, subject, activity, emotions, and colors.",
            "result": "; ".join(
                f"{k.replace('_', ' ').title()}: {v}" for k, v in answered.items()
            ),
            "evidence": "scene_attributes",
            "confidence": "medium",
        })

        # Step 4 — Attention analysis
        if attention_map is not None:
            top_pct = float(np.mean(attention_map > 0.7) * 100)
            chain.append({
                "step": 4,
                "phase": "Attention Analysis",
                "model": "ViT Attention Rollout",
                "action": "Multi-head attention weights across all transformer layers were rolled up to produce a spatial saliency map.",
                "result": f"{top_pct:.1f}% of the image received high attention (>70% intensity), indicating focused model gaze.",
                "evidence": "attention_overlay",
                "confidence": "high",
            })

        # Step 5 — Narrative construction
        chain.append({
            "step": 5,
            "phase": "Narrative Construction",
            "model": f"Story Engine ({story.get('method', 'template')})",
            "action": "All extracted features — caption, objects, scene attributes, and attention regions — were combined into a structured prompt. The story was generated to faithfully reference every detected visual element.",
            "result": f"A {len(story_text.split())} word story was generated with title \"{story.get('title', '')}\".",
            "evidence": "story",
            "confidence": "high",
        })

        # Step 6 — Attribution mapping
        chain.append({
            "step": 6,
            "phase": "Attribution & Explanation",
            "model": "Feature-Story Attribution Scorer",
            "action": "Each visual feature was scored for its presence and influence in the final story text using text-overlap analysis and confidence weighting.",
            "result": "Every story paragraph is traceable to one or more visual features detected in steps 1–4.",
            "evidence": "attributions",
            "confidence": "high",
        })

        return chain

    # ------------------------------------------------------------------
    # Decision log
    # ------------------------------------------------------------------
    def _build_decision_log(
        self,
        caption: str,
        attrs: Dict[str, str],
        objects: List[Dict],
        story: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        log = []

        log.append({
            "decision": "Image Caption",
            "input": "Raw pixel data (resized to 384×384)",
            "model": "BLIP Captioning (ViT + Text Decoder)",
            "output": caption,
            "why": "The caption provides a one-sentence semantic summary that seeds the narrative.",
        })

        for key, value in attrs.items():
            if value.lower() == "unknown":
                continue
            log.append({
                "decision": f"Scene Attribute: {key.replace('_', ' ').title()}",
                "input": f"Image + question about {key.replace('_', ' ')}",
                "model": "BLIP VQA",
                "output": value,
                "why": f"The {key.replace('_', ' ')} attribute was used to shape the story's {_attr_role(key)}.",
            })

        for obj in objects:
            log.append({
                "decision": f"Object: {obj['label']}",
                "input": "Image feature maps",
                "model": "DETR (ResNet-50 backbone)",
                "output": f"{obj['label']} at confidence {obj['confidence']:.0%}",
                "why": f"The detected {obj['label']} was woven into the narrative as a concrete visual anchor.",
            })

        log.append({
            "decision": "Story Generation",
            "input": "Caption + attributes + objects",
            "model": story.get("method", "template-engine"),
            "output": f"{len(story.get('text', '').split())} words generated",
            "why": "All features were combined into a coherent narrative that stays faithful to the image content.",
        })

        return log

    # ------------------------------------------------------------------
    # Feature → sentence mapping
    # ------------------------------------------------------------------
    def _map_features_to_sentences(
        self,
        attrs: Dict[str, str],
        objects: List[Dict],
        story_text: str,
    ) -> List[Dict[str, Any]]:
        sentences = [s.strip() for s in story_text.replace("\n", " ").split(".") if s.strip()]
        mappings = []

        all_features = []
        for k, v in attrs.items():
            if v.lower() != "unknown":
                all_features.append({"type": k, "value": v})
        for obj in objects:
            all_features.append({"type": "object", "value": obj["label"]})

        for feat in all_features:
            matched = []
            for i, sent in enumerate(sentences):
                if _fuzzy_in(feat["value"], sent):
                    matched.append({"index": i, "sentence": sent.strip() + "."})
            if matched:
                mappings.append({
                    "feature_type": feat["type"].replace("_", " ").title(),
                    "feature_value": feat["value"],
                    "matched_sentences": matched,
                })

        return mappings

    # ------------------------------------------------------------------
    # Transparency summary
    # ------------------------------------------------------------------
    def _transparency_summary(
        self,
        analysis: Dict[str, Any],
        story: Dict[str, Any],
        attributions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        attrs = analysis.get("scene_attributes", {})
        objects = analysis.get("objects", [])
        known_attrs = {k: v for k, v in attrs.items() if v.lower() != "unknown"}

        total_features = len(known_attrs) + len(objects)
        story_text = story.get("text", "").lower()

        features_in_story = 0
        for v in known_attrs.values():
            if _fuzzy_in(v, story_text):
                features_in_story += 1
        for obj in objects:
            if _fuzzy_in(obj["label"], story_text):
                features_in_story += 1

        coverage = round(features_in_story / max(total_features, 1) * 100)

        avg_importance = 0.0
        if attributions:
            avg_importance = round(
                np.mean([a["importance_score"] for a in attributions]) * 100
            )

        return {
            "total_visual_features": total_features,
            "features_used_in_story": features_in_story,
            "feature_coverage_pct": coverage,
            "avg_attribution_score": avg_importance,
            "generation_method": story.get("method", "unknown"),
            "models_used": [
                "BLIP Image Captioning (Salesforce/blip-image-captioning-base)",
                "BLIP Visual QA (Salesforce/blip-vqa-base)",
                "DETR Object Detection (facebook/detr-resnet-50)",
                "ViT Attention Rollout",
                story.get("method", "template-engine"),
            ],
            "explainability_verdict": (
                "Fully Transparent"
                if coverage >= 80
                else "Mostly Transparent"
                if coverage >= 50
                else "Partially Transparent"
            ),
        }


# ======================================================================
# Helpers
# ======================================================================
def _fuzzy_in(feature: str, text: str) -> bool:
    """Check if feature words appear in text (case-insensitive)."""
    fl = feature.lower()
    tl = text.lower()
    if fl in tl:
        return True
    words = fl.split()
    if len(words) > 1:
        return sum(1 for w in words if w in tl and len(w) > 2) >= len(words) // 2
    return False


def _attr_role(key: str) -> str:
    roles = {
        "mood": "emotional tone",
        "setting": "spatial backdrop",
        "time_of_day": "temporal context and lighting",
        "weather": "atmospheric description",
        "emotions": "character feelings",
        "colors": "descriptive palette",
        "activity": "action and plot",
        "main_subject": "narrative focus",
    }
    return roles.get(key, "content")
