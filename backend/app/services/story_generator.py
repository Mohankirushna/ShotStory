"""
Story generation service  –  chain-of-paragraphs with Flan-T5-Large.

Produces a **3rd-person narrative story** (300-400 words) by generating
four paragraphs sequentially.  Each paragraph prompt feeds context from
the previous paragraphs so the narrative stays coherent.

Uses Apple-Silicon MPS / CUDA when available for fast inference.
"""

import random
import re
import logging
from typing import Dict, Any, Optional, List

from app.config import settings

logger = logging.getLogger(__name__)

# Words that signal a VQA answer is effectively empty / useless
_GENERIC = {
    "unknown", "none", "no", "yes", "n/a", "na", "nothing", "",
    "not sure", "unsure", "i don't know",
}

# ======================================================================
# Subject / pronoun helpers
# ======================================================================
_PERSON_WORDS = {
    "boy", "girl", "man", "woman", "child", "kid", "baby", "person",
    "people", "lady", "gentleman", "teen", "teenager", "toddler",
    "group", "couple", "family", "friend", "friends", "crowd",
    "player", "worker", "athlete", "student", "dog", "cat",
}
_HE_WORDS  = {"boy", "man", "gentleman", "teen", "teenager", "guy"}
_SHE_WORDS = {"girl", "woman", "lady"}
_THEY_WORDS = {"people", "group", "couple", "family", "friends", "crowd"}


def _pronoun(subject: str) -> tuple[str, str, str]:
    sl = subject.lower()
    for w in _HE_WORDS:
        if w in sl:
            return ("he", "his", "him")
    for w in _SHE_WORDS:
        if w in sl:
            return ("she", "her", "her")
    for w in _THEY_WORDS:
        if w in sl:
            return ("they", "their", "them")
    return ("it", "its", "it")


def _is_person(subject: str) -> bool:
    return any(w in subject.lower() for w in _PERSON_WORDS)


def _name_for(subject: str) -> str:
    sl = subject.lower()
    male   = ["Arjun", "Leo", "Sam", "Ethan", "Ravi", "Noah", "Kai", "Omar"]
    female = ["Maya", "Lily", "Priya", "Aisha", "Zara", "Emma", "Nina", "Sara"]
    neutral = ["Alex", "Jordan", "Riley", "Avery", "Sage"]
    for w in _HE_WORDS:
        if w in sl:
            return random.choice(male)
    for w in _SHE_WORDS:
        if w in sl:
            return random.choice(female)
    return random.choice(neutral)


def _subject_from_caption(caption: str) -> str:
    cap = caption.strip().lower()
    for art in ("a ", "an ", "the "):
        if cap.startswith(art):
            cap = cap[len(art):]
            break
    for prep in (" on ", " in ", " at ", " with ", " near ", " beside ",
                 " against ", " under ", " over ", " behind ", " next to "):
        idx = cap.find(prep)
        if idx > 0:
            cap = cap[:idx]
            break
    return cap.strip() or "scene"


# ======================================================================
# Story Generator
# ======================================================================
class StoryGenerator:
    """Chain-of-paragraphs story generator using Flan-T5-Large on MPS/CUDA/CPU."""

    _instance = None

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ load
    def load_model(self):
        if self._loaded:
            return
        try:
            import os, torch
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            model_name = settings.STORY_MODEL
            logger.info("Loading story model (%s)...", model_name)

            # Pick best available device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            logger.info("Story model device: %s", self.device)

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
            except OSError:
                logger.info("Cache miss – downloading %s...", model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Story model ready on %s!", self.device)
        except Exception as exc:
            logger.error("Could not load story model: %s", exc)
            self.model = None
            self.tokenizer = None
        self._loaded = True

    # ------------------------------------------------------------------ ctx
    @staticmethod
    def _ctx(analysis: Dict[str, Any]) -> Dict[str, str]:
        caption = analysis.get("caption", "a scene")
        a = analysis.get("scene_attributes", {})
        objs = [o["label"] for o in analysis.get("objects", [])[:8]]

        def _v(key, fb=""):
            v = a.get(key, "").strip()
            return v if v.lower() not in _GENERIC else fb

        subject = _v("main_subject") or _subject_from_caption(caption)
        return {
            "caption": caption, "subject": subject,
            "setting": _v("setting"), "time": _v("time_of_day"),
            "mood": _v("mood"), "weather": _v("weather"),
            "emotions": _v("emotions"), "colors": _v("colors"),
            "activity": _v("activity"),
            "objects": objs, "obj_str": ", ".join(objs) if objs else "",
        }

    # ------------------------------------------------------------------ title
    @staticmethod
    def _generate_title(c: Dict[str, str]) -> str:
        subject = c["subject"] or "Scene"
        word = subject.strip()
        if word and word[0].islower():
            word = word[0].upper() + word[1:]
        if not word.lower().startswith(("a ", "an ", "the ")):
            word = f"the {word.capitalize()}"
        else:
            word = word.capitalize()
        tpls = [f"The Story of {word}", f"A Moment with {word}"]
        if c["mood"]:
            tpls.append(f"A {c['mood'].title()} Day")
        if c["setting"]:
            tpls.append(f"The {c['setting'].title()} Tale")
        return random.choice(tpls)

    # ------------------------------------------------------------------ generate one chunk
    def _generate_chunk(self, prompt: str, max_tokens: int = 150,
                        temperature: float = 0.9) -> str:
        """Generate a single paragraph-sized chunk of text."""
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=60,
                temperature=temperature,
                top_p=0.92,
                top_k=50,
                do_sample=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                num_beams=1,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        # Trim at last sentence boundary
        last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last > 0:
            text = text[:last + 1]
        return text

    # ------------------------------------------------------------------ build prompts
    def _build_paragraph_prompts(self, c: Dict[str, str]) -> List[str]:
        """Return six prompts — one per paragraph of the story."""
        subject = c["subject"] or "figure"
        person = _is_person(subject)
        name = _name_for(subject) if person else None
        observer = random.choice(["Alex", "Jordan", "Sam", "Riley"])
        char = name if person else observer
        setting = c["setting"] or "quiet place"
        mood = c["mood"] or "calm"
        caption = c["caption"]

        # Compose a visual-detail sentence
        vis = [f"The scene: {caption}."]
        if c["weather"]:
            vis.append(f"Weather is {c['weather']}.")
        if c["colors"]:
            vis.append(f"Colors: {c['colors']}.")
        if c["obj_str"]:
            vis.append(f"Objects nearby: {c['obj_str']}.")
        visual = " ".join(vis)

        length_instr = "Write at least 60 words. Use multiple sentences with vivid sensory details."

        if person:
            pro, poss, _ = _pronoun(subject)
            act = c["activity"] or "taking a quiet walk"
            prompts = [
                # P1: Arrival / scene setting
                (
                    f"Write the first paragraph of a literary third-person short story. "
                    f"{char} is a {subject} who arrives at a {setting} during "
                    f"{c['time'] or 'the afternoon'}. {visual} "
                    f"Describe what {char} sees, hears, and feels as {pro} takes in "
                    f"the scene. The mood is {mood}. {length_instr}"
                ),
                # P2: Deeper scene exploration
                (
                    f"Write the second paragraph of a literary third-person story about "
                    f"{char} at the {setting}. {char} walks through the area and "
                    f"notices small, specific details — the texture of the ground, "
                    f"distant sounds, the quality of the light, the temperature of the air. "
                    f"{{prev}} {length_instr}"
                ),
                # P3: Activity / engagement
                (
                    f"Write the third paragraph of a literary third-person story about "
                    f"{char} at the {setting}. {char} begins {act}. "
                    f"Describe {poss} movements and the way {pro} interacts with the "
                    f"environment. Include what {pro} touches, hears close by, "
                    f"and how {poss} body feels. {{prev}} {length_instr}"
                ),
                # P4: Memory / inner world
                (
                    f"Write the fourth paragraph of a literary third-person story about "
                    f"{char} at the {setting}. Something in the scene triggers a "
                    f"personal memory for {char}. Describe the memory in detail — who "
                    f"was there, what happened, why it matters now. Show {poss} "
                    f"emotions. {{prev}} {length_instr}"
                ),
                # P5: Turning point
                (
                    f"Write the fifth paragraph of a literary third-person story about "
                    f"{char} at the {setting}. {char} has a quiet realization or "
                    f"shift in perspective. Something {pro} has been carrying — a worry, "
                    f"a question, a regret — begins to loosen. Describe this change "
                    f"through {poss} thoughts and body language. {{prev}} {length_instr}"
                ),
                # P6: Closing
                (
                    f"Write the final paragraph of a literary third-person story about "
                    f"{char} at the {setting}. {char} prepares to leave but carries "
                    f"something new with {_pronoun(subject)[2]}. End with a quiet, "
                    f"hopeful image. {{prev}} {length_instr}"
                ),
            ]
        else:
            prompts = [
                (
                    f"Write the first paragraph of a literary third-person short story. "
                    f"{observer} walks through a {setting} and discovers a {subject}. "
                    f"{visual} The mood is {mood}. Describe the scene and what draws "
                    f"{observer}'s attention. {length_instr}"
                ),
                (
                    f"Write the second paragraph of a literary third-person story about "
                    f"{observer} and the {subject} at the {setting}. "
                    f"{observer} moves closer and examines the {subject}. Describe its "
                    f"texture, color, shape, and how light falls on it. What sounds are "
                    f"in the background? {{prev}} {length_instr}"
                ),
                (
                    f"Write the third paragraph of a literary third-person story about "
                    f"{observer} and the {subject}. {observer} reaches out and touches "
                    f"the {subject}. Describe the physical sensation and what it reminds "
                    f"them of. {{prev}} {length_instr}"
                ),
                (
                    f"Write the fourth paragraph of a literary third-person story about "
                    f"{observer} and the {subject}. The {subject} triggers a childhood "
                    f"memory for {observer}. Describe the memory vividly — the place, "
                    f"the people, the emotions. {{prev}} {length_instr}"
                ),
                (
                    f"Write the fifth paragraph of a literary third-person story about "
                    f"{observer} and the {subject}. {observer} realizes something "
                    f"important about life, change, or time. Describe this quiet "
                    f"epiphany through thoughts and physical sensations. "
                    f"{{prev}} {length_instr}"
                ),
                (
                    f"Write the final paragraph of a literary third-person story about "
                    f"{observer} and the {subject}. {observer} leaves the {setting} "
                    f"changed. End on a meaningful, hopeful image. "
                    f"{{prev}} {length_instr}"
                ),
            ]

        return prompts

    # ------------------------------------------------------------------ post-process
    @staticmethod
    def _clean(text: str) -> str:
        """Light cleanup on a single paragraph."""
        meta = (
            "Caption:", "Setting:", "Time:", "Weather:", "Mood:",
            "Objects:", "Colors:", "Activity:", "Story:", "---",
            "Visible objects:", "Dominant colors:", "Write a",
            "Write the", "Continue the", "Objects visible:",
            "Time of day:", "Note:", "Image:", "Paragraph",
        )
        lines = text.split("\n")
        lines = [l for l in lines if not any(l.strip().startswith(m) for m in meta)]
        text = " ".join(l.strip() for l in lines if l.strip())
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    # ------------------------------------------------------------------ public generate
    def generate(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        self.load_model()
        c = self._ctx(analysis)
        title = self._generate_title(c)

        if not self.model or not self.tokenizer:
            return {
                "title": title,
                "text": "Story generation model could not be loaded.",
                "method": "error",
            }

        prompts = self._build_paragraph_prompts(c)
        paragraphs: List[str] = []

        for i, raw_prompt in enumerate(prompts):
            try:
                # For prompts 2-4, inject summary of previous paragraphs
                if "{prev}" in raw_prompt:
                    prev_summary = " ".join(paragraphs[-2:])  # last 2 paragraphs as context
                    if len(prev_summary) > 300:
                        prev_summary = prev_summary[-300:]
                    prompt = raw_prompt.replace(
                        "{prev}",
                        f"Story so far: {prev_summary}"
                    )
                else:
                    prompt = raw_prompt

                temp = settings.STORY_TEMPERATURE + (i * 0.02)
                chunk = self._generate_chunk(prompt, max_tokens=200, temperature=temp)
                chunk = self._clean(chunk)

                if chunk and len(chunk) > 30:
                    paragraphs.append(chunk)
                    logger.info("Paragraph %d: %d words", i + 1, len(chunk.split()))
                else:
                    logger.warning("Paragraph %d too short, skipping", i + 1)

            except Exception as exc:
                logger.warning("Paragraph %d failed: %s", i + 1, exc)

        if len(paragraphs) >= 2:
            story = "\n\n".join(paragraphs)
            wc = len(story.split())
            logger.info("Final story: %d words, %d paragraphs", wc, len(paragraphs))
            return {"title": title, "text": story, "method": "flan-t5"}

        return {
            "title": title,
            "text": "The story could not be generated. Please try again.",
            "method": "error",
        }
