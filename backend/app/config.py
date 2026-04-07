"""Configuration for the ShotStory backend."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = "ShotStory"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Model configuration
    CAPTION_MODEL: str = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
    VQA_MODEL: str = os.getenv("VQA_MODEL", "Salesforce/blip-vqa-base")
    DETECTION_MODEL: str = os.getenv("DETECTION_MODEL", "facebook/detr-resnet-50")
    STORY_MODEL: str = os.getenv("STORY_MODEL", "google/flan-t5-large")

    # Detection threshold (lower for illustrations / flat-style images)
    DETECTION_THRESHOLD: float = float(os.getenv("DETECTION_THRESHOLD", "0.5"))

    # Story generation parameters
    STORY_MAX_TOKENS: int = int(os.getenv("STORY_MAX_TOKENS", "512"))
    STORY_TEMPERATURE: float = float(os.getenv("STORY_TEMPERATURE", "0.9"))

    # Image limits
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    IMAGE_RESIZE_MAX: int = 800  # Max dimension for processing

    # Startup behavior
    PRELOAD_MODELS: bool = os.getenv("PRELOAD_MODELS", "false").lower() == "true"


settings = Settings()
