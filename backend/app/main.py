"""FastAPI application for ShotStory."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes.api import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} starting up...")
    
    if settings.PRELOAD_MODELS:
        logger.info("Preloading models during startup (PRELOAD_MODELS=true)...")
        try:
            from app.services.image_analyzer import ImageAnalyzer
            from app.services.story_generator import StoryGenerator
            
            ImageAnalyzer.get_instance().load_models()
            StoryGenerator.get_instance().load_model()
            logger.info("✓ All models preloaded successfully!")
        except Exception as e:
            logger.error(f"Failed to preload models: {e}")
            logger.info("Models will load on first request instead.")
    else:
        logger.info("Models will be loaded on first request (lazy loading).")
        logger.info("Set PRELOAD_MODELS=true in .env to preload during startup.")
    
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
