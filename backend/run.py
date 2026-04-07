"""Entry point for the ShotStory backend server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=300,  # 5 minutes for model loading on first request
        timeout_graceful_shutdown=30,
    )
