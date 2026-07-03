from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def add_cors_middleware(app: FastAPI) -> None:
    settings = get_settings()
    
    logger.info("Adding CORS middleware", extra={
        "origins": settings.cors_origins,
        "methods": settings.cors_methods
    })
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )
