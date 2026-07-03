from functools import lru_cache
from fastapi import Depends
from services.image_service import ImageService
from services.inference_service import InferenceService
from utils.preds_mapper import PREDS_MAPPER


@lru_cache()
def get_image_service() -> ImageService:
    """Get cached image service instance"""
    return ImageService()


@lru_cache()
def get_inference_service() -> InferenceService:
    """Get cached inference service instance"""
    return InferenceService(preds_mapper=PREDS_MAPPER)
