from functools import lru_cache
from fastapi import Depends
from config import get_settings
from db.redis import RedisClient
from repository.redis import RedisRepository
from services.image_service import ImageService
from services.inference_service import InferenceService
from services.model_info_service import ModelInfoService
from utils.preds_mapper import PREDS_MAPPER


@lru_cache()
def get_image_service() -> ImageService:
    """Get cached image service instance"""
    return ImageService()


@lru_cache()
def get_redis_client() -> RedisClient:
    """Get cached Redis client instance"""
    settings = get_settings()
    return RedisClient(settings)


@lru_cache()
def get_redis_repository() -> RedisRepository:
    """Get cached Redis repository instance"""
    redis_client = get_redis_client()
    return RedisRepository(redis_client)


@lru_cache()
def get_inference_service() -> InferenceService:
    """Get cached inference service instance"""
    redis_repo = get_redis_repository()
    return InferenceService(preds_mapper=PREDS_MAPPER, redis_repo=redis_repo)


@lru_cache()
def get_model_info_service() -> ModelInfoService:
    """Get cached model info service instance"""
    return ModelInfoService()
