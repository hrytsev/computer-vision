from fastapi import HTTPException
from typing import Dict
from PIL import Image
import torch
import torch.nn.functional as F
import io
import hashlib
import json
from api.deps.model_provider import get_model_provider
from repository.redis import RedisRepository
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class InferenceService:
    """Service for CV model inference"""
    
    def __init__(self, preds_mapper: Dict[int, str], redis_repo: RedisRepository):
        self.model_provider = get_model_provider()
        self.preds_mapper = preds_mapper
        self.redis_repo = redis_repo
        self.settings = get_settings()
    
    def _generate_cache_key(self, image_bytes: bytes) -> str:
        """Generate a cache key from image bytes using SHA256 hash"""
        return hashlib.sha256(image_bytes).hexdigest()
    
    async def predict(self, image_bytes: bytes) -> dict:
        """
        Run inference on an image with caching.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(image_bytes)
            logger.info("Checking cache", extra={"cache_key": cache_key})
            
            # Check cache first
            cached_result = self.redis_repo.get(cache_key)
            if cached_result:
                logger.info("Cache hit", extra={"cache_key": cache_key})
                return json.loads(cached_result)
            
            logger.info("Cache miss, running inference")
            
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            transform = self.model_provider.transform
            tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Move to device
            device = self.model_provider.device
            tensor = tensor.to(device)
            
            # Get model and run inference
            model = self.model_provider.get_model()
            
            with torch.no_grad():
                output = model(tensor)
            
            # Convert output to list for JSON serialization
            logits = output.cpu().numpy().tolist()
            
            # Calculate softmax for confidence scores
            softmax_probs = F.softmax(output, dim=1)
            confidence = softmax_probs.cpu().numpy().tolist()
            
            prediction_idx = output.argmax(dim=1).item()
            prediction_label = self.preds_mapper.get(prediction_idx, "unknown")
            prediction_confidence = softmax_probs[0][prediction_idx].item()
            
            result = {
                "logits": logits,
                "confidence": confidence,
                "prediction": prediction_label,
                "prediction_confidence": prediction_confidence,
                "shape": list(output.shape)
            }
            
            # Save to cache with TTL from config
            self.redis_repo.set(cache_key, json.dumps(result), ttl=self.settings.cache_ttl)
            
            logger.info("Inference completed successfully", extra={
                "output_shape": output.shape,
                "cache_key": cache_key
            })
            
            return result
            
        except Exception as e:
            logger.error("Inference failed", extra={"error": str(e)})
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )
