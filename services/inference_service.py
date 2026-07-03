from fastapi import HTTPException
from typing import Optional
from PIL import Image
import torch
import io
from api.deps.model_provider import get_model_provider
from utils.logger import get_logger

logger = get_logger(__name__)


class InferenceService:
    """Service for CV model inference"""
    
    def __init__(self):
        self.model_provider = get_model_provider()
    
    async def predict(self, image_bytes: bytes) -> dict:
        """
        Run inference on an image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with prediction results
        """
        try:
            logger.info("Starting inference")
            
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
            predictions = output.cpu().numpy().tolist()
            
            logger.info("Inference completed successfully", extra={
                "output_shape": output.shape
            })
            
            return {
                "predictions": predictions,
                "shape": list(output.shape)
            }
            
        except Exception as e:
            logger.error("Inference failed", extra={"error": str(e)})
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )
