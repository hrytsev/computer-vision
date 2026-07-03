from functools import lru_cache
from typing import Protocol
import torch
from torchvision import transforms
from PIL import Image
from config import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelProtocol(Protocol):
    """Protocol for model interface"""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class ModelProvider:
    """Dependency provider for loading and managing the CV model"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: ModelProtocol | None = None
        self._transform: transforms.Compose | None = None
        self._device: torch.device = torch.device(settings.device)
        
    @property
    def transform(self) -> transforms.Compose:
        """Get the image transform pipeline"""
        if self._transform is None:
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            logger.info("Transform pipeline initialized")
        return self._transform
    
    @property
    def device(self) -> torch.device:
        """Get the device for model inference"""
        return self._device
    
    def load_model(self) -> ModelProtocol:
        """Load the model from pth file"""
        if self._model is not None:
            return self._model
            
        logger.info("Loading model", extra={"path": self.settings.model_path})
        
        try:
            self._model = torch.load(self.settings.model_path, map_location=self._device)
            self._model.eval()
            logger.info("Model loaded successfully", extra={
                "device": str(self._device)
            })
            return self._model
        except Exception as e:
            logger.error("Failed to load model", extra={"error": str(e)})
            raise
    
    def get_model(self) -> ModelProtocol:
        """Get the loaded model instance"""
        return self.load_model()


@lru_cache()
def get_model_provider() -> ModelProvider:
    """Get cached model provider instance"""
    from config import get_settings
    settings = get_settings()
    return ModelProvider(settings)
