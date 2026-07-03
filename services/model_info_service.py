import json
from pathlib import Path
from typing import Dict, Any
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelInfoService:
    """Service for retrieving model information from JSON and config"""
    
    def __init__(self):
        self.settings = get_settings()
        self._model_info: Dict[str, Any] | None = None
    
    def _load_model_info(self) -> Dict[str, Any]:
        """Load model info from JSON file"""
        if self._model_info is not None:
            return self._model_info
        
        model_path = Path(self.settings.model_path)
        json_path = model_path.parent / f"{model_path.stem}.json"
        
        logger.info("Loading model info", extra={"json_path": str(json_path)})
        
        try:
            if json_path.exists():
                with open(json_path, 'r') as f:
                    self._model_info = json.load(f)
                logger.info("Model info loaded successfully")
            else:
                logger.warning("Model info JSON not found, returning empty dict")
                self._model_info = {}
        except Exception as e:
            logger.error("Failed to load model info", extra={"error": str(e)})
            self._model_info = {}
        
        return self._model_info
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get complete model information combining JSON and config.
        
        Returns:
            Dictionary with model information including:
            - name: model name from config
            - info: additional info from JSON file
            - device: device from config
            - model_path: model path from config
        """
        json_info = self._load_model_info()
        
        return {
            "name": self.settings.model_path.split('/')[-1].replace('.pth', ''),
            "info": json_info,
            "device": self.settings.device,
            "model_path": self.settings.model_path
        }
