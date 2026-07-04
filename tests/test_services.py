import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from io import BytesIO
from PIL import Image
import hashlib
import json

from services.image_service import ImageService, settings_context
from services.inference_service import InferenceService
from services.model_info_service import ModelInfoService
from config import Settings
from fastapi import UploadFile


class TestImageService:
    """Unit tests for ImageService"""
    
    @pytest.fixture
    def image_service(self):
        """Create ImageService instance"""
        return ImageService()
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings"""
        return Settings(
            max_image_size_mb=10.0,
            allowed_image_formats=["image/jpeg", "image/png"]
        )
    
    @pytest.fixture
    def valid_jpeg_bytes(self):
        """Create valid JPEG bytes with correct magic bytes"""
        # JPEG magic bytes: FF D8 FF
        return bytes([0xFF, 0xD8, 0xFF] + [0] * 100)
    
    @pytest.fixture
    def valid_png_bytes(self):
        """Create valid PNG bytes with correct magic bytes"""
        # PNG magic bytes: 89 50 4E 47 44 49 48
        return bytes([0x89, 0x50, 0x4E, 0x47, 0x44, 0x49, 0x48] + [0] * 100)
    
    def test_validate_magic_bytes_valid_jpeg(self, image_service):
        """Test JPEG magic bytes validation with valid JPEG"""
        # Valid JPEG magic bytes
        content = bytes([0xFF, 0xD8, 0xFF] + [0] * 5)
        
        # Should not raise exception
        image_service._validate_magic_bytes(content, "image/jpeg")
    
    def test_validate_magic_bytes_invalid_jpeg(self, image_service):
        """Test JPEG magic bytes validation with invalid JPEG"""
        # Invalid JPEG magic bytes
        content = bytes([0x00, 0x00, 0x00] + [0] * 5)
        
        with pytest.raises(Exception) as exc_info:
            image_service._validate_magic_bytes(content, "image/jpeg")
        
        assert "magic bytes do not match" in str(exc_info.value)
    
    def test_validate_magic_bytes_valid_png(self, image_service):
        """Test PNG magic bytes validation with valid PNG"""
        # Valid PNG magic bytes
        content = bytes([0x89, 0x50, 0x4E, 0x47, 0x44, 0x49, 0x48] + [0])
        
        # Should not raise exception
        image_service._validate_magic_bytes(content, "image/png")
    
    def test_validate_magic_bytes_invalid_png(self, image_service):
        """Test PNG magic bytes validation with invalid PNG"""
        # Invalid PNG magic bytes
        content = bytes([0x00, 0x00, 0x00] + [0] * 5)
        
        with pytest.raises(Exception) as exc_info:
            image_service._validate_magic_bytes(content, "image/png")
        
        assert "magic bytes do not match" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_and_process_image_success(self, image_service, mock_settings, valid_jpeg_bytes):
        """Test successful image validation and processing"""
        # Set settings in context
        token = settings_context.set(mock_settings)
        
        try:
            # Create mock upload file
            file = Mock(spec=UploadFile)
            file.filename = "test.jpg"
            file.content_type = "image/jpeg"
            file.read = AsyncMock(return_value=valid_jpeg_bytes)
            file.seek = AsyncMock()
            
            result = await image_service.validate_and_process_image(file)
            
            assert result["filename"] == "test.jpg"
            assert result["content_type"] == "image/jpeg"
            assert "size_mb" in result
            assert "content" in result
            assert result["size_mb"] < mock_settings.max_image_size_mb
        finally:
            settings_context.reset(token)
    
    @pytest.mark.asyncio
    async def test_validate_and_process_image_invalid_type(self, image_service, mock_settings, valid_jpeg_bytes):
        """Test image validation with invalid content type"""
        token = settings_context.set(mock_settings)
        
        try:
            file = Mock(spec=UploadFile)
            file.filename = "test.txt"
            file.content_type = "text/plain"
            file.read = AsyncMock(return_value=valid_jpeg_bytes)
            
            with pytest.raises(Exception) as exc_info:
                await image_service.validate_and_process_image(file)
            
            assert "Invalid file type" in str(exc_info.value)
        finally:
            settings_context.reset(token)
    
    @pytest.mark.asyncio
    async def test_validate_and_process_image_invalid_extension(self, image_service, mock_settings, valid_jpeg_bytes):
        """Test image validation with invalid file extension"""
        token = settings_context.set(mock_settings)
        
        try:
            file = Mock(spec=UploadFile)
            file.filename = "test.bmp"
            file.content_type = "image/jpeg"
            file.read = AsyncMock(return_value=valid_jpeg_bytes)
            
            with pytest.raises(Exception) as exc_info:
                await image_service.validate_and_process_image(file)
            
            assert "Invalid file extension" in str(exc_info.value)
        finally:
            settings_context.reset(token)
    
    @pytest.mark.asyncio
    async def test_validate_and_process_image_missing_filename(self, image_service, mock_settings, valid_jpeg_bytes):
        """Test image validation with missing filename"""
        token = settings_context.set(mock_settings)
        
        try:
            file = Mock(spec=UploadFile)
            file.filename = None
            file.content_type = "image/jpeg"
            file.read = AsyncMock(return_value=valid_jpeg_bytes)
            
            with pytest.raises(Exception) as exc_info:
                await image_service.validate_and_process_image(file)
            
            assert "Filename is required" in str(exc_info.value)
        finally:
            settings_context.reset(token)
    
    @pytest.mark.asyncio
    async def test_validate_and_process_image_size_exceeded(self, image_service, mock_settings):
        """Test image validation when size exceeds limit"""
        # Create large content with valid JPEG magic bytes
        # JPEG magic bytes: FF D8 FF
        large_content = bytes([0xFF, 0xD8, 0xFF]) + b"x" * (11 * 1024 * 1024)  # 11 MB
        
        token = settings_context.set(mock_settings)
        
        try:
            file = Mock(spec=UploadFile)
            file.filename = "large.jpg"
            file.content_type = "image/jpeg"
            file.read = AsyncMock(return_value=large_content)
            file.seek = AsyncMock()
            
            with pytest.raises(Exception) as exc_info:
                await image_service.validate_and_process_image(file)
            
            assert "exceeds maximum allowed size" in str(exc_info.value)
        finally:
            settings_context.reset(token)


class TestInferenceService:
    """Unit tests for InferenceService"""
    
    @pytest.fixture
    def inference_service(self, mock_model_provider, preds_mapper, mock_redis_repo):
        """Create InferenceService instance"""
        return InferenceService(preds_mapper, mock_redis_repo)
    
    def test_generate_cache_key(self, inference_service):
        """Test cache key generation"""
        image_bytes = b"test image data"
        cache_key = inference_service._generate_cache_key(image_bytes)
        
        # Should be SHA256 hash
        expected_key = hashlib.sha256(image_bytes).hexdigest()
        assert cache_key == expected_key
        assert len(cache_key) == 64  # SHA256 produces 64 hex characters
    
    def test_generate_cache_key_different_inputs(self, inference_service):
        """Test that different inputs produce different cache keys"""
        image_bytes1 = b"test image data 1"
        image_bytes2 = b"test image data 2"
        
        cache_key1 = inference_service._generate_cache_key(image_bytes1)
        cache_key2 = inference_service._generate_cache_key(image_bytes2)
        
        assert cache_key1 != cache_key2
    
    def test_generate_cache_key_same_inputs(self, inference_service):
        """Test that same inputs produce same cache keys"""
        image_bytes = b"test image data"
        
        cache_key1 = inference_service._generate_cache_key(image_bytes)
        cache_key2 = inference_service._generate_cache_key(image_bytes)
        
        assert cache_key1 == cache_key2
    
    @pytest.mark.asyncio
    async def test_predict_cache_hit(self, inference_service, mock_redis_repo, sample_image_bytes):
        """Test prediction when cache hit occurs"""
        # Setup cache hit
        cached_result = json.dumps({
            "logits": [[1.0, -1.0]],
            "confidence": [[0.73, 0.27]],
            "prediction": "fractured",
            "prediction_confidence": 0.73,
            "shape": [1, 2]
        })
        mock_redis_repo.get.return_value = cached_result
        
        result = await inference_service.predict(sample_image_bytes)
        
        assert result["prediction"] == "fractured"
        mock_redis_repo.get.assert_called_once()
        # Should not call model if cache hit
        mock_redis_repo.set.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_predict_cache_miss(self, inference_service, mock_redis_repo, mock_model_provider, sample_image_bytes):
        """Test prediction when cache miss occurs"""
        import torch
        import torch.nn.functional as F
        
        # Setup cache miss
        mock_redis_repo.get.return_value = None
        
        # Mock model output
        mock_output = torch.tensor([[2.5, -1.0]])
        mock_model_provider.get_model.return_value = Mock(__call__=Mock(return_value=mock_output))
        
        result = await inference_service.predict(sample_image_bytes)
        
        assert "prediction" in result
        assert "confidence" in result
        mock_redis_repo.get.assert_called_once()
        mock_redis_repo.set.assert_called_once()


class TestModelInfoService:
    """Unit tests for ModelInfoService"""
    
    @pytest.fixture
    def model_info_service(self, test_settings):
        """Create ModelInfoService instance with test settings"""
        with patch('services.model_info_service.get_settings', return_value=test_settings):
            return ModelInfoService()
    
    def test_get_model_info(self, model_info_service):
        """Test getting model info"""
        result = model_info_service.get_model_info()
        
        assert "name" in result
        assert "info" in result
        assert "device" in result
        assert "model_path" in result
    
    def test_load_model_info_cached(self, model_info_service):
        """Test that model info is cached after first load"""
        # First call
        result1 = model_info_service._load_model_info()
        
        # Second call should return cached result
        result2 = model_info_service._load_model_info()
        
        assert result1 is result2
    
    def test_get_model_info_structure(self, model_info_service):
        """Test that model info has correct structure"""
        result = model_info_service.get_model_info()
        
        # Check that all expected keys are present
        expected_keys = ["name", "info", "device", "model_path"]
        for key in expected_keys:
            assert key in result
        
        # Check types
        assert isinstance(result["name"], str)
        assert isinstance(result["info"], dict)
        assert isinstance(result["device"], str)
        assert isinstance(result["model_path"], str)
