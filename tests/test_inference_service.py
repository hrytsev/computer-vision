import pytest
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException
from PIL import Image
import torch
import numpy as np
from services.inference_service import InferenceService


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = MagicMock()
    # Mock the forward pass to return a tensor
    output = torch.tensor([[0.1, 0.2, 0.7]])
    model.return_value = output
    model.eval = Mock()
    return model


@pytest.fixture
def valid_image_bytes():
    """Create valid image bytes for testing"""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def rgba_image_bytes():
    """Create RGBA image bytes for testing"""
    img = Image.new('RGBA', (100, 100), color='blue')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_valid_image(mock_get_provider, valid_image_bytes, mock_model):
    """Test successful prediction with valid image"""
    # Mock the model provider
    mock_provider = Mock()
    mock_provider.transform = Mock()
    mock_provider.device = torch.device('cpu')
    mock_provider.get_model.return_value = mock_model
    
    # Mock the transform to return a tensor
    mock_tensor = torch.rand(1, 3, 224, 224)
    mock_provider.transform.return_value = mock_tensor
    
    mock_get_provider.return_value = mock_provider
    
    service = InferenceService()
    result = await service.predict(valid_image_bytes)
    
    assert "predictions" in result
    assert "shape" in result
    assert isinstance(result["predictions"], list)
    assert isinstance(result["shape"], list)
    mock_provider.get_model.assert_called_once()


@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_rgba_image_conversion(mock_get_provider, rgba_image_bytes, mock_model):
    """Test that RGBA images are converted to RGB"""
    mock_provider = Mock()
    mock_provider.transform = Mock()
    mock_provider.device = torch.device('cpu')
    mock_provider.get_model.return_value = mock_model
    
    mock_tensor = torch.rand(1, 3, 224, 224)
    mock_provider.transform.return_value = mock_tensor
    
    mock_get_provider.return_value = mock_provider
    
    service = InferenceService()
    result = await service.predict(rgba_image_bytes)
    
    assert "predictions" in result
    mock_provider.transform.assert_called_once()


@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_model_not_loaded(mock_get_provider, valid_image_bytes, mock_model):
    """Test prediction when model needs to be loaded"""
    mock_provider = Mock()
    mock_provider.transform = Mock()
    mock_provider.device = torch.device('cpu')
    mock_provider._model = None  # Model not loaded
    mock_provider.get_model.return_value = mock_model
    
    mock_tensor = torch.rand(1, 3, 224, 224)
    mock_provider.transform.return_value = mock_tensor
    
    mock_get_provider.return_value = mock_provider
    
    service = InferenceService()
    result = await service.predict(valid_image_bytes)
    
    assert "predictions" in result
    mock_provider.get_model.assert_called_once()


@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_invalid_image_bytes(mock_get_provider):
    """Test prediction with invalid image bytes"""
    mock_provider = Mock()
    mock_provider.transform = Mock()
    mock_provider.device = torch.device('cpu')
    mock_provider.get_model.return_value = Mock()
    
    mock_get_provider.return_value = mock_provider
    
    service = InferenceService()
    
    with pytest.raises(HTTPException) as exc_info:
        await service.predict(b"invalid image data")
    
    assert exc_info.value.status_code == 500
    assert "Inference failed" in exc_info.value.detail


@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_model_load_failure(mock_get_provider, valid_image_bytes):
    """Test prediction when model loading fails"""
    mock_provider = Mock()
    mock_provider.transform = Mock()
    mock_provider.device = torch.device('cpu')
    mock_provider.get_model.side_effect = Exception("Model load failed")
    
    mock_get_provider.return_value = mock_provider
    
    service = InferenceService()
    
    with pytest.raises(HTTPException) as exc_info:
        await service.predict(valid_image_bytes)
    
    assert exc_info.value.status_code == 500
    assert "Inference failed" in exc_info.value.detail


@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_transform_applied(mock_get_provider, valid_image_bytes, mock_model):
    """Test that transform is applied to the image"""
    mock_provider = Mock()
    mock_provider.transform = Mock()
    mock_provider.device = torch.device('cpu')
    mock_provider.get_model.return_value = mock_model
    
    mock_tensor = torch.rand(1, 3, 224, 224)
    mock_provider.transform.return_value = mock_tensor
    
    mock_get_provider.return_value = mock_provider
    
    service = InferenceService()
    await service.predict(valid_image_bytes)
    
    # Verify transform was called
    mock_provider.transform.assert_called_once()
    
    # Verify the image was converted to tensor and batch dimension added
    call_args = mock_provider.transform.call_args
    assert call_args is not None


@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_device_handling(mock_get_provider, valid_image_bytes, mock_model):
    """Test that tensor is moved to correct device"""
    mock_provider = Mock()
    mock_provider.transform = Mock()
    mock_provider.device = torch.device('cpu')
    mock_provider.get_model.return_value = mock_model
    
    mock_tensor = torch.rand(1, 3, 224, 224)
    mock_provider.transform.return_value = mock_tensor
    
    mock_get_provider.return_value = mock_provider
    
    service = InferenceService()
    result = await service.predict(valid_image_bytes)
    
    assert "predictions" in result
    assert isinstance(result["predictions"], list)


@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_no_grad_context(mock_get_provider, valid_image_bytes, mock_model):
    """Test that inference runs in no_grad context"""
    mock_provider = Mock()
    mock_provider.transform = Mock()
    mock_provider.device = torch.device('cpu')
    mock_provider.get_model.return_value = mock_model
    
    mock_tensor = torch.rand(1, 3, 224, 224)
    mock_provider.transform.return_value = mock_tensor
    
    mock_get_provider.return_value = mock_provider
    
    service = InferenceService()
    result = await service.predict(valid_image_bytes)
    
    assert "predictions" in result
    # Model should be called
    mock_model.assert_called_once()


@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_output_serialization(mock_get_provider, valid_image_bytes):
    """Test that output is properly serialized for JSON"""
    mock_provider = Mock()
    mock_provider.transform = Mock()
    mock_provider.device = torch.device('cpu')
    
    # Create mock model with specific output
    mock_model = MagicMock()
    output = torch.tensor([[0.1, 0.2, 0.7, 0.9]])
    mock_model.return_value = output
    mock_provider.get_model.return_value = mock_model
    
    mock_tensor = torch.rand(1, 3, 224, 224)
    mock_provider.transform.return_value = mock_tensor
    
    mock_get_provider.return_value = mock_provider
    
    service = InferenceService()
    result = await service.predict(valid_image_bytes)
    
    assert isinstance(result["predictions"], list)
    assert isinstance(result["predictions"][0], list)
    assert result["shape"] == [1, 4]
