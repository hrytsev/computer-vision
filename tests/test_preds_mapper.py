import pytest
from services.inference_service import InferenceService
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image
from io import BytesIO

PREDS_MAPPER = {
    0: "fractured",
    1: "not fractured"
}

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = MagicMock()
    # Mock the forward pass to return a tensor for "fractured"
    output = torch.tensor([[0.9, 0.1]])
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

@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_fractured(mock_get_provider, valid_image_bytes, mock_model):
    """Test prediction for 'fractured' class"""
    mock_provider = Mock()
    mock_provider.transform = Mock(return_value=torch.rand(1, 3, 224, 224))
    mock_provider.device = torch.device('cpu')
    mock_provider.get_model.return_value = mock_model
    mock_get_provider.return_value = mock_provider

    service = InferenceService(preds_mapper=PREDS_MAPPER)
    result = await service.predict(valid_image_bytes)

    assert result["prediction"] == "fractured"

@pytest.mark.asyncio
@patch('services.inference_service.get_model_provider')
async def test_predict_not_fractured(mock_get_provider, valid_image_bytes, mock_model):
    """Test prediction for 'not fractured' class"""
    # Update model output for "not fractured"
    mock_model.return_value = torch.tensor([[0.1, 0.9]])

    mock_provider = Mock()
    mock_provider.transform = Mock(return_value=torch.rand(1, 3, 224, 224))
    mock_provider.device = torch.device('cpu')
    mock_provider.get_model.return_value = mock_model
    mock_get_provider.return_value = mock_provider

    service = InferenceService(preds_mapper=PREDS_MAPPER)
    result = await service.predict(valid_image_bytes)

    assert result["prediction"] == "not fractured"
