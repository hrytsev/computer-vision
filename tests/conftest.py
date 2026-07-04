import pytest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
import torch

from main import app
from config import get_settings
from api.deps.service_deps import get_image_service, get_inference_service, get_model_info_service
from services.image_service import ImageService
from services.inference_service import InferenceService
from services.model_info_service import ModelInfoService
from repository.redis import RedisRepository


@pytest.fixture
def test_settings():
    """Override settings for testing"""
    settings = get_settings()
    settings.debug = True
    settings.redis_host = "localhost"
    settings.redis_port = 6379
    settings.redis_db = 1  # Use separate DB for tests
    settings.cache_ttl = 1
    return settings


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def sample_fractured_image_path():
    """Path to sample fractured image"""
    return Path(__file__).parent / "resources" / "bones" / "fractured" / "002555.png"


@pytest.fixture
def sample_not_fractured_image_path():
    """Path to sample not fractured image"""
    return Path(__file__).parent / "resources" / "bones" / "not fractured" / "IMG0003941.jpg"


@pytest.fixture
def sample_image_bytes(sample_fractured_image_path):
    """Read sample image as bytes"""
    return sample_fractured_image_path.read_bytes()


@pytest.fixture
def mock_redis_repo():
    """Mock Redis repository"""
    repo = Mock(spec=RedisRepository)
    repo.get = Mock(return_value=None)
    repo.set = Mock()
    return repo


@pytest.fixture
def mock_model():
    """Mock PyTorch model"""
    model = Mock()
    model.eval = Mock()
    
    # Mock forward pass to return realistic output
    def mock_forward(x):
        # Return tensor with shape [batch_size, 2] for binary classification
        batch_size = x.shape[0] if len(x.shape) > 0 else 1
        return torch.randn(batch_size, 2)
    
    model.__call__ = mock_forward
    return model


@pytest.fixture
def mock_model_provider(mock_model):
    """Mock model provider"""
    provider = Mock()
    provider.get_model = Mock(return_value=mock_model)
    provider.transform = Mock()
    provider.device = torch.device("cpu")
    return provider


@pytest.fixture
def preds_mapper():
    """Prediction mapper"""
    return {0: "fractured", 1: "not fractured"}


@pytest.fixture
def mock_inference_service(mock_model_provider, preds_mapper, mock_redis_repo):
    """Mock inference service"""
    service = Mock(spec=InferenceService)
    service.predict = AsyncMock()
    service.model_provider = mock_model_provider
    service.preds_mapper = preds_mapper
    service.redis_repo = mock_redis_repo
    return service


@pytest.fixture
def mock_image_service():
    """Mock image service"""
    service = Mock(spec=ImageService)
    service.validate_and_process_image = AsyncMock()
    return service


@pytest.fixture
def mock_model_info_service():
    """Mock model info service"""
    service = Mock(spec=ModelInfoService)
    service.get_model_info = Mock(return_value={
        "model_name": "ResNet18",
        "num_classes": 2,
        "input_shape": [224, 224, 3],
        "classes": ["fractured", "not fractured"]
    })
    return service


@pytest.fixture
def override_dependencies(mock_image_service, mock_inference_service, mock_model_info_service):
    """Override FastAPI dependencies for testing"""
    def override_image_service():
        return mock_image_service
    
    def override_inference_service():
        return mock_inference_service
    
    def override_model_info_service():
        return mock_model_info_service
    
    app.dependency_overrides[get_image_service] = override_image_service
    app.dependency_overrides[get_inference_service] = override_inference_service
    app.dependency_overrides[get_model_info_service] = override_model_info_service
    
    yield
    
    app.dependency_overrides.clear()


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
