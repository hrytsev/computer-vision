import pytest
from io import BytesIO
from unittest.mock import Mock, AsyncMock
from fastapi import HTTPException
from PIL import Image
from services.image_service import ImageService
from contextvars import ContextVar
from config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    settings = Settings()
    settings.max_image_size_mb = 10.0
    settings.allowed_image_formats = ["image/jpeg", "image/png"]
    return settings


@pytest.fixture
def image_service(mock_settings):
    """Create image service instance with mocked settings"""
    from services.image_service import settings_context
    settings_context.set(mock_settings)
    return ImageService()


@pytest.fixture
def valid_jpg_image():
    """Create a valid JPG image for testing"""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def valid_png_image():
    """Create a valid PNG image for testing"""
    img = Image.new('RGB', (100, 100), color='blue')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def oversized_image():
    """Create an oversized image (>10MB) for testing"""
    # Create a large byte array that exceeds 10MB
    large_content = b'x' * (11 * 1024 * 1024)  # 11MB
    img_bytes = BytesIO(large_content)
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def mock_upload_file():
    """Create a mock UploadFile object"""
    def _create_upload_file(filename, content_type, content):
        mock_file = Mock()
        mock_file.filename = filename
        mock_file.content_type = content_type
        mock_file.read = AsyncMock(return_value=content)
        mock_file.seek = AsyncMock()
        return mock_file
    return _create_upload_file


@pytest.mark.asyncio
async def test_validate_and_process_valid_jpg(image_service, valid_jpg_image, mock_upload_file):
    """Test validation of a valid JPG image"""
    content = valid_jpg_image.read()
    valid_jpg_image.seek(0)
    
    upload_file = mock_upload_file("test.jpg", "image/jpeg", content)
    
    result = await image_service.validate_and_process_image(upload_file)
    
    assert result["filename"] == "test.jpg"
    assert result["content_type"] == "image/jpeg"
    assert result["size_mb"] < 10.0
    assert result["content"] == content


@pytest.mark.asyncio
async def test_validate_and_process_valid_png(image_service, valid_png_image, mock_upload_file):
    """Test validation of a valid PNG image"""
    content = valid_png_image.read()
    valid_png_image.seek(0)
    
    upload_file = mock_upload_file("test.png", "image/png", content)
    
    result = await image_service.validate_and_process_image(upload_file)
    
    assert result["filename"] == "test.png"
    assert result["content_type"] == "image/png"
    assert result["size_mb"] < 10.0
    assert result["content"] == content


@pytest.mark.asyncio
async def test_validate_and_process_oversized_image(image_service, oversized_image, mock_upload_file):
    """Test validation rejects oversized image"""
    content = oversized_image.read()
    oversized_image.seek(0)
    
    upload_file = mock_upload_file("large.jpg", "image/jpeg", content)
    
    with pytest.raises(HTTPException) as exc_info:
        await image_service.validate_and_process_image(upload_file)
    
    assert exc_info.value.status_code == 400
    assert "File size exceeds maximum allowed size" in exc_info.value.detail


@pytest.mark.asyncio
async def test_validate_and_process_invalid_content_type(image_service, mock_upload_file):
    """Test validation rejects invalid content type"""
    content = b"fake image content"
    upload_file = mock_upload_file("test.gif", "image/gif", content)
    
    with pytest.raises(HTTPException) as exc_info:
        await image_service.validate_and_process_image(upload_file)
    
    assert exc_info.value.status_code == 400
    assert "Invalid file type" in exc_info.value.detail


@pytest.mark.asyncio
async def test_validate_and_process_invalid_extension(image_service, valid_jpg_image, mock_upload_file):
    """Test validation rejects invalid file extension"""
    content = valid_jpg_image.read()
    valid_jpg_image.seek(0)
    
    upload_file = mock_upload_file("test.gif", "image/jpeg", content)
    
    with pytest.raises(HTTPException) as exc_info:
        await image_service.validate_and_process_image(upload_file)
    
    assert exc_info.value.status_code == 400
    assert "Invalid file extension" in exc_info.value.detail


@pytest.mark.asyncio
async def test_validate_and_process_missing_filename(image_service, valid_jpg_image, mock_upload_file):
    """Test validation rejects missing filename"""
    content = valid_jpg_image.read()
    valid_jpg_image.seek(0)
    
    upload_file = mock_upload_file(None, "image/jpeg", content)
    
    with pytest.raises(HTTPException) as exc_info:
        await image_service.validate_and_process_image(upload_file)
    
    assert exc_info.value.status_code == 400
    assert "Filename is required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_validate_and_process_lowercase_jpeg_extension(image_service, valid_jpg_image, mock_upload_file):
    """Test validation accepts .jpeg extension"""
    content = valid_jpg_image.read()
    valid_jpg_image.seek(0)
    
    upload_file = mock_upload_file("test.jpeg", "image/jpeg", content)
    
    result = await image_service.validate_and_process_image(upload_file)
    
    assert result["filename"] == "test.jpeg"
    assert result["content_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_validate_and_process_uppercase_extension(image_service, valid_jpg_image, mock_upload_file):
    """Test validation accepts uppercase extensions"""
    content = valid_jpg_image.read()
    valid_jpg_image.seek(0)
    
    upload_file = mock_upload_file("test.JPG", "image/jpeg", content)
    
    result = await image_service.validate_and_process_image(upload_file)
    
    assert result["filename"] == "test.JPG"
    assert result["content_type"] == "image/jpeg"
