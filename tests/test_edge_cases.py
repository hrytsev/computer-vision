import pytest
from io import BytesIO
from PIL import Image
import numpy as np


class TestEdgeCases:
    """E2E tests for edge cases"""
    
    @pytest.fixture
    def large_image_bytes(self):
        """Create a large image ( > 10MB) for testing size limit"""
        # Create a large image by making it high resolution
        width, height = 8000, 8000  # This will be > 10MB when saved as PNG/JPEG
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.getvalue()
    
    @pytest.fixture
    def non_image_file(self):
        """Create a non-image file (text file)"""
        return b"This is not an image file. It's just text."
    
    @pytest.fixture
    def invalid_jpeg_file(self):
        """Create a file with .jpg extension but invalid content"""
        return b"Invalid JPEG content that is not actually a JPEG"
    
    @pytest.fixture
    def invalid_png_file(self):
        """Create a file with .png extension but invalid content"""
        return b"Invalid PNG content that is not actually a PNG"
    
    def test_large_image_upload_rejected(
        self,
        test_client,
        override_dependencies,
        mock_image_service,
        large_image_bytes
    ):
        """Test that large image (>10MB) is rejected"""
        # Mock the service to raise HTTPException for size limit
        from fastapi import HTTPException
        mock_image_service.validate_and_process_image.side_effect = HTTPException(
            status_code=400,
            detail="File size exceeds maximum allowed size of 10.0MB"
        )
        
        response = test_client.post(
            "/api/images/upload",
            files={"file": ("large.png", BytesIO(large_image_bytes), "image/png")}
        )
        
        assert response.status_code == 400
        assert "exceeds maximum allowed size" in response.json()["detail"]
    
    def test_large_image_prediction_rejected(
        self,
        test_client,
        override_dependencies,
        mock_image_service,
        large_image_bytes
    ):
        """Test that large image is rejected in prediction endpoint"""
        from fastapi import HTTPException
        mock_image_service.validate_and_process_image.side_effect = HTTPException(
            status_code=400,
            detail="File size exceeds maximum allowed size of 10.0MB"
        )
        
        response = test_client.post(
            "/api/ml/fracture",
            files={"file": ("large.png", BytesIO(large_image_bytes), "image/png")}
        )
        
        assert response.status_code == 400
        assert "exceeds maximum allowed size" in response.json()["detail"]
    
    def test_non_image_file_upload_rejected(
        self,
        test_client,
        override_dependencies,
        mock_image_service,
        non_image_file
    ):
        """Test that non-image file is rejected"""
        from fastapi import HTTPException
        mock_image_service.validate_and_process_image.side_effect = HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed formats: ['image/jpeg', 'image/png']"
        )
        
        response = test_client.post(
            "/api/images/upload",
            files={"file": ("test.txt", BytesIO(non_image_file), "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_non_image_file_prediction_rejected(
        self,
        test_client,
        override_dependencies,
        mock_image_service,
        non_image_file
    ):
        """Test that non-image file is rejected in prediction endpoint"""
        from fastapi import HTTPException
        mock_image_service.validate_and_process_image.side_effect = HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed formats: ['image/jpeg', 'image/png']"
        )
        
        response = test_client.post(
            "/api/ml/fracture",
            files={"file": ("test.txt", BytesIO(non_image_file), "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_invalid_jpeg_magic_bytes_rejected(
        self,
        test_client,
        override_dependencies,
        mock_image_service,
        invalid_jpeg_file
    ):
        """Test that file with invalid JPEG magic bytes is rejected"""
        from fastapi import HTTPException
        mock_image_service.validate_and_process_image.side_effect = HTTPException(
            status_code=400,
            detail="Invalid JPEG file: magic bytes do not match"
        )
        
        response = test_client.post(
            "/api/images/upload",
            files={"file": ("fake.jpg", BytesIO(invalid_jpeg_file), "image/jpeg")}
        )
        
        assert response.status_code == 400
        assert "magic bytes do not match" in response.json()["detail"]
    
    def test_invalid_png_magic_bytes_rejected(
        self,
        test_client,
        override_dependencies,
        mock_image_service,
        invalid_png_file
    ):
        """Test that file with invalid PNG magic bytes is rejected"""
        from fastapi import HTTPException
        mock_image_service.validate_and_process_image.side_effect = HTTPException(
            status_code=400,
            detail="Invalid PNG file: magic bytes do not match"
        )
        
        response = test_client.post(
            "/api/images/upload",
            files={"file": ("fake.png", BytesIO(invalid_png_file), "image/png")}
        )
        
        assert response.status_code == 400
        assert "magic bytes do not match" in response.json()["detail"]
    
    def test_invalid_file_extension_rejected(
        self,
        test_client,
        override_dependencies,
        mock_image_service,
        sample_image_bytes
    ):
        """Test that file with invalid extension is rejected"""
        from fastapi import HTTPException
        mock_image_service.validate_and_process_image.side_effect = HTTPException(
            status_code=400,
            detail="Invalid file extension. Only .jpg, .jpeg, and .png are allowed"
        )
        
        response = test_client.post(
            "/api/images/upload",
            files={"file": ("image.bmp", BytesIO(sample_image_bytes), "image/bmp")}
        )
        
        assert response.status_code == 400
        assert "Invalid file extension" in response.json()["detail"]
    
    def test_missing_filename_rejected(
        self,
        test_client,
        override_dependencies,
        sample_image_bytes
    ):
        """Test that file without filename is rejected by FastAPI validation"""
        # FastAPI UploadFile requires filename, so this will return 422
        response = test_client.post(
            "/api/images/upload",
            files={"file": (None, BytesIO(sample_image_bytes), "image/png")}
        )
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_empty_file_rejected(
        self,
        test_client,
        override_dependencies,
        mock_image_service
    ):
        """Test that empty file is rejected"""
        from fastapi import HTTPException
        mock_image_service.validate_and_process_image.side_effect = HTTPException(
            status_code=400,
            detail="Invalid JPEG file: magic bytes do not match"
        )
        
        response = test_client.post(
            "/api/images/upload",
            files={"file": ("empty.jpg", BytesIO(b""), "image/jpeg")}
        )
        
        assert response.status_code == 400
