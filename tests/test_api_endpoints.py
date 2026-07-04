

class TestAPIEndpoints:
    """E2E tests for API endpoints"""
    
    def test_health_endpoint(self, test_client):
        """Test /health endpoint returns healthy status"""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API info"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_model_info_endpoint(self, test_client, override_dependencies):
        """Test /api/ml/model-info endpoint"""
        response = test_client.get("/api/ml/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "num_classes" in data
        assert "input_shape" in data
        assert "classes" in data
    
    def test_fracture_prediction_with_fractured_image(
        self, 
        test_client, 
        override_dependencies,
        mock_inference_service,
        mock_image_service,
        sample_fractured_image_path,
        sample_image_bytes
    ):
        """Test fracture prediction endpoint with fractured image"""
        # Setup mocks
        mock_image_service.validate_and_process_image.return_value = {
            "filename": "002555.png",
            "content_type": "image/png",
            "size_mb": 0.37,
            "content": sample_image_bytes
        }
        
        # Mock inference to return fractured prediction
        mock_inference_service.predict.return_value = {
            "logits": [[2.5, -1.0]],
            "confidence": [[0.95, 0.05]],
            "prediction": "fractured",
            "prediction_confidence": 0.95,
            "shape": [1, 2]
        }
        
        with open(sample_fractured_image_path, "rb") as f:
            response = test_client.post(
                "/api/ml/fracture",
                files={"file": ("002555.png", f, "image/png")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "fractured"
        assert data["prediction_confidence"] > 0.5
        assert "logits" in data
        assert "confidence" in data
        assert "shape" in data
        assert "request_id" in data
        
        # Verify mocks were called
        mock_image_service.validate_and_process_image.assert_called_once()
        mock_inference_service.predict.assert_called_once()
    
    def test_fracture_prediction_with_not_fractured_image(
        self,
        test_client,
        override_dependencies,
        mock_inference_service,
        mock_image_service,
        sample_not_fractured_image_path,
        sample_image_bytes
    ):
        """Test fracture prediction endpoint with not fractured image"""
        # Setup mocks
        mock_image_service.validate_and_process_image.return_value = {
            "filename": "IMG0003941.jpg",
            "content_type": "image/jpeg",
            "size_mb": 0.02,
            "content": sample_image_bytes
        }
        
        # Mock inference to return not fractured prediction
        mock_inference_service.predict.return_value = {
            "logits": [[-1.0, 2.5]],
            "confidence": [[0.05, 0.95]],
            "prediction": "not fractured",
            "prediction_confidence": 0.95,
            "shape": [1, 2]
        }
        
        with open(sample_not_fractured_image_path, "rb") as f:
            response = test_client.post(
                "/api/ml/fracture",
                files={"file": ("IMG0003941.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "not fractured"
        assert data["prediction_confidence"] > 0.5
    
    def test_upload_image_endpoint(
        self,
        test_client,
        override_dependencies,
        mock_image_service,
        sample_fractured_image_path,
        sample_image_bytes
    ):
        """Test image upload endpoint"""
        mock_image_service.validate_and_process_image.return_value = {
            "filename": "002555.png",
            "content_type": "image/png",
            "size_mb": 0.37,
            "content": sample_image_bytes
        }
        
        with open(sample_fractured_image_path, "rb") as f:
            response = test_client.post(
                "/api/images/upload",
                files={"file": ("002555.png", f, "image/png")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Image uploaded successfully"
        assert "filename" in data
        assert "content_type" in data
        assert "size_mb" in data
        assert "request_id" in data
    
    def test_fracture_prediction_missing_file(self, test_client, override_dependencies):
        """Test fracture prediction without file returns error"""
        response = test_client.post("/api/ml/fracture")
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_upload_image_missing_file(self, test_client, override_dependencies):
        """Test image upload without file returns error"""
        response = test_client.post("/api/images/upload")
        
        assert response.status_code == 422  # Unprocessable Entity
