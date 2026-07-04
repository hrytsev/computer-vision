import pytest
import torch
from pathlib import Path
from PIL import Image
import io

from models.resnet18_fractured.resnet18 import ResNet18
from torchvision import transforms


class TestModelInference:
    """Unit tests for ResNet18 model inference"""
    
    @pytest.fixture
    def model(self):
        """Load the actual model for testing"""
        model = ResNet18(num_classes=2)
        model.eval()
        return model
    
    @pytest.fixture
    def transform(self):
        """Image transform pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    @pytest.fixture
    def fractured_image_tensor(self, sample_fractured_image_path, transform):
        """Load and transform fractured image"""
        image = Image.open(sample_fractured_image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = transform(image).unsqueeze(0)
        return tensor
    
    @pytest.fixture
    def not_fractured_image_tensor(self, sample_not_fractured_image_path, transform):
        """Load and transform not fractured image"""
        image = Image.open(sample_not_fractured_image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = transform(image).unsqueeze(0)
        return tensor
    
    def test_model_output_shape(self, model, fractured_image_tensor):
        """Test that model outputs correct shape"""
        with torch.no_grad():
            output = model(fractured_image_tensor)
        
        assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    
    def test_fractured_image_prediction(self, model, fractured_image_tensor):
        """Test that fractured image is predicted as fractured"""
        with torch.no_grad():
            output = model(fractured_image_tensor)
        
        prediction_idx = output.argmax(dim=1).item()
        prediction_label = "fractured" if prediction_idx == 0 else "not fractured"
        
        assert prediction_idx == 0, f"Expected fractured (0), got {prediction_label} ({prediction_idx})"
    
    def test_not_fractured_image_prediction(self, model, not_fractured_image_tensor):
        """Test that not fractured image is predicted as not fractured"""
        with torch.no_grad():
            output = model(not_fractured_image_tensor)
        
        prediction_idx = output.argmax(dim=1).item()
        prediction_label = "fractured" if prediction_idx == 0 else "not fractured"
        
        assert prediction_idx == 1, f"Expected not fractured (1), got {prediction_label} ({prediction_idx})"
    
    def test_model_forward_pass(self, model):
        """Test that model forward pass works with random input"""
        random_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(random_input)
        
        assert output is not None
        assert output.shape == (1, 2)
    
    def test_model_batch_processing(self, model):
        """Test that model can process batch of images"""
        batch_input = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            output = model(batch_input)
        
        assert output.shape == (4, 2)
    
    def test_model_output_type(self, model, fractured_image_tensor):
        """Test that model output is a tensor"""
        with torch.no_grad():
            output = model(fractured_image_tensor)
        
        assert isinstance(output, torch.Tensor)
    
    def test_model_inference_mode(self, model, fractured_image_tensor):
        """Test that model is in eval mode and doesn't track gradients"""
        assert not model.training, "Model should be in eval mode"
        
        with torch.no_grad():
            output = model(fractured_image_tensor)
        
        assert not output.requires_grad
    
    def test_prediction_confidence(self, model, fractured_image_tensor):
        """Test that prediction confidence is between 0 and 1"""
        import torch.nn.functional as F
        
        with torch.no_grad():
            output = model(fractured_image_tensor)
        
        softmax_probs = F.softmax(output, dim=1)
        prediction_idx = output.argmax(dim=1).item()
        confidence = softmax_probs[0][prediction_idx].item()
        
        assert 0 <= confidence <= 1, f"Confidence should be between 0 and 1, got {confidence}"
