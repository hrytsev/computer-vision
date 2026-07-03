# ResNet Fracture API

FastAPI application for bone fracture detection using a custom ResNet18 model trained on X-ray images.

## Motivation

This project addresses the need for automated bone fracture detection in medical imaging. By leveraging deep learning, the system can assist radiologists and medical professionals in quickly identifying fractures from X-ray images, potentially reducing diagnosis time and improving patient outcomes. The binary classification model distinguishes between fractured and non-fractured bones, providing a valuable tool for preliminary screening.

## Features

- Image upload endpoint (JPEG/PNG formats)
- File size validation
- CORS support
- Rate limiting
- Request ID tracking for debugging
- PyTorch integration for ML models
- Binary fracture classification (fractured/not fractured)

## Architecture

### Model Architecture

The project uses a custom ResNet18 implementation with the following components:

- **BasicBlock**: Residual building block with two 3x3 convolutional layers, batch normalization, and ReLU activation. Includes skip connections with optional downsampling for dimension matching.

- **ResNet18**: 18-layer residual network consisting of:
  - Initial 7x7 convolution with stride 2 (64 channels)
  - Max pooling layer
  - Four residual layers with progressive channel expansion:
    - Layer 1: 64 → 64 channels (2 blocks)
    - Layer 2: 64 → 128 channels (2 blocks, stride 2)
    - Layer 3: 128 → 256 channels (2 blocks, stride 2)
    - Layer 4: 256 → 512 channels (2 blocks, stride 2)
  - Adaptive average pooling
  - Fully connected layer for classification (2 output classes)

### System Architecture

The application follows a layered architecture:

- **API Layer** (`api/`): FastAPI routers handling HTTP requests
- **Service Layer** (`services/`): Business logic for image processing and inference
- **Model Layer** (`models/`): PyTorch model definitions and weights
- **Middleware Layer** (`middlewares/`): Cross-cutting concerns (CORS, rate limiting, request tracking)
- **Utility Layer** (`utils/`): Logging and prediction mapping
- **Configuration** (`config.py`): Centralized settings management

## Metrics

The model was trained on the Fracture Multi-region X-ray Dataset and achieved the following performance metrics on the validation set:

- **Accuracy**: ~99% (final training accuracy)
- **Training Loss**: Decreased from 0.4296 to 0.0181 over 26 epochs
- **Training Progress**: Model achieved >98% accuracy by epoch 6 and maintained >99% accuracy in later epochs

### Training Details

- **Dataset**: Fracture Multi-region X-ray Data (Kaggle)
- **Classes**: 2 (fractured, not fractured)
- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate: 1e-3)
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: 30 (with checkpointing)
- **Device**: GPU (CUDA) training, CPU inference

## Installation

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
# On fish shell: source .venv/bin/activate.fish
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root to override default settings:

```env
APP_NAME=ResNet Fracture API
APP_VERSION=1.0.0
DEBUG=False
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=["*"]
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
MAX_IMAGE_SIZE_MB=10.0
```

## Running the Application

Start the server:
```bash
uvicorn main:app --reload
```

Or use the built-in main:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Docker

Build the Docker image:
```bash
docker build -t resnet-fracture-api .
```

Run the Docker container:
```bash
docker run -p 8000:8000 resnet-fracture-api
```

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

### Test the health endpoint:
```bash
curl http://localhost:8000/health
```

### Test image upload:
```bash
curl -X POST "http://localhost:8000/api/images/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### Test fracture prediction:
```bash
curl -X POST "http://localhost:8000/api/ml/fracture" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### Test with Python:
```python
import requests

url = "http://localhost:8000/api/images/upload"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Project Structure

```
resnet-fracture/
├── api/
│   ├── __init__.py
│   ├── deps/
│   │   ├── __init__.py
│   │   ├── model_provider.py    # Model loading and management
│   │   └── service_deps.py      # Dependency injection
│   └── image_router.py          # API routes for images and ML
├── db/
│   ├── __init__.py
│   └── redis.py                 # Redis database connection
├── middlewares/
│   ├── __init__.py
│   ├── cors.py                  # CORS middleware
│   ├── rate_limit.py            # Rate limiting
│   └── request_id.py            # Request ID tracking
├── models/
│   ├── __init__.py
│   ├── resnet18.py              # ResNet18 model definition
│   ├── resnet18_fractured.pth   # Trained model weights
│   ├── checkpoint.pth           # Training checkpoint
│   └── resnet18_fractured/      # Model artifacts
├── notebooks/
│   └── resnet18_bones.ipynb     # Training notebook
├── repository/
│   ├── __init__.py
│   └── redis.py                 # Redis repository layer
├── services/
│   ├── __init__.py
│   ├── image_service.py         # Image validation and processing
│   └── inference_service.py     # ML inference logic
├── tests/
│   ├── __init__.py
│   ├── resources/
│   │   └── bones/               # Test resources
│   ├── test_image_service.py    # Image service tests
│   ├── test_inference_service.py # Inference service tests
│   └── test_preds_mapper.py     # Prediction mapper tests
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Logging configuration
│   └── preds_mapper.py          # Class label mapping
├── config.py                    # Configuration settings
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── Dockerfile                   # Docker configuration
└── README.md                    # This file
```
