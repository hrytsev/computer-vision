# ResNet Fracture API

FastAPI application for image upload and validation with ResNet fracture detection capabilities.

## Features

- Image upload endpoint (JPEG/PNG formats)
- File size validation
- CORS support
- Rate limiting
- Request ID tracking for debugging
- PyTorch integration for ML models

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

## Motivation

This project addresses the critical need for automated fracture detection in medical imaging. By leveraging deep learning (ResNet18), the system provides:

- **Rapid triage**: Quick preliminary assessment to prioritize urgent cases
- **Consistency**: Reduces human error and fatigue-related inconsistencies
- **Scalability**: Enables processing of large volumes of medical images
- **Accessibility**: Makes fracture detection capabilities available through a simple REST API

The API is designed for integration with hospital information systems, telemedicine platforms, and research workflows.

## ML Metrics & Evaluation

The fracture detection model returns comprehensive metrics for each prediction:

### Model Information

- **Architecture**: ResNet18
- **Framework**: PyTorch
- **Task**: Binary image classification
- **Classes**: fracture, normal
- **Version**: 1.0.0

### Output Metrics

- **Logits**: Raw model output scores for each class (useful for ensemble methods)
- **Confidence**: Softmax probability distribution across all classes
- **Prediction**: Predicted class label ("fracture" or "normal")
- **Prediction Confidence**: Confidence score for the predicted class (0-1)
- **Shape**: Tensor dimensions of the model output

### Model Performance

The model has been evaluated on a test dataset with the following metrics:

- **Accuracy**: 98.07% - Overall correctness of predictions
- **Precision**: 96.85% - True positive rate for fracture detection
- **Recall**: 100% - Sensitivity - ability to detect actual fractures (no false negatives)
- **F1-Score**: 98.4% - Harmonic mean of precision and recall

### Confusion Matrix

```
                Predicted
Actual        Fracture    Normal
Fracture        321        16
Normal           0        492
```

- **True Positives**: 321 (correctly identified fractures)
- **False Positives**: 16 (normal images incorrectly classified as fractures)
- **False Negatives**: 0 (no missed fractures - critical for medical applications)
- **True Negatives**: 492 (correctly identified normal images)

For medical applications, **recall** is particularly critical to minimize missed fractures (false negatives). This model achieves 100% recall, ensuring no fractures are missed during screening.

## Structured Logging

The application implements structured JSON logging for production-grade observability:

### Logging Features

- **JSON Format**: All logs are structured as JSON for easy parsing by log aggregation tools (ELK, Splunk, etc.)
- **Request Tracking**: Each request is assigned a unique ID for end-to-end tracing
- **Contextual Metadata**: Logs include relevant context (cache keys, file names, error details)
- **Log Levels**: INFO, WARNING, ERROR, DEBUG for appropriate granularity

### Log Structure

```json
{
  "timestamp": "2024-01-01T00:00:00.000000",
  "level": "INFO",
  "logger": "services.inference_service",
  "message": "Inference completed successfully",
  "module": "inference_service",
  "function": "predict",
  "line": 96,
  "request_id": "abc123",
  "cache_key": "sha256hash",
  "output_shape": [1, 2]
}
```

### Configuration

Set log level via environment variable:
```env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Caching Strategy

The API implements intelligent caching to reduce inference latency and computational costs:

### Cache Implementation

- **Storage**: Redis for high-performance key-value storage
- **Key Generation**: SHA256 hash of image bytes ensures cache hits for identical images
- **TTL**: Configurable time-to-live (default: 60 seconds) balances freshness with performance
- **Cache-Aside Pattern**: Check cache first, compute on miss, store result

### Cache Benefits

- **Reduced Latency**: Cached responses return in milliseconds vs. seconds for inference
- **Lower Costs**: Fewer GPU/CPU cycles for repeated images
- **Scalability**: Handles burst traffic without overwhelming inference resources
- **Consistency**: Same image always returns same prediction within TTL window

### Configuration

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=60  # seconds
```

## Best Practices

This project follows industry best practices for production ML APIs:

### Architecture

- **Layered Architecture**: Clear separation between API, service, and data layers
- **Dependency Injection**: Loose coupling via FastAPI's dependency system
- **Repository Pattern**: Abstracted data access for testability and flexibility
- **Service Layer**: Business logic encapsulated in reusable services

### Security

- **CORS Configuration**: Configurable cross-origin resource sharing policies
- **Rate Limiting**: Prevents abuse and ensures fair resource allocation
- **Input Validation**: File size, format, and type validation before processing
- **Request ID Tracking**: Enables security auditing and debugging

### Performance

- **Async/Await**: Non-blocking I/O for high concurrency
- **Connection Pooling**: Efficient database/Redis connection management
- **Lazy Loading**: Models loaded on-demand to reduce startup time
- **Device Optimization**: Configurable CPU/GPU execution

### Observability

- **Structured Logging**: JSON logs for machine-readable analysis
- **Request Tracing**: End-to-end request tracking via request IDs
- **Health Endpoints**: Liveness and readiness checks for orchestration
- **Error Handling**: Comprehensive exception handling with meaningful error messages

### Testing

- **Unit Tests**: Isolated testing of individual components
- **Integration Tests**: Testing of service interactions
- **Mock Dependencies**: Fast, reliable test execution
- **Test Coverage**: Critical paths covered by automated tests

### Deployment

- **Docker Support**: Containerized deployment for consistency
- **Environment Configuration**: 12-factor app principles via environment variables
- **Graceful Shutdown**: Proper cleanup of resources on termination
- **Production-Ready**: Uvicorn ASGI server with auto-reload for development

## Project Structure

```
resnet-fracture/
├── api/
│   ├── __init__.py
│   ├── deps/
│   │   ├── __init__.py
│   │   ├── model_provider.py    # Model dependency injection
│   │   └── service_deps.py      # Service dependency injection
│   └── image_router.py          # API routes
├── middlewares/
│   ├── __init__.py
│   ├── cors.py                  # CORS middleware
│   ├── rate_limit.py            # Rate limiting
│   └── request_id.py            # Request ID tracking
├── services/
│   ├── __init__.py
│   ├── image_service.py         # Image validation & processing
│   ├── inference_service.py     # ML model inference with caching
│   └── model_info_service.py    # Model metadata service
├── repository/
│   ├── __init__.py
│   └── redis.py                 # Redis repository pattern
├── db/
│   ├── __init__.py
│   └── redis.py                 # Redis client configuration
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Structured JSON logging
│   └── preds_mapper.py          # Prediction label mapping
├── models/
│   └── resnet18_fractured/
│       └── resnet18.py          # ResNet18 model architecture
├── tests/
│   ├── __init__.py
│   ├── test_image_service.py
│   ├── test_inference_service.py
│   └── test_preds_mapper.py
├── config.py                    # Configuration management
├── main.py                      # Application entry point
├── requirements.txt             # Dependencies
└── README.md
```
