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
│   └── image_router.py      # API routes
├── middlewares/
│   ├── __init__.py
│   ├── cors.py              # CORS middleware
│   ├── rate_limit.py        # Rate limiting
│   └── request_id.py        # Request ID tracking
├── services/
│   ├── __init__.py
│   └── image_service.py     # Business logic
├── config.py                # Configuration
├── main.py                  # Application entry point
├── requirements.txt         # Dependencies
└── README.md
```
