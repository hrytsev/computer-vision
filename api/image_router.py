from fastapi import APIRouter, UploadFile, File, Request, Depends
from config import get_settings
from services.image_service import ImageService, settings_context
from services.inference_service import InferenceService
from api.deps.service_deps import get_image_service, get_inference_service
from contextvars import copy_context
from utils.logger import get_logger

router = APIRouter(prefix="/api/images", tags=["images"])
ml_router = APIRouter(prefix="/api/ml", tags=["ml"])
logger = get_logger(__name__)


@router.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    image_service: ImageService = Depends(get_image_service)
):
    """
    Upload and validate an image file.
    Accepts JPEG and PNG formats with size validation.
    """
    
    logger.info("Received image upload request", extra={
        "file_name": file.filename,
        "content_type": file.content_type
    })
    
    # Set settings in context for this request
    settings = get_settings()
    token = settings_context.set(settings)
    
    try:
        # Delegate business logic to service
        image_metadata = await image_service.validate_and_process_image(file)
        
        # Get request ID for tracking
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.info("Image upload completed successfully", extra={
            "file_name": image_metadata["filename"],
            "size_mb": image_metadata["size_mb"]
        })
        
        return {
            "message": "Image uploaded successfully",
            "filename": image_metadata["filename"],
            "content_type": image_metadata["content_type"],
            "size_mb": image_metadata["size_mb"],
            "request_id": request_id
        }
    except Exception as e:
        logger.error("Image upload failed", extra={
            "file_name": file.filename,
            "error": str(e)
        })
        raise
    finally:
        # Clean up context
        settings_context.reset(token)


@ml_router.post("/fracture")
async def predict_fracture(
    request: Request,
    file: UploadFile = File(...),
    image_service: ImageService = Depends(get_image_service),
    inference_service: InferenceService = Depends(get_inference_service)
):
    """
    Predict fracture from an image using ResNet18 model.
    Accepts JPEG and PNG formats, returns model logits.
    """
    
    logger.info("Received fracture prediction request", extra={
        "file_name": file.filename,
        "content_type": file.content_type
    })
    
    # Set settings in context for this request
    settings = get_settings()
    token = settings_context.set(settings)
    
    try:
        # Validate and process image using ImageService
        image_metadata = await image_service.validate_and_process_image(file)
        
        # Get image bytes for inference
        image_bytes = image_metadata["content"]
        
        # Run inference using InferenceService
        inference_result = await inference_service.predict(image_bytes)
        
        # Get request ID for tracking
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.info("Fracture prediction completed successfully", extra={
            "file_name": image_metadata["filename"],
            "request_id": request_id
        })
        
        return {
            "logits":inference_result["logits"],
            "prediction": inference_result["prediction"],
            "shape": inference_result["shape"],
            "filename": image_metadata["filename"],
            "request_id": request_id
        }
    except Exception as e:
        logger.error("Fracture prediction failed", extra={
            "file_name": file.filename,
            "error": str(e)
        })
        raise
    finally:
        # Clean up context
        settings_context.reset(token)
