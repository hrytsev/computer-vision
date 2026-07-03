from fastapi import APIRouter, UploadFile, File, Request
from config import get_settings
from services.image_service import ImageService, settings_context
from contextvars import copy_context
from utils.logger import get_logger

router = APIRouter(prefix="/api/images", tags=["images"])
logger = get_logger(__name__)


@router.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...)
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
        # Create service instance
        image_service = ImageService()
        
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
