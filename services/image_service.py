from fastapi import UploadFile, HTTPException
from config import Settings
from pydantic import BaseModel
from contextvars import ContextVar
from utils.logger import get_logger

# Magic bytes for image format validation
JPEG_MAGIC = [0xFF, 0xD8, 0xFF]
PNG_MAGIC = [0x89, 0x50, 0x4E, 0x47, 0x44, 0x49, 0x48]

# Context variable for settings
settings_context: ContextVar[Settings] = ContextVar('settings_context')

logger = get_logger(__name__)


class ImageService(BaseModel):
    """Service for image validation and processing using Pydantic context"""
    
    @classmethod
    def get_settings(cls) -> Settings:
        """Get settings from context"""
        return settings_context.get()
    
    def _validate_magic_bytes(self, content: bytes, content_type: str) -> None:
        """
        Validate file magic bytes match the declared content type.
        
        Args:
            content: Raw file content bytes
            content_type: Declared content type (e.g., 'image/jpeg')
            
        Raises:
            HTTPException: If magic bytes don't match the content type
        """
        # Get first few bytes for magic byte check
        header = list(content[:8])
        
        if content_type == "image/jpeg":
            if header[:3] != JPEG_MAGIC:
                logger.warning("JPEG magic bytes mismatch", extra={"header": header[:3]})
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JPEG file: magic bytes do not match"
                )
        elif content_type == "image/png":
            if header[:7] != PNG_MAGIC:
                logger.warning("PNG magic bytes mismatch", extra={"header": header[:7]})
                raise HTTPException(
                    status_code=400,
                    detail="Invalid PNG file: magic bytes do not match"
                )
    
    async def validate_and_process_image(self, file: UploadFile) -> dict:
        """
        Validate and process an uploaded image file.
        Returns metadata about the validated image.
        """
        settings = self.get_settings()
        
        logger.info("Starting image validation", extra={
            "file_name": file.filename,
            "content_type": file.content_type
        })
        
        # Validate file type
        if file.content_type not in settings.allowed_image_formats:
            logger.warning("Invalid file type", extra={
                "content_type": file.content_type,
                "allowed_formats": settings.allowed_image_formats
            })
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed formats: {settings.allowed_image_formats}"
            )
        
        # Validate file extension
        if not file.filename:
            logger.warning("Filename is missing")
            raise HTTPException(status_code=400, detail="Filename is required")
        
        filename_lower = file.filename.lower()
        if not (filename_lower.endswith(('.jpg', '.jpeg')) or filename_lower.endswith('.png')):
            logger.warning("Invalid file extension", extra={"file_name": file.filename})
            raise HTTPException(
                status_code=400,
                detail="Invalid file extension. Only .jpg, .jpeg, and .png are allowed"
            )
        
        # Read file content to validate size and magic bytes
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        # Validate magic bytes
        self._validate_magic_bytes(content, file.content_type)
        
        logger.info("File size validated", extra={"size_mb": round(file_size_mb, 2)})
        
        if file_size_mb > settings.max_image_size_mb:
            logger.warning("File size exceeds limit", extra={
                "size_mb": round(file_size_mb, 2),
                "max_size_mb": settings.max_image_size_mb
            })
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {settings.max_image_size_mb}MB"
            )
        
        # Reset file pointer for further processing if needed
        await file.seek(0)
        
        logger.info("Image validation completed successfully", extra={
            "file_name": file.filename,
            "size_mb": round(file_size_mb, 2)
        })
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_mb": round(file_size_mb, 2),
            "content": content
        }
