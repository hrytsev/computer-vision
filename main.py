from fastapi import FastAPI, Request
from slowapi.errors import RateLimitExceeded
from config import get_settings
from middlewares.cors import add_cors_middleware
from middlewares.rate_limit import limiter, get_rate_limit_exceeded_handler
from middlewares.request_id import RequestIDMiddleware
from api.image_router import router as image_router, ml_router
from utils.logger import setup_logging, get_logger

settings = get_settings()

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

logger.info("Initializing application", extra={
    "app_name": settings.app_name,
    "version": settings.app_version,
    "debug": settings.debug
})

# Add CORS middleware
add_cors_middleware(app)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, get_rate_limit_exceeded_handler())

# Add request ID middleware
app.add_middleware(RequestIDMiddleware)

# Include routers
app.include_router(image_router)
app.include_router(ml_router)


@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    logger.info("Root endpoint accessed")
    return {
        "message": "ResNet Fracture API",
        "version": settings.app_version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server", extra={
        "host": settings.host,
        "port": settings.port
    })
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
