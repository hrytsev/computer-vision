import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from utils.logger import request_id_context, get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(self.header_name)
        
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store request_id in request state for tracking through the flow
        request.state.request_id = request_id
        
        # Set request_id in context for global access (logging, services, etc.)
        token = request_id_context.set(request_id)
        
        logger.info("Request received", extra={
            "method": request.method,
            "path": request.url.path,
            "request_id": request_id
        })
        
        try:
            response: Response = await call_next(request)
            response.headers[self.header_name] = request_id
            
            logger.info("Request completed", extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "request_id": request_id
            })
            
            return response
        except Exception as e:
            logger.error("Request failed", extra={
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "request_id": request_id
            })
            raise
        finally:
            # Clean up context
            request_id_context.reset(token)
