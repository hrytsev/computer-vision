from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from config import get_settings
from utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_requests}/{settings.rate_limit_period}seconds"]
)


def get_rate_limit_exceeded_handler():
    logger.warning("Rate limit exceeded handler registered")
    return _rate_limit_exceeded_handler
