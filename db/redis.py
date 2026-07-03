import redis
from typing import Optional
from config import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Redis client wrapper for connection management"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[redis.Redis] = None
        
    def connect(self) -> redis.Redis:
        """Establish connection to Redis"""
        if self._client is None:
            logger.info("Connecting to Redis", extra={
                "host": self.settings.redis_host,
                "port": self.settings.redis_port,
                "db": self.settings.redis_db
            })
            self._client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                decode_responses=True
            )
            # Test connection
            self._client.ping()
            logger.info("Redis connection established")
        return self._client
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client instance"""
        if self._client is None:
            return self.connect()
        return self._client
