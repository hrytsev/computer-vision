from typing import Optional, Any
from db.redis import RedisClient
from utils.logger import get_logger

logger = get_logger(__name__)


class RedisRepository:
    """Repository pattern for Redis operations"""
    
    def __init__(self, redis_client: RedisClient):
        self.redis_client = redis_client
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a key-value pair in Redis with optional TTL.
        
        Args:
            key: Redis key
            value: Value to store
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful
        """
        logger.debug("Setting Redis key", extra={"key": key, "ttl": ttl})
        if ttl:
            return self.redis_client.client.setex(key, ttl, value)
        return self.redis_client.client.set(key, value)
    
    def get(self, key: str) -> Optional[str]:
        """
        Get a value from Redis by key.
        
        Args:
            key: Redis key
            
        Returns:
            Value if exists, None otherwise
        """
        logger.debug("Getting Redis key", extra={"key": key})
        return self.redis_client.client.get(key)
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Redis key
            
        Returns:
            True if key was deleted
        """
        logger.debug("Deleting Redis key", extra={"key": key})
        return bool(self.redis_client.client.delete(key))
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Redis key
            
        Returns:
            True if key exists
        """
        logger.debug("Checking Redis key existence", extra={"key": key})
        return bool(self.redis_client.client.exists(key))
