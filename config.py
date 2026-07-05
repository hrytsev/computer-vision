from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "ResNet Fracture API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS settings
    # Comma-separated list of allowed origins, e.g., "http://localhost:3000,https://example.com"
    cors_origins: list[str] = []
    cors_methods: list[str] = ["GET", "POST"]
    cors_headers: list[str] = ["Content-Type", "Authorization"]
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    # Image validation
    max_image_size_mb: float = 10.0
    allowed_image_formats: list[str] = ["image/jpeg", "image/png"]
    
    # Model settings
    model_path: str = "models/resnet18_fractured/resnet18_fractured.pth"
    device: str = "cpu"
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Cache settings
    cache_ttl: int = 60  # seconds

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
