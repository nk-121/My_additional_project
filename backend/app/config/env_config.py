from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    allowed_origins: list[str] = ["*"]
    database_url: str
    #redis_url: str = "redis://localhost:6379/0"
    #log_level: str = "INFO"
    environment: str = "development"
    
    # Rate limiting
    #rate_limit_enabled: bool = True
    #max_requests_per_minute: int = 100
    
    # Security
    min_password_length: int = 8
    max_request_size_mb: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()