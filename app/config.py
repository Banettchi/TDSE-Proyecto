"""
Configuration settings for the Melanoma Screening Platform.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "DermAI - Plataforma de Screening Dermatológico"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./dermai.db"

    # JWT
    SECRET_KEY: str = "dermai-secret-key-change-in-production-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ML Models
    MODEL_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_weights")
    DEFAULT_MODEL: str = "efficientnet"  # efficientnet, resnet, vit
    IMAGE_SIZE_EFFICIENTNET: int = 299
    IMAGE_SIZE_RESNET: int = 224
    IMAGE_SIZE_VIT: int = 224
    NUM_CLASSES: int = 7  # HAM10000: akiec, bcc, bkl, df, mel, nv, vasc
    CONFIDENCE_THRESHOLD: float = 0.5

    # HAM10000 class labels
    CLASS_NAMES: list = [
        "Queratosis actínica (akiec)",
        "Carcinoma basocelular (bcc)",
        "Queratosis benigna (bkl)",
        "Dermatofibroma (df)",
        "Melanoma (mel)",
        "Nevo melanocítico (nv)",
        "Lesión vascular (vasc)"
    ]

    # Risk mapping: which classes are high risk
    HIGH_RISK_CLASSES: list = [0, 1, 4]  # akiec, bcc, mel
    MELANOMA_CLASS_INDEX: int = 4

    # Upload
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
