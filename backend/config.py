"""
Configuration settings for the Voice Sales Agent MVP
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class Config:
    """Application configuration."""

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "doorindex")

    # Embedding model - must match what was used to create Pinecone index
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    # SMTP Configuration
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "sonudevmail16@gmail.com")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")

    # Business Configuration
    CLIENT_EMAIL: str = os.getenv("CLIENT_EMAIL", "abhinav.sarkar@jellyfishtechnologies.com")
    COMPANY_NAME: str = os.getenv("COMPANY_NAME", "ABC Doors & Windows")

    # Realtime API Configuration
    REALTIME_MODEL: str = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
    VOICE: str = os.getenv("VOICE", "alloy")

    # Agent-specific voices (OpenAI Realtime API voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar)
    GREETING_AGENT_VOICE: str = os.getenv("GREETING_AGENT_VOICE", "coral")  # Warm, friendly voice
    SALES_AGENT_VOICE: str = os.getenv("SALES_AGENT_VOICE", "ash")  # Professional voice

    # MongoDB Configuration
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "voice_sales_agent")

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


# Global config instance
config = Config()
