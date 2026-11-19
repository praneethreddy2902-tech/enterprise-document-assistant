"""Validated settings; importing this module makes no network or database calls."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
import os

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_NAMES = frozenset({'answer', 'grade', 'intent', 'rerank', 'rewrite'})


class ModelConfigurationError(ValueError):
    """An actionable, secret-free model configuration error."""


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    database_url: SecretStr = SecretStr('postgresql://rag:ragsecret@localhost:5434/ragdb')
    openai_api_key: SecretStr = SecretStr('')
    demo_mode: bool = False
    # Changing embedding model requires reindexing, even at the same dimension.
    embedding_model: Literal['text-embedding-3-small'] = 'text-embedding-3-small'
    embedding_dimensions: Literal[1536] = 1536
    answer_model: str = Field(default='gpt-4o-mini', min_length=1)
    helper_model: str = Field(default='gpt-4o-mini', min_length=1)
    chunk_size: int = Field(default=600, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    candidates: int = Field(default=12, gt=0)
    top_k: int = Field(default=4, gt=0)
    min_rerank_score: float = Field(default=4, ge=0, le=10)
    high_confidence_score: float = Field(default=8, ge=0, le=10)
    rrf_constant: int = Field(default=60, gt=0)
    max_attempts: int = Field(default=2, ge=1, le=2)
    stale_after_days: int = Field(default=180, gt=0)

    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        try:
            parsed = urlsplit(value.get_secret_value())
            valid = (parsed.scheme in {'postgresql', 'postgres'} and parsed.hostname
                     and parsed.path.strip('/') and (parsed.port is None or parsed.port > 0))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError('DATABASE_URL must be a PostgreSQL URL with a host and database')
        return value

    @model_validator(mode='after')
    def validate_limits(self) -> 'Settings':
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('CHUNK_OVERLAP must be smaller than CHUNK_SIZE')
        if self.top_k > self.candidates:
            raise ValueError('TOP_K must not exceed CANDIDATES')
        if self.high_confidence_score < self.min_rerank_score:
            raise ValueError('HIGH_CONFIDENCE_SCORE must be at least MIN_RERANK_SCORE')
        return self

    def require_openai_key(self) -> str:
        key = self.openai_api_key.get_secret_value().strip()
        placeholders = {'sk-...', 'your_openai_api_key', 'your_openai_api_key_here',
                        'your-api-key', 'your_api_key', 'replace_me'}
        if not key or key.lower() in placeholders or key.lower().startswith(('sk-your', 'sk-replace')):
            raise ModelConfigurationError(
                'Set a real OPENAI_API_KEY in the project .env file, then restart the server. '
                'The current key is missing or a placeholder.')
        return key


def load_settings(env_file: Path | None = PROJECT_ROOT / '.env') -> Settings:
    """Environment variables override .env; neither source mutates os.environ."""
    values = dotenv_values(env_file, interpolate=False) if env_file is not None else {}
    values.update(os.environ)
    return Settings(**{
        name: values[name.upper()]
        for name in Settings.model_fields
        if values.get(name.upper()) is not None
    })


def prompt(name: str) -> str:
    """Load a known prompt and fail clearly while its scaffold is still empty."""
    if name not in PROMPT_NAMES:
        raise ValueError(f'Unknown prompt: {name!r}')
    text = (PROJECT_ROOT / 'prompts' / f'{name}.txt').read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError(f'Prompt {name!r} has not been implemented yet')
    return text
