"""Metadata validation and the plain data objects passed between services."""

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Visibility(StrEnum):
    PUBLIC = 'public'
    INTERNAL = 'internal'
    RESTRICTED = 'restricted'


class PolicyStatus(StrEnum):
    ACTIVE = 'active'
    SUPERSEDED = 'superseded'


class PolicyMetadata(BaseModel):
    """Required Markdown front matter, matching the project PDF's field names."""

    model_config = ConfigDict(extra='forbid', frozen=True, str_strip_whitespace=True)

    title: str = Field(min_length=1)
    department: str = Field(min_length=1)
    visibility: Visibility
    version: int = Field(strict=True, gt=0)
    status: PolicyStatus
    effective: date
    updated: date

    @field_validator('department')
    @classmethod
    def normalize_department(cls, value: str) -> str:
        return value.lower()

    @field_validator('effective', 'updated', mode='before')
    @classmethod
    def validate_date_input(cls, value: object) -> object:
        # Accept YAML dates and ISO strings, not numeric epoch timestamps.
        if type(value) is date:
            return value
        if isinstance(value, str):
            try:
                parsed = date.fromisoformat(value)
                if parsed.isoformat() == value:
                    return parsed
            except ValueError:
                pass
        raise ValueError('Use an ISO date in YYYY-MM-DD format')


@dataclass(frozen=True)
class User:
    username: str
    department: str
    is_employee: bool

    def describe_access(self) -> str:
        if not self.is_employee:
            return 'public documents only'
        return f'public and internal documents, plus restricted {self.department} documents'


@dataclass(frozen=True)
class SourceDocument:
    source: str
    metadata: PolicyMetadata
    content: str
    content_hash: str


@dataclass(frozen=True)
class Chunk:
    section: str
    content: str


@dataclass(frozen=True)
class UploadResult:
    filename: str
    document_id: int
    title: str
    version: int
    outcome: Literal['inserted', 'replaced', 'duplicate']
    chunks: int


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    content: str
    section: str
    source: str
    title: str
    version: int
    status: PolicyStatus
    updated_on: date
    score: float = 0.0

    @property
    def label(self) -> str:
        return f'{self.title} v{self.version} > {self.section}'


@dataclass(frozen=True)
class Citation:
    chunk_id: int
    label: str
    source: str
    quote: str
    verified: bool


@dataclass
class Answer:
    text: str
    answered: bool
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    grounding_errors: list[str] = field(default_factory=list)

    @property
    def fully_grounded(self) -> bool:
        # A refusal or an answer without citations cannot claim grounding.
        return (self.answered and bool(self.citations) and not self.grounding_errors
                and all(c.verified for c in self.citations))
