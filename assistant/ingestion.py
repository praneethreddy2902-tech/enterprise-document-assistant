"""Markdown parsing, section-aware splitting, embedding, and atomic storage."""

from hashlib import sha256
from pathlib import PurePosixPath
from typing import Protocol

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from psycopg import Connection
from pydantic import ValidationError
import yaml

from assistant import repository
from assistant.config import Settings, load_settings
from assistant.models import Chunk, PolicyMetadata, SourceDocument, UploadResult


class IngestionError(ValueError):
    """Invalid policy content; safe to return as a validation error in the API."""


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class _MetadataLoader(yaml.SafeLoader):
    """Reject ambiguous duplicate keys instead of silently accepting the last."""


def _mapping(loader: _MetadataLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise IngestionError('Metadata keys must be field names')
        if key in result:
            raise IngestionError(f'Duplicate metadata field: {key}')
        result[key] = loader.construct_object(value_node)
    return result


_MetadataLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)

#parse()--> document metadata,actual doc content,content hash
def parse(text: str, filename: str) -> SourceDocument:
    """Validate front matter and hash the entire normalized upload, not its name."""
    source = PurePosixPath(filename.replace('\\', '/')).name
    if not source or PurePosixPath(source).suffix.lower() != '.md':
        raise IngestionError('Upload a Markdown file with a .md extension')
    normalized = text.removeprefix('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    if '\x00' in normalized or '\x00' in source:
        raise IngestionError('Policy text and filename must not contain null characters')
    lines = normalized.split('\n')
    if lines[0].strip() != '---':
        raise IngestionError('Start the file with a --- metadata block')
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == '---'), None)
    if end is None:
        raise IngestionError('Close the metadata block with --- before the policy text')
    try:
        raw = yaml.load('\n'.join(lines[1:end]), Loader=_MetadataLoader)
    except IngestionError:
        raise
    except (yaml.YAMLError, ValueError) as exc:
        raise IngestionError('Invalid YAML in the policy metadata block') from exc
    if not isinstance(raw, dict):
        raise IngestionError('Metadata must contain named fields such as title and visibility')
    try:
        metadata = PolicyMetadata.model_validate(raw)
    except ValidationError as exc:
        details = '; '.join(f'{".".join(map(str, e["loc"]))}: {e["msg"]}' for e in exc.errors())
        raise IngestionError(f'Invalid policy metadata: {details}') from exc
    body = '\n'.join(lines[end + 1:]).strip()
    if not body:
        raise IngestionError('Add policy text after the metadata block')
    return SourceDocument(source=source, metadata=metadata, content=body,
                          content_hash=sha256(normalized.encode('utf-8')).hexdigest())


def split(document: SourceDocument, settings: Settings | None = None) -> list[Chunk]:
    settings = settings or Settings()
    headings = MarkdownHeaderTextSplitter(headers_to_split_on=[('##', 'section')])
    by_size = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
                                            chunk_overlap=settings.chunk_overlap)
    chunks = [Chunk(section=section.metadata.get('section', document.metadata.title), content=piece)
              for section in headings.split_text(document.content)
              for piece in by_size.split_text(section.page_content) if piece.strip()]
    if not chunks:
        raise IngestionError('The policy has no indexable text beneath its headings')
    return chunks


class Ingestor:
    def __init__(self, settings: Settings | None = None, *, embeddings: EmbeddingProvider | None = None):
        self.settings = settings or load_settings()
        self._embeddings = embeddings

    def _provider(self) -> EmbeddingProvider:
        if self._embeddings is None:
            if self.settings.demo_mode:
                from assistant.demo import DemoEmbeddings
                self._embeddings = DemoEmbeddings()
                return self._embeddings
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                api_key=self.settings.require_openai_key(), model=self.settings.embedding_model,
                dimensions=self.settings.embedding_dimensions,
            )
        return self._embeddings

    def store(self, connection: Connection, text: str, filename: str) -> UploadResult:
        """Validate and dedupe before any paid call; replace only after embedding succeeds.

        The caller owns the connection and its outer transaction. The repository
        rechecks duplicates under locks to handle concurrent uploads safely.
        """
        document = parse(text, filename)
        existing = repository.find_by_hash(connection, document.content_hash)
        if existing:
            return UploadResult(document.source, existing['id'], existing['title'],
                                existing['version'], 'duplicate', 0)
        chunks = split(document, self.settings)
        vectors = self._provider().embed_documents([chunk.content for chunk in chunks])
        stored = repository.store_document(connection, document, chunks, vectors)
        return UploadResult(document.source, stored['document_id'], document.metadata.title,
                            document.metadata.version, stored['outcome'],
                            0 if stored['outcome'] == 'duplicate' else len(chunks))
