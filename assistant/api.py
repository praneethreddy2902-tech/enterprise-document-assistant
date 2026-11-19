"""Local Phase 1 HTTP interface. Request usernames are demo identities, not auth."""

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAIError
from psycopg import Connection, Error as DatabaseError
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assistant import repository
from assistant.repository import PolicyOwnershipError
from assistant.config import PROJECT_ROOT, ModelConfigurationError, Settings, load_settings
from assistant.ingestion import IngestionError, Ingestor, parse
from assistant.models import User
from assistant.pipeline import Pipeline
from assistant.retrieval import RerankError

MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def get_connection():
    # This context commits uploads and question audits before the response returns.
    with repository.pool().connection() as connection:
        yield connection


ConnectionDependency = Annotated[Connection, Depends(get_connection, scope='function')]


def _user(connection: Connection, username: str) -> User:
    try:
        return repository.get_user(connection, username)
    except ValueError as exc:
        raise HTTPException(404, 'Unknown demo user') from exc


class AskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    username: str = Field(default='guest', min_length=1, max_length=100)


def create_app(*, pipeline: Pipeline | None = None, ingestor: Ingestor | None = None,
               settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            repository.close_pool()

    app = FastAPI(title='Enterprise Document Assistant', lifespan=lifespan)
    settings = settings or load_settings()
    app.state.pipeline = pipeline if pipeline is not None else Pipeline(settings)
    app.state.ingestor = ingestor if ingestor is not None else Ingestor(settings)

    @app.get('/config')
    def configuration():
        return dict(demo_mode=settings.demo_mode)

    @app.middleware('http')
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['X-Frame-Options'] = 'DENY'
        return response

    async def database_unavailable(request: Request, exc: Exception):
        return JSONResponse(status_code=503, content={'detail': 'Database unavailable. Check Docker and database setup.'})

    app.add_exception_handler(DatabaseError, database_unavailable)
    app.add_exception_handler(PoolTimeout, database_unavailable)

    async def model_error(request: Request, exc: Exception):
        return JSONResponse(status_code=502, content={'detail': 'The model request failed. Please try again.'})

    app.add_exception_handler(OpenAIError, model_error)
    app.add_exception_handler(ValidationError, model_error)
    app.add_exception_handler(RerankError, model_error)

    @app.exception_handler(IngestionError)
    async def upload_error(request: Request, exc: IngestionError):
        return JSONResponse(status_code=422, content={'detail': str(exc)})

    @app.exception_handler(PolicyOwnershipError)
    async def ownership_conflict(request: Request, exc: PolicyOwnershipError):
        return JSONResponse(status_code=409, content={'detail': 'This policy version belongs to another department.'})

    @app.exception_handler(ValueError)
    async def operation_error(request: Request, exc: ValueError):
        # Do not expose provider responses, connection strings, or key values.
        return JSONResponse(status_code=503, content={'detail': 'Operation unavailable. Check server configuration.'})

    @app.exception_handler(ModelConfigurationError)
    async def configuration_error(request: Request, exc: ModelConfigurationError):
        return JSONResponse(status_code=503, content={'detail': str(exc)})

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(PROJECT_ROOT / 'web' / 'index.html', media_type='text/html')

    @app.get('/health')
    def health(connection: ConnectionDependency):
        return dict(status='ok', demo_mode=settings.demo_mode, **repository.health(connection))

    @app.get('/documents')
    def documents(connection: ConnectionDependency,
                  username: Annotated[str, Query(min_length=1, max_length=100)] = 'guest'):
        return repository.list_documents(connection, _user(connection, username))

    @app.post('/documents')
    def upload(connection: ConnectionDependency, file: Annotated[UploadFile, File()],
               username: Annotated[str, Form(min_length=1, max_length=100)] = 'guest'):
        actor = _user(connection, username)
        if not actor.is_employee:
            raise HTTPException(403, 'Only employees can upload policies')
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Policy uploads must be no larger than 2 MiB')
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise HTTPException(422, 'Upload a UTF-8 encoded Markdown file') from exc
        document = parse(text, file.filename or '')
        if document.metadata.department != actor.department:
            raise HTTPException(403, 'You can only upload policies owned by your department')
        existing = repository.find_by_version(connection, document.metadata.title, document.metadata.version)
        if existing and existing['department'] != actor.department:
            raise HTTPException(403, 'This policy version belongs to another department')
        # Model work and SQL run in FastAPI's thread pool, never the event loop.
        return dict(asdict(app.state.ingestor.store(connection, text, file.filename)), demo_mode=settings.demo_mode)

    @app.post('/ask')
    def ask(body: AskRequest, connection: ConnectionDependency):
        actor = _user(connection, body.username)
        answer = app.state.pipeline.ask(connection, body.question, actor.username)
        return dict(asdict(answer), fully_grounded=answer.fully_grounded, demo_mode=settings.demo_mode)

    return app


app = create_app()
