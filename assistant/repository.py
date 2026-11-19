"""Database operations. Callers own connections; writes use atomic transactions.

Use pool().connection() in the API lifecycle and close_pool() on shutdown.
Document lookup/write functions are trusted ingestion operations. User-facing
listing and search always apply the same SQL visibility predicate.
"""

from collections.abc import Sequence
from datetime import date
from functools import lru_cache
import math

from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row, tuple_row
from psycopg_pool import ConnectionPool

from assistant.config import PROJECT_ROOT, load_settings
from assistant.models import Chunk, Hit, PolicyStatus, SourceDocument, User


class PolicyOwnershipError(ValueError):
    """A title/version cannot be reassigned to another owning department."""


def _configure(connection: Connection) -> None:
    register_vector(connection)
    connection.commit()
    if load_settings().demo_mode:
        initialize_demo_schema(connection)


def initialize_demo_schema(connection: Connection) -> None:
    """Keep deterministic demo vectors separate from public OpenAI tables.

    Serialized bootstrap is safe for multiple pooled connections. No documents
    are copied or automatically indexed. The connection retains this search path.
    """
    with connection.transaction():
        connection.execute("SELECT pg_advisory_xact_lock(hashtextextended('bootstrap:assistant_demo', 0))")
        connection.execute('CREATE SCHEMA IF NOT EXISTS assistant_demo')
        connection.execute("SELECT set_config('search_path', 'assistant_demo,public', false)")
        script = (PROJECT_ROOT / 'schema.sql').read_text(encoding='utf-8')
        # pgvector was initialized before the pool opened; all application SQL
        # objects are created in the selected schema, inside this transaction.
        script = script[script.index('CREATE TABLE'):].rstrip().removesuffix('COMMIT;')
        connection.execute(script)


@lru_cache(maxsize=1)
def pool() -> ConnectionPool:
    """Create the process pool lazily, after schema.sql has enabled pgvector."""
    return ConnectionPool(
        load_settings().database_url.get_secret_value(),
        min_size=1, max_size=5, timeout=10, open=True,
        configure=_configure, check=ConnectionPool.check_connection,
    )


def close_pool() -> None:
    if pool.cache_info().currsize:
        pool().close()
        pool.cache_clear()


def get_user(connection: Connection, username: str) -> User:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT username, department, is_employee FROM users WHERE username = %s', (username,))
        row = cursor.fetchone()
    if row is None:
        raise ValueError('Unknown user')
    return User(**row)


# Even a nonemployee with a matching department cannot see restricted documents.
_ACCESS = """(
    d.visibility = 'public'
    OR (%(is_employee)s AND d.visibility = 'internal')
    OR (%(is_employee)s AND d.visibility = 'restricted' AND d.department = %(department)s)
)"""
_ELIGIBLE = _ACCESS + """
AND d.effective_from <= %(today)s
AND (d.status = 'active' OR %(include_history)s)
"""


def _access_params(user: User, include_history: bool, today: date | None) -> dict:
    return dict(is_employee=user.is_employee, department=user.department,
                include_history=include_history, today=today or date.today())


def list_documents(connection: Connection, user: User) -> list[dict]:
    """List only metadata the user may read, including historical versions."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute('''SELECT d.id, d.source, d.title, d.department, d.visibility,
            d.version, d.status, d.effective_from, d.updated_on, d.indexed_at
            FROM documents d WHERE ''' + _ACCESS + ' ORDER BY d.title, d.version DESC',
            _access_params(user, True, None))
        return cursor.fetchall()


def find_by_hash(connection: Connection, content_hash: str) -> dict | None:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM documents WHERE content_hash = %s', (content_hash,))
        return cursor.fetchone()


def find_by_version(connection: Connection, title: str, version: int) -> dict | None:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM documents WHERE title = %s AND version = %s', (title, version))
        return cursor.fetchone()


def _vector(values: Sequence[float]) -> str:
    """Use an explicit vector cast so list adaptation cannot become a SQL array."""
    if len(values) != 1536:
        raise ValueError('Embeddings must contain 1536 dimensions')
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector) or not any(vector):
        raise ValueError('Embeddings must be finite and nonzero for cosine search')
    return '[' + ','.join(map(str, vector)) + ']'


def store_document(connection: Connection, document: SourceDocument,
                   chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> dict:
    """Atomically insert or replace one title/version, or skip a duplicate hash.

    Explicit front-matter status is preserved, as in the reference project.
    Ingestion must mark old policies superseded when publishing a new version.
    Embedding calls must happen before this transaction, never while holding locks.
    """
    if not chunks or len(chunks) != len(embeddings):
        raise ValueError('Provide one embedding per chunk and at least one chunk')
    vectors = [_vector(values) for values in embeddings]
    metadata = document.metadata
    with connection.transaction(), connection.cursor(row_factory=dict_row) as cursor:
        # Deterministic advisory locks serialize competing uploads of the same
        # content or policy identity; all upload writers must use this function.
        keys = sorted({f'hash:{document.content_hash}', f'title:{metadata.title}'})
        for key in keys:
            cursor.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))', (key,))
        existing = find_by_hash(connection, document.content_hash)
        if existing:
            return dict(document_id=existing['id'], outcome='duplicate')
        previous = find_by_version(connection, metadata.title, metadata.version)
        if previous and previous['department'] != metadata.department:
            raise PolicyOwnershipError('A policy version cannot change its owning department')
        if previous:
            cursor.execute('DELETE FROM documents WHERE id = %s', (previous['id'],))
        cursor.execute('''INSERT INTO documents
            (source, title, department, visibility, version, status, effective_from, updated_on, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id''',
            (document.source, metadata.title, metadata.department, metadata.visibility.value,
             metadata.version, metadata.status.value, metadata.effective, metadata.updated,
             document.content_hash))
        document_id = cursor.fetchone()['id']
        cursor.executemany('''INSERT INTO chunks(document_id, section, content, embedding)
            VALUES (%s, %s, %s, %s::vector)''',
            [(document_id, chunk.section, chunk.content, vector)
             for chunk, vector in zip(chunks, vectors, strict=True)])
    return dict(document_id=document_id, outcome='replaced' if previous else 'inserted')


def searchable_chunks(connection: Connection, user: User, include_history: bool = False,
                      *, today: date | None = None) -> dict:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute('''SELECT (SELECT count(*) FROM chunks) AS total,
            count(*) AS searchable FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE ''' + _ELIGIBLE, _access_params(user, include_history, today))
        return cursor.fetchone()


def hybrid_search(connection: Connection, question: str, query_vector: Sequence[float],
                  user: User, include_history: bool = False, *, candidates: int = 12,
                  rrf_constant: int = 60, today: date | None = None) -> list[Hit]:
    if not question.strip():
        raise ValueError('Question must not be empty')
    if candidates < 1 or rrf_constant < 1:
        raise ValueError('Candidates and RRF constant must be positive')
    params = _access_params(user, include_history, today)
    params.update(question=question, query_vector=_vector(query_vector),
                  candidates=candidates, rrf_constant=rrf_constant)
    # Both branches rank only readable rows. MATERIALIZED makes the boundary
    # explicit; text is never retrieved first and filtered in Python afterwards.
    sql = '''WITH readable AS MATERIALIZED (
        SELECT d.id FROM documents d WHERE ''' + _ELIGIBLE + '''
    ), semantic AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(query_vector)s::vector, c.id) AS rank
        FROM chunks c JOIN readable r ON r.id = c.document_id
        ORDER BY c.embedding <=> %(query_vector)s::vector, c.id LIMIT %(candidates)s
    ), keyword AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY ts_rank(c.search_text, q.query) DESC, c.id) AS rank
        FROM chunks c JOIN readable r ON r.id = c.document_id
        CROSS JOIN plainto_tsquery('english', %(question)s) AS q(query)
        WHERE c.search_text @@ q.query
        ORDER BY ts_rank(c.search_text, q.query) DESC, c.id LIMIT %(candidates)s
    ), fused AS (
        SELECT id, SUM(1.0 / (%(rrf_constant)s + rank)) AS score
        FROM (SELECT * FROM semantic UNION ALL SELECT * FROM keyword) ranked
        GROUP BY id ORDER BY score DESC, id LIMIT %(candidates)s
    )
    SELECT c.id AS chunk_id, c.content, c.section, d.source, d.title, d.version,
        d.status, d.updated_on, f.score
    FROM fused f JOIN chunks c ON c.id = f.id JOIN documents d ON d.id = c.document_id
    ORDER BY f.score DESC, c.id'''
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(sql, params)
        return [Hit(**dict(row, status=PolicyStatus(row['status']), score=float(row['score'])))
                for row in cursor.fetchall()]


def record_event(connection: Connection, *, username: str, question: str,
                 answered: bool, grounded: bool, sources: str = 'none') -> int:
    with connection.transaction(), connection.cursor(row_factory=tuple_row) as cursor:
        cursor.execute('''INSERT INTO audit_events(username, question, answered, grounded, sources)
            VALUES (%s, %s, %s, %s, %s) RETURNING id''',
            (username, question, answered, grounded, sources))
        return cursor.fetchone()[0]


def health(connection: Connection) -> dict:
    """Connection and extension check; policy counts belong to user-scoped calls."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT current_database() AS database, extversion AS pgvector_version "
                       "FROM pg_extension WHERE extname = 'vector'")
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError('pgvector is not enabled; apply schema.sql first')
    return row
