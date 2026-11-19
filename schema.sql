-- Initial schema. Re-running bootstraps missing objects; it is not a migration
-- mechanism for changing existing columns or constraints.
BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY CHECK (btrim(username) <> ''),
    department TEXT NOT NULL CHECK (btrim(department) <> ''),
    is_employee BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL CHECK (btrim(source) <> ''),
    title TEXT NOT NULL CHECK (btrim(title) <> ''),
    department TEXT NOT NULL CHECK (btrim(department) <> ''),
    visibility TEXT NOT NULL CHECK (visibility IN ('public', 'internal', 'restricted')),
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded')),
    effective_from DATE NOT NULL,
    updated_on DATE NOT NULL,
    -- SHA-256 of normalized policy content, computed during ingestion.
    content_hash TEXT NOT NULL UNIQUE CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (title, version)
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section TEXT NOT NULL CHECK (btrim(section) <> ''),
    content TEXT NOT NULL CHECK (btrim(content) <> ''),
    -- text-embedding-3-small at its default dimension. Model changes require reindexing.
    embedding VECTOR(1536) NOT NULL,
    search_text TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING gin (search_text);
CREATE INDEX IF NOT EXISTS documents_access_idx
    ON documents(visibility, department, status, effective_from);

CREATE TABLE IF NOT EXISTS audit_events (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL REFERENCES users(username),
    question TEXT NOT NULL CHECK (btrim(question) <> ''),
    answered BOOLEAN NOT NULL,
    grounded BOOLEAN NOT NULL,
    sources TEXT NOT NULL DEFAULT 'none',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audit_events_user_time_idx
    ON audit_events(username, created_at DESC);

-- Demo identities; existing identities are never overwritten on startup.
INSERT INTO users (username, department, is_employee) VALUES
    ('alice', 'engineering', TRUE),
    ('bob', 'hr', TRUE),
    ('carol', 'finance', TRUE),
    ('guest', 'none', FALSE)
ON CONFLICT (username) DO NOTHING;

COMMIT;
