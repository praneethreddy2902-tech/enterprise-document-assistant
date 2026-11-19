# Enterprise Document Assistant

A learning-focused enterprise RAG application for answering policy
questions using permission-aware retrieval and traceable citations.

## Phase 1

- Markdown policy ingestion and metadata validation
- Section-aware chunking
- PostgreSQL and pgvector storage
- Permission-filtered hybrid retrieval
- Answers with supporting quotations
- Version-aware retrieval and audit records
- FastAPI and a simple HTML interface

## Current status

Local PostgreSQL and pgvector setup and the initial database schema are configured.
Validated settings, shared data models, the PostgreSQL repository, and Markdown
ingestion, retrieval with reranking, and cited answer generation are implemented.
Pipeline orchestration, the FastAPI endpoints, and a simple HTML upload/chat
interface are implemented. Live OpenAI calls remain unverified.

## Development environment

Activate the virtual environment:

    source .venv/bin/activate

Start Docker Desktop, then validate and start the database:

    docker compose config --quiet
    docker compose up -d --wait
    docker compose ps

PostgreSQL is available at `localhost:5434` with database `ragdb`, user
`rag`, and password `ragsecret`. These credentials are for local development.
Port 5434 is used because port 5433 was already occupied on the development host.
The named `pgdata` volume preserves database contents across container restarts.

If `.env` does not exist, copy `.env.example` to `.env`. Otherwise, add any
missing settings from the example without replacing your existing secrets.
Set `OPENAI_API_KEY` for live model operations, or enable `DEMO_MODE=true` for
the offline demonstration described below.

Enable pgvector and inspect the connection:

    docker compose exec -T db psql -U rag -d ragdb -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"
    docker compose exec -T db psql -U rag -d ragdb -c "SELECT current_database(), current_user, extversion FROM pg_extension WHERE extname = 'vector';"

Stop the database while keeping its data:

    docker compose down

## Database schema

Apply the schema after starting the database:

    docker compose exec -T db psql -U rag -d ragdb -v ON_ERROR_STOP=1 < schema.sql

This creates `users`, `documents`, `chunks`, and `audit_events`, plus HNSW
vector and GIN full-text indexes. It seeds Alice (engineering), Bob (HR),
Carol (finance), and a guest. These are demo identities, not authentication.

The script can run repeatedly without deleting data or overwriting users.
Future changes to existing tables require migrations.

Documents have a unique content hash and a unique title/version pair. Chunks
store 1536-dimensional embeddings and a generated English full-text index.
Policy status is supplied explicitly in upload metadata. Retrieval enforces
visibility and current-policy eligibility in SQL before either search selects
candidates; the schema alone does not enforce those access rules.

Run database smoke checks (all test data is rolled back):

    docker compose exec -T db psql -U rag -d ragdb -v ON_ERROR_STOP=1 < tests/schema_smoke.sql

## Configuration and data models

`assistant.config.load_settings()` reads the project `.env`, with environment
variables taking precedence. It does not change the process environment or call
external services. Settings use the defaults documented in `.env.example` and
reject incompatible chunking, retrieval limits, and embedding dimensions.
Database credentials and the OpenAI key are masked in settings representations.
An API key is required only when a model-backed operation explicitly requests it.

`assistant.models.PolicyMetadata` validates the PDF's required front-matter
fields: `title`, `department`, `visibility`, `version`, `status`, `effective`,
and `updated`. Dates use `YYYY-MM-DD`. Department names are normalized to
lowercase. Shared dataclasses represent users, source documents, chunks, search
hits, citations, and answers. Citation labels include the policy version.
Quotation checks and claim coverage still belong to the generation layer.

Run the configuration and model tests without OpenAI or Docker:

    .venv/bin/python -m unittest discover -s tests -v

## Repository

`assistant.repository` provides a lazy connection pool with connection checks,
per-connection pgvector registration, and an explicit `close_pool()` for shutdown.
Apply `schema.sql` before opening the pool. Use `with pool().connection() as conn`
to borrow a connection; the context commits successful work and rolls back errors.

Trusted ingestion calls `find_by_hash`, `find_by_version`, and `store_document`.
The store operation skips duplicate hashes and atomically replaces an edited
title/version and its chunks. Concurrent uploads are serialized by content hash
and policy title. Compute embeddings before calling it. It preserves the supplied
active/superseded status; publication must explicitly update old policy metadata.

For user requests, load the identity with `get_user` and pass that trusted user
to `list_documents`, `searchable_chunks`, and `hybrid_search`. Both vector and
keyword candidate selection apply SQL visibility filters. Restricted access
requires employee status and the owning department. Normal queries exclude
superseded and future-effective policies; historical queries include superseded
policies but still exclude future-effective policies. Full-text search uses
PostgreSQL `ts_rank`, rather than BM25. Search results carry RRF scores for later
reranking. `record_event` stores answer outcomes and source labels.

Run all tests including PostgreSQL integration tests:

    RUN_DATABASE_TESTS=1 .venv/bin/python -m unittest discover -s tests -v

Integration tests create a temporary schema per test and roll back all fixtures.
They require a running database and the repository dependencies from `uv.lock`.

## Markdown ingestion

`assistant.ingestion.parse` requires a `.md` filename, a YAML front-matter block
with all seven metadata fields, and a nonempty body. See
`fixtures/policies/leave_policy.md` for a fictional demonstration policy.
Malformed YAML, duplicate fields, unsupported fields, and invalid metadata are
rejected before embedding. Filename paths are stripped; files are not saved to disk.

The content hash covers the entire upload including metadata, after normalizing
line endings and removing an optional UTF-8 BOM. Renaming a file does not change
the hash; changing its visibility does. `split` separates `##` sections before
applying the configured character size and overlap. Text before a section uses
the policy title as its section label.

`Ingestor.store` checks for duplicates before creating the OpenAI client, then
embeds all chunks in one provider call and stores them atomically. Duplicate
uploads report zero newly indexed chunks. A failed embedding call leaves an
existing policy untouched. Concurrent duplicate uploads can still incur redundant
embedding calls, but storage checks again under locks and indexes the content once.

To index the sample using your configured OpenAI key (incurs embedding usage),
run this Python from the project root:

```python
from pathlib import Path
from assistant.ingestion import Ingestor
from assistant.repository import pool, close_pool

path = Path('fixtures/policies/leave_policy.md')
try:
    with pool().connection() as connection:
        result = Ingestor().store(connection, path.read_text(encoding='utf-8'), path.name)
    print(result)
finally:
    close_pool()
```

Uploads preserve explicit `active` or `superseded` metadata. To retire an old
version, reupload it with `status: superseded`; uploading a new active version
does not automatically change another version's status.

Tests inject a fake embedding provider and make no OpenAI requests. The opt-in
database suite covers upload, renamed duplicates, edits, and permission-aware
retrieval in a rolled-back test schema.

## Retrieval and reranking

`assistant.retrieval.Retriever.search(connection, question, user)` embeds the
question and calls the repository's permission-filtered hybrid search. It sends
only those candidates to a structured-output reranker using `HELPER_MODEL` and
`prompts/rerank.txt`. Load the user through `repository.get_user` first.

The reranker assigns integer scores from 0 to 10. Scores below
`MIN_RERANK_SCORE` (default 4) are dropped; at most `TOP_K` (default 4) chunks
survive. Ties preserve fused candidate order. Returned hits have reranker scores
and retain their policy version, section, source, and update date.

Empty search results skip reranking. Missing ratings are discarded; invalid or
duplicate candidate numbers and malformed scores raise an error. Provider errors
propagate rather than falling back to unreranked results. Question and policy
content are sent as JSON data separately from reranker instructions.

`include_history=True` allows superseded versions through repository search.
The pipeline will decide history intent and perform the one query-rewrite retry;
neither behavior is implemented in this layer. Tests inject fake embeddings and
reranker responses, including a PostgreSQL test confirming that inaccessible
content never reaches the reranker. Live OpenAI calls are not yet verified.

## Answer generation

`assistant.generation.Generator.generate(question, hits)` accepts only hits already
permission-filtered by retrieval. It uses `ANSWER_MODEL` and `prompts/answer.txt`
to request a structured draft with inline numbered references and supporting
quotations. Citation labels, filenames, and chunk IDs come from the retrieved
hits, not from model-generated metadata.

Quotation checks normalize whitespace and case, then check the quote against its
referenced chunk. Missing citations, unknown chunk numbers, empty answers, and
mismatched inline references fail grounding. `Answer.grounding_errors` records
these failures, and `fully_grounded` is false when any remain. These are string
and reference checks, not proof of semantic support or complete claim coverage.

A failed grounding check triggers one repair attempt using the same evidence.
After a second failure, the draft is returned with an explicit untrusted warning.
This is separate from the pipeline's planned query-rewrite retry. Empty evidence
skips the model; model refusals return a standard insufficient-evidence message.
Malformed structured output and provider failures propagate to the caller.

Citing a superseded policy adds a historical-policy warning. Each cited policy
older than `STALE_AFTER_DAYS` adds a freshness warning, even if another cited
policy is fresh. Uncited policies do not produce warnings. Generation tests use
fake model responses; live OpenAI generation remains unverified.

## Question pipeline

`assistant.pipeline.Pipeline.ask(connection, question, username)` loads the user
from PostgreSQL, detects history intent, and retrieves only eligible evidence.
Use a connection context that commits on success to persist its audit record:

```python
from assistant.pipeline import Pipeline
from assistant.repository import pool, close_pool

try:
    with pool().connection() as connection:
        answer = Pipeline().ask(connection, 'How much annual leave?', 'bob')
    print(answer.text)
    print(answer.warnings)
finally:
    close_pool()
```

This invokes OpenAI when configured. Demo usernames are identities for learning;
an authenticated API must obtain the username from a trusted session or token.

Evidence scoring at least `HIGH_CONFIDENCE_SCORE` skips the extra grading call.
Otherwise the evidence grader decides whether the excerpts answer the original
question. Insufficient evidence triggers at most one rewritten search using the
grader's missing-facts description. Rewrites retain the same identity, history
flag, and policy date, and generation always receives the original question.
After two failed searches the pipeline refuses instead of guessing. No searchable
documents skips retrieval and rewrite calls entirely.

Each completed answer or refusal inserts one audit row with the user, original
question, answered/grounded flags, and source labels. Audit failure prevents a
success response. The caller owns the transaction: provider/validation failures
raise and do not create completed-answer audit rows; committing is required to
persist successful outcomes. Error-request auditing is not implemented yet.
The returned trace describes intent, access counts, retrieval attempts, grading,
rewriting, generation checks, and the audit insert. Citation repair remains a
separate bounded operation in the generator.

## Run the application

Install dependencies, start the database, apply the schema, and launch the API:

    uv sync --group dev
    docker compose up -d --wait
    docker compose exec -T db psql -U rag -d ragdb -v ON_ERROR_STOP=1 < schema.sql
    .venv/bin/python -m uvicorn assistant.api:app --reload --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000 for the interface or http://127.0.0.1:8000/docs for
interactive API documentation. Set your OpenAI key in `.env` for live model calls.
Select Bob (HR) to upload `fixtures/policies/leave_policy.md`, then ask how much
annual leave employees receive. Switch to Guest to confirm the internal policy
is not available. Uploading or asking may incur OpenAI usage.

The API provides `GET /`, `GET /health`, `GET /documents?username=guest`,
`POST /documents` (multipart `file` and `username`), and `POST /ask` (JSON
`question` and `username`). Health checks database connectivity and pgvector.
Listing requires a known identity and returns only readable document metadata.

Uploads require an employee identity, UTF-8 Markdown up to 2 MiB, and an owning
department matching that identity. Guest uploads and edits to another
department's existing version are rejected. Upload outcomes report inserted,
replaced, or duplicate, together with the newly indexed chunk count.
Question responses include text, citations, warnings, grounding errors, a trace,
and the computed `fully_grounded` flag. The connection dependency commits before
the response is sent, so commit failures cannot report successful indexing or
auditing. Database and provider errors return generic messages without secrets.

This is a local learning interface: usernames are supplied by the client and
can be impersonated. The selector is not login or production authentication.
Replace it with a trusted session/token before a real deployment. The server
defaults above bind to loopback, and responses are marked `no-store`.
The interface renders policy and model text using text nodes, clears previous
results on identity changes, and ignores responses from a previous identity.

API tests and the PostgreSQL upload-to-answer integration test use fake model
responses. Visual browser verification remains pending; HTML delivery and
JavaScript syntax have been checked.

## Run without an API key

Set this in the project `.env` file and restart the server:

    DEMO_MODE=true

Then run:

    docker compose up -d --wait
    .venv/bin/python -m uvicorn assistant.api:app --reload --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000. A prominent demo banner confirms that no API key is
required. `GET /config` and `GET /health` report `demo_mode: true`; question and
upload responses carry the same flag. Each completed demo question includes a
warning and trace entry identifying the scripted providers.

Upload `fixtures/policies/leave_policy.md` as Bob, then ask "How much annual
leave?" Upload the public refund policy as Carol and ask about refunds as Guest.
Upload both travel versions as Carol to compare the current hotel limit with
version 2. The full sample list is in `fixtures/README.md`. The application
still requires PostgreSQL, metadata validation, department ownership, and user
permissions; no sample directory is scanned or automatically indexed.

Demo embeddings are deterministic hashed word features. Reranking and evidence
grading use word overlap. Intent uses simple historical-word/version patterns.
Generation selects an exact matching sentence from the retrieved text and labels
it "Demo excerpt". Citation verification, warnings, retry bounds, and audits run
through the existing pipeline. This is a demonstration of application behavior,
not a language model or a test of semantic retrieval quality. Synonyms, nuanced
questions, negation, and comparisons may not be interpreted correctly.

The connection pool automatically initializes and selects the `assistant_demo`
schema when demo mode is enabled. Demo documents, fake vectors, users, and audits
are separate from the normal `public` schema; existing live documents are not
copied or changed. The demo library starts empty and persists across restarts.
After obtaining an API key, set `DEMO_MODE=false` and restart the server. Normal
mode uses the existing live library; upload demo-only documents again to generate
real embeddings. Never copy fake demo vectors into the live tables. Leave the
mode fixed for a running process and always restart when changing it.

The default in `.env.example` remains `false`. Normal OpenAI behavior does not
silently switch to fake providers when a key is missing or a request fails.
Database integration tests verify demo uploads and duplicates, authorized answers,
guest and department restrictions, historical citations, unrelated-question
refusals, audit rows, and storage isolation without permitting an API key call.
