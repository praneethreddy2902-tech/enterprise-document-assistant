"""Opt-in real PostgreSQL tests, isolated in a schema rolled back per test."""

from datetime import date, timedelta
from hashlib import sha256
import os
from pathlib import Path
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from assistant.config import load_settings
from assistant.models import Chunk, PolicyMetadata, SourceDocument, User
from assistant import repository as repo


@unittest.skipUnless(os.environ.get('RUN_DATABASE_TESTS') == '1', 'Set RUN_DATABASE_TESTS=1 for PostgreSQL tests')
class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.connection = psycopg.connect(load_settings().database_url.get_secret_value(), autocommit=True,
                                           row_factory=dict_row)
        self.addCleanup(self.connection.close)
        self.connection.execute('BEGIN')
        self.addCleanup(lambda: self.connection.execute('ROLLBACK'))
        schema = 'repository_test_' + uuid4().hex
        self.connection.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        self.connection.execute(sql.SQL('SET LOCAL search_path TO {}, public').format(sql.Identifier(schema)))
        script = (Path(__file__).resolve().parents[1] / 'schema.sql').read_text()
        script = script[script.index('CREATE TABLE'):].removesuffix('\n').removesuffix('COMMIT;')
        self.connection.execute(script)
        self.vector = [1.0] + [0.0] * 1535
        self.guest = repo.get_user(self.connection, 'guest')
        self.alice = repo.get_user(self.connection, 'alice')
        self.bob = repo.get_user(self.connection, 'bob')

    def document(self, title='Leave', version=1, visibility='internal', department='hr',
                 status='active', effective=None, content='Annual leave is twenty days.'):
        metadata = PolicyMetadata(title=title, version=version, visibility=visibility,
                                  department=department, status=status,
                                  effective=effective or date.today(), updated=date.today())
        digest = sha256((metadata.model_dump_json() + content).encode()).hexdigest()
        return SourceDocument('policy.md', metadata, content, digest)

    def store(self, **kwargs):
        doc = self.document(**kwargs)
        return repo.store_document(self.connection, doc, [Chunk('Allowance', doc.content)], [self.vector])

    def search(self, user, **kwargs):
        return repo.hybrid_search(self.connection, 'annual leave', self.vector, user, **kwargs)

    def test_permissions_apply_before_candidate_limit(self):
        self.store(title='Engineering', visibility='restricted', department='engineering')
        self.store(title='HR', visibility='restricted')
        self.store(title='Internal')
        self.store(title='Public', visibility='public')
        self.assertEqual({hit.title for hit in self.search(self.guest)}, {'Public'})
        self.assertEqual({hit.title for hit in self.search(self.alice)}, {'Engineering', 'Internal', 'Public'})
        self.assertEqual({hit.title for hit in self.search(self.bob)}, {'HR', 'Internal', 'Public'})
        # An earlier inaccessible row cannot consume a one-row candidate limit.
        self.assertEqual(self.search(self.guest, candidates=1)[0].title, 'Public')
        outsider = User('outsider', 'engineering', False)
        self.assertEqual({hit.title for hit in self.search(outsider)}, {'Public'})
        self.assertEqual({row['title'] for row in repo.list_documents(self.connection, self.guest)}, {'Public'})
        self.assertEqual(repo.searchable_chunks(self.connection, self.guest), {'total': 4, 'searchable': 1})

    def test_history_and_future_effective_policy_filter(self):
        self.store(version=1, status='superseded')
        self.store(version=2)
        self.store(version=3, effective=date.today() + timedelta(days=1))
        self.assertEqual([h.version for h in self.search(self.bob)], [2])
        self.assertEqual({h.version for h in self.search(self.bob, include_history=True)}, {1, 2})

    def test_duplicate_and_edit_replace_chunks(self):
        first = self.store()
        self.assertEqual(self.store()['outcome'], 'duplicate')
        changed = self.document(content='Annual leave is thirty days.')
        result = repo.store_document(self.connection, changed, [Chunk('New section', changed.content)], [self.vector])
        self.assertEqual(result['outcome'], 'replaced')
        self.assertIsNone(repo.find_by_hash(self.connection, self.document().content_hash))
        self.assertNotEqual(first['document_id'], result['document_id'])
        self.assertEqual(repo.find_by_version(self.connection, 'Leave', 1)['id'], result['document_id'])
        hits = self.search(self.bob)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].section, 'New section')

    def test_failed_replacement_preserves_previous_policy(self):
        first = self.store()
        with self.assertRaises(psycopg.errors.CheckViolation):
            repo.store_document(self.connection, self.document(content='Edited text'),
                                [Chunk('', 'Edited text')], [self.vector])
        self.assertEqual(repo.find_by_version(self.connection, 'Leave', 1)['id'], first['document_id'])
        self.assertEqual(self.search(self.bob)[0].content, 'Annual leave is twenty days.')

    def test_policy_owner_cannot_change_during_replacement(self):
        original = self.store()
        changed = self.document(department='engineering')
        with self.assertRaises(repo.PolicyOwnershipError):
            repo.store_document(self.connection, changed, [Chunk('Leave', changed.content)], [self.vector])
        self.assertEqual(repo.find_by_version(self.connection, 'Leave', 1)['id'], original['document_id'])

    def test_rrf_adds_both_ranks_and_supports_semantic_only(self):
        self.store()
        hit = self.search(self.bob)[0]
        self.assertAlmostEqual(hit.score, 2 / 61)
        semantic = repo.hybrid_search(self.connection, 'unmatchedterm', self.vector, self.bob)
        self.assertAlmostEqual(semantic[0].score, 1 / 61)

    def test_audit_health_and_unknown_user(self):
        self.assertIn('pgvector_version', repo.health(self.connection))
        with self.assertRaisesRegex(ValueError, 'Unknown user'):
            repo.get_user(self.connection, "alice' OR TRUE --")
        event_id = repo.record_event(self.connection, username='bob', question="What's the policy?",
                                    answered=False, grounded=False)
        row = self.connection.execute('SELECT * FROM audit_events WHERE id = %s', (event_id,)).fetchone()
        self.assertEqual(row['sources'], 'none')
        self.assertFalse(row['answered'])

    def test_invalid_vectors_and_chunk_counts(self):
        for vector in ([1, 2], [0.0] * 1536, [float('nan')] * 1536):
            with self.subTest(vector_length=len(vector)), self.assertRaises(ValueError):
                repo.hybrid_search(self.connection, 'leave', vector, self.bob)
        with self.assertRaises(ValueError):
            repo.store_document(self.connection, self.document(), [], [])

    def test_pool_lifecycle(self):
        self.addCleanup(repo.close_pool)
        first = repo.pool()
        first.wait(timeout=5)
        with first.connection() as connection:
            self.assertIn('pgvector_version', repo.health(connection))
        self.assertIs(repo.pool(), first)
        repo.close_pool()
        self.assertTrue(first.closed)
        second = repo.pool()
        self.assertIsNot(second, first)
        second.wait(timeout=5)

    def test_ingestion_round_trip_and_duplicate_without_model_call(self):
        from unittest.mock import Mock
        from assistant.ingestion import Ingestor

        path = Path(__file__).resolve().parents[1] / 'fixtures/policies/leave_policy.md'
        text = path.read_text()
        embedder = Mock()
        embedder.embed_documents.side_effect = lambda texts: [self.vector for _ in texts]
        ingestor = Ingestor(load_settings(), embeddings=embedder)
        first = ingestor.store(self.connection, text, path.name)
        duplicate = ingestor.store(self.connection, text.replace('\n', '\r\n'), 'renamed.md')
        self.assertEqual(first.outcome, 'inserted')
        self.assertEqual(duplicate.outcome, 'duplicate')
        self.assertEqual(duplicate.document_id, first.document_id)
        self.assertEqual(embedder.embed_documents.call_count, 1)
        self.assertEqual(self.search(self.guest), [])
        self.assertTrue(any('twenty days' in hit.content for hit in self.search(self.alice)))
        edited = ingestor.store(self.connection, text.replace('twenty days', 'thirty days'), path.name)
        self.assertEqual(edited.outcome, 'replaced')
        hits = self.search(self.bob)
        self.assertTrue(any('thirty days' in hit.content for hit in hits))
        self.assertFalse(any('twenty days' in hit.content for hit in hits))

    def test_retrieval_passes_only_authorized_candidates_to_reranker(self):
        import json
        from unittest.mock import Mock
        from assistant.retrieval import Retriever

        self.store(title='Engineering secret', visibility='restricted', department='engineering')
        self.store(title='Internal leave')
        self.store(title='Public leave', visibility='public')
        self.store(title='Old public leave', visibility='public', status='superseded')
        embeddings = Mock()
        embeddings.embed_query.return_value = self.vector
        reranker = Mock()
        def rate(messages):
            candidates = json.loads(messages[1][1])['candidates']
            return {'ratings': [{'index': c['index'], 'score': 9} for c in candidates]}
        reranker.invoke.side_effect = rate
        retriever = Retriever(load_settings(), embeddings=embeddings, reranker=reranker)
        hits = retriever.search(self.connection, 'annual leave', self.guest)
        self.assertEqual([h.title for h in hits], ['Public leave'])
        sent = json.loads(reranker.invoke.call_args.args[0][1][1])['candidates']
        self.assertEqual([c['label'] for c in sent], ['Public leave v1 > Allowance'])
        self.assertEqual(hits[0].score, 9)
        historical = retriever.search(self.connection, 'annual leave', self.guest, include_history=True)
        self.assertEqual({h.title for h in historical}, {'Public leave', 'Old public leave'})

    def test_generation_uses_retrieved_source_and_verifies_quote(self):
        import json
        from unittest.mock import Mock
        from assistant.generation import Generator

        self.store(title='Public leave', visibility='public')
        self.store(title='Private leave', visibility='restricted')
        hits = self.search(self.guest)
        model = Mock()
        model.invoke.return_value = dict(
            answer='Annual leave is twenty days. [1]', answered=True,
            citations=[dict(chunk_number=1, quote='Annual leave is twenty days.')])
        answer = Generator(load_settings(), model=model).generate('How much leave?', hits)
        self.assertTrue(answer.fully_grounded)
        self.assertEqual(answer.citations[0].label, 'Public leave v1 > Allowance')
        self.assertEqual(answer.citations[0].chunk_id, hits[0].chunk_id)
        payload = json.loads(model.invoke.call_args.args[0][1][1])
        self.assertEqual([c['label'] for c in payload['chunks']], ['Public leave v1 > Allowance'])

    def test_pipeline_end_to_end_audits_answer_and_empty_access_refusal(self):
        import json
        from unittest.mock import Mock
        from assistant.generation import Generator
        from assistant.pipeline import Pipeline
        from assistant.retrieval import Retriever

        self.store(title='Restricted leave', visibility='restricted')
        embeddings, reranker, model, intent = Mock(), Mock(), Mock(), Mock()
        embeddings.embed_query.return_value = self.vector
        reranker.invoke.side_effect = lambda messages: {'ratings': [
            {'index': c['index'], 'score': 9}
            for c in json.loads(messages[1][1])['candidates']]}
        intent.invoke.return_value = dict(wants_history=False)
        model.invoke.return_value = dict(answer='Annual leave is twenty days. [1]', answered=True,
            citations=[dict(chunk_number=1, quote='Annual leave is twenty days.')])
        settings = load_settings()
        pipeline = Pipeline(settings, intent=intent,
            retriever=Retriever(settings, embeddings=embeddings, reranker=reranker),
            generator=Generator(settings, model=model))
        answer = pipeline.ask(self.connection, 'How much annual leave?', 'bob')
        self.assertTrue(answer.fully_grounded)
        refusal = pipeline.ask(self.connection, 'How much annual leave?', 'guest')
        self.assertFalse(refusal.answered)
        self.assertEqual(embeddings.embed_query.call_count, 1)
        self.assertEqual(model.invoke.call_count, 1)
        events = self.connection.execute('SELECT * FROM audit_events ORDER BY id').fetchall()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['username'], 'bob')
        self.assertTrue(events[0]['grounded'])
        self.assertEqual(events[0]['sources'], 'Restricted leave v1 > Allowance')
        self.assertEqual(events[1]['username'], 'guest')
        self.assertFalse(events[1]['answered'])
        self.assertEqual(events[1]['sources'], 'none')

    def test_api_upload_list_ask_and_audit_round_trip(self):
        import json
        from unittest.mock import Mock
        from fastapi.testclient import TestClient
        from assistant.api import create_app, get_connection
        from assistant.generation import Generator
        from assistant.ingestion import Ingestor
        from assistant.pipeline import Pipeline
        from assistant.retrieval import Retriever

        embeddings, reranker, model, intent = Mock(), Mock(), Mock(), Mock()
        embeddings.embed_documents.side_effect = lambda texts: [self.vector for _ in texts]
        embeddings.embed_query.return_value = self.vector
        reranker.invoke.side_effect = lambda messages: {'ratings': [
            {'index': c['index'], 'score': 9}
            for c in json.loads(messages[1][1])['candidates']]}
        intent.invoke.return_value = dict(wants_history=False)
        def draft(messages):
            chunks = json.loads(messages[1][1])['chunks']
            chunk = next(c for c in chunks if 'twenty days' in c['content'])
            number = chunk['chunk_number']
            return dict(answer=f'Employees receive twenty days of annual leave. [{number}]',
                        answered=True, citations=[dict(chunk_number=number,
                        quote='Employees receive twenty days of annual leave per calendar year.')])
        model.invoke.side_effect = draft
        settings = load_settings()
        pipeline = Pipeline(settings, intent=intent,
            retriever=Retriever(settings, embeddings=embeddings, reranker=reranker),
            generator=Generator(settings, model=model))
        app = create_app(pipeline=pipeline, ingestor=Ingestor(settings, embeddings=embeddings))
        app.dependency_overrides[get_connection] = lambda: self.connection
        client = TestClient(app)
        path = Path(__file__).resolve().parents[1] / 'fixtures/policies/leave_policy.md'
        uploaded = client.post('/documents', data={'username': 'bob'}, files={'file': (path.name, path.read_bytes())})
        self.assertEqual(uploaded.status_code, 200)
        self.assertEqual(uploaded.json()['outcome'], 'inserted')
        self.assertEqual(client.get('/documents?username=guest').json(), [])
        self.assertEqual(len(client.get('/documents?username=bob').json()), 1)
        response = client.post('/ask', json={'username': 'bob', 'question': 'How much annual leave?'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['fully_grounded'])
        refusal = client.post('/ask', json={'username': 'guest', 'question': 'How much annual leave?'})
        self.assertFalse(refusal.json()['answered'])
        count = self.connection.execute('SELECT count(*) AS count FROM audit_events').fetchone()['count']
        self.assertEqual(count, 2)

    def test_demo_api_without_key_permissions_history_duplicates_and_audit(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from assistant.api import create_app, get_connection
        from assistant.config import Settings

        settings = Settings(demo_mode=True)
        app = create_app(settings=settings)
        app.dependency_overrides[get_connection] = lambda: self.connection
        client = TestClient(app)
        directory = Path(__file__).resolve().parents[1] / 'fixtures/policies'
        with patch.object(Settings, 'require_openai_key', side_effect=AssertionError('No API key permitted')):
            for filename, owner in [('leave_policy.md', 'bob'), ('security_policy.md', 'alice'),
                                    ('refund_policy.md', 'carol'), ('travel_policy_v2.md', 'carol'),
                                    ('travel_policy_v3.md', 'carol'), ('remote_work_policy.md', 'bob')]:
                response = client.post('/documents', data={'username': owner},
                    files={'file': (filename, (directory / filename).read_bytes())})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json()['demo_mode'])
            duplicate = client.post('/documents', data={'username': 'bob'},
                files={'file': ('renamed.md', (directory / 'leave_policy.md').read_bytes())})
            self.assertEqual(duplicate.json()['outcome'], 'duplicate')
            cases = [
                ('bob', 'How much annual leave?', True, 'twenty days'),
                ('guest', 'How much annual leave?', False, None),
                ('guest', 'What is the refund policy?', True, 'refund'),
                ('alice', 'Can engineers upload source code to external AI tools?', True, 'must not upload'),
                ('bob', 'Can engineers upload source code to external AI tools?', False, None),
                ('bob', 'What is the current domestic hotel reimbursement limit?', True, '4500'),
                ('bob', 'What was the domestic hotel reimbursement limit in version 2?', True, '3000'),
                ('bob', 'What is the policy on pets at the office?', False, None),
            ]
            for owner, question, expected, phrase in cases:
                with self.subTest(question=question, owner=owner):
                    response = client.post('/ask', json={'username': owner, 'question': question})
                    self.assertEqual(response.status_code, 200, response.text)
                    result = response.json()
                    self.assertEqual(result['answered'], expected, result)
                    self.assertTrue(result['demo_mode'])
                    self.assertTrue(any('Demo mode' in warning for warning in result['warnings']))
                    if phrase:
                        self.assertIn(phrase, result['text'])
                        self.assertTrue(result['fully_grounded'])
                    if 'version 2' in question:
                        self.assertTrue(any('superseded' in warning for warning in result['warnings']))
            events = self.connection.execute('SELECT count(*) AS count FROM audit_events').fetchone()['count']
            self.assertEqual(events, len(cases))

    def test_demo_schema_keeps_existing_embeddings_separate(self):
        from psycopg import sql

        original_schema = self.connection.execute('SELECT current_schema() AS name').fetchone()['name']
        self.store(title='Isolation ' + original_schema)
        repo.initialize_demo_schema(self.connection)
        self.assertEqual(self.connection.execute('SELECT current_schema() AS name').fetchone()['name'], 'assistant_demo')
        self.assertIsNone(repo.find_by_version(self.connection, 'Isolation ' + original_schema, 1))
        self.connection.execute(sql.SQL('SET LOCAL search_path TO {}, public').format(sql.Identifier(original_schema)))
        self.assertIsNotNone(repo.find_by_version(self.connection, 'Isolation ' + original_schema, 1))
