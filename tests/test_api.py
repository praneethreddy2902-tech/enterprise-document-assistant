from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from psycopg import OperationalError

from assistant.api import MAX_UPLOAD_BYTES, create_app, get_connection
from assistant.config import ModelConfigurationError, Settings
from assistant.models import Answer, Citation, UploadResult, User

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/policies/leave_policy.md'


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.connection = Mock()
        self.pipeline, self.ingestor = Mock(), Mock()
        self.app = create_app(pipeline=self.pipeline, ingestor=self.ingestor)
        self.app.dependency_overrides[get_connection] = lambda: self.connection
        self.client = TestClient(self.app)
        self.patch = patch('assistant.api.repository')
        self.repo = self.patch.start()
        self.addCleanup(self.patch.stop)
        users = {name: User(name, dept, employee) for name, dept, employee in (
            ('guest', 'none', False), ('bob', 'hr', True), ('alice', 'engineering', True))}
        def user(connection, name):
            if name not in users:
                raise ValueError('Unknown user')
            return users[name]
        self.repo.get_user.side_effect = user
        self.repo.find_by_version.return_value = None
        self.repo.health.return_value = dict(database='ragdb', pgvector_version='0.8.7')
        self.repo.list_documents.return_value = []
        self.ingestor.store.return_value = UploadResult('leave.md', 1, 'Leave policy', 1, 'inserted', 3)
        self.pipeline.ask.return_value = Answer('Twenty days.', True, citations=[
            Citation(1, 'Leave policy v1 > Leave', 'leave.md', 'Twenty days.', True)])

    def upload(self, content=None, filename='leave.md', username='bob'):
        return self.client.post('/documents', data={'username': username},
                                files={'file': (filename, FIXTURE.read_bytes() if content is None else content)})

    def test_home_and_health(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Your policies. Clear answers.', response.text)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(self.client.get('/health').json()['status'], 'ok')

    def test_listing_uses_database_identity(self):
        response = self.client.get('/documents?username=bob')
        self.assertEqual(response.json(), [])
        self.assertEqual(self.repo.list_documents.call_args.args[1].department, 'hr')
        self.assertEqual(self.client.get('/documents?username=unknown').status_code, 404)

    def test_upload_is_validated_and_indexes(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['outcome'], 'inserted')
        self.ingestor.store.assert_called_once()

    def test_guest_and_department_uploads_are_rejected(self):
        self.assertEqual(self.upload(username='guest').status_code, 403)
        self.assertEqual(self.upload(username='alice').status_code, 403)
        self.repo.find_by_version.return_value = {'department': 'engineering'}
        self.assertEqual(self.upload().status_code, 403)
        self.ingestor.store.assert_not_called()

    def test_invalid_and_oversize_files_never_reach_ingestor(self):
        for content, filename, expected in (
            (b'No metadata', 'leave.md', 422), (b'\xff', 'leave.md', 422),
            (b'content', 'leave.pdf', 422), (b'a' * (MAX_UPLOAD_BYTES + 1), 'leave.md', 413),
        ):
            with self.subTest(filename=filename, expected=expected):
                self.assertEqual(self.upload(content, filename).status_code, expected)
        self.ingestor.store.assert_not_called()

    def test_ask_serializes_citations_and_grounding(self):
        response = self.client.post('/ask', json={'question': 'How much leave?', 'username': 'bob'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['fully_grounded'])
        self.assertTrue(response.json()['citations'][0]['verified'])
        self.pipeline.ask.assert_called_once_with(self.connection, 'How much leave?', 'bob')

    def test_blank_unknown_and_extra_fields_are_rejected(self):
        self.assertEqual(self.client.post('/ask', json={'question': '  '}).status_code, 422)
        self.assertEqual(self.client.post('/ask', json={'question': 'x', 'username': 'unknown'}).status_code, 404)
        self.assertEqual(self.client.post('/ask', json={'question': 'x', 'department': 'hr'}).status_code, 422)
        self.pipeline.ask.assert_not_called()

    def test_database_and_configuration_errors_hide_details(self):
        self.repo.health.side_effect = OperationalError('secret database URL')
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('secret', response.text)
        self.pipeline.ask.side_effect = ValueError('secret key')
        response = self.client.post('/ask', json={'question': 'Hello'})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('secret', response.text)

    def test_commit_failure_prevents_success_response(self):
        self.app.dependency_overrides.clear()
        context = self.repo.pool.return_value.connection.return_value
        context.__enter__.return_value = self.connection
        context.__exit__.side_effect = OperationalError('Commit failed')
        response = self.client.post('/ask', json={'question': 'Hello'})
        self.assertEqual(response.status_code, 503)

    def test_model_configuration_error_is_actionable(self):
        self.pipeline.ask.side_effect = ModelConfigurationError('Set a real OPENAI_API_KEY in .env')
        response = self.client.post('/ask', json={'question': 'Hello'})
        self.assertEqual(response.status_code, 503)
        self.assertIn('OPENAI_API_KEY', response.json()['detail'])

    def test_config_reports_explicit_demo_mode_without_database(self):
        app = create_app(settings=Settings(demo_mode=True))
        response = TestClient(app).get('/config')
        self.assertEqual(response.json(), {'demo_mode': True})
        normal = create_app(settings=Settings())
        self.assertEqual(TestClient(normal).get('/config').json(), {'demo_mode': False})
