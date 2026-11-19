import os
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from assistant.config import Settings, load_settings, prompt
from assistant.models import Answer, Citation, Hit, PolicyMetadata, PolicyStatus, User


class ConfigurationTests(unittest.TestCase):
    def test_invalid_tuning_is_rejected(self):
        for values in (
            {'chunk_overlap': 600}, {'top_k': 13}, {'max_attempts': 3},
            {'min_rerank_score': 9, 'high_confidence_score': 8},
            {'embedding_dimensions': 768}, {'database_url': 'sqlite:///db'},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Settings(**values)

    def test_environment_overrides_file_without_mutation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('CHUNK_SIZE=800\nOPENAI_API_KEY=test-secret\n')
            with patch.dict(os.environ, {'CHUNK_SIZE': '900'}, clear=True):
                settings = load_settings(path)
                self.assertEqual(settings.chunk_size, 900)
                self.assertEqual(settings.require_openai_key(), 'test-secret')
                self.assertNotIn('test-secret', repr(settings))
                self.assertNotIn('OPENAI_API_KEY', os.environ)

    def test_missing_key_only_blocks_model_operations(self):
        settings = Settings()
        with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
            settings.require_openai_key()

    def test_placeholder_keys_are_rejected_without_echoing_secrets(self):
        for key in ('sk-...', 'your_openai_api_key', 'sk-your-secret'):
            with self.subTest(key=key), self.assertRaises(ValueError) as caught:
                Settings(openai_api_key=key).require_openai_key()
            self.assertNotIn(key, str(caught.exception))

    def test_prompt_rejects_path_traversal_and_empty_scaffold(self):
        with self.assertRaises(ValueError):
            prompt('../.env')
        with patch('assistant.config.Path.read_text', return_value='   '):
            with self.assertRaisesRegex(ValueError, 'not been implemented'):
                prompt('answer')


class ModelTests(unittest.TestCase):
    def metadata(self, **overrides):
        return PolicyMetadata(**dict({
            'title': ' Leave policy ', 'department': ' HR ',
            'visibility': 'internal', 'version': 1, 'status': 'active',
            'effective': '2026-06-30', 'updated': date(2026, 6, 30),
        }, **overrides))

    def test_metadata_normalization(self):
        metadata = self.metadata()
        self.assertEqual(metadata.title, 'Leave policy')
        self.assertEqual(metadata.department, 'hr')
        self.assertEqual(metadata.effective, date(2026, 6, 30))

    def test_invalid_metadata_is_rejected(self):
        for values in (
            {'visibility': 'private'}, {'status': 'draft'}, {'version': True},
            {'version': 0}, {'department': ' '}, {'updated': '2026-02-30'},
            {'effective': 123456}, {'unexpected': 'field'},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                self.metadata(**values)

    def test_access_description_and_versioned_label(self):
        self.assertEqual(User('guest', 'hr', False).describe_access(), 'public documents only')
        self.assertIn('restricted hr', User('bob', 'hr', True).describe_access())
        hit = Hit(1, 'Policy text', 'Leave', 'leave.md', 'Leave policy', 2,
                  PolicyStatus.SUPERSEDED, date(2026, 6, 30))
        self.assertEqual(hit.label, 'Leave policy v2 > Leave')

    def test_grounding_requires_answer_and_verified_citations(self):
        valid = Citation(1, 'Leave policy v1 > Leave', 'leave.md', 'Exact quote.', True)
        invalid = Citation(2, 'Leave policy v1 > Leave', 'leave.md', 'Invented quote.', False)
        self.assertFalse(Answer('No evidence', False).fully_grounded)
        self.assertFalse(Answer('Unsupported', True).fully_grounded)
        self.assertFalse(Answer('Mixed', True, [valid, invalid]).fully_grounded)
        self.assertTrue(Answer('Supported', True, [valid]).fully_grounded)
        first, second = Answer('a', False), Answer('b', False)
        first.trace.append('step')
        self.assertEqual(second.trace, [])


if __name__ == '__main__':
    unittest.main()
