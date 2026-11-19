from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from assistant.config import Settings
from assistant.ingestion import IngestionError, Ingestor, parse, split

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/policies/leave_policy.md'


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.text = FIXTURE.read_text()

    def test_hash_ignores_filename_bom_and_line_endings(self):
        first = parse(self.text, 'leave.md')
        second = parse('\ufeff' + self.text.replace('\n', '\r\n'), '../renamed.md')
        third = parse(self.text.replace('\n', '\r'), 'C:\\uploads\\policy.md')
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.content_hash, third.content_hash)
        self.assertEqual(first.content_hash, sha256(self.text.encode()).hexdigest())
        self.assertEqual(second.source, 'renamed.md')
        self.assertEqual(third.source, 'policy.md')
        self.assertNotIn('visibility: internal', first.content)

    def test_metadata_changes_affect_hash(self):
        original = parse(self.text, 'leave.md')
        changed = parse(self.text.replace('visibility: internal', 'visibility: restricted'), 'leave.md')
        self.assertNotEqual(original.content_hash, changed.content_hash)

    def test_invalid_uploads(self):
        cases = [
            ('', 'empty.md'), ('No metadata', 'leave.md'), ('---\ntitle: Test', 'leave.md'),
            (self.text, 'leave.txt'), ('---\n- list\n---\nBody', 'leave.md'),
            (self.text.replace('visibility: internal', 'visibility: private'), 'leave.md'),
            (self.text.replace('version: 1', 'version: true'), 'leave.md'),
            (self.text.replace('title: Leave policy', 'title: Leave policy\ntitle: Other'), 'leave.md'),
            (self.text.replace('updated: 2026-06-30', 'updated: 2026-02-30'), 'leave.md'),
            ('---\ntitle: [broken\n---\nBody', 'leave.md'),
            ('---\ntitle: !!python/object:builtins.object {}\n---\nBody', 'leave.md'),
            (self.text + '\x00', 'leave.md'),
            (self.text.split('\n# Leave policy')[0] + '\n', 'leave.md'),
        ]
        for text, filename in cases:
            with self.subTest(text=text[:60], filename=filename), self.assertRaises(IngestionError):
                parse(text, filename)

    def test_sections_preserve_labels_and_split_long_content(self):
        text = self.text + '\n## Extended rules\n\n' + ('Employees request annual leave. ' * 100)
        chunks = split(parse(text, 'leave.md'), Settings(chunk_size=100, chunk_overlap=20))
        self.assertTrue(all(0 < len(c.content) <= 100 for c in chunks))
        self.assertIn('Annual leave', {c.section for c in chunks})
        self.assertIn('Sick leave', {c.section for c in chunks})
        self.assertGreater(len([c for c in chunks if c.section == 'Extended rules']), 1)
        self.assertFalse(any('visibility: internal' in c.content for c in chunks))

    def test_no_section_uses_title_and_empty_section_is_rejected(self):
        header = self.text.split('\n# Leave policy')[0]
        chunks = split(parse(header + '\nPlain policy text.', 'leave.md'))
        self.assertEqual(chunks[0].section, 'Leave policy')
        with self.assertRaises(IngestionError):
            split(parse(header + '\n## Empty section\n', 'leave.md'))

    def test_duplicate_skips_embeddings_and_writes(self):
        embedder = Mock()
        with patch('assistant.ingestion.repository') as repository:
            repository.find_by_hash.return_value = {'id': 7, 'title': 'Leave policy', 'version': 1}
            result = Ingestor(Settings(), embeddings=embedder).store(Mock(), self.text, 'leave.md')
            self.assertEqual((result.outcome, result.document_id, result.chunks), ('duplicate', 7, 0))
            embedder.embed_documents.assert_not_called()
            repository.store_document.assert_not_called()

    def test_invalid_policy_never_calls_database_or_model(self):
        embedder = Mock()
        with patch('assistant.ingestion.repository') as repository:
            with self.assertRaises(IngestionError):
                Ingestor(Settings(), embeddings=embedder).store(Mock(), 'Invalid', 'leave.md')
            repository.find_by_hash.assert_not_called()
            embedder.embed_documents.assert_not_called()

    def test_embedding_failure_never_replaces_existing_policy(self):
        embedder = Mock()
        embedder.embed_documents.side_effect = RuntimeError('Provider unavailable')
        with patch('assistant.ingestion.repository') as repository:
            repository.find_by_hash.return_value = None
            with self.assertRaisesRegex(RuntimeError, 'Provider unavailable'):
                Ingestor(Settings(), embeddings=embedder).store(Mock(), self.text, 'leave.md')
            repository.store_document.assert_not_called()

    def test_embedding_batch_and_racing_duplicate(self):
        embedder = Mock()
        expected = split(parse(self.text, 'leave.md'))
        embedder.embed_documents.return_value = [[1.0] + [0.0] * 1535 for _ in expected]
        with patch('assistant.ingestion.repository') as repository:
            repository.find_by_hash.return_value = None
            repository.store_document.return_value = {'document_id': 8, 'outcome': 'inserted'}
            ingestor = Ingestor(Settings(), embeddings=embedder)
            result = ingestor.store(Mock(), self.text, 'leave.md')
            embedder.embed_documents.assert_called_once_with([c.content for c in expected])
            self.assertEqual(result.chunks, len(expected))
            repository.store_document.return_value = {'document_id': 8, 'outcome': 'duplicate'}
            self.assertEqual(ingestor.store(Mock(), self.text, 'leave.md').chunks, 0)
