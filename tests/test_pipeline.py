from datetime import date
import json
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from assistant.config import Settings
from assistant.models import Answer, Citation, Hit, PolicyStatus, User
from assistant.pipeline import Pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.repo_patch = patch('assistant.pipeline.repository')
        self.repo = self.repo_patch.start()
        self.addCleanup(self.repo_patch.stop)
        self.user = User('bob', 'hr', True)
        self.repo.get_user.return_value = self.user
        self.repo.searchable_chunks.return_value = dict(searchable=1, total=3)
        self.retriever, self.generator = Mock(), Mock()
        self.intent, self.grader, self.rewriter = Mock(), Mock(), Mock()
        self.intent.invoke.return_value = dict(wants_history=False)
        self.grader.invoke.return_value = dict(sufficient=True, missing='')
        self.rewriter.invoke.return_value = dict(query='annual leave allowance')
        self.hit = Hit(1, 'Twenty days.', 'Leave', 'leave.md', 'Leave', 1,
                       PolicyStatus.ACTIVE, date(2026, 6, 30), 9)
        self.retriever.search.return_value = [self.hit]
        self.generator.generate.return_value = Answer('Twenty days.', True, citations=[
            Citation(1, self.hit.label, 'leave.md', 'Twenty days.', True)], trace=['generated'])
        self.pipeline = Pipeline(Settings(), retriever=self.retriever, generator=self.generator,
                                 intent=self.intent, grader=self.grader, rewriter=self.rewriter)
        self.connection = Mock()
        self.today = date(2026, 11, 17)

    def ask(self):
        return self.pipeline.ask(self.connection, 'How much leave?', 'bob', today=self.today)

    def test_high_confidence_skips_grader_and_audits_once(self):
        answer = self.ask()
        self.assertTrue(answer.fully_grounded)
        self.grader.invoke.assert_not_called()
        self.rewriter.invoke.assert_not_called()
        self.repo.record_event.assert_called_once_with(
            self.connection, username='bob', question='How much leave?', answered=True,
            grounded=True, sources=self.hit.label)
        self.assertIn('generated', answer.trace)

    def test_history_and_trusted_identity_are_preserved(self):
        self.intent.invoke.return_value = dict(wants_history=True)
        self.ask()
        self.retriever.search.assert_called_once_with(
            self.connection, 'How much leave?', self.user, True, today=self.today)
        self.generator.generate.assert_called_once_with(
            'How much leave?', [self.hit], today=self.today, include_history=True)

    def test_low_confidence_is_graded_against_original_question(self):
        from dataclasses import replace
        self.retriever.search.return_value = [replace(self.hit, score=5)]
        self.ask()
        self.grader.invoke.assert_called_once()
        payload = json.loads(self.grader.invoke.call_args.args[0][1][1])
        self.assertEqual(payload['question'], 'How much leave?')
        self.assertEqual(payload['chunks'][0]['content'], 'Twenty days.')

    def test_one_rewrite_then_generation_uses_original_question(self):
        self.retriever.search.side_effect = [[], [self.hit]]
        self.ask()
        self.assertEqual(self.retriever.search.call_count, 2)
        self.assertEqual(self.retriever.search.call_args.args[1], 'annual leave allowance')
        self.assertEqual(self.generator.generate.call_args.args[0], 'How much leave?')
        self.rewriter.invoke.assert_called_once()
        payload = json.loads(self.rewriter.invoke.call_args.args[0][1][1])
        self.assertIn('No relevant evidence', payload['missing'])
        self.repo.record_event.assert_called_once()

    def test_grader_missing_information_guides_rewrite(self):
        from dataclasses import replace
        self.retriever.search.side_effect = [[replace(self.hit, score=5)], [self.hit]]
        self.grader.invoke.return_value = dict(sufficient=False, missing='Annual entitlement is missing')
        self.ask()
        payload = json.loads(self.rewriter.invoke.call_args.args[0][1][1])
        self.assertEqual(payload['missing'], 'Annual entitlement is missing')

    def test_exhaustion_refuses_and_audits(self):
        self.retriever.search.return_value = []
        answer = self.ask()
        self.assertFalse(answer.answered)
        self.assertEqual(self.retriever.search.call_count, 2)
        self.rewriter.invoke.assert_called_once()
        self.generator.generate.assert_not_called()
        self.assertEqual(self.repo.record_event.call_args.kwargs['sources'], 'none')

    def test_empty_library_skips_retrieval_but_records_refusal(self):
        self.repo.searchable_chunks.return_value = dict(searchable=0, total=3)
        self.assertFalse(self.ask().answered)
        self.retriever.search.assert_not_called()
        self.rewriter.invoke.assert_not_called()
        self.repo.record_event.assert_called_once()

    def test_config_can_disable_rewrite(self):
        self.pipeline.settings = Settings(max_attempts=1)
        self.retriever.search.return_value = []
        self.assertFalse(self.ask().answered)
        self.retriever.search.assert_called_once()
        self.rewriter.invoke.assert_not_called()

    def test_ungrounded_generation_is_audited_as_ungrounded(self):
        self.generator.generate.return_value = Answer('Untrusted.', True, grounding_errors=['Missing quote'])
        self.assertFalse(self.ask().fully_grounded)
        self.assertFalse(self.repo.record_event.call_args.kwargs['grounded'])

    def test_unknown_user_and_blank_question_do_not_call_models(self):
        self.repo.get_user.side_effect = ValueError('Unknown user')
        with self.assertRaises(ValueError):
            self.ask()
        self.intent.invoke.assert_not_called()
        self.repo.get_user.reset_mock()
        with self.assertRaises(ValueError):
            self.pipeline.ask(self.connection, ' ', 'bob')
        self.repo.get_user.assert_not_called()

    def test_invalid_rewrite_and_provider_errors_propagate(self):
        self.retriever.search.return_value = []
        self.rewriter.invoke.return_value = dict(query=' ')
        with self.assertRaises(ValidationError):
            self.ask()
        self.repo.record_event.assert_not_called()
        self.intent.invoke.side_effect = RuntimeError('Unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Unavailable'):
            self.ask()

    def test_audit_failure_prevents_success_response(self):
        self.repo.record_event.side_effect = RuntimeError('Audit unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Audit unavailable'):
            self.ask()
