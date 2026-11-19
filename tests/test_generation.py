from dataclasses import replace
from datetime import date, timedelta
import json
import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from assistant.config import Settings
from assistant.generation import Generator, REFUSAL
from assistant.models import Hit, PolicyStatus


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.today = date(2026, 11, 17)
        self.hit = Hit(10, 'Employees receive twenty days of annual leave.', 'Annual leave',
                       'leave.md', 'Leave policy', 1, PolicyStatus.ACTIVE, self.today, 9)
        self.model = Mock()
        self.generator = Generator(Settings(), model=self.model)

    def draft(self, quote=None, answer='Employees receive twenty days. [1]', number=1):
        return dict(answer=answer, answered=True, citations=[dict(
            chunk_number=number, quote=quote or self.hit.content)])

    def generate(self, hits=None, **kwargs):
        return self.generator.generate('How much annual leave?', [self.hit] if hits is None else hits,
                                       today=self.today, **kwargs)

    def test_grounded_answer_has_trusted_metadata_and_no_warning(self):
        self.model.invoke.return_value = self.draft()
        answer = self.generate()
        self.assertTrue(answer.fully_grounded)
        self.assertIn('[Leave policy v1 > Annual leave]', answer.text)
        self.assertEqual(answer.citations[0].source, 'leave.md')
        self.assertEqual(answer.citations[0].chunk_id, 10)
        self.assertEqual(answer.warnings, [])
        self.assertEqual(self.model.invoke.call_count, 1)

    def test_quote_normalization_preserves_punctuation(self):
        self.model.invoke.return_value = self.draft(quote='EMPLOYEES receive\n twenty days of annual leave.')
        self.assertTrue(self.generate().fully_grounded)
        self.model.invoke.return_value = self.draft(quote='Employees receive twenty days of annual leave!')
        answer = self.generate()
        self.assertFalse(answer.fully_grounded)
        self.assertFalse(answer.citations[0].verified)

    def test_one_repair_attempt_can_correct_bad_quote(self):
        self.model.invoke.side_effect = [self.draft(quote='Employees receive fifty days.'), self.draft()]
        answer = self.generate()
        self.assertTrue(answer.fully_grounded)
        self.assertEqual(self.model.invoke.call_count, 2)
        self.assertEqual(len(answer.trace), 2)
        self.assertIn('previous draft failed', self.model.invoke.call_args.args[0][-1][1])

    def test_failed_repair_is_flagged_and_bounded(self):
        self.model.invoke.return_value = self.draft(quote='Invented quotation.')
        answer = self.generate()
        self.assertFalse(answer.fully_grounded)
        self.assertEqual(self.model.invoke.call_count, 2)
        self.assertTrue(answer.grounding_errors)
        self.assertIn('not trusted', answer.warnings[0])

    def test_unknown_citation_cannot_hide_behind_valid_citation(self):
        draft = self.draft()
        draft['citations'].append(dict(chunk_number=2, quote='Invented quote.'))
        self.model.invoke.return_value = draft
        answer = self.generate()
        self.assertTrue(answer.citations[0].verified)
        self.assertFalse(answer.fully_grounded)

    def test_missing_citations_or_inline_references_are_ungrounded(self):
        cases = [self.draft(answer='Uncited answer.'), self.draft(answer='Answer [2]'),
                 dict(answer='Answer [1]', answered=True, citations=[]), self.draft(answer='')]
        for draft in cases:
            with self.subTest(draft=draft):
                self.model.invoke.return_value = draft
                self.assertFalse(self.generate().fully_grounded)

    def test_no_hits_and_refusals_do_not_expose_model_text(self):
        answer = self.generate(hits=[])
        self.assertFalse(answer.answered)
        self.model.invoke.assert_not_called()
        self.model.invoke.return_value = dict(answer='Untrusted unsupported text', answered=False, citations=[])
        answer = self.generate()
        self.assertEqual(answer.text, REFUSAL)
        self.assertEqual(answer.citations, [])
        self.assertFalse(answer.fully_grounded)

    def test_freshness_and_superseded_warnings_use_cited_sources(self):
        old = replace(self.hit, status=PolicyStatus.SUPERSEDED,
                      updated_on=self.today - timedelta(days=181))
        fresh = replace(self.hit, chunk_id=11, title='Fresh policy')
        draft = self.draft(answer='Old and fresh evidence. [1] [2]')
        draft['citations'].append(dict(chunk_number=2, quote=fresh.content))
        self.model.invoke.return_value = draft
        answer = self.generate(hits=[old, fresh], include_history=True)
        self.assertTrue(answer.fully_grounded)
        self.assertEqual(len(answer.warnings), 2)
        self.assertTrue(any('181 days' in warning for warning in answer.warnings))
        self.assertTrue(any('superseded' in warning for warning in answer.warnings))
        self.model.invoke.return_value = self.draft()
        self.assertEqual(self.generate(hits=[fresh, old]).warnings, [])

    def test_staleness_boundary_and_duplicate_policy_warning(self):
        self.model.invoke.return_value = self.draft()
        boundary = replace(self.hit, updated_on=self.today - timedelta(days=180))
        self.assertEqual(self.generate(hits=[boundary]).warnings, [])
        old = replace(boundary, updated_on=self.today - timedelta(days=181))
        draft = self.draft(answer='Evidence [1] [2]')
        draft['citations'].append(dict(chunk_number=2, quote=old.content))
        self.model.invoke.return_value = draft
        self.assertEqual(len(self.generate(hits=[old, replace(old, chunk_id=11)]).warnings), 1)

    def test_prompt_separates_data_and_includes_history(self):
        self.model.invoke.return_value = self.draft()
        self.generate(include_history=True)
        messages = self.model.invoke.call_args.args[0]
        self.assertEqual(messages[0][0], 'system')
        payload = json.loads(messages[1][1])
        self.assertTrue(payload['include_history'])
        self.assertEqual(payload['chunks'][0]['content'], self.hit.content)

    def test_invalid_response_and_provider_failures_propagate(self):
        self.model.invoke.return_value = dict(answer='Answer', answered=True,
                                             citations=[dict(chunk_number=True, quote=' ')])
        with self.assertRaises(ValidationError):
            self.generate()
        self.model.invoke.side_effect = RuntimeError('Provider unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Provider unavailable'):
            self.generate()

    def test_empty_question_never_calls_model(self):
        with self.assertRaises(ValueError):
            self.generator.generate(' ', [self.hit])
        self.model.invoke.assert_not_called()
