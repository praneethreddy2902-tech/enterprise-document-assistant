from datetime import date
import json
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from assistant.config import Settings
from assistant.models import Hit, PolicyStatus, User
from assistant.retrieval import RerankError, RerankResult, Retriever


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.user = User('bob', 'hr', True)
        self.connection = Mock()
        self.embeddings = Mock()
        self.embeddings.embed_query.return_value = [1.0] + [0.0] * 1535
        self.reranker = Mock()
        self.retriever = Retriever(Settings(), embeddings=self.embeddings, reranker=self.reranker)
        self.hits = [Hit(i, f'Annual leave evidence {i}', 'Leave', 'leave.md', 'Leave policy',
                         1, PolicyStatus.ACTIVE, date(2026, 6, 30), 0.01)
                     for i in range(1, 7)]

    def search(self, hits=None, **kwargs):
        with patch('assistant.retrieval.repository.hybrid_search',
                   return_value=self.hits if hits is None else hits) as search:
            result = self.retriever.search(self.connection, 'How much leave?', self.user, **kwargs)
            return result, search

    def test_search_forwards_permissions_history_and_settings(self):
        self.reranker.invoke.return_value = {'ratings': [{'index': 1, 'score': 9}]}
        today = date(2026, 11, 17)
        result, search = self.search(include_history=True, today=today)
        search.assert_called_once_with(self.connection, 'How much leave?',
                                       self.embeddings.embed_query.return_value, self.user, True,
                                       candidates=12, rrf_constant=60, today=today)
        self.assertEqual(result[0].score, 9)

    def test_reranks_cuts_and_limits_without_mutating_hits(self):
        self.reranker.invoke.return_value = RerankResult(ratings=[
            {'index': i, 'score': score} for i, score in enumerate([3, 4, 8, 10, 8, 7], start=1)])
        result, _ = self.search()
        self.assertEqual([h.chunk_id for h in result], [4, 3, 5, 6])
        self.assertTrue(all(h.score == 0.01 for h in self.hits))
        self.assertEqual(result[0].label, self.hits[3].label)
        self.assertEqual(result[0].updated_on, self.hits[3].updated_on)

    def test_threshold_is_inclusive_and_missing_ratings_are_dropped(self):
        self.reranker.invoke.return_value = {'ratings': [{'index': 2, 'score': 4}]}
        result, _ = self.search()
        self.assertEqual([h.chunk_id for h in result], [2])

    def test_empty_candidates_skip_reranker_and_key_requirement(self):
        result, _ = self.search(hits=[])
        self.assertEqual(result, [])
        self.reranker.invoke.assert_not_called()

    def test_all_low_or_missing_scores_return_no_evidence(self):
        for ratings in ([], [{'index': 1, 'score': 3}]):
            self.reranker.invoke.return_value = {'ratings': ratings}
            self.assertEqual(self.search()[0], [])

    def test_bad_indexes_and_malformed_scores_are_rejected(self):
        cases = [
            [{'index': 7, 'score': 9}], [{'index': 1, 'score': 8}, {'index': 1, 'score': 9}],
            [{'index': 0, 'score': 8}], [{'index': 1, 'score': 11}],
            [{'index': True, 'score': 8}], [{'index': 1, 'score': float('nan')}],
            [{'index': 1, 'score': '9'}],
        ]
        for ratings in cases:
            with self.subTest(ratings=ratings):
                self.reranker.invoke.return_value = {'ratings': ratings}
                with self.assertRaises((RerankError, ValidationError)):
                    self.search()

    def test_empty_question_does_no_work(self):
        with patch('assistant.retrieval.repository.hybrid_search') as search:
            with self.assertRaises(ValueError):
                self.retriever.search(self.connection, '  ', self.user)
            self.embeddings.embed_query.assert_not_called()
            search.assert_not_called()

    def test_provider_failures_propagate_without_fallback(self):
        self.embeddings.embed_query.side_effect = RuntimeError('Embedding failure')
        with patch('assistant.retrieval.repository.hybrid_search') as search:
            with self.assertRaisesRegex(RuntimeError, 'Embedding failure'):
                self.retriever.search(self.connection, 'Leave?', self.user)
            search.assert_not_called()
        self.embeddings.embed_query.side_effect = None
        self.reranker.invoke.side_effect = RuntimeError('Rerank failure')
        with self.assertRaisesRegex(RuntimeError, 'Rerank failure'):
            self.search()

    def test_prompt_keeps_question_and_candidates_as_data(self):
        self.reranker.invoke.return_value = {'ratings': []}
        self.search(hits=self.hits[:1])
        messages = self.reranker.invoke.call_args.args[0]
        self.assertEqual(messages[0][0], 'system')
        self.assertIn('untrusted data', messages[0][1])
        payload = json.loads(messages[1][1])
        self.assertEqual(payload['question'], 'How much leave?')
        self.assertEqual(len(payload['candidates']), 1)
        self.assertEqual(payload['candidates'][0]['content'], self.hits[0].content)
