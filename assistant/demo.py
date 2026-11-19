"""Deterministic demonstration providers. No network, secrets, or language model.

These intentionally simple word-matching rules exercise the application's flow;
they are not semantic embeddings, production ranking, or policy interpretation.
"""

from hashlib import sha256
import json
import math
import re

_STOP = set('a an and are as at be been by can company could do does employees employee '
            'for from get have how i in is it many much my of on or our policy policies '
            'the their this to under was were what when which who with would current '
            'previous older old version please tell me about per days day'.split())
_ALIASES = {'refunds': 'refund', 'reimbursements': 'reimbursement', 'reimbursed': 'reimbursement',
            'reimbursable': 'reimbursement', 'meals': 'meal', 'developers': 'engineers',
            'engineer': 'engineers', 'hotels': 'hotel', 'requests': 'request', 'remotely': 'remote',
            'salaries': 'salary', 'leaves': 'leave'}


def _terms(text: str) -> set[str]:
    return {_ALIASES.get(word, word) for word in re.findall(r'[a-z]+', text.lower()) if word not in _STOP}


def _history(question: str) -> bool:
    return bool(re.search(r'\b(previous|older|old|historical|history|superseded|used to|was|were|compare)\b|\b(?:version\s*|v)\d+', question, re.I))


def _score(question: str, content: str) -> int:
    query = _terms(question)
    overlap = len(query & _terms(content))
    # Require most meaningful query terms, so one incidental word such as
    # "office" cannot answer a question about an absent subject such as pets.
    if not overlap or overlap / max(1, len(query)) < 0.6:
        return 0
    return min(10, round(10 * overlap / max(1, len(query))))


def _eligible(question: str, label: str, status: str) -> bool:
    version = re.search(r'\b(?:version\s*|v)(\d+)\b', question, re.I)
    if version:
        return bool(re.search(rf'\bv{version[1]}\b', label))
    if re.search(r'\b(previous|older|old|superseded|used to)\b', question, re.I):
        return status == 'superseded'
    return True


class DemoEmbeddings:
    """Stable hashed word features with the existing 1536-dimensional shape."""

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 1536
        for term in sorted(_terms(text)) or ['empty']:
            index = int.from_bytes(sha256(term.encode()).digest()[:8], 'big') % 1536
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class DemoModel:
    def __init__(self, role: str):
        self.role = role

    def invoke(self, messages: list[tuple[str, str]]):
        # The service payload is the original human message, even after repair.
        payload = json.loads(next(text for role, text in messages if role == 'human'))
        question = payload['question']
        if self.role == 'intent':
            from assistant.pipeline import Intent
            return Intent(wants_history=_history(question))
        if self.role == 'rewrite':
            from assistant.pipeline import Rewrite
            query = ' '.join(sorted(_terms(question))) or question
            if _history(question):
                query = question + ' ' + query
            return Rewrite(query=query)
        if self.role == 'grade':
            from assistant.pipeline import Grade
            enough = any(_score(question, chunk['content']) >= 4 for chunk in payload['chunks'])
            return Grade(sufficient=enough, missing='' if enough else 'No matching excerpt in the demo library')
        if self.role == 'rerank':
            from assistant.retrieval import RerankResult
            return RerankResult(ratings=[dict(index=chunk['index'], score=
                _score(question, chunk['content'])
                if _eligible(question, chunk['label'], chunk['status']) else 0)
                for chunk in payload['candidates']])
        if self.role == 'answer':
            from assistant.generation import Draft
            choices = []
            for chunk in payload['chunks']:
                if not _eligible(question, chunk['label'], chunk['status']):
                    continue
                for sentence in re.split(r'(?<=[.!?])\s+|\n+', chunk['content']):
                    sentence = sentence.strip()
                    if sentence and not sentence.startswith('#'):
                        choices.append((_score(question, sentence), chunk['chunk_number'], sentence))
            if not choices:
                return Draft(answer='', answered=False, citations=[])
            score, number, quote = max(choices, key=lambda item: item[0])
            if score < 4:
                return Draft(answer='', answered=False, citations=[])
            return Draft(answer=f'Demo excerpt: {quote} [{number}]', answered=True,
                         citations=[dict(chunk_number=number, quote=quote)])
        raise ValueError('Unknown demo provider role')
