"""Coordinate permission-aware retrieval, one rewrite, generation, and audit."""

from datetime import date
import json
from typing import Protocol

from psycopg import Connection
from pydantic import BaseModel, ConfigDict, Field

from assistant import repository
from assistant.config import Settings, load_settings, prompt
from assistant.generation import Generator, REFUSAL
from assistant.models import Answer, Hit, User
from assistant.retrieval import Retriever


class Intent(BaseModel):
    model_config = ConfigDict(extra='forbid')
    wants_history: bool = Field(strict=True)


class Grade(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sufficient: bool = Field(strict=True)
    missing: str


class Rewrite(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    query: str = Field(min_length=1)


class Helper(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> BaseModel: ...


class Pipeline:
    def __init__(self, settings: Settings | None = None, *, retriever: Retriever | None = None,
                 generator: Generator | None = None, intent: Helper | None = None,
                 grader: Helper | None = None, rewriter: Helper | None = None):
        self.settings = settings or load_settings()
        self.retriever = retriever if retriever is not None else Retriever(self.settings)
        self.generator = generator if generator is not None else Generator(self.settings)
        self._helpers = {'intent': intent, 'grade': grader, 'rewrite': rewriter}

    def _helper(self, name: str) -> Helper:
        if self._helpers[name] is None:
            if self.settings.demo_mode:
                from assistant.demo import DemoModel
                self._helpers[name] = DemoModel(name)
                return self._helpers[name]
            from langchain_openai import ChatOpenAI

            schema = {'intent': Intent, 'grade': Grade, 'rewrite': Rewrite}[name]
            self._helpers[name] = ChatOpenAI(
                api_key=self.settings.require_openai_key(), model=self.settings.helper_model,
                temperature=0,
            ).with_structured_output(schema)
        return self._helpers[name]

    def _invoke(self, name: str, payload: dict):
        return self._helper(name).invoke([
            ('system', prompt(name)), ('human', json.dumps(payload, ensure_ascii=False)),
        ])

    def ask(self, connection: Connection, question: str, username: str,
            *, today: date | None = None) -> Answer:
        """Caller commits the connection to persist the completed question's audit.

        Unknown users and provider/database failures raise; they are not converted
        into misleading insufficient-evidence refusals. No audit is committed here.
        """
        if not question.strip():
            raise ValueError('Question must not be empty')
        user = repository.get_user(connection, username)
        today = today or date.today()
        history = Intent.model_validate(self._invoke('intent', {'question': question})).wants_history
        counts = repository.searchable_chunks(connection, user, history, today=today)
        trace = [f'user {user.username} ({user.department}): {user.describe_access()}',
                 f'searchable for this user: {counts["searchable"]} of {counts["total"]} chunks',
                 f'history requested: {history}']
        answer = self._retrieve_and_answer(connection, question, user, history, today, trace)
        if self.settings.demo_mode:
            answer.warnings.insert(0, 'Demo mode: fake embeddings and scripted excerpt selection; '
                                   'this is not an AI-generated policy answer.')
            answer.trace.insert(0, 'demo mode: deterministic local providers; no OpenAI calls')
        # Do not return a completed answer if its audit insert fails.
        repository.record_event(
            connection, username=user.username, question=question, answered=answer.answered,
            grounded=answer.fully_grounded,
            sources=', '.join(dict.fromkeys(c.label for c in answer.citations)) or 'none',
        )
        answer.trace.append(f'audit recorded: answered={answer.answered}, grounded={answer.fully_grounded}')
        return answer

    def _retrieve_and_answer(self, connection: Connection, question: str, user: User,
                             history: bool, today: date, trace: list[str]) -> Answer:
        query = question
        # Empty libraries need neither embedding nor rewrite calls.
        if trace[1].startswith('searchable for this user: 0 of '):
            return Answer(REFUSAL, False, trace=trace + ['retrieval skipped: no searchable chunks'])
        for attempt in range(1, self.settings.max_attempts + 1):
            hits = self.retriever.search(connection, query, user, history, today=today)
            trace.append(f'attempt {attempt}: searched {query!r}, kept {len(hits)} chunks')
            if hits and max(hit.score for hit in hits) >= self.settings.high_confidence_score:
                trace.append(f'attempt {attempt}: high-confidence evidence; grading skipped')
                return self._generate(question, hits, history, today, trace)
            if hits:
                grade = Grade.model_validate(self._invoke('grade', {
                    'question': question, 'include_history': history,
                    'chunks': [dict(label=h.label, status=h.status.value, content=h.content) for h in hits],
                }))
            else:
                grade = Grade(sufficient=False, missing='No relevant evidence was found')
            trace.append(f'attempt {attempt}: evidence sufficient={grade.sufficient}')
            if grade.sufficient:
                return self._generate(question, hits, history, today, trace)
            if attempt < self.settings.max_attempts:
                query = Rewrite.model_validate(self._invoke('rewrite', {
                    'question': question, 'missing': grade.missing,
                    'include_history': history,
                })).query
                trace.append('query rewritten for one retry')
        return Answer(REFUSAL, False, trace=trace + ['retrieval exhausted; no answer guessed'])

    def _generate(self, question: str, hits: list[Hit], history: bool,
                  today: date, trace: list[str]) -> Answer:
        answer = self.generator.generate(question, hits, today=today, include_history=history)
        answer.trace = trace + answer.trace
        return answer
