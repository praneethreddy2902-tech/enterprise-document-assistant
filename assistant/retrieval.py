"""Query embedding and reranking of permission-filtered repository candidates."""

from dataclasses import replace
from datetime import date
import json
from typing import Protocol

from psycopg import Connection
from pydantic import BaseModel, ConfigDict, Field

from assistant import repository
from assistant.config import Settings, load_settings, prompt
from assistant.models import Hit, User


class QueryEmbeddings(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class Rating(BaseModel):
    model_config = ConfigDict(extra='forbid')
    index: int = Field(strict=True, ge=1, description='One-based candidate number')
    score: int = Field(strict=True, ge=0, le=10, description='Relevance from 0 to 10')


class RerankResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ratings: list[Rating]


class Reranker(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> RerankResult: ...


class RerankError(ValueError):
    """The model returned ratings that cannot be mapped safely to candidates."""

#convert user questions into embedding vector 
class Retriever:
    def __init__(self, settings: Settings | None = None, *,
                 embeddings: QueryEmbeddings | None = None, reranker: Reranker | None = None):
        self.settings = settings or load_settings()
        self._embeddings = embeddings
        self._reranker = reranker

    def _embedding_provider(self) -> QueryEmbeddings:
        if self._embeddings is None:
            if self.settings.demo_mode:
                from assistant.demo import DemoEmbeddings
                self._embeddings = DemoEmbeddings()
                return self._embeddings
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                api_key=self.settings.require_openai_key(), model=self.settings.embedding_model,
                dimensions=self.settings.embedding_dimensions,
            )
        return self._embeddings

    def _rerank_provider(self) -> Reranker:
        if self._reranker is None:
            if self.settings.demo_mode:
                from assistant.demo import DemoModel
                self._reranker = DemoModel('rerank')
                return self._reranker
            from langchain_openai import ChatOpenAI

            self._reranker = ChatOpenAI(
                api_key=self.settings.require_openai_key(), model=self.settings.helper_model,
                temperature=0,
            ).with_structured_output(RerankResult)
        return self._reranker

    def search(self, connection: Connection, question: str, user: User,
               include_history: bool = False, *, today: date | None = None) -> list[Hit]:
        """The user must come from repository.get_user, never request metadata.

        Provider failures propagate to the caller. Unreranked candidates are never
        silently used as relevant evidence when the reranker fails.
        """
        if not question.strip():
            raise ValueError('Question must not be empty')
        vector = self._embedding_provider().embed_query(question)
        candidates = repository.hybrid_search(
            connection, question, vector, user, include_history,
            candidates=self.settings.candidates, rrf_constant=self.settings.rrf_constant,
            today=today,
        )
        if not candidates:
            return []
        return self._rerank(question, candidates)

    def _rerank(self, question: str, candidates: list[Hit]) -> list[Hit]:
        # Keep policy text and question in a data message, separate from instructions.
        payload = json.dumps({
            'question': question,
            'candidates': [dict(index=i, label=hit.label, status=hit.status.value, content=hit.content)
                           for i, hit in enumerate(candidates, start=1)],
        }, ensure_ascii=False)
        result = RerankResult.model_validate(self._rerank_provider().invoke([
            ('system', prompt('rerank')), ('human', payload),
        ]))
        seen = set()
        scored = []
        for rating in result.ratings:
            if rating.index > len(candidates) or rating.index in seen:
                raise RerankError('Reranker returned an unknown or duplicate candidate number')
            seen.add(rating.index)
            if rating.score >= self.settings.min_rerank_score:
                # Hit is immutable. Preserve repository results and all citation metadata.
                scored.append((rating.index, replace(candidates[rating.index - 1], score=float(rating.score))))
        # Missing ratings are discarded. Ties preserve the original fused ranking.
        scored.sort(key=lambda item: (-item[1].score, item[0]))
        return [hit for _, hit in scored[:self.settings.top_k]]
