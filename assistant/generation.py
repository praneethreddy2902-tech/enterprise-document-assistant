"""Generate a cited draft and verify quotations without a second judge model."""

from datetime import date
import json
import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from assistant.config import Settings, load_settings, prompt
from assistant.models import Answer, Citation, Hit, PolicyStatus

REFUSAL = 'I could not find enough evidence in the available documents to answer this question.'

# we want llm to tell us; which chunk supports the answer & what exact sentence supports it 
class DraftCitation(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    chunk_number: int = Field(strict=True, ge=1)
    quote: str = Field(min_length=1, description='Supporting sentence copied exactly from this chunk')


class Draft(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    answer: str
    answered: bool = Field(strict=True)
    citations: list[DraftCitation]


class AnswerProvider(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> Draft: ...


def _flatten(text: str) -> str:
    return ' '.join(text.split()).casefold()


class Generator:
    def __init__(self, settings: Settings | None = None, *, model: AnswerProvider | None = None):
        self.settings = settings or load_settings()
        self._model = model

    def _provider(self) -> AnswerProvider:
        if self._model is None:
            if self.settings.demo_mode:
                from assistant.demo import DemoModel
                self._model = DemoModel('answer')
                return self._model
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(
                api_key=self.settings.require_openai_key(), model=self.settings.answer_model,
                temperature=0,
            ).with_structured_output(Draft)
        return self._model

    def generate(self, question: str, hits: list[Hit], *, today: date | None = None,
                 include_history: bool = False) -> Answer:
        """Accept only hits already permission-filtered by retrieval.

        One citation-repair attempt is separate from the pipeline's query rewrite.
        After a second grounding failure, return the draft with explicit warnings
        and fully_grounded=False, following the reference project's behavior.
        """
        if not question.strip():
            raise ValueError('Question must not be empty')
        if not hits:
            return Answer(REFUSAL, False, trace=['generation skipped: no relevant chunks'])
        today = today or date.today()
        messages = [('system', prompt('answer')), ('human', json.dumps({
            'question': question, 'include_history': include_history,
            'chunks': [dict(chunk_number=i, label=h.label, status=h.status.value,
                            updated_on=h.updated_on.isoformat(), content=h.content)
                       for i, h in enumerate(hits, start=1)],
        }, ensure_ascii=False))]
        trace = []
        for attempt in range(1, 3):
            draft = Draft.model_validate(self._provider().invoke(messages))
            if not draft.answered:
                return Answer(REFUSAL, False, trace=trace + [f'generation attempt {attempt}: insufficient evidence'])
            answer = self._verify(draft, hits)
            trace.append(f'generation attempt {attempt}: {len(answer.citations)} citation(s), '
                         f'grounded={answer.fully_grounded}')
            if answer.fully_grounded:
                break
            if attempt == 1:
                messages = messages + [('system',
                    'The previous draft failed citation checks. Every claim needs an inline [N] '
                    'reference and a corresponding citation. Copy each supporting sentence '
                    'verbatim from its numbered chunk, including its first and last words. '
                    'Use only the provided chunk numbers. If evidence is insufficient, set '
                    'answered to false. Validation errors: ' + '; '.join(answer.grounding_errors))]
        answer.trace = trace
        if not answer.fully_grounded:
            answer.warnings.append('Citation checks failed; this answer is not trusted.')
        cited_ids = {citation.chunk_id for citation in answer.citations}
        cited = [hit for hit in hits if hit.chunk_id in cited_ids]
        if any(hit.status == PolicyStatus.SUPERSEDED for hit in cited):
            answer.warnings.append('This quotes a superseded version. It is history, not current policy.')
        # Warn for each stale cited policy, even when another cited policy is fresh.
        seen = set()
        for hit in cited:
            age = (today - hit.updated_on).days
            identity = (hit.title, hit.version)
            if age > self.settings.stale_after_days and identity not in seen:
                seen.add(identity)
                answer.warnings.append(f'{hit.title} v{hit.version} was last updated {age} days ago; '
                                       'check whether it is still current.')
        return answer

    @staticmethod
    def _verify(draft: Draft, hits: list[Hit]) -> Answer:
        errors = []
        citations = []
        numbers = set()
        if not draft.answer.strip():
            errors.append('Answer text is empty')
        if not draft.citations:
            errors.append('No supporting citations were provided')
        for item in draft.citations:
            if item.chunk_number > len(hits):
                errors.append('A citation references an unknown chunk number')
                continue
            hit = hits[item.chunk_number - 1]
            verified = _flatten(item.quote) in _flatten(hit.content)
            numbers.add(item.chunk_number)
            citations.append(Citation(hit.chunk_id, hit.label, hit.source, item.quote, verified))
            if not verified:
                errors.append(f'Quote for chunk {item.chunk_number} was not found in its source')
        inline = {int(number) for number in re.findall(r'\[(\d+)\]', draft.answer)}
        if inline != numbers or not inline:
            errors.append('Inline [N] references must match the supporting citation numbers')
        # Labels are supplied by our code; the model cannot invent a source name.
        text = re.sub(r'\[(\d+)\]',
                      lambda match: f'[{hits[int(match[1]) - 1].label}]'
                      if 1 <= int(match[1]) <= len(hits) else match[0], draft.answer)
        return Answer(text, True, citations=citations, grounding_errors=errors)
