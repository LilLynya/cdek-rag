from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .knowledge import KnowledgeBase

Country = Literal["germany", "france"]

CLARIFICATION_QUESTION = (
    "Уточните, пожалуйста, по какой локации программы вас интересует информация: "
    "Германия (Берлин) или Франция (Париж)?"
)


class AgentState(TypedDict, total=False):
    """Состояние графа. Сохраняется через MemorySaver по `thread_id` (= session_id)."""

    messages: Annotated[list[BaseMessage], add_messages]
    country: Optional[Country]
    needs_clarification: bool
    sources: list[str]
    retrieved_context: str


class QueryAnalysis(BaseModel):
    """Структурированный результат анализа запроса."""

    requires_country: bool = Field(
        description=(
            "True, если для ответа нужна информация про конкретную страну "
            "(стипендия, налоги, рабочее время, виза, валюта). "
            "False для общих вопросов (подача заявки, дедлайны, бенефиты, отбор)."
        )
    )
    detected_country: Optional[Country] = Field(
        default=None,
        description=(
            "Страна, упомянутая в текущем сообщении или ранее в диалоге "
            "(germany | france | null)."
        ),
    )


ANALYZE_SYSTEM_PROMPT = """\
Ты — анализатор запросов для бота программы международной стажировки CdekStart.
Программа доступна в двух локациях: Германия (Берлин) и Франция (Париж).

Твои задачи:
1. По всему диалогу определи, нужна ли информация о конкретной стране, чтобы ответить на ПОСЛЕДНЕЕ сообщение пользователя.
2. Если страна явно или косвенно упомянута (в текущем сообщении или ранее), верни её код: "germany" или "france".
   - Маркеры Германии: Германия, Берлин, Berlin, Germany, немецк*, Deutschland.
   - Маркеры Франции: Франция, Париж, Paris, France, французск*.
3. Если страна не упоминалась — верни detected_country = null.

Тематика, ЗАВИСЯЩАЯ от страны: стипендия, налоги, рабочий день/часы, виза, валюта, конкретный город, локация.
Тематика, НЕ зависящая от страны: общая информация о программе, как подать заявку, дедлайны, бенефиты, отбор, язык программы.
"""

ANSWER_SYSTEM_PROMPT = """\
Ты — консультант программы международной стажировки CdekStart.
Отвечай по-русски, кратко и по делу.

СТРОГИЕ ПРАВИЛА:
1. Используй ТОЛЬКО факты из блока «КОНТЕКСТ» ниже. Не выдумывай, не дополняй знаниями из интернета.
2. Если в контексте нет ответа — честно скажи: «В моей базе знаний нет информации по этому вопросу».
3. Не упоминай имена файлов и не цитируй служебные пометки источников в ответе.
4. Если контекст содержит данные сразу по нескольким странам, отвечай только по той, которую обсуждает пользователь.
5. Не задавай встречных вопросов — это уже сделано на этапе уточнения.

КОНТЕКСТ:
{context}
"""


def _make_analyze_node(llm: BaseChatModel):
    structured = llm.with_structured_output(QueryAnalysis)

    def analyze(state: AgentState) -> dict:
        messages: list[BaseMessage] = [SystemMessage(content=ANALYZE_SYSTEM_PROMPT)]
        prev_country = state.get("country")
        if prev_country:
            messages.append(
                SystemMessage(
                    content=(
                        f"Контекст диалога: ранее обсуждалась страна — {prev_country}. "
                        "Если пользователь не сменил тему страны, используй это значение."
                    )
                )
            )
        messages.extend(state.get("messages", []))

        try:
            analysis: QueryAnalysis = structured.invoke(messages)
        except Exception:
            # На случай если LLM не справился со structured output —
            # лучше честно отдать запрос в retrieve без фильтра, чем падать.
            analysis = QueryAnalysis(requires_country=False, detected_country=prev_country)

        country = analysis.detected_country or prev_country
        needs_clarification = analysis.requires_country and country is None

        return {
            "country": country,
            "needs_clarification": needs_clarification,
        }

    return analyze


def _clarify_node(state: AgentState) -> dict:
    return {
        "messages": [AIMessage(content=CLARIFICATION_QUESTION)],
        "sources": [],
        "retrieved_context": "",
    }


def _make_retrieve_node(kb: KnowledgeBase):
    def retrieve(state: AgentState) -> dict:
        last_user_message = next(
            (m.content for m in reversed(state.get("messages", [])) if m.type == "human"),
            "",
        )
        docs = kb.search(last_user_message, country=state.get("country"))
        context = "\n\n".join(
            f"[Источник: {d.metadata['source']}]\n{d.page_content}" for d in docs
        )
        return {
            "sources": [d.metadata["source"] for d in docs],
            "retrieved_context": context,
        }

    return retrieve


def _make_answer_node(llm: BaseChatModel):
    def answer(state: AgentState) -> dict:
        system = ANSWER_SYSTEM_PROMPT.format(context=state.get("retrieved_context", ""))
        messages: list[BaseMessage] = [SystemMessage(content=system)]
        messages.extend(state.get("messages", []))
        response = llm.invoke(messages)
        return {"messages": [response]}

    return answer


def _route_after_analyze(state: AgentState) -> str:
    return "clarify" if state.get("needs_clarification") else "retrieve"


def build_graph(llm: BaseChatModel, kb: KnowledgeBase, *, checkpointer=None):
    """Собирает и компилирует граф. По умолчанию использует in-memory checkpointer."""
    builder = StateGraph(AgentState)
    builder.add_node("analyze", _make_analyze_node(llm))
    builder.add_node("clarify", _clarify_node)
    builder.add_node("retrieve", _make_retrieve_node(kb))
    builder.add_node("answer", _make_answer_node(llm))

    builder.add_edge(START, "analyze")
    builder.add_conditional_edges(
        "analyze",
        _route_after_analyze,
        {"clarify": "clarify", "retrieve": "retrieve"},
    )
    builder.add_edge("clarify", END)
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
