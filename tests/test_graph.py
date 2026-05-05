
from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import QueryAnalysis, build_graph
from app.knowledge import KnowledgeBase


class StubChatModel(FakeListChatModel):

    analyses: list[QueryAnalysis] = []

    def with_structured_output(self, schema: Any, **_: Any):  # type: ignore[override]
        analyses = list(self.analyses)

        class _Runner:
            def invoke(self, _messages):
                if not analyses:
                    return QueryAnalysis(requires_country=False, detected_country=None)
                return analyses.pop(0)

        return _Runner()


def _kb() -> KnowledgeBase:
    return KnowledgeBase(data_dir="data")


def test_clarification_when_country_required_but_missing():
    llm = StubChatModel(responses=["should not be used"])
    llm.analyses = [QueryAnalysis(requires_country=True, detected_country=None)]
    graph = build_graph(llm, _kb())

    result = graph.invoke(
        {"messages": [HumanMessage(content="Какая ставка стипендии?")]},
        config={"configurable": {"thread_id": "t1"}},
    )

    assert result["needs_clarification"] is True
    assert "Германия" in result["messages"][-1].content
    assert "Франция" in result["messages"][-1].content
    assert result.get("sources") == []


def test_answer_with_country_filters_other_country_files():
    llm = StubChatModel(responses=["Стипендия в Германии — 1200 евро в месяц."])
    llm.analyses = [QueryAnalysis(requires_country=True, detected_country="germany")]
    graph = build_graph(llm, _kb())

    result = graph.invoke(
        {"messages": [HumanMessage(content="Какая стипендия в Германии?")]},
        config={"configurable": {"thread_id": "t2"}},
    )

    assert result["needs_clarification"] is False
    assert "france_rules.txt" not in result["sources"]
    assert "germany_rules.txt" in result["sources"]
    assert isinstance(result["messages"][-1], AIMessage)


def test_general_question_does_not_require_country():
    llm = StubChatModel(responses=["Дедлайн — 25 апреля."])
    llm.analyses = [QueryAnalysis(requires_country=False, detected_country=None)]
    graph = build_graph(llm, _kb())

    result = graph.invoke(
        {"messages": [HumanMessage(content="Когда дедлайн подачи документов?")]},
        config={"configurable": {"thread_id": "t3"}},
    )

    assert result["needs_clarification"] is False
    assert "deadlines.txt" in result["sources"]


def test_country_persists_across_turns():
    llm = StubChatModel(
        responses=[
            "Стипендия — 1300 евро.",  # ответ на первый вопрос
            "Налог во Франции — 20%.",  # ответ на follow-up без указания страны
        ]
    )
    llm.analyses = [
        QueryAnalysis(requires_country=True, detected_country="france"),
        QueryAnalysis(requires_country=True, detected_country="france"),
    ]
    graph = build_graph(llm, _kb())
    cfg = {"configurable": {"thread_id": "t4"}}

    graph.invoke(
        {"messages": [HumanMessage(content="Какая стипендия во Франции?")]}, config=cfg
    )
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="А какой там налог?")]}, config=cfg
    )

    assert result2["needs_clarification"] is False
    assert "france_rules.txt" in result2["sources"]
    assert "germany_rules.txt" not in result2["sources"]
