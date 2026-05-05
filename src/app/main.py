from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage

from .config import settings
from .graph import build_graph
from .knowledge import KnowledgeBase
from .llm import get_llm
from .schemas import ChatRequest, ChatResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cdekstart.api")


_runtime: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "Инициализация: provider=%s model=%s data_dir=%s",
        settings.llm_provider,
        settings.llm_model,
        settings.data_dir,
    )
    kb = KnowledgeBase()
    llm = get_llm()
    graph = build_graph(llm, kb)
    _runtime.update(kb=kb, llm=llm, graph=graph)
    logger.info("Загружено документов: %d", len(kb.documents))
    try:
        yield
    finally:
        _runtime.clear()


app = FastAPI(
    title="CdekStart RAG Bot",
    version="0.1.0",
    description=(
        "Контекстный RAG-агент на LangGraph для консультаций по программе "
        "международной стажировки CdekStart. Помнит диалог и задаёт уточняющие "
        "вопросы, если для ответа не хватает информации о стране."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest) -> ChatResponse:
    graph = _runtime.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Сервис ещё инициализируется")

    session_id = req.session_id or str(uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
    except Exception as exc:
        logger.exception("Ошибка обработки запроса")
        raise HTTPException(status_code=500, detail=f"LLM/graph error: {exc}") from exc

    last_msg = result["messages"][-1]
    return ChatResponse(
        session_id=session_id,
        reply=getattr(last_msg, "content", str(last_msg)),
        needs_clarification=bool(result.get("needs_clarification", False)),
        sources=list(result.get("sources") or []),
    )
