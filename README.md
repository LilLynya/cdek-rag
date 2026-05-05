# CdekStart RAG Bot

Контекстный RAG-агент на **LangGraph + FastAPI**, консультирующий пользователей по программе международной стажировки **CdekStart**. Бот:

- помнит историю диалога (per-session checkpointing в LangGraph),
- задаёт уточняющий вопрос, если запрос неоднозначен (например, не указана страна — Германия или Франция),
- отвечает **строго по базе знаний**, не выдумывая фактов; если ответа нет — честно говорит об этом,
- поддерживает любую LLM (OpenAI, Anthropic, Ollama, любой OpenAI-совместимый эндпоинт).



---

## Запуск


### Через Docker Compose (основной способ)

```bash
docker-compose up --build
```

После запуска:

- Health-check: <http://localhost:8000/health>
- Swagger UI: <http://localhost:8000/docs>

### 3. Локально (для разработки)

```bash
python -m venv .venv && source .venv/bin/activate    
# .venv\Scripts\activate                              

pip install -r requirements.txt
PYTHONPATH=src uvicorn app.main:app --reload
```

---

## 🔌 API

### `POST /chat`

**Request:**

```json
{
  "message": "Какая ставка стипендии?",
  "session_id": "optional-uuid"
}
```

`session_id` опционален. Если не передан — бот сгенерирует новый и вернёт в ответе. **Передавайте этот id в последующих запросах**, чтобы сохранить контекст диалога.

**Response:**

```json
{
  "session_id": "5b1c7d96-...-...",
  "reply": "Уточните, пожалуйста, по какой локации программы вас интересует информация: Германия (Берлин) или Франция (Париж)?",
  "needs_clarification": true,
  "sources": []
}
```

| Поле | Описание |
|---|---|
| `session_id` | Идентификатор сессии для продолжения диалога |
| `reply` | Текст ответа бота |
| `needs_clarification` | `true`, если бот задаёт уточняющий вопрос |
| `sources` | Список имён файлов БЗ, использованных для ответа |

### Примеры (curl)

**1. Уточняющий вопрос про страну:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Какая ставка стипендии?"}'
```

```json
{
  "session_id": "ab12...",
  "reply": "Уточните, пожалуйста, по какой локации... Германия (Берлин) или Франция (Париж)?",
  "needs_clarification": true,
  "sources": []
}
```

**2. Ответ с учётом контекста (передаём тот же `session_id`):**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"ab12...","message":"Германия"}'
```

```json
{
  "session_id": "ab12...",
  "reply": "В Германии ставка стипендии составляет 1200 евро в месяц.",
  "needs_clarification": false,
  "sources": ["germany_rules.txt", "general_info.txt"]
}
```

**3. Follow-up с сохранением страны в контексте:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"ab12...","message":"А какой налог?"}'
```

→ бот понимает, что речь по-прежнему о Германии (15 %), не задавая вопрос повторно.

**4. Общий вопрос — страна не нужна:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Когда дедлайн подачи документов?"}'
```

→ ответ из `deadlines.txt`, без уточнений.

**5. Вопрос вне базы знаний — без галлюцинаций:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Сколько стажёров берут в год?"}'
```

→ «В моей базе знаний нет информации по этому вопросу.»

---

## Подключение LLM

Провайдер задаётся через переменные окружения. Пакеты `langchain-ollama` и `langchain-anthropic` — опциональные; ставятся при необходимости (`pip install '.[ollama]'`).

### OpenAI (по умолчанию)

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
```

### OpenAI-совместимые API (LM Studio, vLLM, Together, OpenRouter, Groq …)

```env
LLM_PROVIDER=openai
LLM_MODEL=qwen2.5-7b-instruct
LLM_API_KEY=lm-studio        # любая непустая строка для локальных серверов
LLM_BASE_URL=http://host.docker.internal:1234/v1
```

### Локальная Ollama

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
LLM_BASE_URL=http://host.docker.internal:11434
```

> Из контейнера обращайтесь к Ollama хоста через `host.docker.internal` — это уже прописано в `docker-compose.yml` через `extra_hosts`. Не забудьте установить `langchain-ollama` (раскомментируйте в `requirements.txt` или соберите образ с `pip install '.[ollama]'`).

### Anthropic

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-latest
LLM_API_KEY=sk-ant-...
```

---

## Особенности и ограничения

- **Структурированный вывод (`with_structured_output`)** на этапе `analyze` лучше всего работает с моделями, поддерживающими tool/function calling: GPT-4o, GPT-4.1, Claude 3.5+, Llama 3.1+, Qwen 2.5+. Совсем маленькие/старые модели могут не справиться — на этот случай предусмотрен fallback (запрос идёт в retrieval без страновой фильтрации).
- **BM25** не семантический; для коротких русскоязычных текстов работает достаточно, но при существенном расширении базы знаний рекомендуется заменить на векторный поиск (Chroma / FAISS / pgvector) с эмбеддингами.
- **MemorySaver** — in-memory: после рестарта контейнера сессии теряются. Для production переключитесь на `SqliteSaver` или `PostgresSaver` из `langgraph-checkpoint-*`.
- **Без секретов в репозитории**: реальный `.env` исключён через `.gitignore`; в репозиторий коммитится только `.env.example`.

---

##  Тесты

```bash
pip install '.[dev]'
PYTHONPATH=src pytest
```

Тесты используют `FakeListChatModel` и не обращаются к реальной LLM — ключи не нужны.

Покрывают:

- загрузку и метаданные базы знаний;
- страновую фильтрацию (Германия/Франция не «протекают» друг в друга);
- ветку уточняющего вопроса;
- сохранение страны в контексте между ходами диалога.

---

