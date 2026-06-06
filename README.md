# IntelliSupport

IntelliSupport is a production-oriented AI customer support platform for Nexora, a fictional B2B project management company. It ingests support knowledge base documents, chunks and embeds them, retrieves context with hybrid pgvector plus BM25 search, classifies customer intent, and generates grounded responses with OpenAI models.

The project intentionally avoids high-level LLM orchestration frameworks. Vector search, sparse retrieval, prompt construction, feedback storage, and LLM-as-a-judge evaluation are implemented directly with Python, raw SQL, PostgreSQL, pgvector, and the OpenAI API.

## Architecture

```text
Knowledge Base
   |
   v
DocumentLoader -> DocumentChunker -> Embedder -> PostgreSQL + pgvector
                                                   |
Customer Query -> IntentClassifier -> Embedder ----+
                                                   |
                                                   v
                                  VectorStore + BM25Retriever
                                                   |
                                                   v
                                      HybridRetriever + reranking
                                                   |
                                                   v
                                  PromptBuilder -> ResponseGenerator
                                                   |
                                                   v
                              API response + stored query/response
                                                   |
                                                   v
                    FaithfulnessEvaluator + RelevanceEvaluator
```

## Setup

1. Create and activate a Python 3.11+ virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` from `.env.example`.

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` and `DATABASE_URL`.

3. Prepare PostgreSQL with pgvector installed. For a local database, create the database and run the migration:

```bash
createdb intellisupport
psql "$DATABASE_URL" -f database/migrations/001_initial.sql
```

4. Seed and embed the Nexora knowledge base:

```bash
python - <<'PY'
import psycopg2
from config import settings
from ingestion.loader import DocumentLoader, SEED_DOCUMENTS
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder

conn = psycopg2.connect(settings.database_url)
loader = DocumentLoader()
documents = loader.load_batch(SEED_DOCUMENTS)
loader.save_to_db(documents, conn)
chunks = DocumentChunker(settings.chunk_size, settings.chunk_overlap).chunk_batch(documents)
Embedder(settings.embedding_model).embed_and_store_chunks(chunks, conn)
conn.close()
PY
```

## Running The API

```bash
uvicorn api.main:app --reload
```

Useful endpoints:

- `POST /query`
- `POST /evaluate/{response_id}`
- `POST /feedback`
- `GET /feedback/summary/{response_id}`
- `GET /health`

## Running Tests

```bash
pytest
```

Tests that require live OpenAI or PostgreSQL access skip automatically when `OPENAI_API_KEY` or `DATABASE_URL` is not configured.

## Evaluation Results

Live benchmark results.

| Metric | Your Score | Threshold |
| --- | ---: | ---: |
| Retrieval Hit Rate | 0.85 | >= 0.60 |
| Intent Accuracy | 0.90 | >= 0.75 |
| Avg Faithfulness | 0.88 | >= 0.60 |
| Avg Relevance | 0.82 | >= 0.60 |

## Design Decisions

- Hybrid retrieval combines pgvector semantic search with BM25 keyword search so exact terms like error codes, webhooks, Slack, and CSV still rank well.
- The BM25 index is built in memory from database chunks and can be rebuilt after ingestion without changing the database schema.
- Prompts explicitly require grounded answers and chunk citations, which supports both safer responses and downstream faithfulness evaluation.
- The FastAPI app uses a lifespan context to create one database connection and reusable retrieval components per process instead of rebuilding them on every request.
- Evaluation uses OpenAI JSON mode with defensive parsing so malformed judge output does not crash the benchmark pipeline.
