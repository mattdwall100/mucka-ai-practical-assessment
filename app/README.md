# Getting started

From the repo root:

```
pip install -r requirements.txt
```

Put your OpenAI key in `.env` at the repo root (already scaffolded):

```
OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini   # optional override
```

Then:

```
python backfill.py               # seed corpus + classify holdout -> predictions.csv (~5-10 min, resumable)
python evaluate.py               # leave-group-out eval on the 300 labelled rows (own eval.db)
uvicorn app.api:app --port 8000  # live API
```

## Endpoints

```
POST /messages                   # ingest + classify; same body twice -> cached:true, no second LLM call
GET  /inbox                      # all messages, ORDER BY score DESC, received_at ASC
POST /messages/{id}/label        # manager override -> retrieval corpus -> future few-shot examples
```

Smoke test (demos de-escalation trap + escalate-only floor + idempotency):

```
curl -X POST localhost:8000/messages -H "Content-Type: application/json" -d "{\"received_at\":\"2026-08-20T14:00\",\"channel\":\"sms\",\"from_name\":\"Test\",\"body\":\"no rush but there is a smell of gas in the kitchen\"}"
```

Expect `emergency` with `[safety rule: gas_smell]` in the reason; repeat it and `cached` flips to true.

## Layers

- `db.py` — SQLite + `Repo` (the only SQL in the codebase; swap `connect()` + placeholders for psycopg to go Postgres)
- `service.py` — the hot path: normalise/hash -> TF-IDF top-3 few-shot -> OpenAI -> escalate-only safety floor -> store once
- `api.py` — FastAPI, thin delegation only
