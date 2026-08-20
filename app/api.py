"""API layer — thin: parse, delegate to the service, serialise.

Run: uvicorn app.api:app --port 8000
"""
from datetime import datetime
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.db import Repo, connect
from app.service import Retriever, add_label, classify_message

app = FastAPI(title="Mucka triage")
repo = Repo(connect("app.db"))
retriever = Retriever()
retriever.fit(repo.corpus_rows())  # empty corpus is fine: LLM runs zero-shot until labels arrive


class MessageIn(BaseModel):
    message_id: str | None = None
    received_at: datetime
    channel: Literal["sms", "whatsapp", "voicemail"]
    from_name: str
    body: str


class ClassificationOut(BaseModel):
    message_id: str
    urgency: Literal["routine", "same_day", "emergency"]
    intent: list[str]
    score: int
    reason: str
    cached: bool  # idempotency hit


class InboxItem(BaseModel):
    message_id: str
    received_at: datetime
    channel: str
    from_name: str
    body: str
    urgency: str
    score: int
    reason: str


class LabelIn(BaseModel):
    urgency: Literal["routine", "same_day", "emergency"]


@app.post("/messages", response_model=ClassificationOut)
def post_message(m: MessageIn):
    msg = {"message_id": m.message_id or uuid4().hex[:12],
           "received_at": m.received_at.isoformat(),
           "channel": m.channel, "from_name": m.from_name, "body": m.body}
    return classify_message(repo, retriever, msg)


@app.get("/inbox", response_model=list[InboxItem])
def get_inbox():
    # the sorted inbox IS this query — scores are absolute, so no re-ranking job exists
    return repo.inbox()


@app.post("/messages/{message_id}/label")
def post_label(message_id: str, label: LabelIn):
    """Override capture — the adaptivity loop: correction -> corpus -> future shots."""
    try:
        return {"corpus_size": add_label(repo, retriever, message_id, label.urgency)}
    except KeyError:
        raise HTTPException(404, f"unknown message_id {message_id}")


@app.post("/retrain")
def post_retrain():
    """Cold path: retrain from all accumulated data, emit + promote a versioned artifact.
    Sub-second at this volume - no job queue until retraining outgrows a request."""
    from ml.cold import retrain
    return retrain()


@app.get("/config")
def get_config():
    """Metrics of the currently promoted cold artifact."""
    from ml.cold import load_current
    return load_current()[1]
