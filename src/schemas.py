"""Pydantic schemas.

``LLMAnalysis`` is what we ask the model to return (no ids, no numeric bounds: some structured-output
modes do not support min/max, so bounds are enforced afterwards).
``TranscriptAnalysis`` is the validated record we store and analyze.
"""
from typing import Literal

from pydantic import BaseModel, Field

CallType = Literal["support", "external", "internal", "unknown"]
SentimentLabel = Literal["Positive", "Neutral", "Negative", "Mixed"]
AnalysisMode = Literal["llm", "heuristic_demo"]


class LLMEvidence(BaseModel):
    claim: str
    quote: str


class LLMAnalysis(BaseModel):
    primary_topic: str
    secondary_topics: list[str]
    sentiment_label: SentimentLabel
    sentiment_score: float
    customer_sentiment: SentimentLabel | None
    summary: str
    pain_points: list[str]
    feature_requests: list[str]
    renewal_risk_signals: list[str]
    engineering_blockers: list[str]
    action_items: list[str]
    escalation_required: bool
    supporting_evidence: list[LLMEvidence]
    explanation: str


class Evidence(BaseModel):
    claim: str
    quote: str
    verified: bool | None = None  # True when the quote was found verbatim in the transcript


class TranscriptAnalysis(BaseModel):
    transcript_id: str
    call_type: CallType
    primary_topic: str
    secondary_topics: list[str] = Field(default_factory=list)
    sentiment_label: SentimentLabel
    sentiment_score: float = Field(ge=-1.0, le=1.0)  # NOT a calibrated probability
    customer_sentiment: SentimentLabel | None = None
    customer_sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    summary: str
    pain_points: list[str] = Field(default_factory=list)
    feature_requests: list[str] = Field(default_factory=list)
    renewal_risk_signals: list[str] = Field(default_factory=list)
    engineering_blockers: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    escalation_required: bool = False
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    explanation: str
    analysis_mode: AnalysisMode
    llm_model: str | None = None
    topic_was_remapped: bool = False
