"""Pydantic schemas for the API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    manual_id: str
    page: int
    section: str
    text: str
    component_tags: list[str] = Field(default_factory=list)
    symptom_tags: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class VisionObservation(BaseModel):
    issue_type: Optional[str] = None
    color: Optional[str] = None
    origin: Optional[str] = None
    intensity: Optional[str] = None
    additional_observations: list[str] = Field(default_factory=list)
    raw: Optional[str] = None


class Citation(BaseModel):
    page: int
    chunk_id: str


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: str = "low"
    manual_supported: bool = False
    language: Optional[str] = None


class QueryRequest(BaseModel):
    manual_id: str
    query: str
    image_b64: Optional[str] = None
    session_id: Optional[str] = None
    answer_model: Optional[str] = "sarvam"


class QueryMetrics(BaseModel):
    total_ms: int = 0
    vision_ms: int = 0
    rewrite_ms: int = 0
    retrieve_ms: int = 0
    generate_ms: int = 0
    verify_ms: int = 0
    sarvam_input_tokens: int = 0
    sarvam_output_tokens: int = 0
    gemini_input_tokens: int = 0
    gemini_output_tokens: int = 0
    top_retrieval_score: float = 0.0
    num_citations_kept: int = 0


class QueryResponse(BaseModel):
    answer: GroundedAnswer
    vision: Optional[VisionObservation] = None
    rewrites: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    metrics: Optional[QueryMetrics] = None
