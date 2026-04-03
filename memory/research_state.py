"""
Research State — Pydantic models for all state managed by the research flow.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Set, Tuple


class Claim(BaseModel):
    claim: str
    source: str
    publication_date: str | None = None
    source_type: str | None = None
    credibility_score: float = 0.0


class PDFChunk(BaseModel):
    chunk_id: str
    source_file: str
    text: str


class DocumentInsight(BaseModel):
    document_title: str
    key_findings: str

    # Deep traceability
    source_file: str | None = None
    chunk_id: str | None = None
    page_number: int | None = None

    statistics: str | None = None
    methodology: str | None = None
    limitations: str | None = None
    confidence_level: str | None = None


class ConflictRecord(BaseModel):
    issue: str
    conflicting_sources: List[str]
    severity: str  # Expected: High / Medium / Low


class ResearchState(BaseModel):

    # Core query
    query: str = ""

    # Planning output
    research_plan: Dict[str, Any] = Field(default_factory=dict)

    # Retrieved document chunks for RAG context
    retrieved_documents: List[str] = Field(default_factory=list)

    # Evidence stores
    web_claims: List[Claim] = Field(default_factory=list)
    document_insights: List[DocumentInsight] = Field(default_factory=list)

    # Deduplication tracking (FIX: proper Pydantic types)
    document_insights_seen: Set[Tuple[str, str]] = Field(default_factory=set)
    web_sources_seen: Set[str] = Field(default_factory=set)
    chunk_ids_seen: Set[str] = Field(default_factory=set)

    pdf_chunks: List[PDFChunk] = Field(default_factory=list)

    # Evidence mapping for cross-reference
    evidence_map: Dict[str, List[str]] = Field(default_factory=dict)

    # Conflict tracking
    conflicts: List[ConflictRecord] = Field(default_factory=list)
    conflicts_detected: bool = False

    # Recursive control
    recursion_count: int = 0
    max_recursions: int = 1

    # Reasoning trace (important for explainability)
    reasoning_trace: List[str] = Field(default_factory=list)

    # Final outputs
    final_report: str = ""
    confidence_score: float = 0.0
