from typing import List
from urllib.parse import urlparse

from .research_state import (
    ResearchState,
    Claim,
    DocumentInsight,
    ConflictRecord,
    PDFChunk,
)


class KnowledgeStore:

    def __init__(self, state: ResearchState):
        self.state = state

    # -----------------------------------
    # SOURCE AUTHORITY INTELLIGENCE (NEW)
    # -----------------------------------

    def _source_authority_boost(self, url: str) -> float:
        """
        Lightweight authority scoring.

        Returns small boost/penalty:
        + academic / official docs → higher confidence
        - random blogs → lower confidence

        Safe, deterministic, zero dependencies.
        """

        if not url:
            return 0.0

        try:
            domain = urlparse(url).netloc.lower()

            # Academic / research sources
            if any(x in domain for x in [
                "arxiv.org",
                "nature.com",
                "science.org",
                "ieee.org",
                "acm.org",
                "springer.com",
            ]):
                return 0.15

            # Official technical docs
            if any(x in domain for x in [
                "docs.microsoft.com",
                "learn.microsoft.com",
                "cloud.google.com",
                "openai.com",
                "huggingface.co",
            ]):
                return 0.12

            # Company engineering blogs
            if any(x in domain for x in [
                "anthropic.com",
                "deepmind.com",
                "openai.com",
            ]):
                return 0.08

            # Medium / personal blogs
            if "medium.com" in domain or "blog" in domain:
                return -0.05

            return 0.0

        except Exception:
            return 0.0

    # -----------------------------------
    # WEB EVIDENCE MANAGEMENT
    # -----------------------------------

    def add_web_claim(self, claim_data: dict):

        try:
            claim = Claim(**claim_data)

            # Deduplicate by source
            if claim.source in self.state.web_sources_seen:
                return

            self.state.web_claims.append(claim)
            self.state.web_sources_seen.add(claim.source)

            # Evidence map
            if claim.claim not in self.state.evidence_map:
                self.state.evidence_map[claim.claim] = []

            self.state.evidence_map[claim.claim].append(claim.source)

        except Exception:
            self.state.reasoning_trace.append(
                "Web claim validation failed."
            )

    # -----------------------------------
    # RAW PDF CHUNK MANAGEMENT
    # -----------------------------------

    def add_pdf_chunk(self, chunk_id: str, source_file: str, text: str):

        if chunk_id in self.state.chunk_ids_seen:
            return

        try:
            chunk = PDFChunk(
                chunk_id=chunk_id,
                source_file=source_file,
                text=text,
            )

            self.state.pdf_chunks.append(chunk)
            self.state.chunk_ids_seen.add(chunk_id)

        except Exception:
            self.state.reasoning_trace.append(
                "PDF chunk validation failed."
            )

    # -----------------------------------
    # DOCUMENT INSIGHTS
    # -----------------------------------

    def add_document_insight(self, doc_data: dict):

        if not isinstance(doc_data, dict):
            self.add_reasoning_step(
                f"Document insight ignored (not dict): {type(doc_data)}"
            )
            return

        try:
            if "title" in doc_data and "document_title" not in doc_data:
                doc_data["document_title"] = doc_data["title"]

            if "findings" in doc_data and "key_findings" not in doc_data:
                doc_data["key_findings"] = doc_data["findings"]

            doc_data.setdefault("document_title", "Unknown Document")
            doc_data.setdefault("key_findings", "No findings provided")

            insight = DocumentInsight(**doc_data)

            key = (insight.document_title, insight.key_findings)

            if key not in self.state.document_insights_seen:
                self.state.document_insights.append(insight)
                self.state.document_insights_seen.add(key)

        except Exception as e:
            self.add_reasoning_step(
                f"Document insight validation failed: {str(e)}"
            )

    # -----------------------------------
    # CONFLICT MANAGEMENT
    # -----------------------------------

    def register_conflict(self, issue: str, sources: List[str], severity: str):

        severity = severity.capitalize()

        if severity not in ["High", "Medium", "Low"]:
            severity = "Medium"

        conflict = ConflictRecord(
            issue=issue,
            conflicting_sources=sources,
            severity=severity,
        )

        self.state.conflicts.append(conflict)
        self.state.conflicts_detected = True

    def clear_conflicts(self):
        self.state.conflicts = []
        self.state.conflicts_detected = False

    # -----------------------------------
    # RECURSION CONTROL
    # -----------------------------------

    def increment_recursion(self):
        self.state.recursion_count += 1

    def can_recurse(self) -> bool:
        return self.state.recursion_count < self.state.max_recursions

    # -----------------------------------
    # REASONING TRACE
    # -----------------------------------

    def add_reasoning_step(self, step: str):
        self.state.reasoning_trace.append(step)

    # -----------------------------------
    # CONFIDENCE SCORING
    # -----------------------------------

    def calculate_confidence(self):

        if not self.state.web_claims and not self.state.pdf_chunks:
            self.state.confidence_score = 0.0
            return 0.0

        # --- Web credibility (UPGRADED WITH AUTHORITY) ---
        if self.state.web_claims:

            adjusted_scores = []

            for claim in self.state.web_claims:
                base = claim.credibility_score
                boost = self._source_authority_boost(claim.source)
                adjusted = max(0.0, min(base + boost, 1.0))
                adjusted_scores.append(adjusted)

            avg_credibility = sum(adjusted_scores) / len(adjusted_scores)

        else:
            avg_credibility = 0.5

        independent_sources = len(self.state.web_sources_seen)

        # Use insights strength
        pdf_strength = min(len(self.state.document_insights) / 15, 1.0)

        doc_strength = min(len(self.state.document_insights) / 10, 1.0)

        # Conflict penalty
        severity_penalty = 0.0
        for conflict in self.state.conflicts:
            if conflict.severity == "High":
                severity_penalty += 0.15
            elif conflict.severity == "Medium":
                severity_penalty += 0.08
            else:
                severity_penalty += 0.03

        confidence = (
            (avg_credibility * 0.4)
            + (min(independent_sources / 10, 1.0) * 0.25)
            + (pdf_strength * 0.2)
            + (doc_strength * 0.15)
        )

        confidence -= severity_penalty
        confidence = max(0.0, min(confidence, 1.0))

        self.state.confidence_score = round(confidence * 100, 2)

        return self.state.confidence_score
