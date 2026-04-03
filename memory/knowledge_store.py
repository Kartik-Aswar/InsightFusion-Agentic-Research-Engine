"""
Knowledge Store — manages all evidence, conflicts, and confidence scoring.
Acts as the central intelligence layer over ResearchState.
"""

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
    # SOURCE AUTHORITY INTELLIGENCE
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
                "nih.gov",
                "pubmed.ncbi",
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
            ]):
                return 0.08

            # Government / education
            if ".gov" in domain or ".edu" in domain:
                return 0.12

            # Medium / personal blogs
            if "medium.com" in domain or "blog" in domain:
                return -0.05

            return 0.0

        except Exception:
            return 0.0

    # -----------------------------------
    # WEB EVIDENCE MANAGEMENT
    # -----------------------------------

    def add_web_claim(self, claim_data: dict) -> None:

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

        except Exception as e:
            self.state.reasoning_trace.append(
                f"Web claim validation failed: {str(e)}"
            )

    # -----------------------------------
    # RAW PDF CHUNK MANAGEMENT
    # -----------------------------------

    def add_pdf_chunk(self, chunk_id: str, source_file: str, text: str) -> None:

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

        except Exception as e:
            self.state.reasoning_trace.append(
                f"PDF chunk validation failed: {str(e)}"
            )

    # -----------------------------------
    # DOCUMENT INSIGHTS
    # -----------------------------------

    def add_document_insight(self, doc_data: dict) -> None:

        if not isinstance(doc_data, dict):
            self.add_reasoning_step(
                f"Document insight ignored (not dict): {type(doc_data)}"
            )
            return

        try:
            # Normalize common LLM output field variations
            if "title" in doc_data and "document_title" not in doc_data:
                doc_data["document_title"] = doc_data.pop("title")

            if "findings" in doc_data and "key_findings" not in doc_data:
                doc_data["key_findings"] = doc_data.pop("findings")

            doc_data.setdefault("document_title", "Unknown Document")
            doc_data.setdefault("key_findings", "No findings provided")

            # Remove unexpected fields that would fail Pydantic validation
            allowed_fields = {f.alias or name for name, f in DocumentInsight.model_fields.items()}
            cleaned_data = {k: v for k, v in doc_data.items() if k in allowed_fields}

            insight = DocumentInsight(**cleaned_data)

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

    def register_conflict(self, issue: str, sources: List[str], severity: str) -> None:

        severity = severity.capitalize()

        if severity not in ["High", "Medium", "Low"]:
            severity = "Low"

        conflict = ConflictRecord(
            issue=issue,
            conflicting_sources=sources,
            severity=severity,
        )

        self.state.conflicts.append(conflict)
        self.state.conflicts_detected = True

    def clear_conflicts(self) -> None:
        # We NO LONGER clear the actual conflict list so they get saved to conflicts.json
        # We only reset the flag to allow the next iteration to evaluate fresh
        self.state.conflicts_detected = False

    # -----------------------------------
    # RECURSION CONTROL
    # -----------------------------------

    def increment_recursion(self) -> None:
        self.state.recursion_count += 1

    def can_recurse(self) -> bool:
        return self.state.recursion_count < self.state.max_recursions

    # -----------------------------------
    # REASONING TRACE
    # -----------------------------------

    def add_reasoning_step(self, step: str) -> None:
        self.state.reasoning_trace.append(step)

    # -----------------------------------
    # CONFIDENCE SCORING
    # -----------------------------------

    def calculate_confidence(self) -> float:

        if not self.state.web_claims and not self.state.pdf_chunks:
            self.state.confidence_score = 0.0
            return 0.0

        # --- Web credibility (with authority boost) ---
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

        # PDF evidence strength
        pdf_strength = min(len(self.state.pdf_chunks) / 15, 1.0)

        # Document insights strength
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

        # Weighted confidence formula
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
