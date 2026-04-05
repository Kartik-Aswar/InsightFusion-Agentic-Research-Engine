"""
Research Flow — main orchestration pipeline.

Steps:
1. Get query (CLI or injected by Streamlit)
2. Deterministic web search + PDF indexing/retrieval
3. Run CrewAI crew (planner → web → doc → conflict → report)
4. Parse outputs, handle conflicts with recursion
5. Calculate confidence and save final outputs
"""

import json
import os
import glob
import re

from crewai.flow.flow import Flow, start, listen

from memory.research_state import ResearchState
from memory.knowledge_store import KnowledgeStore
from crews.research_crew import ResearchCrew
from agents.web_scout import WebScoutAgent

from tools.pdf_tool import PDFProcessor
from tools.vector_store import VectorStore


# ---------------------------------------------------------
# HELPER: Safe JSON parser for LLM task outputs
# ---------------------------------------------------------

def safe_json_parse(task_outputs, index: int) -> dict | list | None:
    """
    Safely parse JSON from a CrewAI task output at the given index.
    Handles markdown code fences, partial JSON, and malformed output.
    """

    try:
        if index >= len(task_outputs):
            return None

        raw = task_outputs[index].raw.strip()

        # Remove markdown code fences if LLM adds them
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Fallback → extract JSON block (object OR array)
        match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        return None

    except Exception:
        return None


# ---------------------------------------------------------
# TASK OUTPUT KEYS (readable names instead of magic indices)
# ---------------------------------------------------------

TASK_PLANNING = 0
TASK_WEB = 1
TASK_DOCUMENT = 2
TASK_CONFLICT = 3
TASK_REPORT = 4
TASK_RELIABILITY = 5


########################################################
# RESEARCH FLOW
########################################################

class ResearchFlow(Flow[ResearchState]):

    # -----------------------------------------------------
    # INITIALIZATION (Load Heavy Models Once)
    # -----------------------------------------------------

    def __init__(self):
        super().__init__()

        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()

        self.pdf_indexed = False  # Prevent re-indexing during recursion

    # -----------------------------------------------------
    # STEP 1: GET QUERY
    # -----------------------------------------------------

    @start()
    def get_query(self):

        print("\n===== Agentic AI Deep Research System =====\n")

        # If query already provided (Streamlit), skip input
        if self.state.query and self.state.query.strip():
            print(f"Research initiated for: {self.state.query}\n")
            return self.state

        # Otherwise ask from CLI
        self.state.query = input("Enter research query: ").strip()

        self.state.recursion_count = 0
        self.state.conflicts_detected = False

        print(f"\nResearch initiated for: {self.state.query}\n")

        return self.state

    # -----------------------------------------------------
    # STEP 2: MAIN RESEARCH LOOP
    # -----------------------------------------------------

    @listen(get_query)
    def execute_research(self, state: ResearchState):

        knowledge_store = KnowledgeStore(state)

        task_outputs = None

        while True:

            print(f"\n--- Research Iteration {state.recursion_count + 1} ---\n")

            # =====================================================
            # 1️⃣ Deterministic Web Search
            # =====================================================

            try:
                web_wrapper = WebScoutAgent()
                structured_claims = web_wrapper.perform_search(state.query)

                for claim in structured_claims:
                    knowledge_store.add_web_claim(claim)

                knowledge_store.add_reasoning_step(
                    f"Web search completed. {len(structured_claims)} claims found."
                )

            except Exception as e:
                knowledge_store.add_reasoning_step(
                    f"Web search failed: {str(e)}"
                )

            # =====================================================
            # 2️⃣ PDF Processing (Index Only Once)
            # =====================================================

            try:
                if not self.pdf_indexed:

                    pdf_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "..", "input_pdfs"
                    )
                    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

                    # Fallback to relative path
                    if not pdf_files:
                        pdf_files = glob.glob("input_pdfs/*.pdf")

                    if pdf_files:
                        print(f"Processing {len(pdf_files)} PDFs...\n")

                        all_chunks = []
                        metadata = []

                        for pdf_path in pdf_files:

                            pdf_data = self.pdf_processor.extract_text_and_chunks(pdf_path)

                            if "error" in pdf_data:
                                knowledge_store.add_reasoning_step(
                                    f"Error processing PDF: {pdf_path} - {pdf_data['error']}"
                                )
                                continue

                            chunks = pdf_data.get("chunks", [])

                            for idx, chunk_data in enumerate(chunks):

                                chunk_text = chunk_data["text"]
                                page_number = chunk_data["page_number"]

                                chunk_id = f"{os.path.basename(pdf_path)}_chunk_{idx}"

                                knowledge_store.add_pdf_chunk(
                                    chunk_id=chunk_id,
                                    source_file=pdf_path,
                                    text=chunk_text,
                                )

                                all_chunks.append(chunk_text)
                                metadata.append({
                                    "source": str(pdf_path),
                                    "chunk_id": str(chunk_id),
                                    "page_number": int(page_number),
                                })

                        if all_chunks:
                            self.vector_store.add_documents(all_chunks, metadata)

                        self.pdf_indexed = True

                        knowledge_store.add_reasoning_step(
                            f"PDF indexing completed. {len(all_chunks)} chunks from {len(pdf_files)} PDFs indexed."
                        )

                    else:
                        knowledge_store.add_reasoning_step(
                            "No PDFs found in input_pdfs folder."
                        )

                # =====================================================
                # SEMANTIC RETRIEVAL (with metadata for citations)
                # =====================================================

                results = self.vector_store.query(state.query, top_k=20)

                retrieved_chunks = []
                retrieved_metadatas = []

                if results and "documents" in results:
                    retrieved_chunks = results["documents"][0]
                    retrieved_metadatas = results.get("metadatas", [[]])[0]

                print(f"Retrieved {len(retrieved_chunks)} relevant chunks via semantic search")

                # =====================================================
                # ENRICH CHUNKS WITH SOURCE METADATA FOR CITATIONS
                # =====================================================
                # This is critical — without metadata, the agent cannot
                # produce proper citations (filename + page number).

                enriched_chunks = []

                for i, chunk_text in enumerate(retrieved_chunks):

                    if i < len(retrieved_metadatas):
                        meta = retrieved_metadatas[i]
                        source_file = os.path.basename(str(meta.get("source", "Unknown PDF")))
                        page_num = meta.get("page_number", "?")
                    else:
                        source_file = "Unknown PDF"
                        page_num = "?"

                    # Format: include source + page so agent can cite properly
                    enriched = (
                        f"[Source: {source_file}, Page {page_num}]\n"
                        f"{chunk_text}"
                    )
                    enriched_chunks.append(enriched)

                # Store enriched chunks (with metadata) for crew
                state.retrieved_documents = enriched_chunks[:20]

                knowledge_store.add_reasoning_step(
                    f"Semantic retrieval done. {len(enriched_chunks)} chunks with source metadata prepared."
                )

            except Exception as e:
                knowledge_store.add_reasoning_step(
                    f"PDF processing error: {str(e)}"
                )

            # =====================================================
            # 3️⃣ Run Crew
            # =====================================================

            try:
                crew_builder = ResearchCrew(
                    query=state.query,
                    retrieved_documents=state.retrieved_documents,
                )

                crew, task_map = crew_builder.build()

                result = crew.kickoff()

                task_outputs = result.tasks_output

            except Exception as e:
                knowledge_store.add_reasoning_step(
                    f"Crew execution failed: {str(e)}"
                )
                task_outputs = None
                break

            # =====================================================
            # 4️⃣ Parse Outputs Safely
            # =====================================================

            # Planning
            plan_output = safe_json_parse(task_outputs, TASK_PLANNING)
            if plan_output:
                state.research_plan = plan_output

            # Web claims
            web_output = safe_json_parse(task_outputs, TASK_WEB)
            if web_output:
                if isinstance(web_output, list):
                    for claim in web_output:
                        knowledge_store.add_web_claim(claim)
                elif isinstance(web_output, dict):
                    knowledge_store.add_web_claim(web_output)

            # Document insights
            doc_output = safe_json_parse(task_outputs, TASK_DOCUMENT)
            if doc_output:
                if isinstance(doc_output, dict):
                    knowledge_store.add_document_insight(doc_output)
                elif isinstance(doc_output, list):
                    for doc in doc_output:
                        knowledge_store.add_document_insight(doc)

            # Conflict detection
            conflict_output = safe_json_parse(task_outputs, TASK_CONFLICT)
            if conflict_output and conflict_output.get("conflicts_detected") is True:

                for conflict in conflict_output.get("conflict_details", []):
                    knowledge_store.register_conflict(
                        issue=conflict.get("issue", "Unknown"),
                        sources=conflict.get("conflicting_sources", []),
                        severity=conflict.get("severity", "Medium"),
                    )

                if knowledge_store.can_recurse() and conflict_output.get("conflict_details", []):
                    print("\n⚠ Conflict detected. Adapting query for self-correction...\n")
                    
                    # Make self-correction SMART: dynamically adapt the query
                    # so the web scout and RAG engine specifically search for resolutions!
                    conflict_issues = [c.get("issue") for c in conflict_output.get("conflict_details", [])]
                    conflict_str = " | ".join(conflict_issues)
                    
                    # Give strict instructions to the query to resolve it!
                    state.query = f"{state.query}. (CRITICAL UPDATE: You previously found a conflict. You MUST now prioritize resolving this specific conflict: {conflict_str})"
                    
                    knowledge_store.increment_recursion()
                    knowledge_store.clear_conflicts()
                    continue
                else:
                    print("\n⚠ Conflicts securely logged. Passing to Report Generator for final synthesis...\n")

            break  # Exit loop when conflicts are analyzed or recursion maxed

        # =====================================================
        # 5️⃣ Final Report
        # =====================================================

        # Calculate REAL system confidence
        confidence = knowledge_store.calculate_confidence()

        knowledge_store.add_reasoning_step(
            f"Final confidence score: {confidence}%"
        )

        # Inject confidence into state
        if isinstance(state.research_plan, dict):
            state.research_plan["system_confidence_score"] = confidence
            state.research_plan["confidence_scale"] = "0-100"

        # Extract and enhance report
        if task_outputs and len(task_outputs) > TASK_REPORT:

            report_text = task_outputs[TASK_REPORT].raw.strip()
            report_text += f"\n\n---\nSystem Confidence Score: {confidence}% (Calculated)\n"
            state.final_report = report_text

            if len(task_outputs) > TASK_RELIABILITY:
                state.reliability_report = task_outputs[TASK_RELIABILITY].raw.strip()

        else:
            state.final_report = "Report generation failed. Crew did not return valid output."

        self.save_outputs(state)

        print("\n===== Research Completed =====")
        print(f"Confidence Score: {confidence}%")

        return state

    # -----------------------------------------------------
    # OUTPUT SAVING
    # -----------------------------------------------------

    def save_outputs(self, state: ResearchState) -> None:

        os.makedirs("output", exist_ok=True)

        with open("output/final_report.txt", "w", encoding="utf-8") as f:
            f.write(state.final_report)

        with open("output/reliability_report.txt", "w", encoding="utf-8") as f:
            f.write(state.reliability_report)

        with open("output/reasoning_trace.json", "w", encoding="utf-8") as f:
            json.dump(state.reasoning_trace, f, indent=4)

        with open("output/conflicts.json", "w", encoding="utf-8") as f:
            json.dump(
                [conflict.model_dump() for conflict in state.conflicts],
                f,
                indent=4,
            )

        summary = {
            "query": state.query,
            "confidence_score": state.confidence_score,
            "recursion_count": state.recursion_count,
            "total_web_claims": len(state.web_claims),
            "total_document_insights": len(state.document_insights),
            "total_pdf_chunks": len(state.pdf_chunks),
            "conflicts_found": len(state.conflicts),
        }

        with open("output/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
