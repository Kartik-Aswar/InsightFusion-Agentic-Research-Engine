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
from tools.clustering_tool import InsightClusterer


class ResearchFlow(Flow[ResearchState]):

    # -----------------------------------------------------
    # INITIALIZATION (Load Heavy Models Once)
    # -----------------------------------------------------

    def __init__(self):
        super().__init__()

        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.clusterer = InsightClusterer()

        self.pdf_indexed = False  # Prevent re-indexing during recursion

    # -----------------------------------------------------
    # STEP 1: GET QUERY
    # -----------------------------------------------------

    @start()
    def get_query(self):

        print("\n===== Agentic AI Deep Research System =====\n")

        self.state.query = input("Enter research query: ").strip()
        self.state.recursion_count = 0
        self.state.conflicts_detected = False

        print(f"\nResearch initiated for: {self.state.query}\n")

        return self.state

    # -----------------------------------------------------
    # STEP 2: MAIN LOOP
    # -----------------------------------------------------

    @listen(get_query)
    def execute_research(self, state: ResearchState):

        knowledge_store = KnowledgeStore(state)

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

                knowledge_store.add_reasoning_step("Web search completed.")

            except Exception as e:
                knowledge_store.add_reasoning_step(
                    f"Web search failed: {str(e)}"
                )

            # =====================================================
            # 2️⃣ PDF Processing (Index Only Once)
            # =====================================================

            try:
                if not self.pdf_indexed:

                    pdf_files = glob.glob("input_pdfs/*.pdf")

                    if pdf_files:
                        print(f"Processing {len(pdf_files)} PDFs...\n")

                        all_chunks = []
                        metadata = []

                        for pdf_path in pdf_files:

                            pdf_data = self.pdf_processor.extract_text_and_chunks(pdf_path)

                            if "error" in pdf_data:
                                knowledge_store.add_reasoning_step(
                                    f"Error processing PDF: {pdf_path}"
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
                                    "source": pdf_path,
                                    "chunk_id": chunk_id,
                                    "page_number": page_number
                                })

                        if all_chunks:
                            self.vector_store.add_documents(all_chunks, metadata)

                        self.pdf_indexed = True

                        knowledge_store.add_reasoning_step(
                            "PDF indexing completed."
                        )

                    else:
                        knowledge_store.add_reasoning_step(
                            "No PDFs found in input_pdfs folder."
                        )

                # Semantic Retrieval
                results = self.vector_store.query(state.query)

                retrieved_chunks = []

                if results and "documents" in results:
                    retrieved_chunks = results["documents"][0]

                if retrieved_chunks:

                    clustered = self.clusterer.cluster(retrieved_chunks)

                    for cluster_id, texts in clustered.items():
                        for text in texts:
                            knowledge_store.add_document_insight({
                                "document_title": f"Cluster {cluster_id}",
                                "key_findings": text,
                                "statistics": None,
                                "methodology": None,
                                "limitations": None,
                                "confidence_level": "High"
                            })

            except Exception as e:
                knowledge_store.add_reasoning_step(
                    f"PDF processing error: {str(e)}"
                )

            # =====================================================
            # 3️⃣ Run Crew
            # =====================================================

            try:
                crew_builder = ResearchCrew(state.query)
                crew, task_map = crew_builder.build()
                result = crew.kickoff()
                task_outputs = result.tasks_output
            except Exception as e:
                knowledge_store.add_reasoning_step(
                    f"Crew execution failed: {str(e)}"
                )
                break

            # =====================================================
            # 4️⃣ Parse Outputs Safely
            # =====================================================


            def safe_json_parse(index):
                try:
                    raw = task_outputs[index].raw.strip()

                    # remove markdown if LLM adds it
                    raw = raw.replace("```json", "").replace("```", "").strip()

                    # try direct parse first
                    try:
                        return json.loads(raw)
                    except:
                        pass

                    # fallback → extract JSON block (object OR array)
                    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
                    if match:
                        return json.loads(match.group())

                    return None

                except Exception:
                    return None



            # Planning
            plan_output = safe_json_parse(0)
            if plan_output:
                state.research_plan = plan_output

            # Web claims
            web_output = safe_json_parse(1)
            if web_output:
                for claim in web_output:
                    knowledge_store.add_web_claim(claim)

            # Document insights
            doc_output = safe_json_parse(2)

            if doc_output:

                if isinstance(doc_output, dict):
                    knowledge_store.add_document_insight(doc_output)

                elif isinstance(doc_output, list):
                    for doc in doc_output:
                        knowledge_store.add_document_insight(doc)


            # Conflict detection
            conflict_output = safe_json_parse(3)
            if conflict_output and conflict_output.get("conflicts_detected") is True:

                for conflict in conflict_output.get("conflict_details", []):
                    knowledge_store.register_conflict(
                        issue=conflict.get("issue", "Unknown"),
                        sources=conflict.get("conflicting_sources", []),
                        severity=conflict.get("severity", "Medium"),
                    )

                if knowledge_store.can_recurse():

                    print("\n⚠ Conflict detected. Running recursive research...\n")

                    knowledge_store.increment_recursion()
                    knowledge_store.clear_conflicts()

                    continue

            break  # Exit loop if no recursion


        # =====================================================
        # 5️⃣ Final Report
        # =====================================================

        # First calculate REAL system confidence
        confidence = knowledge_store.calculate_confidence()

        knowledge_store.add_reasoning_step(
            f"Final confidence score: {confidence}%"
        )

        # Inject confidence into state so report can use it
        if isinstance(state.research_plan, dict):
            state.research_plan["system_confidence_score"] = confidence
            state.research_plan["confidence_scale"] = "0-100"

        # Replace confidence section inside report safely
        # replace ONLY confidence line (safe)

        if task_outputs:
            report_text = task_outputs[4].raw.strip()

                # Add system confidence at END (clean & safe)
            report_text += f"\n\n---\nSystem Confidence Score: {confidence}% (Calculated)\n"

            state.final_report = report_text
        else:
            state.final_report = ""
        
        self.save_outputs(state)

        print("\n===== Research Completed =====")
        print(f"Confidence Score: {confidence}%")

        return state

    # -----------------------------------------------------
    # OUTPUT SAVING
    # -----------------------------------------------------

    def save_outputs(self, state: ResearchState):

        os.makedirs("output", exist_ok=True)

        with open("output/final_report.txt", "w", encoding="utf-8") as f:
            f.write(state.final_report)

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
        }

        with open("output/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
