"""
Research Crew — builds a fresh multi-agent crew for each research iteration.

Agents: Planner → Web Scout → Document Specialist → Conflict Detector → Report Generator
Process: Sequential (each task feeds context to the next)
"""

from crewai import Crew, Process, Task

from agents.planner import research_planner
from agents.web_scout import WebScoutAgent
from agents.document_specialist import DocumentSpecialistAgent
from agents.conflict_detector import conflict_detector
from agents.report_generator import report_generator


class ResearchCrew:
    """
    Builds a fresh, fully structured multi-agent research crew
    for each research iteration.
    """

    def __init__(self, query: str, retrieved_documents=None):
        self.query = query
        self.retrieved_documents = retrieved_documents or []

        # Instantiate wrappers per crew instance (thread-safe)
        self.web_agent_wrapper = WebScoutAgent()
        self.document_agent_wrapper = DocumentSpecialistAgent()

        self.web_scout = self.web_agent_wrapper.agent
        self.document_specialist = self.document_agent_wrapper.agent

        # Build document context with source metadata for proper citations
        # Each chunk already contains [Source: filename.pdf, Page N] header
        # from research_flow.py's enrichment step.
        if isinstance(self.retrieved_documents, list):
            docs = self.retrieved_documents[:15]  # Use more chunks for better coverage
        else:
            docs = []

        self.documents_context = "\n\n---\n\n".join(
            [f"### PDF Evidence {i+1}\n{doc}" for i, doc in enumerate(docs)]
        )

        self.num_pdf_evidences = len(docs)

    # -------------------------------------------------
    # TASK FACTORY
    # -------------------------------------------------

    def create_tasks(self) -> dict:

        planning_task = Task(
            description=f"""
                You are given the research query:

                "{self.query}"

                1. Restate the core objective clearly.
                2. Break into 4-6 structured sub-questions.
                3. Identify required data types.
                4. Define validation strategy.
                5. Highlight possible risk areas.

                STRICTLY return ONLY valid JSON:

                {{
                  "research_objective": "...",
                  "sub_questions": ["...", "..."],
                  "data_requirements": ["..."],
                  "validation_strategy": "...",
                  "risk_areas": ["..."]
                }}

                No explanation. No markdown.
            """,
            expected_output="Strict JSON research plan.",
            agent=research_planner,
        )

        web_task = Task(
            description=f"""
                Using the structured research plan generated earlier, perform web research related to:

                "{self.query}"

                IMPORTANT: You MUST use the web search tool to search the internet.
                DO NOT make up results. DO NOT use your training data.
                Perform at least 2-3 different searches with varied keywords.

                Extract verified claims from reliable sources.

                For each claim include:

                - claim: the actual factual statement found
                - source: the FULL URL where this claim was found
                - publication_date: date if available, null otherwise
                - source_type: "Academic", "Official", "News", "Blog", etc.
                - credibility_score: 0.0 to 1.0 based on source reliability

                STRICTLY return ONLY JSON list:

                [
                  {{
                    "claim": "...",
                    "source": "https://...",
                    "publication_date": "...",
                    "source_type": "...",
                    "credibility_score": 0.0
                  }}
                ]

                No markdown. No explanation.
            """,
            expected_output="Strict JSON list of web claims with full source URLs.",
            agent=self.web_scout,
            context=[planning_task],
        )

        document_task = Task(
            description=f"""
                You are a research analyst specialized in reading academic papers.

                Research Question:
                "{self.query}"

                Below are extracted passages from uploaded research PDFs.
                Each passage includes its SOURCE FILENAME and PAGE NUMBER in a header like:
                [Source: filename.pdf, Page N]

                You MUST use this source information for traceability.

                ==========================================================
                RETRIEVED PDF EVIDENCE ({self.num_pdf_evidences} passages)
                ==========================================================

                {self.documents_context}

                ==========================================================
                END OF PDF EVIDENCE
                ==========================================================

                INSTRUCTIONS:
                1. Read EVERY passage above carefully
                2. Extract structured insights from the ACTUAL content
                3. Use the source filename and page number from each passage header
                4. DO NOT invent or hallucinate any information
                5. DO NOT use your general knowledge — ONLY use the text provided above
                6. If a passage is not relevant to the research question, skip it

                Return results in the following JSON format:

                [
                  {{
                    "document_title": "Short title inferred from the passage content",
                    "source_file": "filename.pdf",
                    "page_number": N,
                    "key_findings": "Main technical insight from this passage",
                    "statistics": "Any numbers, percentages, benchmarks mentioned",
                    "methodology": "Method, algorithm, approach described",
                    "limitations": "Any stated weaknesses or constraints",
                    "confidence_level": "High/Medium/Low"
                  }}
                ]

                Rules:
                - ONLY use the provided PDF evidence above
                - Include source_file and page_number from the [Source: ...] headers
                - DO NOT invent documents or citations
                - If information is missing, write "Not specified"
                - Output STRICT JSON only — no explanations outside JSON
            """,
            expected_output="Strict JSON document insights with source filenames and page numbers.",
            agent=self.document_specialist,
            context=[planning_task],
        )

        conflict_task = Task(
            description="""
                Compare structured web claims and document insights from the previous tasks.

                Detect:

                - Contradictory claims between web and PDF sources
                - Statistical inconsistencies (different numbers for the same metric)
                - Outdated or low-credibility evidence
                - Claims in web sources that contradict PDF evidence

                Return STRICT JSON:

                {
                  "conflicts_detected": true/false,
                  "conflict_details": [
                    {
                      "issue": "Clear description of the conflict",
                      "conflicting_sources": ["source1", "source2"],
                      "severity": "High/Medium/Low"
                    }
                  ]
                }

                No explanation outside JSON.
            """,
            expected_output="Strict JSON conflict report.",
            agent=conflict_detector,
            context=[web_task, document_task],
        )

        report_task = Task(
            description=f"""
                Generate a structured academic research report for:

                "{self.query}"

                Below is the SAME retrieved PDF evidence that was analyzed.
                Each passage has its SOURCE FILE and PAGE NUMBER in the header.

                ==========================================================
                RETRIEVED PDF EVIDENCE ({self.num_pdf_evidences} passages)
                ==========================================================

                {self.documents_context}

                ==========================================================
                END OF PDF EVIDENCE
                ==========================================================

                You also have access to:
                - Web claims with source URLs (from previous task)
                - Document insights with source files and pages (from previous task)
                - Conflict analysis results (from previous task)

                REPORT STRUCTURE:

                1. **Executive Summary**
                   Brief overview of key findings.

                2. **Research Objective**
                   Restate the research question and scope.

                3. **Key Findings** (THIS IS THE MOST IMPORTANT SECTION)
                   Present findings organized by theme.

                   CITATION RULES:
                   - For PDF evidence: cite as (Source: filename.pdf, Page N)
                   - For web evidence: cite as (Source: URL)
                   - Every factual claim MUST have a citation
                   - Use the actual filenames from the [Source: ...] headers above
                   - Do NOT use generic labels like "PDF Evidence 1"

                   Example citation format:
                   "Transformer models achieve 95% accuracy on benchmark X
                   (Source: 2308.10848v3.pdf, Page 12)."

                4. **Cross-Source Analysis**
                   Compare findings from PDFs vs web sources.
                   Identify areas of agreement and disagreement.

                5. **Conflict Analysis**
                   Summarize any conflicts detected between sources.

                6. **Limitations**
                   Discuss limitations of the evidence and analysis.

                7. **Conclusion**
                   Summarize the overall answer to the research question.

                8. **Research Reliability Discussion**
                   Analyze the reliability of the findings based on the nature and consistency of the primary evidence sources, citing the specific PDF filenames and web resources used. Discuss the strength of the evidence and acknowledge any analytical limitations.

                9. **References**
                   List all sources cited in the report:
                   - PDF sources with filenames and relevant pages
                   - Web sources with full URLs

                CRITICAL RULES:
                - The PDF evidence above is your PRIMARY source — use it FIRST
                - READ the actual PDF text provided and extract specific facts, numbers, and findings
                - DO NOT say "the PDFs were fragmented" or "could not extract meaningful data"
                - DO NOT ignore the PDF evidence
                - Cite with actual filenames (e.g., "2308.10848v3.pdf") not generic labels
                - Web claims are SECONDARY supporting evidence
                - Write in formal academic tone
                - Do NOT output JSON — write a proper narrative report
                - Do NOT include a confidence score (the system adds it separately)
            """,
            expected_output="Structured academic research report with proper source citations using filenames and page numbers.",
            agent=report_generator,
            context=[web_task, document_task, conflict_task],
        )

        return {
            "planning": planning_task,
            "web": web_task,
            "document": document_task,
            "conflict": conflict_task,
            "report": report_task,
        }

    # -------------------------------------------------
    # BUILD CREW
    # -------------------------------------------------

    def build(self):

        tasks = self.create_tasks()

        crew = Crew(
            agents=[
                research_planner,
                self.web_scout,
                self.document_specialist,
                conflict_detector,
                report_generator,
            ],
            tasks=list(tasks.values()),
            process=Process.sequential,
            verbose=True,
            max_rpm=4,  # Rate limit for Gemini free tier (5 RPM limit)
        )

        return crew, tasks
