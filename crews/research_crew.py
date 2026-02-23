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

    def __init__(self, query: str):
        self.query = query

        # Instantiate wrappers per crew instance (safer)
        self.web_agent_wrapper = WebScoutAgent()
        self.document_agent_wrapper = DocumentSpecialistAgent()

        self.web_scout = self.web_agent_wrapper.agent
        self.document_specialist = self.document_agent_wrapper.agent

    # -------------------------------------------------
    # TASK FACTORY
    # -------------------------------------------------

    def create_tasks(self):

        planning_task = Task(
            description=f"""
You are given the research query:

"{self.query}"

1. Restate the core objective clearly.
2. Break into 4–6 structured sub-questions.
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
            agent=research_planner
        )

        web_task = Task(
            description=f"""
Using the structured research plan, extract verified web claims
related to:

"{self.query}"

For each claim include:

- claim
- source
- publication_date
- source_type
- credibility_score (0–1)

STRICTLY return ONLY JSON list:

[
  {{
    "claim": "...",
    "source": "...",
    "publication_date": "...",
    "source_type": "...",
    "credibility_score": 0.0
  }}
]

No markdown. No explanation.
""",
            expected_output="Strict JSON list of web claims.",
            agent=self.web_scout,
            context=[planning_task]
        )

        document_task = Task(
            description=f"""
Analyze relevant academic or technical evidence related to:

"{self.query}"

Extract structured insights:

[
  {{
    "document_title": "...",
    "key_findings": "...",
    "statistics": "...",
    "methodology": "...",
    "limitations": "...",
    "confidence_level": "High/Medium/Low"
  }}
]

STRICT JSON only.
No commentary.
""",
            expected_output="Strict JSON document insights.",
            agent=self.document_specialist,
            context=[planning_task]
        )

        conflict_task = Task(
            description="""
Compare structured web claims and document insights.

Detect:

- Contradictory claims
- Statistical inconsistencies
- Outdated or low-credibility evidence

Return STRICT JSON:

{
  "conflicts_detected": true/false,
  "conflict_details": [
    {
      "issue": "...",
      "conflicting_sources": ["...", "..."],
      "severity": "High/Medium/Low"
    }
  ]
}

No explanation outside JSON.
""",
            expected_output="Strict JSON conflict report.",
            agent=conflict_detector,
            context=[web_task, document_task]
        )

        report_task = Task(
            description=f"""
Generate a structured academic research report for:

"{self.query}"

Structure:

1. Executive Summary
2. Research Objective
3. Key Findings (cite sources inline)
4. Cross-Source Analysis
5. Conflict Explanation
6. Limitations
7. Conclusion
8. Confidence Assessment

Integrate:
- Web claims
- Document insights
- Conflict analysis

Write in formal academic tone.
Do NOT output JSON.
""",
            expected_output="Final structured research report.",
            agent=report_generator,
            context=[web_task, document_task, conflict_task]
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
                report_generator
            ],
            tasks=list(tasks.values()),
            process=Process.sequential,
            verbose=True
        )

        return crew, tasks
