"""
Document Specialist Agent — extracts structured insights from academic PDFs.
"""

from crewai import Agent
from .base_llm import llm
from tools.pdf_tool import PDFProcessor


class DocumentSpecialistAgent:

    def __init__(self):
        self.pdf_processor = PDFProcessor()

        self.agent = Agent(
            role="Document Intelligence Specialist",
            goal=(
                "Extract structured insights from academic PDF passages including "
                "methodology, statistics, findings, limitations, and source traceability. "
                "ALWAYS preserve the source filename and page number from each passage header."
            ),
            backstory=(
                "You analyze academic documents carefully and extract "
                "evidence-based structured insights with full source traceability. "
                "You always note which file and page each finding came from."
            ),
            llm=llm,
            verbose=True,
        )

    def analyze_pdf(self, file_path: str) -> dict:
        """
        Deterministic PDF extraction with chunk support.
        """
        return self.pdf_processor.extract_text_and_chunks(file_path)
