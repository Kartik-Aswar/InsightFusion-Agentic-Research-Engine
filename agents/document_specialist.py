from crewai import Agent
from .base_llm import llm
from tools.pdf_tool import PDFProcessor


class DocumentSpecialistAgent:

    def __init__(self):
        self.pdf_processor = PDFProcessor()

        self.agent = Agent(
            role="Document Intelligence Specialist",
            goal=(
                "Extract structured insights from academic PDFs including "
                "methodology, statistics, findings, and limitations."
            ),
            backstory=(
                "You analyze academic documents carefully and extract "
                "evidence-based structured insights."
            ),
            llm=llm,
            verbose=True
        )

    def analyze_pdf(self, file_path: str):
        """
        Deterministic PDF extraction with chunk support.
        """
        return self.pdf_processor.extract_text_and_chunks(file_path)
