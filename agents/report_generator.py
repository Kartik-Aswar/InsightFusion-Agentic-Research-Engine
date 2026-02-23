from crewai import Agent
from .base_llm import llm

report_generator = Agent(
    role="Research Report Synthesizer",
    
    goal=(
    "Generate structured academic research reports using web claims, "
    "document insights, and conflict analysis. "
    "When mentioning confidence, always use the system-provided confidence "
    "score in percentage format (0-100). Do NOT invent new scales like /5 or /10."
    "Use system confidence only. Do not estimate confidence yourself."
),

    
    backstory=(
        "You are a senior academic researcher skilled in synthesizing "
        "evidence into coherent, structured, and citation-backed reports. "
        "You clearly highlight limitations and uncertainty when needed."
    ),
    
    verbose=True,
    allow_delegation=False,
    llm=llm
)
