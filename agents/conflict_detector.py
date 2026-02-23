from crewai import Agent
from .base_llm import llm

conflict_detector = Agent(
    role="Conflict Detection and Self-Correction Agent",
    
    goal=(
    "Compare multi-source structured evidence (web claims, document insights), "
    "detect contradictions, statistical inconsistencies, outdated claims, "
    "and return structured JSON indicating conflict severity."
        ),

    
    backstory=(
        "You specialize in multi-source triangulation. You detect conflicts "
        "between sources, identify outdated or unreliable claims, "
        "and recommend corrective research loops when inconsistencies are found."
    ),
    
    verbose=True,
    allow_delegation=False,
    llm=llm
)
