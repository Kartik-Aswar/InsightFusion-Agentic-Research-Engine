from crewai import LLM
from dotenv import load_dotenv
import os

load_dotenv()

# Centralized LLM configuration
llm = LLM(
    model="gemini-2.5-flash",
    temperature=0.1,  # Lower temp for research accuracy
    max_tokens=4000
)
