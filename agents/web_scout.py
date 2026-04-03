"""
Web Scout Agent — performs live web research and extracts structured claims.
"""

from crewai import Agent
from .base_llm import llm

from tools.web_search_tool import WebSearchToolWrapper
from tools.credibility_tool import CredibilityScorer


class WebScoutAgent:

    def __init__(self):
        self.search_tool = WebSearchToolWrapper()
        self.credibility_scorer = CredibilityScorer()

        self.agent = Agent(
            role="Live Web Intelligence Scout",
            goal="Perform live web research and extract structured factual claims. ALWAYS use web search tool to gather real internet data.",
            backstory=(
                "You specialize in identifying authoritative sources "
                "and extracting verifiable claims from structured web results. "
                "ALWAYS use the search tool for real web data."
            ),
            tools=[self.search_tool.tool],
            llm=llm,
            verbose=True,
        )

    def perform_search(self, query: str) -> list:
        """
        Deterministic web search (outside of CrewAI agent loop).
        Calls Serper API directly and scores credibility.
        """

        raw_results = self.search_tool.search(query)

        structured_claims = []

        for item in raw_results:

            if "error" in item:
                continue

            snippet = item.get("snippet") or item.get("title")

            if not snippet:
                continue

            credibility = self.credibility_scorer.score(
                url=item.get("url", ""),
                publication_date=None,
            )

            structured_claims.append({
                "claim": snippet,
                "source": item.get("url", ""),
                "publication_date": None,
                "source_type": "Web",
                "credibility_score": credibility,
            })

        return structured_claims
