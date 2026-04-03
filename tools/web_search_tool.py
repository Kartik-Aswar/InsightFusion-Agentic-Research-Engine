"""
Web Search Tool — wraps Serper API for both CrewAI agent use and deterministic search.

FIX: The deterministic `search()` method now calls the Serper API directly via
requests instead of relying on SerperDevTool.run() which returns formatted strings
that can't be parsed as JSON dicts.
"""

import os
import json
import requests
from crewai_tools import SerperDevTool
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()


class WebSearchToolWrapper:

    def __init__(self):
        # SerperDevTool is used as a CrewAI tool (passed to agents)
        self.tool = SerperDevTool()

        # API key for direct HTTP calls in deterministic search
        self._api_key = os.environ.get("SERPER_API_KEY", "")

    def search(self, query: str) -> List[Dict]:
        """
        Deterministic web search using Serper API directly.
        Returns structured results as a list of dicts.

        This is called OUTSIDE of CrewAI's agent loop for
        pre-processing web evidence before crew execution.
        """

        if not query:
            return []

        if not self._api_key:
            return [{"error": "SERPER_API_KEY not set in environment"}]

        try:
            response = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": 10},
                timeout=15,
            )

            response.raise_for_status()
            data = response.json()

            structured_results = []

            for item in data.get("organic", []):
                structured_results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                })

            return structured_results

        except requests.exceptions.Timeout:
            return [{"error": "Serper API request timed out"}]

        except requests.exceptions.HTTPError as e:
            return [{"error": f"Serper API HTTP error: {e.response.status_code}"}]

        except Exception as e:
            return [{"error": f"Web search failed: {str(e)}"}]
