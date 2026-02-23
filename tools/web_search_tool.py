from crewai_tools import SerperDevTool
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

class WebSearchToolWrapper:

    def __init__(self):
        self.tool = SerperDevTool()

    def search(self, query: str) -> List[Dict]:

        if not query:
            return []

        try:
            results = self.tool.run(query)

            #  Handle string output (very common)
            if isinstance(results, str):
                import json
                try:
                    results = json.loads(results)
                except:
                    return [{"error": "Serper returned string output"}]

            structured_results = []

            for item in results.get("organic", []):
                structured_results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                })

            return structured_results

        except Exception as e:
            return [{"error": str(e)}]
