from datetime import datetime
from urllib.parse import urlparse


class CredibilityScorer:

    def __init__(self):

        self.high_authority_domains = [
            ".gov",
            ".edu",
            "nature.com",
            "sciencedirect.com",
            "ieee.org",
            "springer.com",
            "who.int",
            "worldbank.org"
        ]

        self.low_authority_indicators = [
            "blog",
            "medium.com",
            "wordpress",
            "opinion"
        ]

    def score(self, url: str, publication_date: str | None = None) -> float:

        if not url:
            return 0.0

        score = 0.5

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # HTTPS boost
        if parsed.scheme == "https":
            score += 0.05

        for trusted in self.high_authority_domains:
            if trusted in domain:
                score += 0.3

        for weak in self.low_authority_indicators:
            if weak in domain:
                score -= 0.2

        # Recency factor
        if publication_date:
            try:
                year = int(publication_date[:4])
                current_year = datetime.now().year

                age = current_year - year

                if age <= 2:
                    score += 0.1
                elif age > 5:
                    score -= 0.1
            except:
                pass

        return round(max(0.0, min(score, 1.0)), 2)
