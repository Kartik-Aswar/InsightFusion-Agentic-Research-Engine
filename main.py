"""
CLI entry point for InsightFusion — Agentic AI Deep Research System.
"""

import sys
import os
import time

from dotenv import load_dotenv
load_dotenv()

# Ensure root path is included
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup dual logging (terminal + file)
from utils.logging_util import setup_logging
log_filename = setup_logging()

from flows.research_flow import ResearchFlow


# ---------------------------------------------------------
# SYSTEM INITIALIZATION
# ---------------------------------------------------------

def ensure_directories() -> None:
    """Ensure required system directories exist."""

    required_dirs = [
        "input_pdfs",
        "output",
        "vector_db",
        "logs",
    ]

    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)


def print_banner() -> None:
    print("\n" + "=" * 60)
    print("  INSIGHTFUSION — AGENTIC AI DEEP RESEARCH SYSTEM")
    print("=" * 60)
    print("Autonomous | Multi-Agent | Self-Correcting")
    print(f"Log file: {log_filename}")
    print("=" * 60 + "\n")


def print_completion_summary(start_time: float) -> None:
    elapsed = round(time.time() - start_time, 2)

    print("\n" + "-" * 60)
    print("Research Session Completed")
    print(f"Execution Time: {elapsed} seconds")
    print("Outputs saved in /output directory")
    print("-" * 60 + "\n")


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def main() -> None:

    start_time = time.time()

    try:
        ensure_directories()
        print_banner()

        flow = ResearchFlow()
        flow.kickoff()

        print_completion_summary(start_time)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")

    except Exception as e:
        print(f"\nUnexpected system error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
