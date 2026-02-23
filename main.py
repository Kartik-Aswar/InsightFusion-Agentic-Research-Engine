import sys
import os
import time

from dotenv import load_dotenv
load_dotenv()



##########################################################   for logs #########################################
# Create logs folder
from datetime import datetime
os.makedirs("logs", exist_ok=True)

# Unique log file per run
log_filename = datetime.now().strftime("logs/run_%Y%m%d_%H%M%S.log")

log_file = open(log_filename, "w", encoding="utf-8")

# Redirect terminal output → file + terminal
class Tee:
    def __init__(self, *files):
        self.files = files
        self._original_stream = files[0]

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    #  REQUIRED for Rich / CrewAI
    def isatty(self):
        return self._original_stream.isatty()

    #  Some libs check this too
    @property
    def encoding(self):
        return self._original_stream.encoding


sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

##################################################### for logs end #######################################################

# Ensure root path is included
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flows.research_flow import ResearchFlow


# ---------------------------------------------------------
# SYSTEM INITIALIZATION
# ---------------------------------------------------------

def ensure_directories():
    """
    Ensure required system directories exist.
    """

    required_dirs = [
        "input_pdfs",
        "output",
        "vector_db"
    ]

    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)


def print_banner():
    print("\n" + "=" * 60)
    print("  AGENTIC AI DEEP RESEARCH SYSTEM")
    print("=" * 60)
    print("Autonomous | Multi-Agent | Self-Correcting")
    print("=" * 60 + "\n")


def print_completion_summary(start_time):
    elapsed = round(time.time() - start_time, 2)

    print("\n" + "-" * 60)
    print("Research Session Completed")
    print(f"Execution Time: {elapsed} seconds")
    print("Outputs saved in /output directory")
    print("-" * 60 + "\n")


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def main():

    start_time = time.time()

    try:
        ensure_directories()
        print_banner()

        flow = ResearchFlow()
        flow.kickoff()

        print_completion_summary(start_time)

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")

    except Exception as e:
        print("\nUnexpected system error occurred:")
        print(str(e))


if __name__ == "__main__":
    main()
