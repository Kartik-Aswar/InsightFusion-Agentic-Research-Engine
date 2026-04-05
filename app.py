"""
Streamlit UI for InsightFusion — Agentic AI Deep Research System.
"""

import streamlit as st
import base64
import sys
import os
import json
import shutil
from datetime import datetime

# =================================================
# TERMINAL LOGGING SETUP (STREAMLIT SAFE)
# =================================================

if "logging_initialized" not in st.session_state:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.logging_util import setup_logging
    log_filename = setup_logging()
    st.session_state.logging_initialized = True
    st.session_state.log_filename = log_filename
    print("=" * 60)
    print("INSIGHTFUSION — AGENTIC AI DEEP RESEARCH (STREAMLIT)")
    print("=" * 60)
    print(f"Logging started → {log_filename}")
    print("=" * 60)

# =================================================
# IMPORTS AFTER LOGGING
# =================================================
from flows.research_flow import ResearchFlow

# -------------------------------------------------
# PAGE CONFIG & CSS
# -------------------------------------------------
st.set_page_config(
    page_title="InsightFusion — AI Research",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    /* Gradient text for main title */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #4f46e5, #9333ea, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 5px;
    }
    
    /* Modernized metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #4f46e5 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #6b7280 !important;
    }
    
    /* Sleek container styles */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Subtle hover effect for elements */
    .stExpander:hover {
        border-color: #9333ea !important;
        transition: border-color 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.title("⚙️ System Controls")

with st.sidebar.expander("🔑 Configure API Keys", expanded=False):
    st.markdown("Enter your API credentials to power the multi-agent system engines.")
    gemini_key = st.text_input("Gemini API Key", type="password")
    serper_key = st.text_input("Serper API Key", type="password")

if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["GOOGLE_API_KEY"] = gemini_key
if serper_key:
    os.environ["SERPER_API_KEY"] = serper_key

st.sidebar.divider()

# PDF UPLOAD
st.sidebar.subheader("📄 Upload Academic Vectors")
uploaded_files = st.sidebar.file_uploader(
    "Drag and drop academic papers or documents",
    type=["pdf"],
    accept_multiple_files=True,
)

os.makedirs("input_pdfs", exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join("input_pdfs", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
            
    st.sidebar.success(f"✅ {len(uploaded_files)} PDFs securely uploaded")
    for file in uploaded_files:
        st.sidebar.caption(f"📎 {file.name}")

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear Pipeline Data", use_container_width=True):
    shutil.rmtree("input_pdfs", ignore_errors=True)
    shutil.rmtree("vector_db", ignore_errors=True)
    shutil.rmtree("output", ignore_errors=True)
    os.makedirs("input_pdfs", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    st.sidebar.success("✅ Output, PDFs, and Vector DB fully cleared")

# -------------------------------------------------
# TOP NAVIGATION BAR
# -------------------------------------------------

nav_selection = st.radio(
    "Navigation", 
    ["🏠 Research Engine", "👨‍💻 Developers", "📞 Contact Us"], 
    horizontal=True, 
    label_visibility="collapsed"
)

if nav_selection == "👨‍💻 Developers":
    st.markdown("<div style='text-align: center;'><h2 class='gradient-text'>Meet the Developers</h2></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.write("This project was engineered as a comprehensive demonstration of Multi-Agent Architectures, Deep Research Retrieval-Augmented Generation (RAG), and autonomous conflict resolution protocols.")
    st.info("The system utilizes the **CrewAI** framework for orchestration, **ChromaDB / Sentence-Transformers** for semantic vector search, and generic LLM bindings via **LiteLLM**.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    team = [
        {"name": "Achyut Shinde", "email": "avshinde018@gmail.com", "prefix": "Achyut Image"},
        {"name": "Bhavesh Choudhary", "email": "choudharybhavesh191@gmail.com", "prefix": "Bhavesh Image"},
        {"name": "Kartik Aswar", "email": "kartikaswar03@gmail.com", "prefix": "Kartik Image"}
    ]
    
    cols = [col1, col2, col3]
    
    for i, member in enumerate(team):
        with cols[i]:
            img_path = None
            jpg_path = f"assets/team/{member['prefix']}.jpg"
            png_path = f"assets/team/{member['prefix']}.png"
            
            if os.path.exists(jpg_path):
                img_path = jpg_path
            elif os.path.exists(png_path):
                img_path = png_path
            
            if img_path:
                with open(img_path, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode()
                ext = img_path.split('.')[-1]
                src = f"data:image/{ext};base64,{b64}"
            else:
                src = "https://via.placeholder.com/300?text=" + member["name"].split()[0]
                
            st.markdown(f'''
                <div style="width: 100%; height: 320px; overflow: hidden; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <img src="{src}" style="width: 100%; height: 100%; object-fit: cover;">
                </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f"#### {member['name']}")
            st.markdown(f"**✉️ {member['email']}**")
            
    st.stop()

elif nav_selection == "📞 Contact Us":
    st.markdown("<div style='text-align: center;'><h2 class='gradient-text'>Contact & Support</h2></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Reach Out for Inquiries")
    st.write("For questions regarding the **InsightFusion** framework, API integrations, or academic implementations, please reach out securely.")
    
    with st.container(border=True):
        st.text_input("Name")
        st.text_input("Email")
        st.text_area("Message")
        st.button("Send Message", type="primary")
    st.stop()

# -------------------------------------------------
# MAIN TITLE AND RAG INPUT (Home Page)
# -------------------------------------------------

st.markdown("""
<div style="text-align: center;">
    <h1 class="gradient-text">🔬 InsightFusion</h1>
    <h3>Agentic Research Engine</h3>
    <p>Autonomous deep-research powered by <b>multi-agent reasoning</b>, <b>hybrid RAG capability</b>, and <b>self-correcting logic</b>.</p>
</div>
""", unsafe_allow_html=True)

st.write("") # Spacer

with st.container(border=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter research question",
            placeholder="🔍 Example: What are the primary architectural differences between AutoGPT and LangGraph?",
            label_visibility="collapsed"
        )
    with col2:
        run_button = st.button("🚀 Run Research", type="primary", use_container_width=True)


# -------------------------------------------------
# RUN RESEARCH PIPELINE
# -------------------------------------------------

if run_button and query:
    print("\n" + "=" * 60)
    print(f"Research Query: {query}")
    print("=" * 60)

    with st.status("🔄 Orchestrating Autonomous Agents...", expanded=True) as status:
        st.write("Initializing Web Scouts and Document Specialists...")
        
        try:
            flow = ResearchFlow()
            flow.state.query = query
            st.write("Executing Multi-Agent reasoning tasks in the background...")
            state = flow.kickoff()
            status.update(label="✅ Deep Research completed successfully!", state="complete", expanded=False)
            
        except Exception as e:
            status.update(label=f"❌ Research failed: {str(e)}", state="error", expanded=True)
            import traceback
            traceback.print_exc()

    print("\n===== Research Completed =====")


elif run_button and not query:
    st.warning("⚠️ Please specify a research question to commence operations.")


# -------------------------------------------------
# DISPLAY OUTPUTS
# -------------------------------------------------

report_path = "output/final_report.txt"
trace_path = "output/reasoning_trace.json"
conflict_path = "output/conflicts.json"
summary_path = "output/summary.json"

if os.path.exists(report_path):
    st.write("") # Spacer

    tabs = st.tabs(["📝 Final Report", "📊 Research Summary", "🛡️ Reliability & References", "⚡ Conflict Audits"])

    # REPORT TAB
    with tabs[0]:
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

        with st.container(border=True):
            st.markdown(report)

        st.write("")
        st.download_button(
            label="📥 Download Markdown Report",
            data=report,
            file_name="InsightFusion_Report.md",
            mime="text/markdown",
            type="primary"
        )

    # SUMMARY TAB
    with tabs[1]:
        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)

            st.markdown("### 📈 Pipeline Telemetry")
            col1, col2, col3 = st.columns(3)

            with col1:
                with st.container(border=True):
                    st.metric("System Confidence", f"{summary.get('confidence_score', 0)}%")
            with col2:
                with st.container(border=True):
                    st.metric("Research Recursions", summary.get("recursion_count", 0))
            with col3:
                with st.container(border=True):
                    st.metric("Verified Web Claims", summary.get("total_web_claims", "N/A"))

            st.divider()
            with st.expander("Raw System Output Json"):
                st.json(summary)

    # RELIABILITY AND REASONING TRACE TAB
    with tabs[2]:
        rel_path = "output/reliability_report.txt"
        if os.path.exists(rel_path):
            with open(rel_path, "r", encoding="utf-8") as f:
                rel_data = f.read()
            st.markdown(rel_data)
            st.divider()

        if os.path.exists(trace_path):
            with open(trace_path, encoding="utf-8") as f:
                trace = json.load(f)

            st.markdown("### 🧠 Agent Chain of Thought")
            st.info("The exact logical steps and actions the agents took to arrive at the final conclusion.")
            for i, step in enumerate(trace):
                with st.expander(f"Step {i+1} Trace", expanded=False):
                    st.write(step)

    # CONFLICTS TAB
    with tabs[3]:
        st.markdown("### ⚡ Historical Evidence Conflicts")
        if os.path.exists(conflict_path):
            with open(conflict_path, encoding="utf-8") as f:
                conflicts = json.load(f)

            if conflicts:
                st.warning("The agents detected the following conflicting claims during research and attempted to self-correct them dynamically.")
                for conflict in conflicts:
                    severity = conflict.get("severity", "Low")
                    issue = conflict.get("issue", "Unknown")
                    sources = ", ".join(conflict.get('conflicting_sources', []))
                    
                    if severity == "High":
                        st.error(f"**High Severity Conflict:** {issue}\n\n**Sources Involved:** {sources}")
                    elif severity == "Medium":
                        st.warning(f"**Medium Severity Conflict:** {issue}\n\n**Sources Involved:** {sources}")
                    else:
                        st.info(f"**Low Severity Conflict:** {issue}\n\n**Sources Involved:** {sources}")
            else:
                st.success(" No structural conflicts detected across evidence sources during this research run.")