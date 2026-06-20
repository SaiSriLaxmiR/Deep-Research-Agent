"""
frontend/app.py

Streamlit frontend for the Deep Research Agent.

Run with:
    streamlit run frontend/app.py

Make sure the FastAPI server is running first:
    uvicorn api.main:app --reload --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "polling" not in st.session_state:
    st.session_state.polling = False
if "result" not in st.session_state:
    st.session_state.result = None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Deep Research Agent")
st.caption("Multi-agent LangGraph system")
st.divider()

# ── Sidebar — query input ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Research Settings")

    query = st.text_area(
        "Research query",
        placeholder="e.g. How is AI changing healthcare diagnostics?",
        height=100,
    )

    depth = st.selectbox(
        "Research depth",
        options=["quick", "standard", "deep"],
        index=1,
        help="quick=3 sub-questions, standard=5, deep=8",
    )

    human_review = st.toggle(
        "Human review before report",
        value=False,
        help="Pause before the Writer so you can review the research context",
    )

    run_btn = st.button(
        "▶ Start Research",
        type="primary",
        disabled=st.session_state.polling,
        use_container_width=True,
    )

    st.divider()

    # Recent jobs
    st.subheader("Recent jobs")
    try:
        jobs_resp = requests.get(f"{API_BASE}/research", timeout=3)
        if jobs_resp.status_code == 200:
            jobs = jobs_resp.json()
            if jobs:
                for j in reversed(jobs[-5:]):
                    status_icon = {
                        "complete": "✅",
                        "running": "⏳",
                        "error": "❌",
                        "queued": "🕐",
                        "awaiting_approval": "👤",
                        "cancelled": "🚫",
                    }.get(j["status"], "•")
                    if st.button(
                        f"{status_icon} {j['query'][:30]}...",
                        key=f"job_{j['job_id']}",
                        use_container_width=True,
                    ):
                        st.session_state.job_id = j["job_id"]
                        st.rerun()
            else:
                st.caption("No jobs yet")
    except Exception:
        st.caption("API not reachable")

# ── Start research ─────────────────────────────────────────────────────────────
if run_btn and query.strip():
    try:
        resp = requests.post(
            f"{API_BASE}/research",
            json={
                "query": query.strip(),
                "depth": depth,
                "human_review": human_review,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.job_id = data["job_id"]
            st.session_state.polling = True
            st.session_state.result = None
            st.rerun()
        else:
            st.error(f"API error: {resp.status_code}")
    except Exception as e:
        st.error(f"Could not connect to API: {e}")

elif run_btn and not query.strip():
    st.sidebar.warning("Please enter a research query first.")

# ── Main area — status + result ───────────────────────────────────────────────
if st.session_state.job_id:
    job_id = st.session_state.job_id

    try:
        resp = requests.get(f"{API_BASE}/research/{job_id}", timeout=5)
        if resp.status_code != 200:
            st.error(f"Job {job_id} not found")
            st.session_state.job_id = None
        else:
            job = resp.json()
            status = job["status"]

            # ── Status bar ────────────────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Status", status.replace("_", " ").title())
            col2.metric("Quality score", f"{job['quality_score']:.0%}" if job["quality_score"] else "—")
            col3.metric("Sources", job["total_sources"] or "—")
            col4.metric("Iterations", job["iterations"] or "—")

            st.divider()

            # ── Running state ─────────────────────────────────────────────────
            if status in ("queued", "running"):
                st.session_state.polling = True
                with st.status("Research in progress...", expanded=True) as s:
                    st.write("🧠 Planner breaking down the query...")
                    st.write("🔍 Parallel researchers searching the web...")
                    st.write("🔗 Observer merging findings...")
                    st.write("🎯 Critic evaluating quality...")
                    st.write("✍️ Writer generating report...")
                    s.update(label="Working... (this takes 30-60 seconds)", state="running")
                time.sleep(5)
                st.rerun()

            # ── Awaiting approval ─────────────────────────────────────────────
            elif status == "awaiting_approval":
                st.session_state.polling = False
                st.warning("👤 Research complete — waiting for your approval before generating the report.")

                if job.get("merged_context"):
                    with st.expander("📋 Research context to review", expanded=True):
                        st.write(job.get("merged_context", ""))

                col_approve, col_reject = st.columns(2)
                with col_approve:
                    if st.button("✅ Approve — Generate Report", type="primary", use_container_width=True):
                        approve_resp = requests.post(
                            f"{API_BASE}/research/{job_id}/approve",
                            json={"approved": True},
                            timeout=5,
                        )
                        if approve_resp.status_code == 200:
                            st.session_state.polling = True
                            st.rerun()
                with col_reject:
                    if st.button("❌ Reject — Cancel", use_container_width=True):
                        requests.post(
                            f"{API_BASE}/research/{job_id}/approve",
                            json={"approved": False},
                            timeout=5,
                        )
                        st.session_state.job_id = None
                        st.rerun()

            # ── Complete ──────────────────────────────────────────────────────
            elif status == "complete":
                st.session_state.polling = False
                st.session_state.result = job

                st.success("✅ Research complete!")

                # Table of contents
                if job.get("report_sections"):
                    st.subheader("Contents")
                    cols = st.columns(len(job["report_sections"]))
                    for i, section in enumerate(job["report_sections"]):
                        cols[i].markdown(f"**{section}**")
                    st.divider()

                # The report
                if job.get("final_report"):
                    st.markdown(job["final_report"])
                    st.divider()

                    # Download button
                    st.download_button(
                        label="⬇️ Download report as markdown",
                        data=job["final_report"],
                        file_name=f"research_{job_id}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

            # ── Error ─────────────────────────────────────────────────────────
            elif status == "error":
                st.session_state.polling = False
                st.error("❌ Research failed. Check the API logs for details.")
                st.code(job.get("error", "Unknown error"))

            # ── Cancelled ─────────────────────────────────────────────────────
            elif status == "cancelled":
                st.session_state.polling = False
                st.info("Research was cancelled.")

    except Exception as e:
        st.error(f"Could not reach API: {e}")
        st.session_state.polling = False

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    ### How it works

    1. **Enter a research query** in the sidebar
    2. **Choose depth** — quick (3 sub-questions), standard (5), or deep (8)
    3. **Click Start Research** — the multi-agent pipeline begins

    The system will:
    - Break your query into focused sub-questions
    - Research each one in **parallel** using web search
    - Merge and evaluate the findings
    - Write a structured report with proper citations

    ---

    ### Architecture

    | Agent | Role |
    |---|---|
    | 🧠 Planner | Breaks query into N sub-questions |
    | 🔍 Researcher × N | Parallel web search per sub-question |
    | 🔗 Observer | Merges findings, builds citation map |
    | 🎯 Critic | Scores quality, triggers re-research if needed |
    | ✍️ Writer | Generates final markdown report |
    """)