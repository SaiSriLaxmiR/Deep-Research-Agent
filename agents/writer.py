"""
agents/writer.py

The Writer is the final agent in the pipeline.
It reads the merged context, citation map, and critic notes,
then produces a structured markdown research report.

Responsibilities:
- Write a complete, well-structured research report in markdown
- Add proper citations from the citation map
- Acknowledge gaps identified by the Critic
- Extract section titles for the frontend to display

What it reads:  state["merged_context"], state["citation_map"],
                state["critic_notes"], state["query"],
                state["total_sources"], state["sub_questions"]
What it writes: state["final_report"], state["report_sections"],
                state["status"]
"""

import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

WRITER_PROMPT = ChatPromptTemplate.from_template("""
You are a professional research writer. Write a comprehensive,
well-structured research report based on the provided research context.

Original query: {query}

Research context:
{merged_context}

Sources used ({total_sources} total):
{sources_text}

Known gaps in the research:
{gaps_text}

Write a complete research report in markdown format with these sections:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points with the most important discoveries)
3. Detailed Analysis (3-4 paragraphs covering different angles)
4. Limitations & Gaps (honest acknowledgment of what was not covered)
5. Sources (numbered list of URLs used)

Rules:
- Use proper markdown headers (##, ###)
- Be specific — use numbers and facts from the context
- Reference sources inline where relevant as [Source N]
- Keep the tone professional and objective
- Total length: 500-800 words
- Start directly with the report — no preamble
""")


def _format_sources_for_writer(citation_map: dict) -> tuple[str, dict]:
    """
    Flatten citation map into a numbered source list.

    Returns:
        sources_text: formatted string for the prompt
        source_index: {url: number} for inline citation
    """
    all_urls = []
    seen = set()

    for urls in citation_map.values():
        for url in urls:
            if url and url not in seen:
                seen.add(url)
                all_urls.append(url)

    source_index = {url: i + 1 for i, url in enumerate(all_urls)}
    sources_text = "\n".join(
        f"{i + 1}. {url}" for i, url in enumerate(all_urls)
    )

    return sources_text or "No sources available", source_index


def _format_gaps_for_writer(critic_notes: list) -> str:
    """Format critic notes into a readable gaps summary."""
    if not critic_notes:
        return "No significant gaps identified."

    gaps = [
        note for note in critic_notes
        if note.get("type") in ("gap", "unverified")
    ]

    if not gaps:
        return "No significant gaps identified."

    return "\n".join(
        f"- [{note.get('severity', 'low').upper()}] {note.get('description', '')}"
        for note in gaps
    )


def _extract_sections(report: str) -> list[str]:
    """
    Extract markdown section titles from the report.
    Used by the frontend to build a table of contents.
    """
    sections = []
    for line in report.splitlines():
        line = line.strip()
        if line.startswith("## "):
            sections.append(line.replace("## ", "").strip())
        elif line.startswith("### "):
            sections.append(line.replace("### ", "").strip())
    return sections


def writer_node(state: dict) -> dict:
    """
    LangGraph node for the Writer agent.

    Reads:  state["merged_context"], state["citation_map"],
            state["critic_notes"], state["query"],
            state["total_sources"]
    Writes: state["final_report"], state["report_sections"],
            state["status"]
    """
    merged_context = state.get("merged_context", "")
    citation_map = state.get("citation_map", {})
    critic_notes = state.get("critic_notes", [])
    query = state["query"]
    total_sources = state.get("total_sources", 0)

    print(f"\n[writer] generating final report for query: '{query[:60]}...'")
    print(f"[writer] context: {len(merged_context)} chars")
    print(f"[writer] sources: {total_sources}")
    print(f"[writer] critic notes: {len(critic_notes)}")

    # Format sources and gaps for the prompt
    sources_text, source_index = _format_sources_for_writer(citation_map)
    gaps_text = _format_gaps_for_writer(critic_notes)

    # Generate the report
    response = llm.invoke(
        WRITER_PROMPT.format_messages(
            query=query,
            merged_context=merged_context,
            total_sources=total_sources,
            sources_text=sources_text,
            gaps_text=gaps_text,
        )
    )
    final_report = response.content.strip()

    # Extract section titles for frontend navigation
    report_sections = _extract_sections(final_report)

    print(f"[writer] report generated: {len(final_report)} chars")
    print(f"[writer] sections: {report_sections}")

    return {
        "final_report": final_report,
        "report_sections": report_sections,
        "status": "complete",
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_state = {
        "query": "What is the impact of LLMs on software engineering jobs?",
        "merged_context": """
            Large language models are transforming software engineering by automating
            code generation, testing, and debugging. GitHub Copilot completes 40% of
            code for some developers. Studies show 55% productivity gains when using
            LLM-powered tools. Engineers now need prompt engineering skills and AI
            tool evaluation ability. Current evidence suggests augmentation rather
            than replacement — no major company has replaced engineers with LLMs.
            New roles like AI engineer and LLM ops specialist are emerging rapidly.
            Developer surveys show 70% now use AI tools daily in their workflow.
        """,
        "citation_map": {
            "sq_1": [
                "https://github.blog/2023-copilot-survey",
                "https://arxiv.org/abs/2308.11432",
            ],
            "sq_2": [
                "https://stackoverflow.blog/2023/ai-skills",
            ],
            "sq_3": [
                "https://hbr.org/2023/ai-software-engineering",
                "https://mckinsey.com/ai-developer-2023",
            ],
        },
        "critic_notes": [
            {
                "type": "gap",
                "severity": "medium",
                "description": "Long-term job displacement not fully explored",
            },
            {
                "type": "gap",
                "severity": "low",
                "description": "Industry-specific impacts not covered",
            },
        ],
        "total_sources": 5,
        "sub_questions": [],
    }

    print("Testing writer node standalone...\n")
    result = writer_node(test_state)

    print(f"\n{'='*60}")
    print(result["final_report"])
    print(f"\n{'='*60}")
    print(f"Sections: {result['report_sections']}")
    print(f"Status: {result['status']}")