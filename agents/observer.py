"""
agents/observer.py

The Observer runs after all parallel Researcher agents finish.
It receives all findings merged into state["findings"] and
compresses them into a single unified context block.

Responsibilities:
- Read all N findings from parallel researchers
- Build a citation map tracking which sources answered which sub-question
- Compress everything into one merged_context text block
- Count total unique sources used

What it reads:  state["findings"], state["search_results"], state["query"]
What it writes: state["merged_context"], state["citation_map"],
                state["total_sources"], state["status"]
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

OBSERVER_PROMPT = ChatPromptTemplate.from_template("""
You are a research synthesiser. You have received findings from
multiple parallel research agents, each covering a different angle
of the same topic.

Original research query: {query}

Research findings:
{findings_text}

Your job:
- Synthesise all findings into ONE unified, coherent context block
- Remove redundancy — if two agents found the same fact, state it once
- Preserve all unique insights from each agent
- Keep all key facts — do not discard information
- Write in flowing prose, not bullet points
- Aim for 300-500 words

Reply with ONLY the synthesised context — no preamble, no labels.
""")


def _format_findings_for_observer(findings: list) -> str:
    """Format all findings into a readable block for the Observer LLM."""
    blocks = []
    for i, f in enumerate(findings, 1):
        facts_text = "\n".join(f"  - {fact}" for fact in f.get("key_facts", []))
        block = (
            f"Finding {i} [{f.get('sub_question_id', '')}]\n"
            f"Sub-question: {f.get('sub_question', '')}\n"
            f"Summary: {f.get('summary', '')}\n"
            f"Key facts:\n{facts_text}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _build_citation_map(findings: list) -> tuple[dict, int]:
    """
    Build a citation map: { sub_question_id: [url1, url2, ...] }
    Also returns total unique source count.
    """
    citation_map = {}
    all_urls = set()

    for f in findings:
        sq_id = f.get("sub_question_id", "unknown")
        sources = f.get("sources", [])
        citation_map[sq_id] = sources
        all_urls.update(s for s in sources if s)

    return citation_map, len(all_urls)


def observer_node(state: dict) -> dict:
    """
    LangGraph node for the Observer agent.

    Reads:  state["findings"], state["query"]
    Writes: state["merged_context"], state["citation_map"],
            state["total_sources"], state["status"]
    """
    findings = state.get("findings", [])
    query = state["query"]

    print(f"\n[observer] merging {len(findings)} findings")

    if not findings:
        print("[observer] no findings to merge — returning empty context")
        return {
            "merged_context": "No research findings were collected.",
            "citation_map": {},
            "total_sources": 0,
            "status": "critiquing",
        }

    # Build citation map and count unique sources
    citation_map, total_sources = _build_citation_map(findings)
    print(f"[observer] {total_sources} unique sources across all findings")

    # Format findings for the LLM
    findings_text = _format_findings_for_observer(findings)

    # Synthesise all findings into one unified context block
    response = llm.invoke(
        OBSERVER_PROMPT.format_messages(
            query=query,
            findings_text=findings_text,
        )
    )
    merged_context = response.content.strip()

    print(f"[observer] merged context: {len(merged_context)} chars")
    print(f"[observer] citation map keys: {list(citation_map.keys())}")

    return {
        "merged_context": merged_context,
        "citation_map": citation_map,
        "total_sources": total_sources,
        "status": "critiquing",
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate what parallel researchers would have produced
    test_findings = [
        {
            "sub_question_id": "sq_1",
            "sub_question": "How are LLMs automating software engineering tasks?",
            "summary": "LLMs automate code generation, testing, and debugging, reducing developer workload significantly.",
            "key_facts": [
                "LLMs can generate boilerplate code in seconds",
                "GitHub Copilot completes 40% of code for some developers",
                "Automated testing using LLMs reduces QA time by up to 30%",
            ],
            "sources": [
                "https://github.blog/2023-06-27-survey-reveals-ais-impact-on-the-developer-experience/",
                "https://arxiv.org/abs/2308.11432",
            ],
        },
        {
            "sub_question_id": "sq_2",
            "sub_question": "What new skills do software engineers need for LLM-era work?",
            "summary": "Engineers now need prompt engineering, AI orchestration, and the ability to evaluate LLM outputs critically.",
            "key_facts": [
                "Prompt engineering is now a core skill for developers",
                "Understanding LLM limitations is critical to avoid bugs",
                "AI tool evaluation and selection is a growing responsibility",
            ],
            "sources": [
                "https://www.acm.org/articles/bulletins/2023/llm-skills",
                "https://stackoverflow.blog/2023/ai-skills-developers",
            ],
        },
        {
            "sub_question_id": "sq_3",
            "sub_question": "Will LLMs replace or augment software engineers?",
            "summary": "Current evidence suggests augmentation over replacement — LLMs handle repetitive tasks while humans focus on architecture and judgment.",
            "key_facts": [
                "No major company has reported replacing engineers with LLMs",
                "Developer productivity increased 55% in studies using LLM tools",
                "New roles like AI engineer and LLM ops are emerging",
            ],
            "sources": [
                "https://hbr.org/2023/09/how-ai-is-changing-software-development",
                "https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/ai-developer",
            ],
        },
    ]

    test_state = {
        "query": "What is the impact of LLMs on software engineering jobs?",
        "findings": test_findings,
        "search_results": [],
    }

    print("Testing observer node standalone...\n")
    result = observer_node(test_state)

    print(f"\n── Merged Context ──────────────────────────────────")
    print(result["merged_context"])
    print(f"\n── Citation Map ────────────────────────────────────")
    for sq_id, urls in result["citation_map"].items():
        print(f"  {sq_id}: {len(urls)} sources")
    print(f"\nTotal unique sources: {result['total_sources']}")