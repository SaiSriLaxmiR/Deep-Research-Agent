"""
agents/researcher.py

The Researcher agent runs once per sub-question, in parallel.

This is the most technically interesting file in the project because
it uses LangGraph's Send API to spawn N copies of itself at runtime —
one per sub-question — without knowing N at compile time.

Responsibilities:
- Receive one sub-question from the Planner's output
- Generate a focused search query for it
- Call DuckDuckGo directly (no tool calling — avoids Groq's flakiness)
- Summarise the results into a structured finding
- Append to state["findings"] (Annotated + operator.add merges all N)

What it reads:  one SubQuestion dict (passed via Send API)
What it writes: findings (appended, not overwritten)
"""

import os
import time
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ddgs import DDGS

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)


# ── Direct DuckDuckGo search — no tool calling ────────────────────────────────
# We call DuckDuckGo as a plain Python function.
# The LLM generates a search query string, we pass it here, get results back.
# No @tool decorator, no function_tool, no JSON schema — just Python.

def search_web(query: str, max_results: int = 5) -> List[dict]:
    """
    Search DuckDuckGo and return a list of results.
    Returns empty list on failure — never crashes the graph.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            time.sleep(1)  # polite delay to avoid rate limiting
            return results
    except Exception as e:
        print(f"[researcher] DuckDuckGo error: {e}")
        return []


# ── Prompts ───────────────────────────────────────────────────────────────────

QUERY_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are a research assistant generating a web search query.

Original research question: {original_query}
Sub-question to research: {sub_question}
Focus area: {focus}

Generate ONE specific, targeted web search query that will find
the most relevant and recent information for this sub-question.

Rules:
- Do NOT use site: filters
- Do NOT use OR operators
- Keep it simple — 5 to 8 words maximum
- Write it as a natural search phrase

Reply with ONLY the search query — no explanation, no quotes.
""")

SUMMARISE_PROMPT = ChatPromptTemplate.from_template("""
You are a research analyst. Summarise these web search results
to answer the following sub-question.

Sub-question: {sub_question}

Search results:
{search_results_text}

Instructions:
- Write a clear 2-3 sentence summary that directly answers the sub-question
- Extract 3-5 specific key facts as bullet points
- Only use information from the search results provided
- If the results don't answer the question well, say so clearly

Reply with EXACTLY this format:

SUMMARY: <2-3 sentence answer to the sub-question>

KEY_FACTS:
- <fact 1>
- <fact 2>
- <fact 3>
""")


def _format_search_results(results: List[dict]) -> str:
    """Format DuckDuckGo results into a readable text block for the LLM."""
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        body = r.get("body", "No content")
        href = r.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")
    return "\n\n".join(lines)


def _parse_researcher_output(text: str) -> tuple[str, List[str]]:
    """Parse SUMMARY and KEY_FACTS from the LLM's output."""
    summary = ""
    key_facts = []
    in_facts = False

    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
        elif line.startswith("KEY_FACTS:"):
            in_facts = True
        elif in_facts and line.startswith("-"):
            fact = line.lstrip("- ").strip()
            if fact:
                key_facts.append(fact)

    return summary, key_facts


# ── Main node function ────────────────────────────────────────────────────────

def researcher_node(state: dict) -> dict:
    """
    LangGraph node for one Researcher agent instance.

    When called via the Send API, state contains:
    - "sub_question": one SubQuestion dict
    - "query": the original research query (for context)

    Returns findings appended to the findings list.
    Because findings uses Annotated[List, operator.add],
    all parallel instances merge their results automatically.
    """
    sub_question = state["sub_question"]
    original_query = state["query"]

    sq_id = sub_question["id"]
    sq_text = sub_question["question"]
    sq_focus = sub_question["focus"]

    print(f"\n[researcher:{sq_id}] researching: '{sq_text[:60]}...'")

    # Step 1: Generate a focused search query using the LLM
    query_response = llm.invoke(
        QUERY_GENERATION_PROMPT.format_messages(
            original_query=original_query,
            sub_question=sq_text,
            focus=sq_focus,
        )
    )
    search_query = query_response.content.strip()
    print(f"[researcher:{sq_id}] search query: '{search_query}'")

    # Step 2: Search DuckDuckGo directly — no tool calling
    raw_results = search_web(search_query, max_results=5)
    print(f"[researcher:{sq_id}] got {len(raw_results)} results")

    # Step 3: Format results for the LLM
    results_text = _format_search_results(raw_results)

    # Step 4: Summarise the results with the LLM
    summary_response = llm.invoke(
        SUMMARISE_PROMPT.format_messages(
            sub_question=sq_text,
            search_results_text=results_text,
        )
    )
    summary, key_facts = _parse_researcher_output(
        summary_response.content.strip()
    )

    # Step 5: Extract source URLs
    sources = [r.get("href", "") for r in raw_results if r.get("href")]

    # Step 6: Build the SearchResult records for state
    search_results = [
        {
            "sub_question_id": sq_id,
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw_results
    ]

    # Step 7: Build the ResearchFinding
    finding = {
        "sub_question_id": sq_id,
        "sub_question": sq_text,
        "summary": summary or f"Research completed for: {sq_text}",
        "key_facts": key_facts or ["No specific facts extracted"],
        "sources": sources,
    }

    print(f"[researcher:{sq_id}] done — summary: '{summary[:80]}...'")

    # Return — operator.add will merge this with other parallel findings
    return {
        "search_results": search_results,
        "findings": [finding],
    }


# ── Send API dispatcher ───────────────────────────────────────────────────────
# This function is called by the graph to dispatch parallel researcher nodes.
# It reads sub_questions from state and returns one Send object per question.
# LangGraph runs all of them in parallel automatically.

def dispatch_researchers(state: dict):
    """
    Called by add_conditional_edges to spawn parallel researcher agents.

    Returns a list of Send objects — one per sub-question.
    LangGraph runs all of them simultaneously.

    This is the Send API pattern — the key to dynamic parallelism.
    """
    from langgraph.types import Send

    sub_questions = state["sub_questions"]
    original_query = state["query"]

    print(f"\n[dispatcher] spawning {len(sub_questions)} parallel researchers")

    return [
        Send(
            "researcher",        # which node to send to
            {
                "sub_question": sq,          # one sub-question per agent
                "query": original_query,     # original query for context
                "search_results": [],        # required by ResearchState
                "findings": [],              # required by ResearchState
            }
        )
        for sq in sub_questions
    ]


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_state = {
        "sub_question": {
            "id": "sq_1",
            "question": "How are large language models automating software engineering tasks?",
            "focus": "Task Automation",
        },
        "query": "What is the impact of LLMs on software engineering jobs?",
        "search_results": [],
        "findings": [],
    }

    print("Testing researcher node standalone...\n")
    result = researcher_node(test_state)

    print(f"\n── Result ──────────────────────────────────────")
    finding = result["findings"][0]
    print(f"Sub-question: {finding['sub_question']}")
    print(f"Summary: {finding['summary']}")
    print(f"Key facts:")
    for f in finding["key_facts"]:
        print(f"  - {f}")
    print(f"Sources ({len(finding['sources'])}):")
    for s in finding["sources"]:
        print(f"  {s}")