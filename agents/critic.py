"""
agents/critic.py

The Critic evaluates the quality of the merged research context.
It identifies gaps, contradictions, or unverified claims and
decides whether the pipeline needs another research iteration.

Responsibilities:
- Read the merged context and original query
- Score research quality from 0.0 to 1.0
- Identify specific gaps, contradictions, or weak areas
- Decide: is this good enough to write a report, or do we need more?

What it reads:  state["merged_context"], state["query"],
                state["sub_questions"], state["iteration"]
What it writes: state["critic_notes"], state["quality_score"],
                state["needs_more_research"], state["status"]
"""

import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

CRITIC_PROMPT = ChatPromptTemplate.from_template("""
You are a critical research evaluator. Assess the quality and
completeness of the following research findings.

Original query: {query}

Research context:
{merged_context}

Evaluate the research on these dimensions:
1. Does it fully answer the original query?
2. Are there important angles that were missed?
3. Are there any contradictions or unverified claims?
4. Is the evidence specific enough (facts, numbers, examples)?

Reply with EXACTLY this format:

SCORE: <a number between 0.0 and 1.0>
NEEDS_MORE: <YES or NO>

GAPS:
- TYPE: gap | SEVERITY: <low/medium/high> | NOTE: <description>
- TYPE: gap | SEVERITY: <low/medium/high> | NOTE: <description>

CONTRADICTIONS:
- TYPE: contradiction | SEVERITY: <low/medium/high> | NOTE: <description>

VERDICT: <one sentence overall assessment>

Rules:
- Score above 0.75 means research is good enough to write a final report
- Score below 0.75 means NEEDS_MORE should be YES
- If this is iteration 2 or more, always set NEEDS_MORE to NO
  (we never run more than 2 research iterations)
- Only list real gaps — if there are none, write GAPS: none
- Only list real contradictions — if none, write CONTRADICTIONS: none
""")


def _parse_critic_output(text: str) -> tuple[float, bool, list, str]:
    """
    Parse the Critic's structured output.

    Returns:
        quality_score: float (0.0 to 1.0)
        needs_more: bool
        critic_notes: list of CriticNote dicts
        verdict: str
    """
    quality_score = 0.75
    needs_more = False
    critic_notes = []
    verdict = ""

    lines = text.strip().splitlines()

    for line in lines:
        line = line.strip()

        # Score
        if line.startswith("SCORE:"):
            try:
                score_str = line.replace("SCORE:", "").strip()
                quality_score = float(score_str)
                quality_score = max(0.0, min(1.0, quality_score))
            except ValueError:
                pass

        # Needs more research
        elif line.startswith("NEEDS_MORE:"):
            value = line.replace("NEEDS_MORE:", "").strip().upper()
            needs_more = value == "YES"

        # Gap or contradiction notes
        elif line.startswith("- TYPE:"):
            note_match = re.match(
                r"-\s*TYPE:\s*(\w+)\s*\|\s*SEVERITY:\s*(\w+)\s*\|\s*NOTE:\s*(.+)",
                line,
                re.IGNORECASE,
            )
            if note_match:
                critic_notes.append({
                    "type": note_match.group(1).lower().strip(),
                    "severity": note_match.group(2).lower().strip(),
                    "description": note_match.group(3).strip(),
                })

        # Verdict
        elif line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()

    return quality_score, needs_more, critic_notes, verdict


def critic_node(state: dict) -> dict:
    """
    LangGraph node for the Critic agent.

    Reads:  state["merged_context"], state["query"], state["iteration"]
    Writes: state["critic_notes"], state["quality_score"],
            state["needs_more_research"], state["status"]
    """
    merged_context = state.get("merged_context", "")
    query = state["query"]
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 2)

    print(f"\n[critic] evaluating research quality (iteration {iteration})")

    if not merged_context or merged_context == "No research findings were collected.":
        print("[critic] no context to evaluate")
        return {
            "critic_notes": [{
                "type": "gap",
                "severity": "high",
                "description": "No research findings were collected",
            }],
            "quality_score": 0.0,
            "needs_more_research": iteration < max_iterations,
            "status": "writing" if iteration >= max_iterations else "planning",
        }

    response = llm.invoke(
        CRITIC_PROMPT.format_messages(
            query=query,
            merged_context=merged_context,
        )
    )
    raw_text = response.content.strip()
    print(f"[critic] LLM response received ({len(raw_text)} chars)")

    quality_score, needs_more, critic_notes, verdict = _parse_critic_output(raw_text)

    # Hard cap: never loop more than max_iterations times
    if iteration >= max_iterations:
        needs_more = False
        print(f"[critic] max iterations ({max_iterations}) reached — forcing to write")

    next_status = "planning" if needs_more else "writing"

    print(f"[critic] score={quality_score:.2f} needs_more={needs_more}")
    print(f"[critic] {len(critic_notes)} notes found")
    print(f"[critic] verdict: '{verdict}'")
    print(f"[critic] next status: '{next_status}'")

    return {
        "critic_notes": critic_notes,
        "quality_score": quality_score,
        "needs_more_research": needs_more,
        "status": next_status,
    }


# ── Routing function (used by graph.py) ──────────────────────────────────────

def route_after_critic(state: dict) -> str:
    """
    Routing function for the conditional edge after the Critic.

    Returns:
        "loop"  → go back to Planner for another research iteration
        "write" → proceed to the Writer agent
    """
    if state.get("needs_more_research") and state.get("iteration", 1) < state.get("max_iterations", 2):
        return "loop"
    return "write"


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_context = """
    Large language models are transforming software engineering by automating
    code generation, testing, and debugging. GitHub Copilot completes 40% of
    code for some developers. Studies show 55% productivity gains.
    Engineers now need prompt engineering skills and AI tool evaluation ability.
    Current evidence suggests augmentation rather than replacement of engineers.
    New roles like AI engineer and LLM ops specialist are emerging.
    """

    test_state = {
        "query": "What is the impact of LLMs on software engineering jobs?",
        "merged_context": test_context,
        "iteration": 1,
        "max_iterations": 2,
    }

    print("Testing critic node standalone...\n")
    result = critic_node(test_state)

    print(f"\n── Critic Result ───────────────────────────────────")
    print(f"Quality score:       {result['quality_score']:.2f}")
    print(f"Needs more research: {result['needs_more_research']}")
    print(f"Notes ({len(result['critic_notes'])}):")
    for note in result["critic_notes"]:
        print(f"  [{note['severity']}] {note['type']}: {note['description']}")
    print(f"Next status: {result['status']}")