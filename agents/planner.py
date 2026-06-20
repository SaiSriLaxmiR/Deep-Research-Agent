"""
agents/planner.py

The Planner is the first agent in the pipeline.
It reads the user's query and depth setting, then generates
a structured list of sub-questions that together cover the topic.

Responsibilities:
- Analyse the query and identify the key angles to research
- Generate N sub-questions based on depth (3/5/8)
- Give each sub-question a clear focus label
- Explain its reasoning so the Observer can use it later

What it reads from state:  query, depth
What it writes to state:   sub_questions, plan_reasoning, status
"""

import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# ── LLM setup ─────────────────────────────────────────────────────────────────
# The Planner uses the same Groq model as all other agents.
# It does NOT use tool calling — just plain text generation + parsing.

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,   # low temperature = more consistent structured output
)

# ── Depth → sub-question count mapping ───────────────────────────────────────
DEPTH_TO_COUNT = {
    "quick": 3,
    "standard": 5,
    "deep": 8,
}

# ── Prompt ────────────────────────────────────────────────────────────────────
PLANNER_PROMPT = ChatPromptTemplate.from_template("""
You are a research planning expert. Your job is to break a research query
into {n} focused sub-questions that together give complete coverage of the topic.

Research query: {query}

Rules:
- Each sub-question must cover a DIFFERENT angle of the topic
- Sub-questions should be specific enough to search for on the web
- Avoid overlap between sub-questions
- Together they should answer the original query completely

Reply with EXACTLY this format and nothing else:

REASONING: <one sentence explaining your decomposition strategy>

SUB_QUESTIONS:
1. QUESTION: <sub-question text> | FOCUS: <2-3 word label>
2. QUESTION: <sub-question text> | FOCUS: <2-3 word label>
3. QUESTION: <sub-question text> | FOCUS: <2-3 word label>
{extra_lines}
""")


def _build_extra_lines(n: int) -> str:
    """Build the format hint lines for questions 4-8 if needed."""
    if n <= 3:
        return ""
    lines = []
    for i in range(4, n + 1):
        lines.append(
            f"{i}. QUESTION: <sub-question text> | FOCUS: <2-3 word label>"
        )
    return "\n".join(lines)


def _parse_planner_output(text: str, n: int) -> tuple[str, list]:
    """
    Parse the LLM's structured output into reasoning + sub_questions list.

    Returns:
        reasoning: str
        sub_questions: list of SubQuestion dicts
    """
    reasoning = ""
    sub_questions = []

    lines = text.strip().splitlines()

    for line in lines:
        line = line.strip()

        # Extract reasoning
        if line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

        # Extract sub-questions — match lines like "1. QUESTION: ... | FOCUS: ..."
        sq_match = re.match(
            r"^\d+\.\s*QUESTION:\s*(.+?)\s*\|\s*FOCUS:\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if sq_match:
            question_text = sq_match.group(1).strip()
            focus_text = sq_match.group(2).strip()
            sub_questions.append({
                "id": f"sq_{len(sub_questions) + 1}",
                "question": question_text,
                "focus": focus_text,
            })

    # Fallback: if parsing failed, create generic sub-questions
    if not sub_questions:
        for i in range(1, n + 1):
            sub_questions.append({
                "id": f"sq_{i}",
                "question": f"Research aspect {i} of: {text[:100]}",
                "focus": f"aspect {i}",
            })

    # Trim to exactly n sub-questions
    sub_questions = sub_questions[:n]

    return reasoning, sub_questions


# ── Main node function ────────────────────────────────────────────────────────

def planner_node(state: dict) -> dict:
    """
    LangGraph node function for the Planner agent.

    Reads:  state["query"], state["depth"], state["iteration"]
    Writes: state["sub_questions"], state["plan_reasoning"], state["status"]
    """
    query = state["query"]
    depth = state.get("depth", "standard")
    iteration = state.get("iteration", 0)

    n = DEPTH_TO_COUNT.get(depth, 5)

    print(f"\n[planner] query='{query[:60]}...' depth={depth} n={n} iteration={iteration}")

    # On second iteration (after critic sends back), reduce sub-questions
    # to only fill identified gaps — not re-research everything
    if iteration > 0:
        critic_notes = state.get("critic_notes", [])
        gap_descriptions = [
            note["description"]
            for note in critic_notes
            if note.get("type") == "gap"
        ]
        if gap_descriptions:
            gaps_text = "\n".join(f"- {g}" for g in gap_descriptions)
            query = (
                f"Fill these specific research gaps about '{query}':\n{gaps_text}"
            )
            n = min(len(gap_descriptions) + 1, n)
            print(f"[planner] iteration {iteration} — targeting {n} gaps")

    # Build and invoke the prompt
    prompt_messages = PLANNER_PROMPT.format_messages(
        query=query,
        n=n,
        extra_lines=_build_extra_lines(n),
    )

    response = llm.invoke(prompt_messages)
    raw_text = response.content.strip()

    print(f"[planner] LLM response received ({len(raw_text)} chars)")

    # Parse the structured output
    reasoning, sub_questions = _parse_planner_output(raw_text, n)

    print(f"[planner] parsed {len(sub_questions)} sub-questions:")
    for sq in sub_questions:
        print(f"  [{sq['id']}] {sq['question'][:60]}... | focus: {sq['focus']}")

    return {
        "sub_questions": sub_questions,
        "plan_reasoning": reasoning,
        "iteration": iteration + 1,
        "status": "researching",
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from core.state import initial_state

    test_query = "What is the impact of large language models on software engineering jobs?"

    for depth in ["quick", "standard"]:
        print(f"\n{'='*60}")
        print(f"Testing planner with depth='{depth}'")
        print("="*60)

        state = initial_state(test_query, depth=depth)
        result = planner_node(state)

        print(f"\nReasoning: {result['plan_reasoning']}")
        print(f"\nSub-questions ({len(result['sub_questions'])}):")
        for sq in result["sub_questions"]:
            print(f"  {sq['id']}: [{sq['focus']}] {sq['question']}")