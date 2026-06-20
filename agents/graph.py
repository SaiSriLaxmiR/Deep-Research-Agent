"""
agents/graph.py

Assembles all five agents into one complete LangGraph pipeline.

The full flow:
    START
      ↓
    planner         — breaks query into sub-questions
      ↓
    [Send API]      — dispatches N parallel researcher agents
      ↓ (parallel)
    researcher × N  — each researches one sub-question
      ↓ (merged by operator.add)
    observer        — merges all findings into one context
      ↓
    critic          — evaluates quality, decides loop or write
      ↓ (conditional)
    ┌── "loop" ──→  planner  (back to top, fills gaps)
    └── "write" ──→ writer   — generates final markdown report
                      ↓
                    END

Key patterns used:
- Send API for dynamic parallel agents (dispatch_researchers)
- Annotated + operator.add for merging parallel outputs
- Conditional edge with loop for quality control
- SQLite checkpointer for persistent state
- interrupt_before writer for optional human review
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from core.state import ResearchState, initial_state
from core.checkpointer import get_checkpointer, get_thread_config, generate_thread_id
from agents.planner import planner_node
from agents.researcher import researcher_node, dispatch_researchers
from agents.observer import observer_node
from agents.critic import critic_node, route_after_critic
from agents.writer import writer_node


def build_graph(human_review: bool = False):
    """
    Build and compile the full research graph.

    Args:
        human_review: if True, graph pauses before writer
                      for a human to review/edit the context
                      before the final report is generated.

    Returns:
        compiled LangGraph graph with SQLite checkpointer
    """
    builder = StateGraph(ResearchState)

    # ── Register all nodes ────────────────────────────────────────────────────
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("observer", observer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("writer", writer_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("planner")

    # ── Planner → parallel researchers via Send API ───────────────────────────
    # dispatch_researchers reads sub_questions from state and returns
    # one Send("researcher", {...}) per sub-question.
    # LangGraph runs all of them simultaneously.
    builder.add_conditional_edges(
        "planner",
        dispatch_researchers,   # returns list of Send objects
    )

    # ── Each researcher → observer ────────────────────────────────────────────
    # All parallel researcher instances feed into the same observer node.
    # operator.add in state merges their findings lists automatically.
    builder.add_edge("researcher", "observer")

    # ── Observer → critic ─────────────────────────────────────────────────────
    builder.add_edge("observer", "critic")

    # ── Critic → loop back to planner OR forward to writer ───────────────────
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "loop": "planner",   # ← quality too low, research more
            "write": "writer"    # ← quality good enough, write report
        }
    )

    # ── Writer → END ─────────────────────────────────────────────────────────
    builder.add_edge("writer", END)

    # ── Compile with checkpointer ─────────────────────────────────────────────
    checkpointer = get_checkpointer()

    if human_review:
        # Pause before writer so a human can review merged_context
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["writer"]
        )
        print("[graph] compiled with human review pause before writer")
    else:
        graph = builder.compile(checkpointer=checkpointer)
        print("[graph] compiled without interrupts")

    return graph


def run_research(
    query: str,
    depth: str = "standard",
    human_review: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the full research pipeline for a given query.

    Args:
        query:        the research question
        depth:        "quick" | "standard" | "deep"
        human_review: pause before writing for human approval
        verbose:      print progress to terminal

    Returns:
        final state dict with final_report, findings, etc.
    """
    graph = build_graph(human_review=human_review)
    thread_id = generate_thread_id(query)
    config = get_thread_config(thread_id)
    state = initial_state(query, depth=depth)

    if verbose:
        print(f"\n{'='*60}")
        print(f"DEEP RESEARCH AGENT")
        print(f"{'='*60}")
        print(f"Query:     {query}")
        print(f"Depth:     {depth}")
        print(f"Thread ID: {thread_id}")
        print(f"{'='*60}\n")

    if human_review:
        # First invoke — runs planner → researchers → observer → critic
        # then PAUSES before writer
        graph.invoke(state, config=config)

        # Show the merged context for human review
        current = graph.get_state(config)
        merged = current.values.get("merged_context", "")
        quality = current.values.get("quality_score", 0.0)

        print(f"\n{'='*60}")
        print(f"HUMAN REVIEW REQUIRED")
        print(f"Quality score: {quality:.2f}")
        print(f"{'─'*60}")
        print(f"Merged context preview:")
        print(merged[:500] + "..." if len(merged) > 500 else merged)
        print(f"{'─'*60}")

        decision = input("\nProceed to generate report? (yes/no): ").strip().lower()

        if decision != "yes":
            print("Report generation cancelled.")
            return current.values

        print("\nResuming graph → generating final report...")
        result = graph.invoke(None, config=config)
    else:
        result = graph.invoke(state, config=config)

    return result


# ── Main entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument("query", nargs="?",
                        default="What is the impact of LLMs on software engineering jobs?",
                        help="Research query")
    parser.add_argument("--depth", choices=["quick", "standard", "deep"],
                        default="quick",
                        help="Research depth (default: quick)")
    parser.add_argument("--review", action="store_true",
                        help="Pause for human review before writing")
    args = parser.parse_args()

    result = run_research(
        query=args.query,
        depth=args.depth,
        human_review=args.review,
    )

    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(result.get("final_report", "No report generated"))
    print(f"\n{'='*60}")
    print(f"Status:          {result.get('status')}")
    print(f"Quality score:   {result.get('quality_score', 0):.2f}")
    print(f"Total sources:   {result.get('total_sources', 0)}")
    print(f"Iterations:      {result.get('iteration', 0)}")
    print(f"Sections:        {result.get('report_sections', [])}")
    