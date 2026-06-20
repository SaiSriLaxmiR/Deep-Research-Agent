from typing import TypedDict, Annotated, List
import operator


class SubQuestion(TypedDict):
    id: str
    question: str
    focus: str


class SearchResult(TypedDict):
    sub_question_id: str
    title: str
    url: str
    snippet: str


class ResearchFinding(TypedDict):
    sub_question_id: str
    sub_question: str
    summary: str
    key_facts: List[str]
    sources: List[str]


class CriticNote(TypedDict):
    type: str
    description: str
    severity: str


class ResearchState(TypedDict):
    query: str
    depth: str
    sub_questions: List[SubQuestion]
    plan_reasoning: str
    search_results: Annotated[List[SearchResult], operator.add]
    findings: Annotated[List[ResearchFinding], operator.add]
    merged_context: str
    citation_map: dict
    total_sources: int
    critic_notes: List[CriticNote]
    quality_score: float
    needs_more_research: bool
    final_report: str
    report_sections: List[str]
    iteration: int
    max_iterations: int
    status: str
    error: str


def initial_state(query: str, depth: str = "standard") -> ResearchState:
    return {
        "query": query,
        "depth": depth,
        "sub_questions": [],
        "plan_reasoning": "",
        "search_results": [],
        "findings": [],
        "merged_context": "",
        "citation_map": {},
        "total_sources": 0,
        "critic_notes": [],
        "quality_score": 0.0,
        "needs_more_research": False,
        "final_report": "",
        "report_sections": [],
        "iteration": 0,
        "max_iterations": 2,
        "status": "planning",
        "error": "",
    }