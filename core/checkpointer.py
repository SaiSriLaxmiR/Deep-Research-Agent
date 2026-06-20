"""
core/checkpointer.py

Sets up the SQLite checkpointer for persistent state storage.

Why SQLite and not MemorySaver:
- MemorySaver lives in RAM — if the process crashes or restarts,
  all research state is lost
- SQLite writes state to a file on disk after every node run
- If the server restarts mid-research, the graph resumes from
  the exact node where it stopped
- Multiple research runs (different thread_ids) are stored
  independently in the same database file

In production you'd swap SqliteSaver for PostgresSaver
and point it at a real database. The API is identical —
only the import and connection string change.
"""

import os
from langgraph.checkpoint.memory import MemorySaver


# ── Database file location ────────────────────────────────────────────────────
# Stored in a /data directory inside the project.
# Create it if it doesn't exist.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "research.db")

os.makedirs(DATA_DIR, exist_ok=True)


def get_checkpointer():
    """
    Returns MemorySaver for development.
    State persists within one Python process — lost on restart.
    Swap for SqliteSaver or PostgresSaver for production.
    """
    return MemorySaver()

def get_thread_config(thread_id: str) -> dict:
    """
    Returns the config dict required by every graph.invoke() call.

    Usage:
        config = get_thread_config("research-abc123")
        graph.invoke(state, config=config)
        graph.invoke(None, config=config)  # resume

    The thread_id uniquely identifies one research run.
    Two different queries should always have different thread_ids.
    The same query resumed after a pause uses the same thread_id.
    """
    return {"configurable": {"thread_id": thread_id}}


def generate_thread_id(query: str) -> str:
    """
    Generates a deterministic thread_id from a query string.
    Useful for deduplication — same query = same thread_id.

    For the API, you'd typically use a UUID instead so each
    request gets a fresh run even if the query is the same.
    """
    import hashlib
    import time
    # combine query hash + timestamp so re-runs don't collide
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    timestamp = str(int(time.time()))[-6:]
    return f"research-{query_hash}-{timestamp}"


# ── Quick connectivity test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Database path: {DB_PATH}")
    checkpointer = get_checkpointer()
    print(f"Checkpointer ready: {type(checkpointer).__name__}")
    thread_id = generate_thread_id("test query")
    print(f"Sample thread_id: {thread_id}")
    config = get_thread_config(thread_id)
    print(f"Sample config: {config}")
    print("\nAll good — checkpointer is working.")