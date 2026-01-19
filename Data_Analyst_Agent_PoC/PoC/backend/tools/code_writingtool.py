
from typing import List


def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas operations on `df` only.
    2. Assign the final result to `result`.
    3. Wrap the snippet in a single ```python code fence (no extra prose).
    """