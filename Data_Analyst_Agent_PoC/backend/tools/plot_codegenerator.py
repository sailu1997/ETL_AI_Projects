from typing import List


def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named `result`.
    3. Create only ONE relevant plot. Set `figsize=(6,4)`, add title/labels.
    4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    """