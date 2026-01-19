import re
import pandas as pd
import numpy as np
import io
import base64
import plotly.graph_objects as go
from matplotlib.figure import Figure
from datetime import datetime, date

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

def make_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to dict, but handle Interval objects first
        df_copy = obj.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'interval':
                # Convert interval objects to strings
                df_copy[col] = df_copy[col].astype(str)
        return df_copy.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        # Handle Series with Interval dtype
        if obj.dtype == 'interval':
            return obj.astype(str).to_dict()
        return obj.to_dict()
    elif isinstance(obj, (float, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, pd.Interval):
        return {
            'left': make_json_serializable(obj.left),
            'right': make_json_serializable(obj.right),
            'closed': obj.closed
        }
    elif isinstance(obj, go.Figure):
        fig_dict = obj.to_dict()
        serialized_fig_dict = make_json_serializable(fig_dict)
        return {
            "type": "plotly",
            "figure_json": serialized_fig_dict
        }
    elif isinstance(obj, Figure):
        buf = io.BytesIO()
        obj.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return {"image_base64": img_base64}
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (str, int, bool, type(None))):
        return obj
    else:
        return str(obj) 