import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pydantic import BaseModel
import io
import base64
import os
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
file_path = os.path.join(UPLOAD_DIR, "uploaded_data.csv")

class ExecutionRequest(BaseModel):
    code: str
    should_plot: bool

@router.post("/executionagent/")
async def ExecutionAgent(payload: ExecutionRequest):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    env = {"pd": pd, "df": pd.read_csv(file_path)}
    if payload.should_plot:
        plt.rcParams["figure.dpi"] = 100  # Set default DPI for all figures
        env["plt"] = plt
        env["io"] = io
    try:
        exec(payload.code, {}, env)
        result = env.get("result", None)
        if isinstance(result, pd.DataFrame):
            result = result.to_dict(orient="records")
        elif isinstance(result, Figure):
            buf = io.BytesIO()
            result.savefig(buf, format="png")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            result = {"image_base64": img_base64}
        return JSONResponse(content={"status": "success", "result": result})
    
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error executing code: {exc}"}
        )