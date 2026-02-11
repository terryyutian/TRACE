from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from trace_revision.pipeline import trace  # <- your working pipeline import

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"

app = FastAPI(title="TRACE: Text Revision Analysis via Change Estimation")


class AnalyzeRequest(BaseModel):
    source_text: str
    target_text: str
    replace_similarity_threshold: float | None = None


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/processing", response_class=HTMLResponse)
def processing() -> str:
    return (WEB_DIR / "processing.html").read_text(encoding="utf-8")


@app.get("/results", response_class=HTMLResponse)
def results() -> str:
    return (WEB_DIR / "results.html").read_text(encoding="utf-8")


@app.get("/assets/app.js", response_class=Response)
def app_js() -> Response:
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    return Response(content=js, media_type="application/javascript")


@app.get("/assets/styles.css", response_class=Response)
def styles_css() -> Response:
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    return Response(content=css, media_type="text/css")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest) -> JSONResponse:
    df, _ = trace(
        req.source_text,
        req.target_text,
        replace_similarity_threshold=req.replace_similarity_threshold or 0.4,
        debug=False,
    )
    rows: List[Dict[str, Any]] = df.fillna("").to_dict(orient="records")
    return JSONResponse({"rows": rows})


@app.post("/api/export")
def api_export(payload: Dict[str, Any]) -> Response:
    rows = payload.get("rows", [])
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="trace_results.csv"'},
    )
