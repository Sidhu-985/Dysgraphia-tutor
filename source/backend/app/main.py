"""
FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000
from inside backend/ (see README.md "Backend setup").
"""

from __future__ import annotations

import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router

app = FastAPI(
    title="Dysgraphia Adaptive Handwriting Tutor -- Backend",
    description=(
        "REST backend for the RL-based adaptive handwriting intervention "
        "system: tablet stroke ingestion, static/dynamic feature "
        "extraction, 12-D state construction, Double DQN policy, safety "
        "layer, and intervention controller."
    ),
    version="0.1.0",
)

# Permissive CORS for local development against a Flutter debug client
# (web/desktop debug builds run from an arbitrary localhost port). Not
# intended as-is for a production deployment -- restrict `allow_origins`
# there.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def _json_safe(value):
    """Recursively replace NaN/inf (which strict JSON cannot encode) with a
    string, so a validation error ABOUT a NaN can itself be reported."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, BaseException):
        return str(value)
    return value


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    # Phase E fix: FastAPI's default handler echoes the offending input. For
    # a NaN coordinate that echo made the 422 response itself unencodable,
    # and the server dropped the connection instead of answering.
    return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})


@app.get("/")
def root():
    return {"name": "dysgraphia-tutor-backend", "docs": "/docs"}
