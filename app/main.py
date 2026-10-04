from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .engine import (
    ValidationError,
    compare,
    compose,
    preview,
    prune,
    rename,
    review,
    revise,
)
from .models import (
    CompareRequest,
    ComposeRequest,
    PreviewRequest,
    PruneRequest,
    RenameRequest,
    ReviewRequest,
    ReviseRequest,
)

app = FastAPI(
    title="Local Data Export Risk Review API",
    description=(
        "Reviews every output column of an export node for sensitive content "
        "provenance and sensitive row-level influence, can revise a blocked "
        "graph into a re-submittable, review-safe export, can prune a graph "
        "down to a chosen subset of export columns, can batch-rename "
        "columns of a source node while keeping every in-graph reference "
        "pointing at the same original columns, can compose an upstream "
        "graph's export into a downstream graph's source position, yielding "
        "one reviewable combined graph, can run a small-sample preview "
        "of an export over caller-provided source rows and filter "
        "conditions, returning sample data only when the export passes "
        "review, and can diff a baseline graph against a candidate graph "
        "to report the export risks the candidate newly introduces."
    ),
    version="1.6.0",
)


@app.exception_handler(ValidationError)
async def _validation_error_handler(_request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": exc.detail()})


@app.exception_handler(Exception)
async def _unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    # Never leak stack traces from a local review service.
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL", "message": str(exc)}},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/review")
async def review_export(request: ReviewRequest) -> dict[str, object]:
    return review(request)


@app.post("/api/revise")
async def revise_export(request: ReviseRequest) -> dict[str, object]:
    return revise(request)


@app.post("/api/prune")
async def prune_export(request: PruneRequest) -> dict[str, object]:
    return prune(request)


@app.post("/api/rename")
async def rename_source_columns(request: RenameRequest) -> dict[str, object]:
    return rename(request)


@app.post("/api/compose")
async def compose_graphs(request: ComposeRequest) -> dict[str, object]:
    return compose(request)


@app.post("/api/preview")
async def preview_export(request: PreviewRequest) -> dict[str, object]:
    return preview(request)


@app.post("/api/compare")
async def compare_exports(request: CompareRequest) -> dict[str, object]:
    return compare(request)
