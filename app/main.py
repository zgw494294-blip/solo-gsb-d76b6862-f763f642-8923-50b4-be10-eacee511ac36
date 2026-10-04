from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
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
    GraphPreviewRequest,
    PreviewRequest,
    PruneRequest,
    RenameRequest,
    ReviewRequest,
    ReviseRequest,
    RosterReplaceRequest,
)
from .storage import GenerationConflictError, NotFoundError, Store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.store = Store(os.environ.get("EXPORT_RISK_DB"))
    try:
        yield
    finally:
        store = getattr(_app.state, "store", None)
        if store is not None:
            store.close()
            _app.state.store = None


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
        "to report the export risks the candidate newly introduces. "
        "Graphs can additionally be registered for persistent storage so "
        "their verdict follows a shared, versioned sensitive-column roster "
        "(in-graph sensitive flags OR roster) and be previewed by graph id."
    ),
    version="1.7.0",
    lifespan=lifespan,
)


def get_store(request: Request) -> Store:
    # Lazily open the store too, so TestClient(app) works without driving the
    # lifespan (the existing tests instantiate it at module import time).
    store = getattr(request.app.state, "store", None)
    if store is None:
        store = Store(os.environ.get("EXPORT_RISK_DB"))
        request.app.state.store = store
    return store


@app.exception_handler(ValidationError)
async def _validation_error_handler(_request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": exc.detail()})


@app.exception_handler(NotFoundError)
async def _not_found_error_handler(_request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": exc.detail()})


@app.exception_handler(GenerationConflictError)
async def _conflict_error_handler(
    _request: Request, exc: GenerationConflictError
) -> JSONResponse:
    return JSONResponse(status_code=409, content={"error": exc.detail()})


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


# ---------------------------------------------------------------------------
# Registered graphs with the shared sensitive-column roster
# ---------------------------------------------------------------------------


@app.post("/api/graphs", status_code=201)
async def register_graph(
    request: ReviewRequest, store: Store = Depends(get_store)
) -> dict[str, object]:
    """Register one whole validated graph; returns its persistent graph id.

    The stored review already combines the graph's own ``sensitive`` flags
    with the shared roster (OR).
    """
    return store.register_graph(request)


@app.get("/api/graphs/{graph_id}")
async def read_graph(
    graph_id: str, store: Store = Depends(get_store)
) -> dict[str, object]:
    """Return the original graph, the current generation and current review."""
    return store.read_graph(graph_id)


@app.post("/api/graphs/{graph_id}/preview")
async def preview_registered_graph(
    graph_id: str,
    request: GraphPreviewRequest,
    store: Store = Depends(get_store),
) -> dict[str, object]:
    """Preview by graph id with only rows/conditions; verdict is the stored one."""
    return store.preview_graph(graph_id, request)


@app.get("/api/sensitive-roster")
async def read_roster(store: Store = Depends(get_store)) -> dict[str, object]:
    """Read the current roster and generation."""
    return store.get_roster()


@app.put("/api/sensitive-roster")
async def replace_roster(
    request: RosterReplaceRequest,
    store: Store = Depends(get_store),
) -> dict[str, object]:
    """Replace the whole roster (expected generation check), re-review graphs.

    On success the generation advances by one and every registered graph's
    stored review is replaced atomically with the new roster; a stale
    ``expected_generation`` answers HTTP 409 and changes nothing.
    """
    return store.replace_roster(
        request.expected_generation,
        [(entry.source_id, entry.column) for entry in request.entries],
    )
