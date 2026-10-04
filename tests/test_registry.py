"""Registered graphs, the shared sensitive-column roster and SQLite persistence."""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage import GenerationConflictError, NotFoundError, Store
from app.models import GraphPreviewRequest, ReviewRequest


# Source -> filter(country) -> projection export, mirroring the README graph.
# Nothing is marked sensitive in the graph itself: roster effects are visible
# as the *only* source of risk in these tests.
GRAPH = {
    "nodes": [
        {
            "id": "users",
            "type": "source",
            "columns": [
                {"name": "uid", "sensitive": False},
                {"name": "email", "sensitive": False},
                {"name": "country", "sensitive": False},
            ],
        },
        {"id": "only_cn", "type": "filter", "input": "users",
         "references": ["country"]},
        {
            "id": "export_users",
            "type": "projection",
            "input": "only_cn",
            "columns": [
                {"name": "user_id", "copy": "uid"},
                {"name": "mail", "copy": "email"},
                {"name": "bucket", "constant": "ALL"},
            ],
        },
    ],
    "export": "export_users",
}

ROWS = {
    "users": [
        {"uid": 1, "email": "a@x.example", "country": "CN"},
        {"uid": 2, "email": "b@x.example", "country": "US"},
    ]
}
CONDITIONS = {"only_cn": {"country": "CN"}}


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "risk.db"))
    yield s
    s.close()


def register(store, graph=GRAPH):
    return store.register_graph(ReviewRequest.model_validate(graph))


# ----------------------------- initial state -------------------------------


def test_roster_starts_empty_at_generation_zero(store):
    roster = store.get_roster()
    assert roster == {"generation": 0, "entries": []}


def test_register_validates_whole_graph_and_returns_id(store):
    invalid = {
        "nodes": [
            {"id": "s", "type": "source",
             "columns": [{"name": "x"}, {"name": "x"}]},
        ],
        "export": "s",
    }
    with pytest.raises(Exception) as exc:
        store.register_graph(ReviewRequest.model_validate(invalid))
    assert exc.value.code == "COLUMN_CONFLICT"

    saved = register(store)
    assert isinstance(saved["graph_id"], str) and len(saved["graph_id"]) >= 16
    assert saved["generation"] == 0
    assert saved["export"] == "export_users"
    # The original graph is returned, in-graph sensitive flags untouched.
    assert saved["nodes"] == GRAPH["nodes"]
    # Empty roster: the clean graph reviews clean.
    assert saved["review"]["blocked"] is False


def test_registered_ids_are_unique(store):
    first = register(store)
    second = register(store)
    assert first["graph_id"] != second["graph_id"]


def test_read_unknown_graph_is_404(store):
    with pytest.raises(NotFoundError):
        store.read_graph("deadbeef")


# ----------------------------- roster semantics ----------------------------


def test_graph_flags_and_roster_take_or(store):
    # A graph with its own sensitive flag stays blocked with an empty roster.
    flagged = {
        "nodes": [
            {"id": "s", "type": "source",
             "columns": [{"name": "secret", "sensitive": True},
                         {"name": "open"}]},
            {"id": "e", "type": "projection", "input": "s",
             "columns": [{"name": "secret", "copy": "secret"},
                         {"name": "open", "copy": "open"}]},
        ],
        "export": "e",
    }
    saved = register(store, flagged)
    assert saved["review"]["blocked"] is True
    assert saved["review"]["columns"][0]["content_reasons"] == [
        {"node_id": "s", "column": "secret"}
    ]

    # The other graph's clean column becomes blocked via the roster alone.
    other = register(store, GRAPH)
    res = store.replace_roster(0, [("users", "email")])
    assert res == {"generation": 1,
                   "entries": [{"source_id": "users", "column": "email"}],
                   "reviewed_graphs": 2}

    reread = store.read_graph(other["graph_id"])
    assert reread["generation"] == 1
    mail = [c for c in reread["review"]["columns"] if c["name"] == "mail"][0]
    assert mail["blocked"] is True
    assert mail["content_reasons"] == [
        {"node_id": "users", "column": "email"}
    ]

    # The graph-flagged graph stays blocked (OR can never un-block a flag).
    still_flagged = store.read_graph(saved["graph_id"])
    assert still_flagged["review"]["blocked"] is True

    # The returned original graph still shows its own (false) flags: the roster
    # never rewrites the stored graph.
    users = [n for n in reread["nodes"] if n["id"] == "users"][0]
    assert users["columns"][1]["sensitive"] is False


def test_roster_entry_may_target_unregistered_source(store):
    register(store)
    res = store.replace_roster(
        0, [("ghost_source", "col"), ("users", "email")]
    )
    assert res["generation"] == 1
    assert res["entries"] == [
        {"source_id": "ghost_source", "column": "col"},
        {"source_id": "users", "column": "email"},
    ]
    reread = store.read_graph(register(store)["graph_id"])
    mail = [c for c in reread["review"]["columns"] if c["name"] == "mail"][0]
    assert mail["blocked"] is True


def test_roster_entry_column_unknown_on_source_is_inert(store):
    saved = register(store)
    store.replace_roster(0, [("users", "nonexistent")])
    reread = store.read_graph(saved["graph_id"])
    assert reread["review"]["blocked"] is False


def test_roster_entry_on_filter_reference_adds_row_influence(store):
    saved = register(store)
    store.replace_roster(0, [("users", "country")])
    review_result = store.read_graph(saved["graph_id"])["review"]
    for column in review_result["columns"]:
        assert column["blocked"] is True
        assert column["row_reasons"] == [
            {"node_id": "users", "column": "country"}
        ]


def test_roster_entry_on_join_key_taints_every_output(store):
    join_graph = {
        "nodes": [
            {"id": "orders", "type": "source", "columns": [
                {"name": "uid_enc"}, {"name": "amount"}]},
            {"id": "users", "type": "source", "columns": [
                {"name": "uid"}, {"name": "name"}]},
            {"id": "j", "type": "join",
             "left_input": "orders", "right_input": "users",
             "left_key": "uid_enc", "right_key": "uid",
             "outputs": [
                 {"alias": "amt", "side": "left", "column": "amount"},
                 {"alias": "uname", "side": "right", "column": "name"}]},
        ],
        "export": "j",
    }
    saved = register(store, join_graph)
    assert saved["review"]["blocked"] is False
    store.replace_roster(0, [("orders", "uid_enc")])
    review_result = store.read_graph(saved["graph_id"])["review"]
    assert review_result["blocked"] is True
    for column in review_result["columns"]:
        assert column["row_reasons"] == [
            {"node_id": "orders", "column": "uid_enc"}
        ]


def test_replacement_reviews_all_registered_graphs_together(store):
    a = register(store, GRAPH)
    # A second graph with a differently named source.
    other = {
        "nodes": [
            {"id": "s", "type": "source",
             "columns": [{"name": "id_card"}, {"name": "score"}]},
            {"id": "e", "type": "projection", "input": "s",
             "columns": [{"name": "score", "copy": "score"}]},
        ],
        "export": "e",
    }
    b = register(store, other)
    res = store.replace_roster(
        0, [("s", "id_card"), ("users", "email")]
    )
    assert res["reviewed_graphs"] == 2
    assert store.read_graph(a["graph_id"])["review"]["blocked"] is True
    # id_card never reaches the export projection; still flagged at source,
    # but the single score column remains clean.
    b_review = store.read_graph(b["graph_id"])["review"]
    assert b_review["columns"][0]["blocked"] is False


def test_failed_replacement_changes_nothing(store, monkeypatch):
    import app.storage as storage_mod

    saved = register(store)
    store.replace_roster(0, [("users", "email")])

    # Fail during the pre-write re-review phase (before any transaction is
    # opened): generation, roster and conclusions must stay as they were.
    def boom(nodes, export, roster):
        raise RuntimeError("simulated review failure")

    monkeypatch.setattr(storage_mod, "review_with_roster", boom)
    with pytest.raises(RuntimeError):
        store.replace_roster(1, [])
    monkeypatch.undo()

    roster = store.get_roster()
    assert roster["generation"] == 1
    assert roster["entries"] == [
        {"source_id": "users", "column": "email"}
    ]
    reread = store.read_graph(saved["graph_id"])
    assert reread["generation"] == 1
    assert reread["review"]["blocked"] is True

    # The store still serves replacements after the failed attempt.
    res = store.replace_roster(1, [])
    assert res["generation"] == 2
    assert store.read_graph(saved["graph_id"])["review"]["blocked"] is False


def test_replace_stale_generation_conflicts_and_changes_nothing(store):
    register(store)
    store.replace_roster(0, [("users", "email")])

    with pytest.raises(GenerationConflictError) as exc:
        store.replace_roster(0, [])
    assert exc.value.expected == 0
    assert exc.value.current == 1

    # The failed replacement changed nothing: roster and generation stand.
    roster = store.get_roster()
    assert roster["generation"] == 1
    assert roster["entries"] == [
        {"source_id": "users", "column": "email"}
    ]

    # Retrying with the current generation succeeds and advances again.
    res = store.replace_roster(1, [])
    assert res["generation"] == 2
    assert res["entries"] == []


def test_clear_roster_restores_safe_conclusion(store):
    saved = register(store)
    store.replace_roster(0, [("users", "email")])
    assert store.read_graph(saved["graph_id"])["review"]["blocked"] is True

    res = store.replace_roster(1, [])
    assert res["generation"] == 2
    assert res["entries"] == []
    assert res["reviewed_graphs"] == 1
    reread = store.read_graph(saved["graph_id"])
    assert reread["generation"] == 2
    assert reread["review"]["blocked"] is False


def test_replace_roster_reviews_graphs_registered_after_later(store):
    # Newly registered graphs are reviewed under the roster current at
    # registration time.
    store.replace_roster(0, [("users", "email")])
    saved = register(store)
    assert saved["generation"] == 1
    assert saved["review"]["blocked"] is True


# ----------------------------- registered preview --------------------------


def test_preview_unknown_graph_is_404(store):
    req = GraphPreviewRequest.model_validate(
        {"rows": ROWS, "conditions": CONDITIONS}
    )
    with pytest.raises(NotFoundError):
        store.preview_graph("nope", req)


def test_preview_success_returns_data_and_generation(store):
    saved = register(store)
    req = GraphPreviewRequest.model_validate(
        {"rows": ROWS, "conditions": CONDITIONS}
    )
    result = store.preview_graph(saved["graph_id"], req)
    assert result["columns"] == ["user_id", "mail", "bucket"]
    assert result["rows"] == [[1, "a@x.example", "ALL"]]
    assert result["generation"] == 0
    assert result["review"]["blocked"] is False


def test_preview_blocked_after_roster_change_uses_current_conclusion(store):
    from app.engine import ValidationError

    saved = register(store)
    req = GraphPreviewRequest.model_validate(
        {"rows": ROWS, "conditions": CONDITIONS}
    )
    store.replace_roster(0, [("users", "email")])
    with pytest.raises(ValidationError) as exc:
        store.preview_graph(saved["graph_id"], req)
    assert exc.value.code == "PREVIEW_BLOCKED"
    assert exc.value.node_id == "export_users"
    assert exc.value.extra["review"]["blocked"] is True
    # The error payload carries the review only, never sample data.
    assert "rows" not in exc.value.detail()
    assert "columns" not in exc.value.detail()
    assert "rows" not in exc.value.extra

    # Clearing the roster restores the safe verdict and data comes back.
    store.replace_roster(1, [])
    result = store.preview_graph(saved["graph_id"], req)
    assert result["rows"] == [[1, "a@x.example", "ALL"]]
    assert result["generation"] == 2
    assert result["review"]["blocked"] is False


def test_preview_row_influence_blocked_carries_current_generation(store):
    from app.engine import ValidationError

    saved = register(store)
    store.replace_roster(0, [("users", "country")])
    req = GraphPreviewRequest.model_validate(
        {"rows": ROWS, "conditions": CONDITIONS}
    )
    with pytest.raises(ValidationError) as exc:
        store.preview_graph(saved["graph_id"], req)
    assert exc.value.code == "PREVIEW_BLOCKED"
    review_result = exc.value.extra["review"]
    assert all(c["blocked"] for c in review_result["columns"])


def test_preview_reuses_original_preview_rules(store):
    from app.engine import ValidationError

    saved = register(store)
    # Missing rows for the only source.
    req = GraphPreviewRequest.model_validate({"rows": {}})
    with pytest.raises(ValidationError) as exc:
        store.preview_graph(saved["graph_id"], req)
    assert exc.value.code == "PREVIEW_SOURCE_COVERAGE"

    # Conditions for a filter that is not in the graph.
    req = GraphPreviewRequest.model_validate(
        {"rows": ROWS, "conditions": {"ghost": {"x": 1}}}
    )
    with pytest.raises(ValidationError) as exc:
        store.preview_graph(saved["graph_id"], req)
    assert exc.value.code == "PREVIEW_FILTER_COVERAGE"


# ----------------------------- persistence ---------------------------------


def test_state_survives_store_reopening(tmp_path):
    path = str(tmp_path / "persist.db")
    first = Store(path)
    saved = first.register_graph(ReviewRequest.model_validate(GRAPH))
    first.replace_roster(0, [("users", "email")])
    graph_id = saved["graph_id"]
    first.close()

    second = Store(path)
    roster = second.get_roster()
    assert roster["generation"] == 1
    assert roster["entries"] == [
        {"source_id": "users", "column": "email"}
    ]
    reread = second.read_graph(graph_id)
    assert reread["nodes"] == GRAPH["nodes"]
    assert reread["review"]["blocked"] is True
    mail = [c for c in reread["review"]["columns"] if c["name"] == "mail"][0]
    assert mail["content_reasons"] == [
        {"node_id": "users", "column": "email"}
    ]
    second.close()


def test_persisted_blocked_preview_then_clear_after_reopen(tmp_path):
    from app.engine import ValidationError

    path = str(tmp_path / "persist2.db")
    first = Store(path)
    saved = first.register_graph(ReviewRequest.model_validate(GRAPH))
    first.replace_roster(0, [("users", "email")])
    graph_id = saved["graph_id"]
    first.close()

    second = Store(path)
    req = GraphPreviewRequest.model_validate(
        {"rows": ROWS, "conditions": CONDITIONS}
    )
    with pytest.raises(ValidationError) as exc:
        second.preview_graph(graph_id, req)
    assert exc.value.code == "PREVIEW_BLOCKED"

    second.replace_roster(1, [])
    result = second.preview_graph(graph_id, req)
    assert result["rows"] == [[1, "a@x.example", "ALL"]]
    assert result["generation"] == 2
    second.close()


# ----------------------------- concurrency ---------------------------------


def test_replacements_and_reads_linearize_without_mixed_states(store):
    """Every observed (generation, conclusion) pair is one committed state."""
    saved = register(store)
    graph_id = saved["graph_id"]

    # Even generations (0, 2, ...): empty roster -> safe.
    # Odd generations (1, 3, ...): roster marks users.email -> blocked.
    errors: list[Exception] = []
    stop = threading.Event()

    def replacer():
        expected = 0
        try:
            while not stop.is_set():
                next_entries = (
                    [("users", "email")] if expected % 2 == 0 else []
                )
                try:
                    res = store.replace_roster(expected, next_entries)
                    expected = res["generation"]
                except GenerationConflictError as exc:
                    # Another logical mover advanced it; adopt the current
                    # generation and continue from the committed state.
                    expected = exc.current
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def reader():
        try:
            for _ in range(200):
                view = store.read_graph(graph_id)
                blocked = view["review"]["blocked"]
                # The verdict must always match the generation read in the
                # same response: no new-roster/old-conclusion mixing.
                assert blocked == (view["generation"] % 2 == 1)
                roster = store.get_roster()
                assert roster["entries"] == (
                    [{"source_id": "users", "column": "email"}]
                    if roster["generation"] % 2 == 1
                    else []
                )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=replacer) for _ in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    # Let the replacers churn for a bit, then stop them; readers finish.
    timer = threading.Timer(1.0, stop.set)
    timer.start()
    for t in threads[:2]:
        t.join()
    stop.set()
    for t in threads[2:]:
        t.join()
    timer.cancel()
    assert errors == []


def test_concurrent_preview_never_returns_data_against_blocked_conclusion(
    store,
):
    from app.engine import ValidationError

    saved = register(store)
    graph_id = saved["graph_id"]
    req = GraphPreviewRequest.model_validate(
        {"rows": ROWS, "conditions": CONDITIONS}
    )

    outcomes: list[str] = []
    stop = threading.Event()
    lock = threading.Lock()

    def replacer():
        expected = 0
        while not stop.is_set():
            try:
                res = store.replace_roster(
                    expected, [("users", "email")] if expected % 2 == 0 else []
                )
                expected = res["generation"]
            except GenerationConflictError as exc:
                expected = exc.current

    def previewer():
        while not stop.is_set():
            try:
                result = store.preview_graph(graph_id, req)
                # Data returned ⇒ the verdict in the very same response must
                # be safe, and must carry the matching even generation.
                assert result["review"]["blocked"] is False
                assert result["generation"] % 2 == 0
                assert result["rows"]
                with lock:
                    outcomes.append("data")
            except ValidationError as exc:
                if exc.code == "PREVIEW_BLOCKED":
                    with lock:
                        outcomes.append("blocked")
            except NotFoundError:  # pragma: no cover
                assert False

    threads = [threading.Thread(target=replacer),
               threading.Thread(target=previewer)]
    for t in threads:
        t.start()
    stop.wait(1.0)
    stop.set()
    for t in threads:
        t.join()
    assert "data" in outcomes and "blocked" in outcomes


# ----------------------------- HTTP layer ----------------------------------


@pytest.fixture
def client(store):
    app.state.store = store
    return TestClient(app)


def test_http_roster_lifecycle(client):
    # Initially empty at generation 0.
    resp = client.get("/api/sensitive-roster")
    assert resp.status_code == 200
    assert resp.json() == {"generation": 0, "entries": []}

    # Register a whole graph (same validation as /api/review).
    resp = client.post("/api/graphs", json=GRAPH)
    assert resp.status_code == 201
    body = resp.json()
    graph_id = body["graph_id"]
    assert body["nodes"] == GRAPH["nodes"]
    assert body["review"]["blocked"] is False
    assert body["generation"] == 0

    # Read it back: original graph + current generation + review.
    resp = client.get(f"/api/graphs/{graph_id}")
    assert resp.status_code == 200
    assert resp.json()["export"] == "export_users"
    assert resp.json()["generation"] == 0

    # Safe preview with rows/conditions only.
    resp = client.post(
        f"/api/graphs/{graph_id}/preview",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 200
    assert resp.json()["rows"] == [[1, "a@x.example", "ALL"]]
    assert resp.json()["generation"] == 0

    # Mark a shared sensitive column: the stored verdict flips.
    resp = client.put(
        "/api/sensitive-roster",
        json={
            "expected_generation": 0,
            "entries": [{"source_id": "users", "column": "email"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["generation"] == 1
    assert resp.json()["reviewed_graphs"] == 1

    # Read now reports the blocked conclusion at generation 1.
    resp = client.get(f"/api/graphs/{graph_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["generation"] == 1
    assert body["review"]["blocked"] is True

    # Preview is refused: 400/PREVIEW_BLOCKED, no data.
    resp = client.post(
        f"/api/graphs/{graph_id}/preview",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "PREVIEW_BLOCKED"
    assert error["review"]["blocked"] is True
    assert "rows" not in error and "columns" not in error

    # Stale generation replacement: 409, nothing changes.
    resp = client.put(
        "/api/sensitive-roster",
        json={"expected_generation": 0, "entries": []},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "ROSTER_GENERATION_CONFLICT"
    assert client.get("/api/sensitive-roster").json()["generation"] == 1

    # Clear with the current generation: safe conclusion returns.
    resp = client.put(
        "/api/sensitive-roster",
        json={"expected_generation": 1, "entries": []},
    )
    assert resp.status_code == 200
    assert resp.json()["generation"] == 2

    resp = client.post(
        f"/api/graphs/{graph_id}/preview",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"] == [[1, "a@x.example", "ALL"]]
    assert data["generation"] == 2
    assert data["review"]["blocked"] is False


def test_http_unknown_graph_ids_are_404(client):
    resp = client.get("/api/graphs/unknown")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "GRAPH_NOT_FOUND"

    resp = client.post(
        "/api/graphs/unknown/preview",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 404


def test_http_register_inherits_whole_graph_validation(client):
    bad = {**GRAPH, "export": "ghost"}
    resp = client.post("/api/graphs", json=bad)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_REFERENCE"


def test_http_register_schema_errors_are_422(client):
    # Unknown field -> schema rejection before graph validation.
    resp = client.post("/api/graphs", json={**GRAPH, "unexpected": 1})
    assert resp.status_code == 422

    # Preview by id must not accept another graph in the body.
    register_resp = client.post("/api/graphs", json=GRAPH)
    graph_id = register_resp.json()["graph_id"]
    resp = client.post(
        f"/api/graphs/{graph_id}/preview",
        json={"nodes": GRAPH["nodes"], "rows": ROWS},
    )
    assert resp.status_code == 422


def test_http_roster_duplicate_entries_are_422(client):
    resp = client.put(
        "/api/sensitive-roster",
        json={
            "expected_generation": 0,
            "entries": [
                {"source_id": "users", "column": "email"},
                {"source_id": "users", "column": "email"},
            ],
        },
    )
    assert resp.status_code == 422
    assert client.get("/api/sensitive-roster").json()["generation"] == 0


def test_http_roster_entries_unknown_fields_are_422(client):
    resp = client.put(
        "/api/sensitive-roster",
        json={
            "expected_generation": 0,
            "entries": [],
            "unexpected": 1,
        },
    )
    assert resp.status_code == 422


def test_http_old_review_endpoint_ignores_roster(client):
    # The shared roster marks users.email, but the legacy /api/review endpoint
    # must keep reviewing by in-graph flags only.
    client.post("/api/graphs", json=GRAPH)
    client.put(
        "/api/sensitive-roster",
        json={
            "expected_generation": 0,
            "entries": [{"source_id": "users", "column": "email"}],
        },
    )
    resp = client.post("/api/review", json=GRAPH)
    assert resp.status_code == 200
    assert resp.json()["blocked"] is False

    # The old /api/preview endpoint likewise ignores the roster.
    resp = client.post(
        "/api/preview",
        json={**GRAPH, "rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 200
    assert resp.json()["rows"] == [[1, "a@x.example", "ALL"]]


def test_http_roster_can_target_unregistered_sources(client):
    client.post("/api/graphs", json=GRAPH)
    resp = client.put(
        "/api/sensitive-roster",
        json={
            "expected_generation": 0,
            "entries": [
                {"source_id": "some_other_system", "column": "ssn"}
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["reviewed_graphs"] == 1
    # Existing graph stays clean: the entry is inert without a matching source.
    graph_id = client.post("/api/graphs", json=GRAPH).json()["graph_id"]
    review_result = client.get(f"/api/graphs/{graph_id}").json()["review"]
    assert review_result["blocked"] is False
