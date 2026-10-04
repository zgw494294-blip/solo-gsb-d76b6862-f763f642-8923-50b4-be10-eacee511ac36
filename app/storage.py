"""SQLite-backed persistence for registered graphs and the shared roster.

Two pieces of state outlive the process:

* the **registered graphs** (the original ``nodes`` + ``export`` as submitted,
  together with their current review result);
* the **shared sensitive-column roster** (a set of ``(source id, column)``
  pairs) with its monotonically increasing **generation** (starting at 0).

A registered graph's effective sensitivity is the graph's own ``sensitive``
flags OR the roster. Replacing the roster therefore re-reviews every
registered graph; the new roster, the advanced generation and every graph's
new review result are committed together, or not at all.

All mutating and reading operations take one process-wide re-entrant lock and
run inside an ``IMMEDIATE`` SQLite transaction, so roster replacement, graph
registration, graph reads and graph previews are linearized in one definite
order: a preview can never mix a new roster with an old review result or run
on a stale "safe" verdict after the replacement committed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from typing import Iterable

from .engine import (
    Origin,
    ValidationError,
    apply_roster,
    preview,
    review_with_roster,
    serialize_node,
)
from .models import GraphPreviewRequest, PreviewRequest, ReviewRequest


class NotFoundError(Exception):
    """The referenced graph id is not registered (HTTP 404)."""

    def __init__(self, graph_id: str):
        super().__init__(f"graph '{graph_id}' is not registered")
        self.graph_id = graph_id

    def detail(self) -> dict[str, object]:
        return {
            "code": "GRAPH_NOT_FOUND",
            "message": f"graph '{self.graph_id}' is not registered",
            "graph_id": self.graph_id,
        }


class GenerationConflictError(Exception):
    """The roster replacement carried a stale expected generation (HTTP 409)."""

    def __init__(self, expected: int, current: int):
        super().__init__(
            f"expected generation {expected} does not match current "
            f"generation {current}"
        )
        self.expected = expected
        self.current = current

    def detail(self) -> dict[str, object]:
        return {
            "code": "ROSTER_GENERATION_CONFLICT",
            "message": (
                f"expected generation {self.expected} does not match the "
                f"current generation {self.current}; re-read the roster and "
                "retry"
            ),
            "expected_generation": self.expected,
            "current_generation": self.current,
        }


def _new_graph_id() -> str:
    return uuid.uuid4().hex


class Store:
    """Thread-safe SQLite store; one instance per process."""

    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get(
            "EXPORT_RISK_DB", "data/export-risk.db"
        )
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        # check_same_thread=False is safe: every use is serialized by
        # ``self._lock``. isolation_level=None puts the connection in
        # autocommit so transactions are opened explicitly (BEGIN IMMEDIATE),
        # which makes the atomic replace-commit boundary unambiguous.
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS graphs (
                    id TEXT PRIMARY KEY,
                    export TEXT NOT NULL,
                    nodes_json TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE TABLE IF NOT EXISTS roster (
                    source_id TEXT NOT NULL,
                    "column" TEXT NOT NULL,
                    PRIMARY KEY (source_id, "column")
                );
                """
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('generation', '0')"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- internal helpers (caller holds the lock) ---------------------------

    def _generation(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'generation'"
        ).fetchone()
        return int(row["value"])

    def _roster_set(self) -> frozenset[Origin]:
        rows = self._conn.execute(
            'SELECT source_id, "column" FROM roster'
        ).fetchall()
        return frozenset((row["source_id"], row["column"]) for row in rows)

    @staticmethod
    def _entries_view(roster: Iterable[Origin]) -> list[dict[str, str]]:
        return [
            {"source_id": source_id, "column": column}
            for source_id, column in sorted(roster)
        ]

    def _fetch_graph(self, graph_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT id, export, nodes_json, review_json FROM graphs WHERE id = ?",
            (graph_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(graph_id)
        return row

    # -- public operations ---------------------------------------------------

    def get_roster(self) -> dict[str, object]:
        with self._lock:
            return {
                "generation": self._generation(),
                "entries": self._entries_view(self._roster_set()),
            }

    def register_graph(
        self, request: ReviewRequest
    ) -> dict[str, object]:
        """Validate the whole graph under the original rules and register it.

        The stored graph is the original (its own ``sensitive`` flags
        untouched); the stored review already reflects roster OR graph.
        Returns the graph id, the original graph, its generation and review.
        """
        # Persist the graph exactly as submitted (serialized in canonical form
        # by the same helper the other endpoints use).
        original_nodes = [serialize_node(n) for n in request.nodes]

        with self._lock:
            roster = self._roster_set()
            review_result = review_with_roster(
                original_nodes, request.export, roster
            )
            graph_id = _new_graph_id()
            generation = self._generation()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO graphs(id, export, nodes_json, review_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        graph_id,
                        request.export,
                        json.dumps(original_nodes),
                        json.dumps(review_result),
                    ),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return {
            "graph_id": graph_id,
            "nodes": original_nodes,
            "export": request.export,
            "generation": generation,
            "review": review_result,
        }

    def read_graph(self, graph_id: str) -> dict[str, object]:
        with self._lock:
            row = self._fetch_graph(graph_id)
            nodes = json.loads(row["nodes_json"])
            review_result = json.loads(row["review_json"])
            generation = self._generation()
        return {
            "graph_id": graph_id,
            "nodes": nodes,
            "export": row["export"],
            "generation": generation,
            "review": review_result,
        }

    def preview_graph(
        self, graph_id: str, request: GraphPreviewRequest
    ) -> dict[str, object]:
        """Preview a registered graph using its *stored, current* verdict.

        The stored review result is the single source of truth for the block
        decision and for the returned ``review``: the preview never recomputes
        risk from a newer roster than the one the stored conclusion came from,
        because reads and roster replacements share one lock.
        """
        with self._lock:
            row = self._fetch_graph(graph_id)
            nodes = json.loads(row["nodes_json"])
            export = row["export"]
            stored_review = json.loads(row["review_json"])
            generation = self._generation()
            roster = self._roster_set()

            # The old preview rules still apply in full (coverage, key sets,
            # scalars, row cap, materialization). Run the engine with the
            # *current* effective sensitivity so its in-transit block guard and
            # the stored verdict are the same verdict by construction.
            payload = {
                "nodes": apply_roster(nodes, roster),
                "export": export,
                "rows": request.rows,
                "conditions": request.conditions,
            }
            try:
                result = preview(PreviewRequest.model_validate(payload))
            except ValidationError as exc:
                # The block guard must speak the stored, current conclusion
                # exactly (identical JSON), never a recomputed approximation.
                if exc.code == "PREVIEW_BLOCKED":
                    raise ValidationError(
                        "PREVIEW_BLOCKED",
                        exc.message,
                        export,
                        {"review": stored_review},
                    )
                raise

        result["review"] = stored_review
        result["generation"] = generation
        return result

    def replace_roster(
        self,
        expected_generation: int,
        entries: Iterable[tuple[str, str]],
    ) -> dict[str, object]:
        """Replace the whole roster and re-review every registered graph.

        Atomic by transaction: the roster rows, the advanced generation and
        every graph's new review result commit together — any failure leaves
        the previous roster, generation and conclusions untouched. A stale
        ``expected_generation`` is rejected with HTTP 409 before any write.
        """
        new_roster = frozenset(entries)

        with self._lock:
            current = self._generation()
            if expected_generation != current:
                raise GenerationConflictError(expected_generation, current)

            # Re-review every graph with the prospective roster before writing
            # anything: structural validity cannot change (roster entries only
            # flip source sensitivity), but any failure still leaves all state
            # untouched.
            graph_rows = self._conn.execute(
                "SELECT id, export, nodes_json FROM graphs"
            ).fetchall()
            new_reviews: dict[str, str] = {}
            for row in graph_rows:
                nodes = json.loads(row["nodes_json"])
                new_review = review_with_roster(
                    nodes, row["export"], new_roster
                )
                new_reviews[row["id"]] = json.dumps(new_review)

            # One atomic commit: roster + generation + every conclusion.
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM roster")
                self._conn.executemany(
                    'INSERT INTO roster(source_id, "column") VALUES (?, ?)',
                    sorted(new_roster),
                )
                self._conn.execute(
                    "UPDATE meta SET value = ? WHERE key = 'generation'",
                    (str(current + 1),),
                )
                self._conn.executemany(
                    "UPDATE graphs SET review_json = ? WHERE id = ?",
                    [
                        (review_json, graph_id)
                        for graph_id, review_json in new_reviews.items()
                    ],
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return {
            "generation": current + 1,
            "entries": self._entries_view(new_roster),
            "reviewed_graphs": len(graph_rows),
        }
