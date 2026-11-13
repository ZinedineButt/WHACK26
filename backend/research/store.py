"""SQLite persistence for research runs (plain sqlite3: one short-lived
connection per call, safe across worker threads)."""
import json
import math
import sqlite3
import datetime as dt

import numpy as np

SCHEMA = """CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY, preset_id TEXT NOT NULL, status TEXT NOT NULL,
    current_stage TEXT, report_json TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""


def to_jsonable(o):
    """numpy -> python, NaN/Inf -> None. Guarantees strict JSON."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def _stage_of(report):
    return report.get("current_stage") or report.get("current_phase")


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = path
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def create(self, run_id, preset_id, report):
        with self._conn() as c:
            c.execute("INSERT INTO research_runs VALUES (?,?,?,?,?,?,?)",
                      (run_id, preset_id, report["status"], _stage_of(report),
                       json.dumps(to_jsonable(report), allow_nan=False), _now(), _now()))

    def save(self, report):
        with self._conn() as c:
            c.execute("UPDATE research_runs SET status=?, current_stage=?, report_json=?, updated_at=? WHERE id=?",
                      (report["status"], _stage_of(report),
                       json.dumps(to_jsonable(report), allow_nan=False), _now(), report["id"]))

    def get(self, run_id):
        with self._conn() as c:
            row = c.execute("SELECT report_json FROM research_runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(row["report_json"]) if row else None

    def active_count(self):
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM research_runs WHERE status IN ('queued','running')").fetchone()[0]

    def list(self, limit=50, offset=0):
        with self._conn() as c:
            rows = c.execute("SELECT id, preset_id, status, current_stage, created_at, updated_at "
                             "FROM research_runs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def fail_interrupted(self):
        """Runs left queued/running by a previous process can never finish."""
        with self._conn() as c:
            rows = c.execute("SELECT id FROM research_runs WHERE status IN ('queued','running')").fetchall()
        for r in rows:
            rep = self.get(r["id"])
            rep["status"] = "failed"
            err = {"code": "interrupted", "message": "Server restarted before the run finished."}
            if rep.get("schema_version", "1").startswith("2"):
                rep["errors"].append(err)
                for h in rep.get("hypotheses", []):
                    if h["status"] == "running":
                        h["status"] = "failed"
            else:
                rep["error"] = err
                for s in rep["stages"]:
                    if s["status"] in ("pending", "running"):
                        s["status"] = "failed" if s["status"] == "running" else "pending"
            self.save(rep)
