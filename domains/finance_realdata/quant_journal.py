"""Append-only attempt history for public evaluator calls, including failures.

This records evaluator invocations, not arbitrary scratch calculations. Research
protocols must still disclose searches performed outside this interface.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid


@contextmanager
def evaluation_attempt(root: Path, arguments: dict):
    path = root / ".quant-trials.sqlite"
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS attempts(
        id TEXT PRIMARY KEY, started_at TEXT NOT NULL, arguments TEXT NOT NULL,
        state TEXT NOT NULL, result TEXT, error TEXT)""")
    connection.execute("""CREATE TABLE IF NOT EXISTS study_registrations(
        family_id TEXT PRIMARY KEY, protocol_hash TEXT NOT NULL,
        protocol_json TEXT NOT NULL, registered_at TEXT NOT NULL, prior_attempts INTEGER NOT NULL)""")
    identifier = str(uuid.uuid4())
    candidate_root = Path(arguments["candidate_root"])
    protocol_path = root / "study-protocol.json"
    if not protocol_path.is_file():
        protocol_path = candidate_root / "study-protocol.json"
    protocol = None
    try:
        protocol = json.loads(protocol_path.read_text("utf-8"))
    except (OSError, ValueError):
        pass
    registration = {"registered": False, "history_complete": False}
    protocol_error = None
    # Commit the declaration before executing this comparison. A family cannot
    # rewrite its protocol after inspecting results, including failed attempts.
    connection.execute("BEGIN IMMEDIATE")
    prior_attempts = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    if isinstance(protocol, dict) and isinstance(protocol.get("family_id"), str) and protocol["family_id"].strip():
        family = protocol["family_id"].strip()
        body = json.dumps(protocol, sort_keys=True, allow_nan=False)
        checksum = hashlib.sha256(body.encode()).hexdigest()
        stored = connection.execute("SELECT protocol_hash,registered_at,prior_attempts FROM study_registrations WHERE family_id=?", (family,)).fetchone()
        if stored and stored[0] != checksum:
            protocol_error = "study family protocol is frozen; register a new family and disclose prior searches"
        if not stored:
            registered_at = datetime.now(timezone.utc).isoformat()
            connection.execute("INSERT INTO study_registrations VALUES(?,?,?,?,?)",
                               (family, checksum, body, registered_at, prior_attempts))
            stored = (checksum, registered_at, prior_attempts)
        registration = {"registered": True, "family_id": family, "protocol_hash": checksum,
                        "registered_at": stored[1], "prior_attempts_at_registration": stored[2],
                        "declared_history_complete": protocol.get("history_complete") is True,
                        "history_complete": False,
                        "limitation": "Local registration freezes this family, not unobserved scratch searches or cleared history."}
    arguments = {**arguments, "study_registration": registration, "input_file_hashes": {
        name: hashlib.sha256((candidate_root / name).read_bytes()).hexdigest()
        for name in ("model.py", "config.json", "study-protocol.json") if (candidate_root / name).is_file()}}
    connection.execute("INSERT INTO attempts VALUES(?,?,?,'running',NULL,NULL)",
                       (identifier, datetime.now(timezone.utc).isoformat(), json.dumps(arguments)))
    connection.commit()
    result = {"study_registration": registration}
    try:
        if protocol_error:
            raise ValueError(protocol_error)
        yield result
    except BaseException as error:
        connection.execute("UPDATE attempts SET state='failed',error=? WHERE id=?", (str(error), identifier))
        connection.commit()
        raise
    else:
        summary = {key: result.get(key) for key in (
            "candidate_hash", "policy_hash", "evaluator_sha256", "runner_sha256", "snapshot_id",
            "snapshot_manifest_sha256", "decision_contract", "screen", "screen_reasons", "scenarios",
            "study_registration", "validation")}
        connection.execute("UPDATE attempts SET state='completed',result=? WHERE id=?", (json.dumps(summary), identifier))
        connection.commit()
        result["local_trial_id"] = identifier
        result["local_registered_trials"] = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    finally:
        connection.close()
