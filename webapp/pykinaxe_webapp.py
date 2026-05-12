"""Flask web backend and persistent job runtime for pyKinaXe.

This module is the operational heart of the web application. It exposes the
HTTP endpoints used by the browser frontend, manages uploaded inputs, stores
job state under the runtime root, and executes analysis jobs through a
persistent FIFO queue.

The runtime model is intentionally filesystem-based:

- locally, ``PYKINAXE_WEB_RUNTIME_ROOT`` can point to ``webapp/runtime/``
- on Hugging Face Spaces, the same variable can point to a mounted Storage
  Bucket path

That design keeps local development and deployment behavior as similar as
possible while still allowing queue state, logs, uploads, and results to live
outside the Python process itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import socket
import threading
import time
import uuid
import sys
import zipfile

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    # The web app is launched as a standalone module, so we make sure the
    # repository root is importable before pulling in the shared pipeline code.
    sys.path.insert(0, str(REPO_ROOT))

from webapp.backend.kx_web_kinase_extraction_pipeline import (
    WebPipelineRun,
    collect_frontend_results,
    create_results_archive,
    resolve_uploaded_data_dir,
    run_web_pipeline,
)


RUNTIME_ROOT = Path(
    os.environ.get("PYKINAXE_WEB_RUNTIME_ROOT", APP_ROOT / "runtime")
).resolve()
JOBS_ROOT = RUNTIME_ROOT / "jobs"
# Keep the runtime on a normal filesystem path. In local development this can
# stay as webapp/runtime; on Hugging Face Spaces the same path can point to an
# attached Storage Bucket mount via PYKINAXE_WEB_RUNTIME_ROOT.
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("PYKINAXE_WEB_MAX_WORKERS", "1")))
JOB_RETENTION_HOURS = float(os.environ.get("PYKINAXE_WEB_JOB_RETENTION_HOURS", "24"))
JOB_MAX_COUNT = int(os.environ.get("PYKINAXE_WEB_JOB_MAX_COUNT", "20"))
# Hard ceiling on how long any job (running or finished) may live on disk.
# Keeps deployments tidy even when browsers crash without sending a beacon.
JOB_MAX_AGE_SECONDS = float(os.environ.get("PYKINAXE_WEB_JOB_MAX_AGE_SECONDS", "3600"))
# How long we tolerate a client being silent before considering a job
# abandoned (browser closed, tab closed, network dropped, laptop slept...).
JOB_HEARTBEAT_TIMEOUT_SECONDS = float(
    os.environ.get("PYKINAXE_WEB_JOB_HEARTBEAT_TIMEOUT_SECONDS", "60")
)
VIEWER_HEARTBEAT_TIMEOUT_SECONDS = float(
    os.environ.get("PYKINAXE_WEB_VIEWER_HEARTBEAT_TIMEOUT_SECONDS", "45")
)
# How often the background reaper walks the job table.
JOB_REAPER_INTERVAL_SECONDS = float(
    os.environ.get("PYKINAXE_WEB_JOB_REAPER_INTERVAL_SECONDS", "15")
)

app = Flask(__name__, template_folder=".", static_folder="static")

# Allow the static frontend (GitHub Pages, local dev, etc.) to call this API
# from a different origin. Comma-separated list, e.g.
#   PYKINAXE_WEB_ALLOWED_ORIGINS="https://you.github.io,http://127.0.0.1:5500"
_allowed_origins_env = os.environ.get(
    "PYKINAXE_WEB_ALLOWED_ORIGINS", ""
).strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else ["*"]
)


@app.after_request
def _add_cors_headers(response):
    """Attach CORS headers to every HTTP response.
    
    Args:
        response: Response processed by this function.
    
    Returns:
        object: Add cors headers.
    """
    origin = request.headers.get("Origin", "")
    if "*" in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def _cors_preflight(_any):
    """Handle CORS preflight requests for API endpoints.
    
    Args:
        _any: Any processed by this function.
    
    Returns:
        tuple: Cors preflight.
    """
    return ("", 204)


app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("PYKINAXE_WEB_MAX_CONTENT_LENGTH", str(1024 * 1024 * 1024))
)
app.config["MAX_FORM_MEMORY_SIZE"] = int(
    os.environ.get("PYKINAXE_WEB_MAX_FORM_MEMORY_SIZE", str(16 * 1024 * 1024))
)
app.config["MAX_FORM_PARTS"] = int(
    os.environ.get("PYKINAXE_WEB_MAX_FORM_PARTS", "200000")
)
jobs_lock = threading.Lock()
jobs_condition = threading.Condition(jobs_lock)
jobs: dict[str, dict[str, object]] = {}
viewer_lock = threading.Lock()
viewer_sessions: dict[str, dict[str, object]] = {}
# These flags keep background infrastructure single-instanced even when the
# module is imported more than once inside the same Python process.
_JOB_REAPER_STARTED = False
_JOB_QUEUE_STARTED = False
_NEXT_QUEUE_SEQUENCE = 1


# ---------------------------------------------------------------------------
# Generic runtime/path helpers
# ---------------------------------------------------------------------------
def _find_available_port(preferred_port: int) -> int:
    # When binding only to localhost in development, try a small fallback list
    # so the app still starts if the preferred port is already busy.
    """Return the first available localhost port from the fallback list.
    
    Args:
        preferred_port (int): Preferred localhost port to try before using fallback values.
    
    Returns:
        int: Found available port.
    """
    for port in (preferred_port, 5050, 5051, 8000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError("Could not find an available localhost port for the web app.")


def _ensure_dir(path: Path) -> Path:
    # Return the same path object so callers can create-and-use in one line.
    """Create the directory if needed and return its path.
    
    Args:
        path (Path): Path value processed by this helper.
    
    Returns:
        Path: Ensured dir.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_runtime_audit(message: str) -> None:
    # The audit log is separate from per-job logs and captures server-level
    # events such as startup, upload failures, and reaper actions.
    """Append one server-level event to the runtime audit log.
    
    Args:
        message (str): Status or log message to record.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    _ensure_dir(RUNTIME_ROOT)
    audit_path = RUNTIME_ROOT / "server_audit.log"
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_timestamp_now()} {message}\n")


def _viewer_identity_from_request() -> tuple[str | None, str | None]:
    """Extract viewer identifiers from one HTTP request.

    Args:
        None.

    Returns:
        tuple[str | None, str | None]: Parsed ``(viewer_id, session_id)`` pair.
    """
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        viewer_id = str(payload.get("viewer_id", "")).strip() or None
        session_id = str(payload.get("session_id", "")).strip() or None
        return viewer_id, session_id

    viewer_id = str(request.values.get("viewer_id", "")).strip() or None
    session_id = str(request.values.get("session_id", "")).strip() or None
    return viewer_id, session_id


def _active_viewer_count_unlocked(now: float | None = None) -> int:
    """Count unique non-stale viewers currently present on the page.

    Args:
        now (float | None): Epoch timestamp used for stale-session filtering.

    Returns:
        int: Number of unique active viewers.
    """
    now_value = time.time() if now is None else now
    active_viewer_ids = {
        str(session_data.get("viewer_id"))
        for session_data in viewer_sessions.values()
        if (
            isinstance(session_data.get("last_seen_at"), (int, float))
            and now_value - float(session_data["last_seen_at"])
            <= VIEWER_HEARTBEAT_TIMEOUT_SECONDS
            and session_data.get("viewer_id")
        )
    }
    return len(active_viewer_ids)


def _touch_viewer_session(viewer_id: str, session_id: str) -> int:
    """Register or refresh one active browser session.

    Args:
        viewer_id (str): Stable browser-level viewer identifier.
        session_id (str): Per-tab browser session identifier.

    Returns:
        int: Updated number of unique active viewers.
    """
    now = time.time()
    with viewer_lock:
        viewer_sessions[session_id] = {
            "viewer_id": viewer_id,
            "last_seen_at": now,
        }
        return _active_viewer_count_unlocked(now=now)


def _release_viewer_session(session_id: str) -> int:
    """Remove one browser session from the active viewer table.

    Args:
        session_id (str): Per-tab browser session identifier to release.

    Returns:
        int: Updated number of unique active viewers after removal.
    """
    now = time.time()
    with viewer_lock:
        viewer_sessions.pop(session_id, None)
        return _active_viewer_count_unlocked(now=now)


def _reap_stale_viewer_sessions() -> None:
    """Drop viewer sessions whose heartbeat has expired.

    Args:
        None.

    Returns:
        None: This function is used for side effects and does not return a value.
    """
    if VIEWER_HEARTBEAT_TIMEOUT_SECONDS <= 0:
        return

    now = time.time()
    stale_session_ids: list[str] = []
    with viewer_lock:
        for session_id, session_data in viewer_sessions.items():
            last_seen_at = session_data.get("last_seen_at")
            if not isinstance(last_seen_at, (int, float)):
                stale_session_ids.append(session_id)
                continue
            if now - float(last_seen_at) > VIEWER_HEARTBEAT_TIMEOUT_SECONDS:
                stale_session_ids.append(session_id)

        for session_id in stale_session_ids:
            viewer_sessions.pop(session_id, None)


def _viewer_count_payload() -> dict[str, int]:
    """Build the JSON payload returned by viewer-presence endpoints.

    Args:
        None.

    Returns:
        dict[str, int]: Mapping containing the current active viewer count.
    """
    with viewer_lock:
        return {"viewer_count": _active_viewer_count_unlocked(now=time.time())}


def _job_dir(job_id: str) -> Path:
    """Return the runtime directory for the given job ID.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        Path: Job dir.
    """
    return JOBS_ROOT / job_id


def _job_state_path(job_id: str) -> Path:
    """Return the persisted job-state file path for the given job ID.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        Path: Job state path.
    """
    return _job_dir(job_id) / "job_state.json"


def _timestamp_now() -> str:
    """Return the current timestamp in timezone-aware ISO format.
    
    Args:
        None.
    
    Returns:
        str: Timestamp now.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _persist_job_unlocked(job_id: str) -> None:
    """Persist one job record into the runtime root.
    
    Keeping job state on disk means local development and a mounted HF Storage
    Bucket use the exact same code path.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    job = jobs.get(job_id)
    if job is None:
        return
    state_path = _job_state_path(job_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, **job}
    state_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _persist_job(job_id: str) -> None:
    # Convenience wrapper for callers that are not already inside the job lock.
    """Persist one job record using the thread-safe wrapper.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    with jobs_lock:
        _persist_job_unlocked(job_id)


# ---------------------------------------------------------------------------
# Queue bookkeeping helpers
# ---------------------------------------------------------------------------
def _notify_job_workers_unlocked() -> None:
    # Wake queue workers whenever queued/running state changes.
    """Wake any queue workers waiting for state changes.
    
    Args:
        None.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    jobs_condition.notify_all()


def _is_active_status(status: object) -> bool:
    """Return whether the job status represents active execution.
    
    Args:
        status (object): Status processed by this function.
    
    Returns:
        bool: Is active status.
    """
    return status in {"starting", "running"}


def _sorted_queued_job_ids_unlocked() -> list[str]:
    # Queue order is explicit and stable: first by assigned FIFO sequence, then
    # by creation time as a fallback for restored or partially populated jobs.
    """Return queued job IDs in their effective FIFO order.
    
    Args:
        None.
    
    Returns:
        list[str]: Sorted queued job IDs unlocked.
    """
    queued_items = [
        (
            int(job_data.get("queue_sequence") or 0),
            job_data.get("created_at") or "",
            job_id,
        )
        for job_id, job_data in jobs.items()
        if job_data.get("status") == "queued" and not job_data.get("released", False)
    ]
    queued_items.sort()
    return [job_id for _, _, job_id in queued_items]


def _sorted_line_job_ids_unlocked() -> list[str]:
    """Return job IDs participating in the user-visible queue/line order.

    Args:
        None.

    Returns:
        list[str]: Sorted line job IDs.
    """
    line_statuses = {
        "waiting_for_upload",
        "upload_ready",
        "uploading",
        "queued",
        "starting",
        "running",
        "completed",
    }
    line_items = [
        (
            int(job_data.get("queue_sequence") or 0),
            job_data.get("created_at") or "",
            job_id,
        )
        for job_id, job_data in jobs.items()
        if job_data.get("status") in line_statuses and not job_data.get("released", False)
    ]
    line_items.sort()
    return [job_id for _, _, job_id in line_items]


def _status_blocks_next_upload_turn(status: object) -> bool:
    """Return whether the given status should block the next user's upload turn.

    Args:
        status (object): Status processed by this function.

    Returns:
        bool: Whether the status blocks the next upload turn.
    """
    return status in {"upload_ready", "uploading", "queued", "starting", "running", "completed"}


def _preserved_waiting_job_ids_unlocked(*, exclude_job_ids: set[str] | None = None) -> set[str]:
    """Return queued ticket IDs whose state files must survive bucket wipes.

    Args:
        exclude_job_ids (set[str] | None): Optional job IDs excluded from the result.

    Returns:
        set[str]: Waiting/upload-ready job IDs to preserve.
    """
    excluded = exclude_job_ids or set()
    return {
        job_id
        for job_id, job_data in jobs.items()
        if job_id not in excluded
        and not job_data.get("released", False)
        and job_data.get("status") in {"waiting_for_upload", "upload_ready"}
    }


def _maybe_promote_next_upload_turn_unlocked() -> str | None:
    """Grant the next waiting browser permission to begin its upload.

    Args:
        None.

    Returns:
        str | None: Promoted job ID, if any.
    """
    for job_data in jobs.values():
        if job_data.get("released", False):
            continue
        if _status_blocks_next_upload_turn(job_data.get("status")):
            return None

    for job_id in _sorted_line_job_ids_unlocked():
        job = jobs.get(job_id)
        if job is None or job.get("released", False):
            continue
        if job.get("status") != "waiting_for_upload":
            continue
        job["status"] = "upload_ready"
        job["message"] = "Your turn has arrived. The browser can now upload PTK/STK data."
        _persist_job_unlocked(job_id)
        return job_id

    return None


def _queue_summary_unlocked(job_id: str) -> tuple[int | None, int | None, int]:
    """Return queue-position details for the requested job.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        tuple[int | None, int | None, int]: Queue summary unlocked.
    """
    job = jobs.get(job_id)
    if job is None:
        return None, None, 0

    line_ids = _sorted_line_job_ids_unlocked()
    if job_id not in line_ids:
        return None, None, len(line_ids)

    queued_index = line_ids.index(job_id)
    jobs_ahead = queued_index
    queue_position = jobs_ahead + 1
    return queue_position, jobs_ahead, len(line_ids)


def _next_queue_sequence_unlocked() -> int:
    """Return the next FIFO queue sequence number.
    
    Args:
        None.
    
    Returns:
        int: Next queue sequence unlocked.
    """
    global _NEXT_QUEUE_SEQUENCE
    sequence = _NEXT_QUEUE_SEQUENCE
    _NEXT_QUEUE_SEQUENCE += 1
    return sequence


def _enqueue_job(job_id: str, *, message: str) -> None:
    """Place a job into the persistent FIFO queue.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
        message (str): Status or log message to record.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    with jobs_condition:
        job = jobs.get(job_id)
        if job is None:
            return
        # Reset any stale execution state before the job enters the queue again.
        # This matters for retried jobs and for jobs re-queued after a server
        # restart; each queue entry should look like a fresh pending run.
        job["status"] = "queued"
        job["message"] = message
        job["started_at"] = None
        job["finished_at"] = None
        job["error"] = None
        job["results"] = None
        job["archive_path"] = None
        job["released"] = False
        job["cancel_after_finish"] = False
        if not isinstance(job.get("queue_sequence"), int):
            job["queue_sequence"] = _next_queue_sequence_unlocked()
        _persist_job_unlocked(job_id)
        _notify_job_workers_unlocked()


def _append_job_log(job_id: str, message: str) -> None:
    """Append one message to the rolling per-job log.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
        message (str): Status or log message to record.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    timestamp = _timestamp_now()
    with jobs_lock:
        if job_id not in jobs:
            return
        # Keep a rolling in-memory/on-disk tail instead of unbounded logs so a
        # long-running job cannot grow its state file forever.
        logs = list(jobs[job_id].get("logs", []))
        logs.append({"timestamp": timestamp, "message": message})
        jobs[job_id]["logs"] = logs[-200:]
        jobs[job_id]["message"] = message
        _persist_job_unlocked(job_id)


def _public_job_payload(job_id: str) -> dict[str, object]:
    """Build the browser-facing status payload for a job.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        dict[str, object]: Public job payload.
    """
    with jobs_lock:
        job = dict(jobs[job_id])
        queue_position, jobs_ahead, queued_jobs_total = _queue_summary_unlocked(job_id)

    # Keep the public payload focused on information the browser actually uses:
    # current state, recent logs, queue position, and result download metadata.
    payload = {
        "job_id": job_id,
        "status": job["status"],
        "message": job.get("message", ""),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
        "logs": list(job.get("logs", [])),
        "queue_position": queue_position,
        "jobs_ahead": jobs_ahead,
        "queued_jobs_total": queued_jobs_total,
    }
    if job.get("results"):
        # Results are persisted as plain relative paths on disk. Convert them to
        # concrete API URLs here so the browser never has to reconstruct paths.
        results = dict(job["results"])
        results["download_url"] = f"/api/jobs/{job_id}/download"
        if job.get("archive_path"):
            results["archive_name"] = Path(str(job["archive_path"])).name
        results["heatmaps"] = [
            {
                **item,
                "url": f"/api/jobs/{job_id}/files/{item['relative_path']}",
            }
            for item in results.get("heatmaps", [])
        ]
        results["kinase_outputs"] = [
            {
                **item,
                "workbook_url": f"/api/jobs/{job_id}/files/{item['relative_path']}",
            }
            for item in results.get("kinase_outputs", [])
        ]
        payload["results"] = results
    return payload


def _job_archive_metadata(job_id: str) -> dict[str, object]:
    """Return the persisted job metadata useful for archive provenance.

    Args:
        job_id (str): Unique identifier of the web-analysis job.

    Returns:
        dict[str, object]: Archive metadata snapshot for this job.
    """
    with jobs_lock:
        job = dict(jobs.get(job_id, {}))
    return {
        "job_kind": job.get("job_kind"),
        "local_paths": job.get("local_paths"),
        "selected_folders": job.get("selected_folders"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }


def _set_job_state(job_id: str, **updates: object) -> None:
    """Update a job record and persist the new state.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
        **updates (object): Additional keyword arguments forwarded by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    with jobs_condition:
        job = jobs.get(job_id)
        if job is None:
            return
        # A single helper keeps state updates consistent across endpoints,
        # worker threads, and cleanup code.
        job.update(updates)
        _persist_job_unlocked(job_id)
        _notify_job_workers_unlocked()


def _is_finished_status(status: object) -> bool:
    """Return whether finished status.
    
    Args:
        status (object): Status processed by this function.
    
    Returns:
        bool: Is finished status.
    """
    return status in {"completed", "failed"}


def _cleanup_old_jobs() -> None:
    """Prune finished jobs and stale on-disk job directories.
    
    Args:
        None.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    _ensure_dir(JOBS_ROOT)
    retention_seconds = max(JOB_RETENTION_HOURS, 0.0) * 3600.0
    now = time.time()

    with jobs_lock:
        # Work from a snapshot first so we keep the lock held only while
        # reading the job table, not while performing filesystem cleanup.
        jobs_snapshot = {job_id: dict(job_data) for job_id, job_data in jobs.items()}
        removable_finished_jobs = [
            (job_id, dict(job_data))
            for job_id, job_data in jobs_snapshot.items()
            if _is_finished_status(job_data.get("status"))
        ]

    removable_finished_jobs.sort(
        key=lambda item: item[1].get("finished_at") or item[1].get("created_at") or ""
    )

    # Phase 1: remove finished jobs that are older than the retention window.
    expired_job_ids = []
    if retention_seconds > 0:
        for job_id, job_data in removable_finished_jobs:
            timestamp_text = job_data.get("finished_at") or job_data.get("created_at")
            if not timestamp_text:
                continue
            try:
                timestamp_value = datetime.fromisoformat(str(timestamp_text)).timestamp()
            except ValueError:
                continue
            if now - timestamp_value > retention_seconds:
                expired_job_ids.append(job_id)

    # Phase 2: if too many finished jobs exist, keep only the newest ones.
    overflow_job_ids = []
    if JOB_MAX_COUNT > 0 and len(removable_finished_jobs) - len(expired_job_ids) > JOB_MAX_COUNT:
        keep_ids = {
            job_id
            for job_id, _ in removable_finished_jobs[-JOB_MAX_COUNT:]
        }
        overflow_job_ids = [
            job_id
            for job_id, _ in removable_finished_jobs
            if job_id not in keep_ids
        ]

    to_remove = list(dict.fromkeys(expired_job_ids + overflow_job_ids))
    if not to_remove:
        to_remove = []

    # Phase 3: reconcile the in-memory table with on-disk leftovers. Active
    # jobs are intentionally skipped here because the queue/reaper manage them
    # with heartbeat- and status-aware rules elsewhere.
    disk_job_dirs = sorted(path for path in JOBS_ROOT.iterdir() if path.is_dir())
    disk_candidates: list[tuple[str, Path, float]] = []
    for path in disk_job_dirs:
        in_memory_job = jobs_snapshot.get(path.name)
        if in_memory_job is not None and not _is_finished_status(in_memory_job.get("status")):
            continue
        mtime = path.stat().st_mtime
        disk_candidates.append((path.name, path, mtime))

    expired_disk_job_ids = []
    if retention_seconds > 0:
        for job_id, path, mtime in disk_candidates:
            if now - mtime > retention_seconds:
                expired_disk_job_ids.append(job_id)

    if JOB_MAX_COUNT > 0 and len(disk_candidates) - len(expired_disk_job_ids) > JOB_MAX_COUNT:
        keep_ids = {
            job_id
            for job_id, _, _ in sorted(disk_candidates, key=lambda item: item[2])[-JOB_MAX_COUNT:]
        }
        overflow_disk_job_ids = [
            job_id
            for job_id, _, _ in disk_candidates
            if job_id not in keep_ids
        ]
    else:
        overflow_disk_job_ids = []

    to_remove = list(dict.fromkeys(to_remove + expired_disk_job_ids + overflow_disk_job_ids))

    for job_id in to_remove:
        shutil.rmtree(_job_dir(job_id), ignore_errors=True)

    with jobs_lock:
        for job_id in to_remove:
            jobs.pop(job_id, None)


def _run_job(job_id: str) -> None:
    """Execute one uploaded-data job from extracted inputs to packaged outputs.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    job_dir = _job_dir(job_id)
    extracted_dir = job_dir / "extracted"
    output_dir = job_dir / "results"

    try:
        with jobs_lock:
            started_at = (
                jobs.get(job_id, {}).get("started_at") or _timestamp_now()
            )
        _set_job_state(
            job_id,
            status="running",
            message="Resolving uploaded PTK/STK folders",
            started_at=started_at,
        )
        _append_job_log(job_id, "Job started.")

        # Uploaded jobs first resolve the true PTK/STK run directories from the
        # extracted upload tree, then run the same scientific pipeline as the
        # terminal workflow.
        ptk_extract_root = _ensure_dir(extracted_dir / "ptk")
        stk_extract_root = _ensure_dir(extracted_dir / "stk")
        ptk_data_dir = resolve_uploaded_data_dir(ptk_extract_root, peptide_type="PTK")
        stk_data_dir = resolve_uploaded_data_dir(stk_extract_root, peptide_type="STK")
        _append_job_log(job_id, "Preparing pyKinaXe for kinase inference.")

        # At this point the input tree is ready, so we can hand off to the
        # shared scientific pipeline implementation.
        _set_job_state(job_id, message="Running pyKinaXe analysis")
        _append_job_log(job_id, "Running pyKinaXe analysis.")
        run = run_web_pipeline(
            job_id=job_id,
            ptk_data_dir=ptk_data_dir,
            stk_data_dir=stk_data_dir,
            output_root=output_dir,
            progress_callback=lambda message: _append_job_log(job_id, message),
        )

        # The backend pipeline writes many files; summarize the important ones
        # into a compact payload tailored to the web frontend.
        _set_job_state(job_id, message="Collecting result summaries")
        _append_job_log(job_id, "Collecting result summaries.")
        results = collect_frontend_results(run)
        archive_path = create_results_archive(
            run,
            job_metadata=_job_archive_metadata(job_id),
        )
        _append_job_log(job_id, f"Results archived into {Path(archive_path).name}")

        _set_job_state(
            job_id,
            status="completed",
            message="Analysis finished",
            finished_at=_timestamp_now(),
            results=results,
            archive_path=str(archive_path),
        )
        _append_job_log(job_id, "Downloading results.")
    except Exception as exc:  # noqa: BLE001
        # Surface the exception string to the browser while still keeping the
        # worker process alive for later jobs.
        _set_job_state(
            job_id,
            status="failed",
            message="Analysis failed",
            finished_at=_timestamp_now(),
            error=str(exc),
        )
        _append_job_log(job_id, f"Job failed: {exc}")


def _run_local_job(job_id: str, ptk_data_dir: Path, stk_data_dir: Path) -> None:
    """Execute one local-path job without the upload/extraction phase.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
        ptk_data_dir (Path): Directory containing or receiving the PTK data.
        stk_data_dir (Path): Directory containing or receiving the STK data.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    job_dir = _job_dir(job_id)
    output_dir = job_dir / "results"

    try:
        with jobs_lock:
            started_at = (
                jobs.get(job_id, {}).get("started_at") or _timestamp_now()
            )
        _set_job_state(
            job_id,
            status="running",
            message="Validating local PTK/STK folders",
            started_at=started_at,
        )
        _append_job_log(job_id, "Local-path job started.")
        _append_job_log(job_id, f"Using PTK directory: {ptk_data_dir}")
        _append_job_log(job_id, f"Using STK directory: {stk_data_dir}")

        # Local-path jobs skip the upload/extraction phase but otherwise share
        # the exact same backend pipeline implementation.
        if not ptk_data_dir.exists() or not ptk_data_dir.is_dir():
            raise FileNotFoundError(f"PTK folder not found: {ptk_data_dir}")
        if not stk_data_dir.exists() or not stk_data_dir.is_dir():
            raise FileNotFoundError(f"STK folder not found: {stk_data_dir}")

        # Once the local directories are validated, execution is identical to
        # the uploaded-data flow.
        _set_job_state(job_id, message="Running pyKinaXe analysis")
        _append_job_log(job_id, "Running pyKinaXe analysis.")
        run = run_web_pipeline(
            job_id=job_id,
            ptk_data_dir=ptk_data_dir,
            stk_data_dir=stk_data_dir,
            output_root=output_dir,
            progress_callback=lambda message: _append_job_log(job_id, message),
        )

        _set_job_state(job_id, message="Collecting result summaries")
        _append_job_log(job_id, "Collecting result summaries.")
        results = collect_frontend_results(run)
        archive_path = create_results_archive(
            run,
            job_metadata=_job_archive_metadata(job_id),
        )
        _append_job_log(job_id, f"Results archived into {Path(archive_path).name}")

        _set_job_state(
            job_id,
            status="completed",
            message="Analysis finished",
            finished_at=_timestamp_now(),
            results=results,
            archive_path=str(archive_path),
        )
        _append_job_log(job_id, "Downloading results.")
    except Exception as exc:  # noqa: BLE001
        _set_job_state(
            job_id,
            status="failed",
            message="Analysis failed",
            finished_at=_timestamp_now(),
            error=str(exc),
        )
        _append_job_log(job_id, f"Job failed: {exc}")


def _save_uploaded_zip(field_name: str, destination: Path) -> None:
    """Save one uploaded archive to the requested destination.
    
    Args:
        field_name (str): Field name used by this function.
        destination (Path): Path-like value for destination.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    uploaded = request.files.get(field_name)
    if uploaded is None or not uploaded.filename:
        raise ValueError(f"Missing uploaded archive for field '{field_name}'.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    uploaded.save(destination)

def _save_uploaded_folder(file_field_name: str, destination_root: Path) -> int:
    """Save a batch of uploaded files while preserving their folder structure.
    
    Args:
        file_field_name (str): File field name used by this function.
        destination_root (Path): Path-like value for destination root.
    
    Returns:
        int: Saved uploaded folder.
    """
    uploaded_files = request.files.getlist(file_field_name)
    relative_paths = request.form.getlist("relative_paths")
    saved_files = 0

    for index, uploaded in enumerate(uploaded_files):
        if uploaded is None or not uploaded.filename:
            continue

        raw_relative_path = ""
        if index < len(relative_paths):
            raw_relative_path = relative_paths[index]
        if not raw_relative_path:
            raw_relative_path = uploaded.filename

        # Reconstruct the client-side folder tree while stripping dangerous
        # path components so uploads cannot escape the intended destination.
        relative_path = Path(raw_relative_path)
        parts = [part for part in relative_path.parts if part not in {"", ".", ".."}]
        if not parts:
            continue

        target_path = destination_root.joinpath(*parts)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        uploaded.save(target_path)
        saved_files += 1

    if saved_files == 0:
        raise ValueError(f"Missing uploaded folder contents for field '{file_field_name}'.")

    return saved_files


def _sanitize_relative_path(raw_relative_path: str) -> Path:
    # Shared safety check for any API that accepts a browser-provided relative
    # file path.
    """Normalize and validate a client-provided relative path.
    
    Args:
        raw_relative_path (str): Path to the raw relative.
    
    Returns:
        Path: Sanitize relative path.
    """
    relative_path = Path(raw_relative_path)
    parts = [part for part in relative_path.parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("Missing or invalid relative file path.")
    return Path(*parts)


def _save_uploaded_raw_file(destination_root: Path, raw_relative_path: str, payload: bytes) -> Path:
    """Persist one raw uploaded file payload under the job tree.
    
    Args:
        destination_root (Path): Path-like value for destination root.
        raw_relative_path (str): Path to the raw relative.
        payload (bytes): Payload processed by this function.
    
    Returns:
        Path: Saved uploaded raw file.
    """
    relative_path = _sanitize_relative_path(raw_relative_path)
    target_path = destination_root.joinpath(*relative_path.parts)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(payload)
    return target_path


def _extract_uploaded_zip_archive(uploaded_archive, destination_root: Path) -> int:
    """Extract an uploaded ZIP archive into the destination tree.
    
    Args:
        uploaded_archive: Uploaded archive processed by this function.
        destination_root (Path): Path-like value for destination root.
    
    Returns:
        int: Extracted uploaded zip archive.
    """
    destination_root.mkdir(parents=True, exist_ok=True)
    extracted_files = 0

    with zipfile.ZipFile(uploaded_archive) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            parts = [part for part in member_path.parts if part not in {"", ".", ".."}]
            if not parts or member.is_dir():
                continue

            # Extract file-by-file instead of using extractall so we stay in
            # control of path sanitization.
            target_path = destination_root.joinpath(*parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files += 1

    if extracted_files == 0:
        raise ValueError("Uploaded archive did not contain any files.")

    return extracted_files


def _resolve_job_file(job_id: str, relative_path: str) -> Path:
    """Resolve and validate a downloadable file inside a job result directory.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
        relative_path (str): Path to the relative.
    
    Returns:
        Path: Resolved job file.
    """
    root = (_job_dir(job_id) / "results").resolve()
    path = (root / relative_path).resolve()
    # Prevent download endpoints from serving arbitrary files outside the job's
    # result tree, even if a crafted relative path is requested.
    if root not in path.parents and path != root:
        raise ValueError("Requested file path is outside the job result directory.")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _create_empty_job(
    job_id: str,
    *,
    job_kind: str = "uploaded",
    local_paths: dict[str, str] | None = None,
    selected_folders: dict[str, str] | None = None,
    initial_status: str = "waiting_for_upload",
    initial_message: str = "Waiting in line for an upload turn",
    queue_sequence: int | None = None,
) -> None:
    """Create the initial in-memory and on-disk state for a new job.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
        job_kind (str): Job kind used by this function.
        local_paths (dict[str, str] | None): Local paths processed by this function.
        selected_folders (dict[str, str] | None): Browser-visible selected folder names for uploaded jobs.
        initial_status (str): Initial job status persisted for this job.
        initial_message (str): Initial job message persisted for this job.
        queue_sequence (int | None): Explicit queue order when already assigned.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    now_iso = _timestamp_now()
    with jobs_lock:
        # Create the canonical job record up front so every later endpoint only
        # updates well-known fields instead of building ad hoc state.
        jobs[job_id] = {
            "job_kind": job_kind,
            "local_paths": local_paths,
            "selected_folders": selected_folders,
            "status": initial_status,
            "message": initial_message,
            "created_at": now_iso,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "results": None,
            "archive_path": None,
            "logs": [],
            "upload_counts": {"ptk": 0, "stk": 0},
            "last_seen_at": now_iso,
            "queue_sequence": queue_sequence,
            "released": False,
            "cancel_after_finish": False,
        }
        _persist_job_unlocked(job_id)


def _touch_job(job_id: str) -> None:
    """Record that the client is still interested in this job.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    now_iso = _timestamp_now()
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["last_seen_at"] = now_iso
            _persist_job_unlocked(job_id)


def _delete_job(job_id: str) -> None:
    # Delete disk state first, then drop the in-memory entry and notify any
    # blocked workers waiting on queue state changes.
    """Remove all runtime state associated with a job.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    with jobs_condition:
        jobs.pop(job_id, None)
        _maybe_promote_next_upload_turn_unlocked()
        _notify_job_workers_unlocked()

def _purge_runtime_storage(*, preserve_job_ids: set[str] | None = None) -> None:
    """Clear bucket-backed runtime artifacts while preserving queue tickets.

    Args:
        preserve_job_ids (set[str] | None): Job IDs whose ``job_state.json``
            files should survive the purge.

    Returns:
        None: This function is used for side effects and does not return a value.
    """
    preserved = set(preserve_job_ids or set())
    _ensure_dir(RUNTIME_ROOT)
    _ensure_dir(JOBS_ROOT)

    for path in list(RUNTIME_ROOT.iterdir()):
        if path == JOBS_ROOT:
            for job_path in list(JOBS_ROOT.iterdir()):
                if not job_path.is_dir():
                    try:
                        job_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue

                if job_path.name not in preserved:
                    shutil.rmtree(job_path, ignore_errors=True)
                    continue

                state_path = job_path / "job_state.json"
                state_bytes = state_path.read_bytes() if state_path.exists() else None
                shutil.rmtree(job_path, ignore_errors=True)
                job_path.mkdir(parents=True, exist_ok=True)
                if state_bytes is not None:
                    state_path.write_bytes(state_bytes)
            continue

        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    _ensure_dir(JOBS_ROOT)


def _prepare_job_runtime_for_upload(job_id: str) -> bool:
    """Clear prior artifacts, then create a clean upload workspace for one job.

    Args:
        job_id (str): Unique identifier of the web-analysis job.

    Returns:
        bool: ``True`` when the job still existed and was prepared.
    """
    with jobs_condition:
        preserve_job_ids = _preserved_waiting_job_ids_unlocked(exclude_job_ids={job_id})

    _purge_runtime_storage(preserve_job_ids=preserve_job_ids)

    job_dir = _job_dir(job_id)
    _ensure_dir(job_dir / "extracted" / "ptk")
    _ensure_dir(job_dir / "extracted" / "stk")
    _ensure_dir(job_dir / "results")

    with jobs_condition:
        job = jobs.get(job_id)
        if job is None:
            return False
        job["status"] = "uploading"
        job["message"] = "Upload turn granted. Uploading PTK/STK folder contents."
        job["started_at"] = None
        job["finished_at"] = None
        job["error"] = None
        job["results"] = None
        job["archive_path"] = None
        job["upload_counts"] = {"ptk": 0, "stk": 0}
        job["released"] = False
        job["cancel_after_finish"] = False
        _persist_job_unlocked(job_id)
        _notify_job_workers_unlocked()
    return True


def _cleanup_finished_job_for_next_turn(job_id: str) -> None:
    """Delete one finished job, wipe artifacts, and open the next upload turn.

    Args:
        job_id (str): Unique identifier of the finished web-analysis job.

    Returns:
        None: This function is used for side effects and does not return a value.
    """
    with jobs_condition:
        jobs.pop(job_id, None)
        preserve_job_ids = _preserved_waiting_job_ids_unlocked()
        _notify_job_workers_unlocked()

    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    _purge_runtime_storage(preserve_job_ids=preserve_job_ids)

    with jobs_condition:
        promoted_job_id = _maybe_promote_next_upload_turn_unlocked()
        _notify_job_workers_unlocked()

    if promoted_job_id:
        _append_job_log(promoted_job_id, "Server storage prepared. Upload can begin now.")


def _iso_to_epoch(value: object) -> float | None:
    """Convert an ISO timestamp string into epoch seconds.
    
    Args:
        value (object): Input value processed by this helper.
    
    Returns:
        float | None: Iso to epoch.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return None


def _load_jobs_from_runtime() -> None:
    """Restore persisted jobs from the runtime directory at startup.
    
    Args:
        None.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    global _NEXT_QUEUE_SEQUENCE

    _ensure_dir(JOBS_ROOT)
    max_queue_sequence = 0
    restored_jobs = 0

    with jobs_condition:
        jobs.clear()

        for job_path in sorted(JOBS_ROOT.iterdir()):
            if not job_path.is_dir():
                continue

            state_path = job_path / "job_state.json"
            if not state_path.exists():
                continue

            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _append_runtime_audit(
                    f"Could not restore job state from {state_path.name}: {exc!r}"
                )
                continue

            job_id = str(payload.get("job_id") or job_path.name)
            payload.pop("job_id", None)
            payload.setdefault("job_kind", "uploaded")
            payload.setdefault("local_paths", None)
            payload.setdefault("selected_folders", None)
            payload.setdefault("status", "waiting_for_upload")
            payload.setdefault("message", "Recovered job state")
            payload.setdefault("created_at", _timestamp_now())
            payload.setdefault("started_at", None)
            payload.setdefault("finished_at", None)
            payload.setdefault("error", None)
            payload.setdefault("results", None)
            payload.setdefault("archive_path", None)
            payload.setdefault("logs", [])
            payload.setdefault("upload_counts", {"ptk": 0, "stk": 0})
            payload.setdefault("last_seen_at", payload["created_at"])
            payload.setdefault("queue_sequence", None)
            payload.setdefault("released", False)
            payload.setdefault("cancel_after_finish", False)

            status = payload.get("status")

            # Upload turns cannot survive a restart because the browser still
            # holds the original File objects; send those jobs back to the
            # front of the upload line instead of pretending partial data is reusable.
            if status in {"upload_ready", "uploading"}:
                payload["status"] = "waiting_for_upload"
                payload["message"] = "Server restarted. Waiting for a new upload turn."
                payload["upload_counts"] = {"ptk": 0, "stk": 0}
                payload["started_at"] = None
                payload["finished_at"] = None
                payload["error"] = None
                payload["results"] = None
                payload["archive_path"] = None
                logs = list(payload.get("logs", []))
                logs.append(
                    {
                        "timestamp": _timestamp_now(),
                        "message": "Server restarted before upload completed. Waiting for a new upload turn.",
                    }
                )
                payload["logs"] = logs[-200:]

            # If the process died while a job was active, we requeue it on
            # startup instead of pretending it finished.
            if _is_active_status(payload.get("status")):
                payload["status"] = "queued"
                payload["started_at"] = None
                payload["finished_at"] = None
                payload["error"] = None
                logs = list(payload.get("logs", []))
                logs.append(
                    {
                        "timestamp": _timestamp_now(),
                        "message": "Server restarted. Job re-queued.",
                    }
                )
                payload["logs"] = logs[-200:]
                payload["message"] = "Server restarted. Job re-queued."

            if not _is_finished_status(payload.get("status")):
                # Treat restored unfinished jobs as still being watched right
                # now; the heartbeat/reaper logic can decide later if they go stale.
                payload["last_seen_at"] = _timestamp_now()

            queue_sequence = payload.get("queue_sequence")
            if isinstance(queue_sequence, int):
                max_queue_sequence = max(max_queue_sequence, queue_sequence)

            jobs[job_id] = payload
            _persist_job_unlocked(job_id)
            restored_jobs += 1

        _NEXT_QUEUE_SEQUENCE = max_queue_sequence + 1
        _maybe_promote_next_upload_turn_unlocked()
        _notify_job_workers_unlocked()

    if restored_jobs:
        _append_runtime_audit(f"Restored {restored_jobs} job(s) from runtime storage.")


def _claim_next_queued_job() -> tuple[str, dict[str, object]] | None:
    """Claim the next queued job when worker capacity is available.
    
    Args:
        None.
    
    Returns:
        tuple[str, dict[str, object]] | None: Claim next queued job.
    """
    with jobs_condition:
        while True:
            queued_ids = _sorted_queued_job_ids_unlocked()
            active_jobs = [
                job_id
                for job_id, job_data in jobs.items()
                if _is_active_status(job_data.get("status"))
            ]

            # The worker threads pull from a strict FIFO queue. Concurrency is
            # controlled here so the queue can remain persistent and explicit
            # instead of relying on ThreadPoolExecutor internals.
            if queued_ids and len(active_jobs) < MAX_CONCURRENT_JOBS:
                job_id = queued_ids[0]
                job = jobs.get(job_id)
                if job is None:
                    continue
                # Mark the job as claimed before releasing the lock so no other
                # worker can pick up the same queued job.
                job["status"] = "starting"
                job["message"] = "Dequeued. Preparing analysis."
                job["started_at"] = job.get("started_at") or _timestamp_now()
                _persist_job_unlocked(job_id)
                return job_id, dict(job)

            jobs_condition.wait(timeout=1.0)


def _cleanup_finished_released_job(job_id: str) -> None:
    """Delete a released job after it has finished running.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        should_delete = bool(job.get("released") or job.get("cancel_after_finish"))

    if should_delete:
        _delete_job(job_id)
        _append_runtime_audit(f"Deleted released job {job_id} after it finished.")


def _run_claimed_job(job_id: str, job_snapshot: dict[str, object]) -> None:
    # Decide once, at dequeue time, which execution path applies to this job.
    """Dispatch a claimed job to the correct execution path.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
        job_snapshot (dict[str, object]): Job snapshot processed by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    if job_snapshot.get("job_kind") == "local":
        local_paths = job_snapshot.get("local_paths") or {}
        ptk_data_dir = Path(str(local_paths.get("ptk", ""))).expanduser().resolve()
        stk_data_dir = Path(str(local_paths.get("stk", ""))).expanduser().resolve()
        _run_local_job(job_id, ptk_data_dir, stk_data_dir)
    else:
        _run_job(job_id)

    _cleanup_finished_released_job(job_id)


def _reap_abandoned_jobs() -> None:
    """Delete jobs the client has stopped polling for, jobs explicitly
    released by the browser (page refresh / tab close / window close), and
    any job (running or not) older than JOB_MAX_AGE_SECONDS.
    
    Args:
        None.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    now = time.time()
    with jobs_lock:
        snapshot = [(job_id, dict(job_data)) for job_id, job_data in jobs.items()]

    to_delete: list[str] = []
    to_release_after_finish: list[str] = []
    for job_id, job_data in snapshot:
        status = job_data.get("status")
        is_active = _is_active_status(status)

        if job_data.get("released"):
            # A released job was explicitly abandoned by the browser. If it is
            # still active, let it finish and then remove it immediately.
            if is_active:
                to_release_after_finish.append(job_id)
            else:
                to_delete.append(job_id)
            continue

        created_epoch = _iso_to_epoch(job_data.get("created_at"))
        if (
            JOB_MAX_AGE_SECONDS > 0
            and created_epoch is not None
            and now - created_epoch > JOB_MAX_AGE_SECONDS
        ):
            # Absolute age is a hard cap independent of heartbeat freshness.
            if is_active:
                to_release_after_finish.append(job_id)
            else:
                to_delete.append(job_id)
            continue

        last_seen_epoch = (
            _iso_to_epoch(job_data.get("last_seen_at"))
            or _iso_to_epoch(job_data.get("created_at"))
        )
        if (
            JOB_HEARTBEAT_TIMEOUT_SECONDS > 0
            and last_seen_epoch is not None
            and now - last_seen_epoch > JOB_HEARTBEAT_TIMEOUT_SECONDS
        ):
            # Heartbeat timeout handles the common case where the tab disappears
            # without explicitly calling the release endpoint.
            if is_active:
                to_release_after_finish.append(job_id)
            else:
                to_delete.append(job_id)

    with jobs_lock:
        for job_id in to_release_after_finish:
            if job_id in jobs:
                jobs[job_id]["released"] = True
                jobs[job_id]["cancel_after_finish"] = True
                _persist_job_unlocked(job_id)

    for job_id in to_delete:
        _delete_job(job_id)
        _append_runtime_audit(f"Reaped abandoned/expired job {job_id}")

    # Also remove orphaned on-disk job directories that no longer have a
    # corresponding in-memory entry (e.g. left over from a previous server
    # process) and are older than the heartbeat timeout.
    if JOBS_ROOT.exists():
        with jobs_lock:
            known_ids = set(jobs.keys())
        for path in JOBS_ROOT.iterdir():
            if not path.is_dir() or path.name in known_ids:
                continue
            mtime = path.stat().st_mtime
            if (
                JOB_HEARTBEAT_TIMEOUT_SECONDS > 0
                and now - mtime > JOB_HEARTBEAT_TIMEOUT_SECONDS
            ):
                shutil.rmtree(path, ignore_errors=True)
                _append_runtime_audit(f"Reaped orphan job directory {path.name}")

    _reap_stale_viewer_sessions()


def _start_job_reaper() -> None:
    """Run the reaper periodically in a daemon thread.
    
    Args:
        None.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    global _JOB_REAPER_STARTED
    if _JOB_REAPER_STARTED:
        return

    def _loop() -> None:
        """Run the background worker loop for this thread.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        while True:
            try:
                _reap_abandoned_jobs()
            except Exception as exc:  # noqa: BLE001
                _append_runtime_audit(f"Reaper error: {exc!r}")
            time.sleep(max(JOB_REAPER_INTERVAL_SECONDS, 1.0))

    thread = threading.Thread(target=_loop, name="job-reaper", daemon=True)
    thread.start()
    _JOB_REAPER_STARTED = True


def _start_job_queue() -> None:
    """Run a persistent FIFO queue backed by runtime storage.
    
    Args:
        None.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    global _JOB_QUEUE_STARTED
    if _JOB_QUEUE_STARTED:
        return

    def _loop(worker_idx: int) -> None:
        """Run the background worker loop for this thread.
        
        Args:
            worker_idx (int): Zero-based index selecting the worker.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        while True:
            try:
                # Block until a queued job becomes available and capacity
                # exists, then run exactly one claimed job to completion.
                claimed = _claim_next_queued_job()
                if claimed is None:
                    continue
                job_id, job_snapshot = claimed
                _append_runtime_audit(
                    f"Queue worker {worker_idx} started job {job_id}."
                )
                _run_claimed_job(job_id, job_snapshot)
            except Exception as exc:  # noqa: BLE001
                _append_runtime_audit(f"Queue worker {worker_idx} error: {exc!r}")
                time.sleep(1.0)

    for worker_idx in range(MAX_CONCURRENT_JOBS):
        thread = threading.Thread(
            target=_loop,
            args=(worker_idx + 1,),
            name=f"job-queue-{worker_idx + 1}",
            daemon=True,
        )
        thread.start()

    _JOB_QUEUE_STARTED = True


def _initialize_webapp_runtime() -> None:
    """Initialize runtime storage, job restoration, and background workers.
    
    Args:
        None.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    _ensure_dir(JOBS_ROOT)
    # Runtime restoration happens before the worker threads start so any jobs
    # found on disk are visible to the queue immediately.
    _load_jobs_from_runtime()
    _start_job_reaper()
    _start_job_queue()


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(exc: RequestEntityTooLarge):
    """Return a JSON payload for oversized API requests.
    
    Args:
        exc (RequestEntityTooLarge): Exc processed by this function.
    
    Returns:
        object: Handle request entity too large.
    """
    if request.path.startswith("/api/"):
        # API callers expect JSON errors instead of Flask's default HTML page.
        _append_runtime_audit(
            f"RequestEntityTooLarge on {request.path}: {exc.description}"
        )
        return jsonify({"error": exc.description}), 413
    return exc


@app.errorhandler(BadRequest)
def handle_bad_request(exc: BadRequest):
    """Return a JSON payload for malformed API requests.
    
    Args:
        exc (BadRequest): Exc processed by this function.
    
    Returns:
        object: Handle bad request.
    """
    if request.path.startswith("/api/"):
        # Keep frontend error handling simple by normalizing malformed requests
        # into JSON responses.
        _append_runtime_audit(f"BadRequest on {request.path}: {exc.description}")
        return jsonify({"error": exc.description}), 400
    return exc


@app.route("/")
def index():
    # Serve the single-page frontend shell.
    """Serve the main single-page web interface.
    
    Args:
        None.
    
    Returns:
        Response: Rendered HTML response for the main web interface.
    """
    return render_template("index.html")


@app.get("/api/health")
def health_check():
    """Return a lightweight readiness payload for frontend warm-up checks.

    Args:
        None.

    Returns:
        object: JSON response describing current runtime readiness.
    """
    _ensure_dir(JOBS_ROOT)
    return jsonify(
        {
            "status": "ok",
            "runtime_root": str(RUNTIME_ROOT),
            "jobs_root": str(JOBS_ROOT),
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        }
    )


@app.post("/api/viewers/heartbeat")
def heartbeat_viewer():
    """Refresh viewer presence for one browser session.

    Args:
        None.

    Returns:
        object: JSON response containing the current active viewer count.
    """
    viewer_id, session_id = _viewer_identity_from_request()
    if not viewer_id or not session_id:
        return jsonify({"error": "viewer_id and session_id are required."}), 400

    viewer_count = _touch_viewer_session(viewer_id, session_id)
    return jsonify({"viewer_count": viewer_count})


@app.post("/api/viewers/release")
def release_viewer():
    """Release one browser session from the active viewer table.

    Args:
        None.

    Returns:
        object: JSON response containing the updated active viewer count.
    """
    _viewer_id, session_id = _viewer_identity_from_request()
    if not session_id:
        return jsonify({"error": "session_id is required."}), 400

    viewer_count = _release_viewer_session(session_id)
    return jsonify({"viewer_count": viewer_count})


# ---------------------------------------------------------------------------
# HTTP endpoints used by the browser frontend
# ---------------------------------------------------------------------------
@app.post("/api/jobs/init")
def init_job():
    """Create an empty queued upload job without sending folder data yet.
    
    Args:
        None.
    
    Returns:
        tuple: Init job.
    """
    payload = request.get_json(silent=True) or {}
    raw_selected_folders = payload.get("selected_folders")
    selected_folders = None
    if isinstance(raw_selected_folders, dict):
        selected_folders = {
            kind: str(value).strip()
            for kind, value in raw_selected_folders.items()
            if kind in {"ptk", "stk"} and str(value).strip()
        } or None

    job_id = uuid.uuid4().hex
    with jobs_condition:
        queue_sequence = _next_queue_sequence_unlocked()
    _create_empty_job(
        job_id,
        queue_sequence=queue_sequence,
        selected_folders=selected_folders,
    )
    _append_job_log(job_id, "Job created.")
    _append_job_log(job_id, "Waiting in line for an upload turn.")
    with jobs_condition:
        promoted_job_id = _maybe_promote_next_upload_turn_unlocked()
        _notify_job_workers_unlocked()
    _append_runtime_audit(f"Created queued upload job {job_id}")
    return jsonify(_public_job_payload(job_id)), 202


@app.post("/api/jobs/<job_id>/begin_upload")
def begin_job_upload(job_id: str):
    """Grant an upload-ready job a clean runtime and accept incoming ZIPs next.

    Args:
        job_id (str): Unique identifier of the web-analysis job.

    Returns:
        tuple: Updated job payload after upload preparation.
    """
    with jobs_condition:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        if job.get("status") != "upload_ready":
            return jsonify({"error": "This job is not at the front of the upload line yet."}), 409

    if not _prepare_job_runtime_for_upload(job_id):
        abort(404)

    _append_job_log(job_id, "Server storage prepared. Upload can begin now.")
    _append_runtime_audit(f"Prepared clean upload runtime for job {job_id}")
    return jsonify(_public_job_payload(job_id)), 200


@app.post("/api/jobs/start_local")
def start_local_job():
    """Create and queue a job that uses local PTK/STK folder paths.
    
    Args:
        None.
    
    Returns:
        tuple: Start local job.
    """
    payload = request.get_json(silent=True) or {}
    ptk_path = str(payload.get("ptk_path", "")).strip()
    stk_path = str(payload.get("stk_path", "")).strip()

    if not ptk_path or not stk_path:
        return jsonify({"error": "Both PTK and STK local folder paths are required."}), 400

    ptk_data_dir = Path(ptk_path).expanduser().resolve()
    stk_data_dir = Path(stk_path).expanduser().resolve()

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    _ensure_dir(job_dir / "results")
    # Local mode is primarily for development/testing, so the browser sends
    # absolute paths instead of uploading a PTK/STK folder tree.
    _create_empty_job(
        job_id,
        job_kind="local",
        local_paths={"ptk": str(ptk_data_dir), "stk": str(stk_data_dir)},
    )
    _enqueue_job(job_id, message="Local folder paths received. Job queued.")
    _append_job_log(job_id, "Job created.")
    _append_job_log(job_id, "Using local path mode.")
    _append_runtime_audit(
        f"Created local-path job {job_id} with PTK={ptk_data_dir} STK={stk_data_dir}"
    )
    return jsonify(_public_job_payload(job_id)), 202


@app.post("/api/jobs/<job_id>/upload")
def upload_job_files(job_id: str):
    """Accept one multipart batch of PTK or STK folder files.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        tuple: Upload job files.
    """
    with jobs_lock:
        if job_id not in jobs:
            abort(404)
        if jobs[job_id].get("status") != "uploading":
            return jsonify({"error": "This job is not currently accepting uploaded files."}), 409

    kind = str(request.args.get("kind", "")).strip().lower()
    if kind not in {"ptk", "stk"}:
        return jsonify({"error": "Upload kind must be 'ptk' or 'stk'."}), 400

    destination = _job_dir(job_id) / "extracted" / kind
    _append_runtime_audit(
        f"Upload request received for job {job_id}, kind={kind}, content_length={request.content_length}"
    )
    try:
        saved_count = _save_uploaded_folder("files", destination)
    except Exception as exc:  # noqa: BLE001
        _append_runtime_audit(
            f"Upload request failed for job {job_id}, kind={kind}: {exc}"
        )
        return jsonify({"error": str(exc)}), 400

    with jobs_lock:
        counts = dict(jobs[job_id].get("upload_counts", {"ptk": 0, "stk": 0}))
        # Uploads can arrive in multiple batches; keep cumulative counts so the
        # UI can show progress before the job is started.
        counts[kind] = int(counts.get(kind, 0)) + saved_count
        jobs[job_id]["upload_counts"] = counts
        jobs[job_id]["message"] = (
            f"Uploaded {counts.get('ptk', 0)} PTK files and {counts.get('stk', 0)} STK files"
        )
        _persist_job_unlocked(job_id)

    _append_job_log(job_id, f"Uploaded {saved_count} {kind.upper()} files in this batch.")
    _append_runtime_audit(
        f"Upload request completed for job {job_id}, kind={kind}, saved_files={saved_count}"
    )
    return jsonify(_public_job_payload(job_id)), 200


@app.post("/api/jobs/<job_id>/upload_raw")
def upload_job_file_raw(job_id: str):
    """Accept one raw PTK or STK file upload request.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        tuple: Upload job file raw.
    """
    with jobs_lock:
        if job_id not in jobs:
            abort(404)
        if jobs[job_id].get("status") != "uploading":
            return jsonify({"error": "This job is not currently accepting uploaded files."}), 409

    kind = str(request.args.get("kind", "")).strip().lower()
    if kind not in {"ptk", "stk"}:
        return jsonify({"error": "Upload kind must be 'ptk' or 'stk'."}), 400

    raw_relative_path = str(request.args.get("relative_path", "")).strip()
    if not raw_relative_path:
        return jsonify({"error": "Missing relative_path query parameter."}), 400

    destination = _job_dir(job_id) / "extracted" / kind
    _append_runtime_audit(
        f"Raw upload request received for job {job_id}, kind={kind}, relative_path={raw_relative_path}, content_length={request.content_length}"
    )

    try:
        payload = request.get_data(cache=False, as_text=False)
        if not payload:
            raise ValueError("Uploaded file body is empty.")
        # This endpoint is the lowest-level upload path: one request contains
        # the raw bytes for one file at one relative path.
        _save_uploaded_raw_file(destination, raw_relative_path, payload)
    except Exception as exc:  # noqa: BLE001
        _append_runtime_audit(
            f"Raw upload request failed for job {job_id}, kind={kind}, relative_path={raw_relative_path}: {exc!r}"
        )
        return jsonify({"error": str(exc)}), 400

    with jobs_lock:
        counts = dict(jobs[job_id].get("upload_counts", {"ptk": 0, "stk": 0}))
        counts[kind] = int(counts.get(kind, 0)) + 1
        jobs[job_id]["upload_counts"] = counts
        jobs[job_id]["message"] = (
            f"Uploaded {counts.get('ptk', 0)} PTK files and {counts.get('stk', 0)} STK files"
        )
        _persist_job_unlocked(job_id)

    _append_job_log(job_id, f"Uploaded 1 {kind.upper()} file in this batch.")
    _append_runtime_audit(
        f"Raw upload request completed for job {job_id}, kind={kind}, relative_path={raw_relative_path}"
    )
    return jsonify(_public_job_payload(job_id)), 200


@app.post("/api/jobs/<job_id>/upload_zip")
def upload_job_zip(job_id: str):
    """Accept and extract one uploaded PTK or STK ZIP archive.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        tuple: Upload job zip.
    """
    with jobs_lock:
        if job_id not in jobs:
            abort(404)
        if jobs[job_id].get("status") != "uploading":
            return jsonify({"error": "This job is not currently accepting uploaded files."}), 409

    kind = str(request.args.get("kind", "")).strip().lower()
    if kind not in {"ptk", "stk"}:
        return jsonify({"error": "Upload kind must be 'ptk' or 'stk'."}), 400

    archive = request.files.get("archive")
    if archive is None or not archive.filename:
        return jsonify({"error": "Missing uploaded ZIP archive."}), 400

    destination = _job_dir(job_id) / "extracted" / kind
    _append_runtime_audit(
        f"ZIP upload request received for job {job_id}, kind={kind}, filename={archive.filename}, content_length={request.content_length}"
    )

    try:
        # ZIP mode is convenient for clients that can bundle the PTK/STK tree
        # on their side before sending it to the server.
        extracted_count = _extract_uploaded_zip_archive(archive, destination)
    except Exception as exc:  # noqa: BLE001
        _append_runtime_audit(
            f"ZIP upload request failed for job {job_id}, kind={kind}: {exc!r}"
        )
        return jsonify({"error": str(exc)}), 400

    with jobs_lock:
        counts = dict(jobs[job_id].get("upload_counts", {"ptk": 0, "stk": 0}))
        counts[kind] = int(counts.get(kind, 0)) + extracted_count
        jobs[job_id]["upload_counts"] = counts
        jobs[job_id]["message"] = (
            f"Uploaded {counts.get('ptk', 0)} PTK files and {counts.get('stk', 0)} STK files"
        )
        _persist_job_unlocked(job_id)

    _append_job_log(job_id, f"Uploaded and extracted {extracted_count} {kind.upper()} files from ZIP archive.")
    _append_runtime_audit(
        f"ZIP upload request completed for job {job_id}, kind={kind}, extracted_files={extracted_count}"
    )
    return jsonify(_public_job_payload(job_id)), 200


@app.post("/api/jobs/<job_id>/start")
def start_job(job_id: str):
    """Queue an uploaded-data job after both PTK and STK inputs exist.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        tuple: Start job.
    """
    with jobs_lock:
        if job_id not in jobs:
            abort(404)
        if jobs[job_id].get("status") != "uploading":
            return jsonify({"error": "This job is not ready to start yet."}), 409
        counts = dict(jobs[job_id].get("upload_counts", {"ptk": 0, "stk": 0}))

    if counts.get("ptk", 0) == 0 or counts.get("stk", 0) == 0:
        return jsonify({"error": "Both PTK and STK folder contents must be uploaded before starting."}), 400

    # Starting a job does not run it immediately; it enters the persistent FIFO
    # queue and the background workers pick it up when capacity is available.
    _enqueue_job(
        job_id,
        message=(
            f"Upload received ({counts.get('ptk', 0)} PTK files, "
            f"{counts.get('stk', 0)} STK files)"
        ),
    )
    _append_job_log(job_id, "Upload complete.")
    _append_job_log(job_id, f"PTK files uploaded: {counts.get('ptk', 0)}")
    _append_job_log(job_id, f"STK files uploaded: {counts.get('stk', 0)}")
    _append_job_log(job_id, "Job queued.")
    return jsonify(_public_job_payload(job_id)), 202


@app.post("/api/jobs")
def create_job():
    """Create and queue a one-shot multipart upload job.
    
    Args:
        None.
    
    Returns:
        tuple: Created job.
    """
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    extracted_dir = _ensure_dir(job_dir / "extracted")
    _ensure_dir(job_dir / "results")

    try:
        # This is the one-shot upload route used when the client can send both
        # PTK and STK folder contents in a single multipart request.
        ptk_count = _save_uploaded_folder("ptk_files", extracted_dir / "ptk")
        stk_count = _save_uploaded_folder("stk_files", extracted_dir / "stk")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

    _create_empty_job(job_id)
    _set_job_state(job_id, upload_counts={"ptk": ptk_count, "stk": stk_count})
    _enqueue_job(
        job_id,
        message=f"Upload received ({ptk_count} PTK files, {stk_count} STK files)",
    )
    _append_job_log(job_id, "Upload received.")
    _append_job_log(job_id, f"PTK files uploaded: {ptk_count}")
    _append_job_log(job_id, f"STK files uploaded: {stk_count}")
    _append_job_log(job_id, "Job queued.")
    return jsonify(_public_job_payload(job_id)), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    """Return the current status payload for one job.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        object: Requested job.
    """
    with jobs_lock:
        if job_id not in jobs:
            abort(404)
    # Polling the status endpoint also counts as a heartbeat.
    _touch_job(job_id)
    return jsonify(_public_job_payload(job_id))


@app.post("/api/jobs/<job_id>/heartbeat")
def heartbeat_job(job_id: str):
    """Refresh the heartbeat timestamp for one watched job.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        tuple: Heartbeat job.
    """
    with jobs_lock:
        if job_id not in jobs:
            abort(404)
    _touch_job(job_id)
    return ("", 204)


@app.post("/api/jobs/<job_id>/release")
def release_job(job_id: str):
    """Client signalled that it is no longer interested in this job
    (page refresh, tab close, window close). Mark for immediate deletion;
    the background reaper picks it up on its next pass.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        tuple: Release job.
    """
    with jobs_condition:
        job = jobs.get(job_id)
        if job is None:
            return ("", 204)
        if _is_active_status(job.get("status")):
            jobs[job_id]["released"] = True
            # Running jobs are not interrupted mid-analysis; instead they are
            # marked for cleanup as soon as the current run finishes.
            jobs[job_id]["cancel_after_finish"] = True
            _persist_job_unlocked(job_id)
            _notify_job_workers_unlocked()
            return ("", 204)

    _delete_job(job_id)
    return ("", 204)


@app.get("/api/jobs/<job_id>/download")
def download_results(job_id: str):
    """Return the packaged result archive for a completed job.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
    
    Returns:
        object: Download results.
    """
    with jobs_lock:
        if job_id not in jobs:
            abort(404)
        archive_path = jobs[job_id].get("archive_path")
        status = jobs[job_id]["status"]
        cleanup_after_download = str(request.args.get("cleanup", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }

    if status != "completed" or archive_path is None:
        abort(404)
    # Only completed jobs expose a downloadable archive.
    archive_name = Path(str(archive_path)).name
    response = send_file(archive_path, as_attachment=True, download_name=archive_name)
    if cleanup_after_download:
        response.call_on_close(lambda: _cleanup_finished_job_for_next_turn(job_id))
    return response


@app.post("/api/jobs/<job_id>/finalize_download")
def finalize_download(job_id: str):
    """Delete one completed job after the browser has received its ZIP archive.

    Args:
        job_id (str): Unique identifier of the web-analysis job.

    Returns:
        tuple: Empty success response after cleanup scheduling/completion.
    """
    with jobs_lock:
        if job_id not in jobs:
            abort(404)
        if jobs[job_id].get("status") != "completed":
            return jsonify({"error": "Results are not ready to finalize yet."}), 409

    _cleanup_finished_job_for_next_turn(job_id)
    return ("", 204)


@app.get("/api/jobs/<job_id>/files/<path:relative_path>")
def get_job_file(job_id: str, relative_path: str):
    """Serve one individual file from a completed job result tree.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
        relative_path (str): Path to the relative.
    
    Returns:
        object: Requested job file.
    """
    with jobs_lock:
        if job_id not in jobs:
            abort(404)
    try:
        path = _resolve_job_file(job_id, relative_path)
    except (ValueError, FileNotFoundError):
        abort(404)
    # Frontend thumbnails, heatmaps, and workbook links are all served through
    # this safe file resolver.
    return send_file(path)


def run_dev_server() -> None:
    """Run the Flask development server with pyKinaXe runtime initialization.
    
    Args:
        None.
    
    Returns:
        None: Starts the development server and blocks until it exits.
    """
    _initialize_webapp_runtime()
    # Default to 7860 for Hugging Face Spaces; locally still respects PYKINAXE_WEB_PORT.
    preferred_port = int(os.environ.get("PYKINAXE_WEB_PORT", "7860"))
    host = os.environ.get("PYKINAXE_WEB_HOST", "0.0.0.0")
    if host == "127.0.0.1":
        # Only auto-search for a free port in localhost mode. In deployed mode
        # the hosting platform expects a specific bound port.
        port = _find_available_port(preferred_port)
    else:
        port = preferred_port
    print(f"Starting pyKinaXe web app on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run_dev_server()
else:
    # Importing under a WSGI server (for example gunicorn) should initialize
    # the same queue/reaper/runtime state as direct local execution.
    _initialize_webapp_runtime()
