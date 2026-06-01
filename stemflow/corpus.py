"""Persistent corpus index for stemflow tracks.

SQLite-backed store keyed by absolute file path. One row per track,
holding the BPM, key, duration, and the mashability features
(`chroma_mean` and `spectral_profile`) from `analyze.compute_features`.
Designed to scale from per-project caching (today's behavior) to
listener-scoped corpus management.

The index is the bridge from one-shot pair scoring (`mashability.
score_pair`) to corpus-wide candidate generation (`find_mashups`).

Default location: `~/.stemflow/corpus.db`. Override with the
`db_path` keyword on any function, or via `STEMFLOW_CORPUS_DB` in the
environment.

Schema is intentionally simple. Features are stored as JSON text for
forward compatibility (one schema version today, no migrations). The
file's modification time is recorded so re-indexing skips unchanged
files unless `force=True`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .analyze import analyze_track, compute_features
from .config import AUDIO_EXTENSIONS
from .mashability import MashabilityScore, score_pair

log = logging.getLogger(__name__)


# ── Path resolution ──────────────────────────────────────────────────────────

def default_db_path() -> Path:
    """Resolve the default corpus database path.

    Order of precedence:
        1. STEMFLOW_CORPUS_DB environment variable (absolute path).
        2. ~/.stemflow/corpus.db.
    """
    env = os.environ.get("STEMFLOW_CORPUS_DB")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".stemflow" / "corpus.db"


# ── Connection / schema ──────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    file_path TEXT PRIMARY KEY,
    file_mtime REAL NOT NULL,
    indexed_at TEXT NOT NULL,
    bpm REAL,
    key TEXT,
    duration_seconds REAL,
    chroma_mean_json TEXT,
    spectral_centroid_mean REAL,
    spectral_bandwidth_mean REAL,
    spectral_rolloff_mean REAL,
    spectral_profile_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_bpm ON tracks(bpm);
CREATE INDEX IF NOT EXISTS idx_tracks_key ON tracks(key);
"""


@contextmanager
def _connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Open a connection with row dicts. Creates the parent dir and
    initializes the schema on first use."""
    path = (db_path or default_db_path()).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> Path:
    """Create the corpus database (and its parent directory) if missing.

    Returns:
        The resolved absolute path of the database file.
    """
    path = (db_path or default_db_path()).expanduser().resolve()
    with _connect(path):
        pass
    return path


# ── Row <-> dict conversion ──────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Deserialize a stored row into a dict with numpy arrays."""
    return {
        "file_path": row["file_path"],
        "file_mtime": row["file_mtime"],
        "indexed_at": row["indexed_at"],
        "bpm": row["bpm"],
        "key": row["key"],
        "duration_seconds": row["duration_seconds"],
        "chroma_mean": (
            np.asarray(json.loads(row["chroma_mean_json"]), dtype=float)
            if row["chroma_mean_json"] else None
        ),
        "spectral_centroid_mean": row["spectral_centroid_mean"],
        "spectral_bandwidth_mean": row["spectral_bandwidth_mean"],
        "spectral_rolloff_mean": row["spectral_rolloff_mean"],
        "spectral_profile": (
            np.asarray(json.loads(row["spectral_profile_json"]), dtype=float)
            if row["spectral_profile_json"] else None
        ),
    }


def _features_dict(row: dict) -> dict:
    """Project a track row to the dict shape `score_pair` expects."""
    return {
        "chroma_mean": row["chroma_mean"],
        "spectral_profile": row["spectral_profile"],
    }


# ── Indexing ─────────────────────────────────────────────────────────────────

def index_track(
    audio_file: Path,
    db_path: Optional[Path] = None,
    force: bool = False,
    device: str = "mps",
) -> Optional[dict]:
    """Analyze a single track and store it in the corpus.

    Runs `analyze_track` (for BPM + key) and `compute_features` (for
    chroma + spectral profile), then upserts the row keyed by absolute
    file path.

    If the file is already indexed and its mtime has not changed, the
    function returns the existing row unchanged unless `force=True`.

    Args:
        audio_file: Path to the audio file.
        db_path: Override the default DB location.
        force: Re-analyze even if the file's mtime matches the cached
            value.
        device: Inference device passed to `analyze_track` for beat
            detection ("mps", "cuda", "cpu").

    Returns:
        The stored row as a dict, or None if the file could not be
        read.
    """
    audio_file = Path(audio_file).expanduser().resolve()
    if not audio_file.is_file():
        log.warning("index_track: not a file: %s", audio_file)
        return None

    mtime = audio_file.stat().st_mtime

    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM tracks WHERE file_path = ?", (str(audio_file),)
        ).fetchone()
        if existing and not force and abs(existing["file_mtime"] - mtime) < 1e-6:
            log.info("index_track: skip unchanged %s", audio_file.name)
            return _row_to_dict(existing)

    log.info("index_track: analyzing %s", audio_file.name)
    analysis = analyze_track(audio_file, device=device)
    features = compute_features(audio_file)

    # Duration via soundfile to avoid loading the whole file again.
    try:
        import soundfile as sf
        info = sf.info(str(audio_file))
        duration = float(info.frames / info.samplerate) if info.samplerate else None
    except Exception:
        duration = None

    row = {
        "file_path": str(audio_file),
        "file_mtime": mtime,
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bpm": float(analysis.bpm),
        "key": analysis.key,
        "duration_seconds": duration,
        "chroma_mean_json": json.dumps(np.asarray(features["chroma_mean"], dtype=float).tolist()),
        "spectral_centroid_mean": float(features["spectral_centroid_mean"]),
        "spectral_bandwidth_mean": float(features["spectral_bandwidth_mean"]),
        "spectral_rolloff_mean": float(features["spectral_rolloff_mean"]),
        "spectral_profile_json": json.dumps(np.asarray(features["spectral_profile"], dtype=float).tolist()),
    }

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tracks (
                file_path, file_mtime, indexed_at, bpm, key, duration_seconds,
                chroma_mean_json, spectral_centroid_mean, spectral_bandwidth_mean,
                spectral_rolloff_mean, spectral_profile_json
            ) VALUES (
                :file_path, :file_mtime, :indexed_at, :bpm, :key, :duration_seconds,
                :chroma_mean_json, :spectral_centroid_mean, :spectral_bandwidth_mean,
                :spectral_rolloff_mean, :spectral_profile_json
            )
            ON CONFLICT(file_path) DO UPDATE SET
                file_mtime = excluded.file_mtime,
                indexed_at = excluded.indexed_at,
                bpm = excluded.bpm,
                key = excluded.key,
                duration_seconds = excluded.duration_seconds,
                chroma_mean_json = excluded.chroma_mean_json,
                spectral_centroid_mean = excluded.spectral_centroid_mean,
                spectral_bandwidth_mean = excluded.spectral_bandwidth_mean,
                spectral_rolloff_mean = excluded.spectral_rolloff_mean,
                spectral_profile_json = excluded.spectral_profile_json
            """,
            row,
        )
        conn.commit()
        stored = conn.execute(
            "SELECT * FROM tracks WHERE file_path = ?", (str(audio_file),)
        ).fetchone()
        return _row_to_dict(stored)


def index_directory(
    directory: Path,
    db_path: Optional[Path] = None,
    recursive: bool = True,
    force: bool = False,
    device: str = "mps",
) -> int:
    """Index every audio file under `directory`.

    Args:
        directory: Directory to scan.
        db_path: Override the default DB location.
        recursive: If True (default), walks subdirectories. If False,
            indexes only direct children.
        force: Re-analyze each file even if unchanged.
        device: Beat-detection device passed through to `index_track`.

    Returns:
        Number of files indexed (including unchanged-skipped ones).
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in directory.glob(pattern)
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    log.info("index_directory: %d audio files under %s", len(files), directory)

    count = 0
    for f in files:
        try:
            if index_track(f, db_path=db_path, force=force, device=device) is not None:
                count += 1
        except Exception as e:
            log.warning("index_directory: failed on %s: %s", f.name, e)
    return count


# ── Read / query ─────────────────────────────────────────────────────────────

def get_track(file_path: Path, db_path: Optional[Path] = None) -> Optional[dict]:
    """Fetch a single indexed track by absolute path."""
    file_path = Path(file_path).expanduser().resolve()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM tracks WHERE file_path = ?", (str(file_path),)
        ).fetchone()
        return _row_to_dict(row) if row else None


def all_tracks(db_path: Optional[Path] = None) -> list[dict]:
    """Return every indexed track as a list of dicts."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM tracks ORDER BY file_path").fetchall()
        return [_row_to_dict(r) for r in rows]


def query(
    db_path: Optional[Path] = None,
    bpm_min: Optional[float] = None,
    bpm_max: Optional[float] = None,
    key: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Filter indexed tracks by simple criteria.

    Args:
        db_path: Override the default DB location.
        bpm_min, bpm_max: Inclusive BPM bounds.
        key: Exact match on key string (e.g., "F# minor").
        limit: Maximum number of rows to return.

    Returns:
        Matching tracks, ordered by file_path.
    """
    clauses, params = [], []
    if bpm_min is not None:
        clauses.append("bpm >= ?")
        params.append(float(bpm_min))
    if bpm_max is not None:
        clauses.append("bpm <= ?")
        params.append(float(bpm_max))
    if key is not None:
        clauses.append("key = ?")
        params.append(key)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM tracks {where} ORDER BY file_path{limit_sql}",
            params,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


# ── Candidate generation ─────────────────────────────────────────────────────

def find_mashups(
    target_file: Path,
    db_path: Optional[Path] = None,
    top: int = 10,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    search_transposes: bool = True,
    exclude_self: bool = True,
) -> list[tuple[MashabilityScore, dict]]:
    """Score every indexed track against `target_file` and return the
    best matches.

    Computes features for the target on the fly (the target does not
    need to be in the corpus already). For each indexed track, calls
    `mashability.score_pair` against the target and collects the
    results. Returns the top `top` matches sorted by total score
    descending.

    Args:
        target_file: Path to the seed track for matching.
        db_path: Override the default DB location.
        top: Maximum number of matches to return.
        weights: (harmonic, rhythmic, spectral) weights for scoring.
        search_transposes: Search semitone shifts for best harmonic.
        exclude_self: If True (default), drop the target's own row
            from the results if it appears in the corpus.

    Returns:
        List of (score, track_dict) tuples, longest-first by score.
    """
    target_file = Path(target_file).expanduser().resolve()
    target_features = compute_features(target_file)
    target_analysis = analyze_track(target_file)

    results: list[tuple[MashabilityScore, dict]] = []
    for track in all_tracks(db_path=db_path):
        if exclude_self and track["file_path"] == str(target_file):
            continue
        if track["chroma_mean"] is None or track["spectral_profile"] is None:
            continue
        score = score_pair(
            target_features,
            _features_dict(track),
            target_analysis.bpm,
            track["bpm"] or 0.0,
            weights=weights,
            search_transposes=search_transposes,
        )
        results.append((score, track))

    results.sort(key=lambda pair: pair[0].total, reverse=True)
    return results[:top]
