"""Tests for the corpus index.

These tests insert synthetic rows directly via SQL rather than going
through `index_track` / `analyze_track`, which would require real
audio files. The DB layer and query semantics are what's under test
here; the analyze layer is exercised separately (and by the integrated
score CLI).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from stemflow.corpus import (
    _connect,
    all_tracks,
    default_db_path,
    get_track,
    init_db,
    query,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _insert_synthetic(
    db_path: Path,
    file_path: str,
    bpm: float = 120.0,
    key: str = "C major",
    chroma=None,
    profile=None,
) -> None:
    """Insert a synthetic track row, bypassing analyze + compute_features."""
    if chroma is None:
        chroma = np.zeros(12, dtype=float)
        chroma[0] = 1.0
    if profile is None:
        profile = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tracks (
                file_path, file_mtime, indexed_at, bpm, key, duration_seconds,
                chroma_mean_json, spectral_centroid_mean, spectral_bandwidth_mean,
                spectral_rolloff_mean, spectral_profile_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_path,
                1234567890.0,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                bpm,
                key,
                180.0,
                json.dumps(np.asarray(chroma, dtype=float).tolist()),
                2000.0,
                1500.0,
                4000.0,
                json.dumps(np.asarray(profile, dtype=float).tolist()),
            ),
        )
        conn.commit()


# ── default_db_path ──────────────────────────────────────────────────────────

class TestDefaultDbPath:
    def test_default_is_under_home(self):
        # Make sure the env override doesn't leak between tests.
        prior = os.environ.pop("STEMFLOW_CORPUS_DB", None)
        try:
            p = default_db_path()
            assert p == Path.home() / ".stemflow" / "corpus.db"
        finally:
            if prior is not None:
                os.environ["STEMFLOW_CORPUS_DB"] = prior

    def test_env_override(self, tmp_path, monkeypatch):
        target = tmp_path / "my_corpus.db"
        monkeypatch.setenv("STEMFLOW_CORPUS_DB", str(target))
        assert default_db_path() == target.resolve()


# ── init_db ──────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_file_and_parent_dir(self, tmp_path):
        db = tmp_path / "nested" / "corpus.db"
        result = init_db(db)
        assert result == db.resolve()
        assert db.exists()

    def test_schema_has_tracks_table(self, tmp_path):
        db = tmp_path / "c.db"
        init_db(db)
        with sqlite3.connect(str(db)) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'"
            )
            assert cur.fetchone() is not None


# ── insert / get / all ───────────────────────────────────────────────────────

class TestRoundTrip:
    def test_round_trip_features(self, tmp_path):
        db = tmp_path / "c.db"
        init_db(db)

        chroma = np.linspace(0, 1, 12)
        profile = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        _insert_synthetic(db, "/x/y/song.wav", chroma=chroma, profile=profile)

        row = get_track("/x/y/song.wav", db_path=db)
        assert row is not None
        assert row["file_path"] == "/x/y/song.wav"
        np.testing.assert_allclose(row["chroma_mean"], chroma)
        np.testing.assert_allclose(row["spectral_profile"], profile)

    def test_get_track_missing_returns_none(self, tmp_path):
        db = tmp_path / "c.db"
        init_db(db)
        assert get_track("/nowhere/song.wav", db_path=db) is None

    def test_all_tracks_orders_by_path(self, tmp_path):
        db = tmp_path / "c.db"
        init_db(db)
        _insert_synthetic(db, "/b/song.wav")
        _insert_synthetic(db, "/a/song.wav")
        _insert_synthetic(db, "/c/song.wav")
        rows = all_tracks(db_path=db)
        assert [r["file_path"] for r in rows] == ["/a/song.wav", "/b/song.wav", "/c/song.wav"]


# ── query filters ────────────────────────────────────────────────────────────

class TestQuery:
    @pytest.fixture
    def populated_db(self, tmp_path):
        db = tmp_path / "c.db"
        init_db(db)
        _insert_synthetic(db, "/slow.wav", bpm=80, key="A minor")
        _insert_synthetic(db, "/mid.wav", bpm=120, key="C major")
        _insert_synthetic(db, "/fast.wav", bpm=160, key="C major")
        return db

    def test_bpm_min(self, populated_db):
        rows = query(db_path=populated_db, bpm_min=100)
        assert [r["file_path"] for r in rows] == ["/fast.wav", "/mid.wav"]

    def test_bpm_max(self, populated_db):
        rows = query(db_path=populated_db, bpm_max=100)
        assert [r["file_path"] for r in rows] == ["/slow.wav"]

    def test_bpm_range(self, populated_db):
        rows = query(db_path=populated_db, bpm_min=100, bpm_max=140)
        assert [r["file_path"] for r in rows] == ["/mid.wav"]

    def test_key_filter(self, populated_db):
        rows = query(db_path=populated_db, key="C major")
        assert [r["file_path"] for r in rows] == ["/fast.wav", "/mid.wav"]

    def test_combined_filters(self, populated_db):
        rows = query(db_path=populated_db, key="C major", bpm_max=130)
        assert [r["file_path"] for r in rows] == ["/mid.wav"]

    def test_limit(self, populated_db):
        rows = query(db_path=populated_db, limit=2)
        assert len(rows) == 2
