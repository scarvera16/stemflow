"""Tests for the analyze module.

Focuses on the fallback paths (librosa-based BPM and key detection),
which run when the optional beat-this! and Essentia dependencies are
not installed. Mocks librosa to avoid needing audio fixtures.
"""

from unittest import mock

import numpy as np
import pytest


# ── detect_bpm librosa fallback ──────────────────────────────────────────────

class TestDetectBpmLibrosaFallback:
    """Cover the regression where librosa 0.10+ returns tempo as a
    1-d numpy array, breaking `float(tempo)` for the older librosa
    0.9.x scalar contract."""

    def _make_patch(self, tempo_value):
        """Build a patch context for the librosa fallback path."""
        # Force the beat-this! short-circuit off so the librosa path runs.
        patches = mock.patch.multiple(
            "stemflow.analyze",
            HAS_BEAT_THIS=False,
        )
        return patches

    def _call_detect(self, tempo_value):
        from stemflow.analyze import detect_bpm

        fake_y = np.zeros(22050, dtype=np.float32)
        fake_sr = 22050
        fake_frames = np.array([10, 20, 30, 40, 50])
        fake_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

        with mock.patch("stemflow.analyze.HAS_BEAT_THIS", False), \
             mock.patch("librosa.load", return_value=(fake_y, fake_sr)), \
             mock.patch("librosa.beat.beat_track", return_value=(tempo_value, fake_frames)), \
             mock.patch("librosa.frames_to_time", return_value=fake_times):
            return detect_bpm("/fake/path.wav")

    def test_scalar_tempo_works(self):
        """librosa 0.9.x style: tempo is a Python float."""
        result = self._call_detect(120.5)
        assert result.bpm == pytest.approx(120.5)

    def test_one_d_array_tempo_works(self):
        """librosa 0.10+ style: tempo is a 1-d numpy array. Regression
        guard for the 'only 0-dimensional arrays can be converted'
        error that surfaced when indexing the corpus."""
        result = self._call_detect(np.array([124.0]))
        assert result.bpm == pytest.approx(124.0)

    def test_zero_d_array_tempo_works(self):
        """0-d numpy scalar (some librosa configurations)."""
        result = self._call_detect(np.array(100.0))
        assert result.bpm == pytest.approx(100.0)


# ── compute_features sanity ──────────────────────────────────────────────────

class TestComputeFeaturesContract:
    """compute_features returns a dict with the keys mashability.score_pair
    expects. Smoke-tests the return shape against a mocked librosa.load."""

    def test_returns_required_keys_and_shapes(self):
        from stemflow.analyze import compute_features

        # Two seconds of zero audio is enough; we're testing plumbing,
        # not numerics. Real audio is exercised by manual smoke tests
        # (stemflow score / index) outside the unit-test boundary.
        fake_sr = 22050
        fake_y = np.zeros(fake_sr * 2, dtype=np.float32) + 0.01  # tiny non-zero

        with mock.patch("librosa.load", return_value=(fake_y, fake_sr)):
            features = compute_features("/fake/path.wav")

        assert set(features.keys()) >= {
            "chroma_mean",
            "spectral_centroid_mean",
            "spectral_bandwidth_mean",
            "spectral_rolloff_mean",
            "spectral_profile",
        }
        assert features["chroma_mean"].shape == (12,)
        assert features["spectral_profile"].shape == (5,)
        assert isinstance(features["spectral_centroid_mean"], float)
