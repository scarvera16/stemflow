"""Tests for the mashability scorer and its components."""

import numpy as np
import pytest

from stemflow.mashability import (
    MashabilityScore,
    harmonic_score,
    rhythmic_score,
    score_pair,
    spectral_score,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _c_major_triad() -> np.ndarray:
    """12-dim chroma with energy on C, E, G."""
    c = np.zeros(12, dtype=float)
    c[[0, 4, 7]] = 1.0
    return c


def _bass_heavy_profile() -> np.ndarray:
    return np.array([0.6, 0.3, 0.05, 0.03, 0.02])


def _treble_heavy_profile() -> np.ndarray:
    return np.array([0.02, 0.03, 0.05, 0.3, 0.6])


# ── harmonic_score ───────────────────────────────────────────────────────────

class TestHarmonic:
    def test_identical_chroma_scores_one(self):
        c = _c_major_triad()
        score, offset = harmonic_score(c, c, search_transposes=True)
        assert score == pytest.approx(1.0)
        assert offset == 0

    def test_transposed_chroma_recovers_offset(self):
        c = _c_major_triad()
        shifted = np.roll(c, 7)  # transpose B up a fifth
        score, offset = harmonic_score(c, shifted, search_transposes=True)
        # +7 and -5 are octave-equivalent; the implementation picks
        # the representation that fits -6..+5 ordering. Either is a
        # valid recovery as long as the score is 1.0.
        assert score == pytest.approx(1.0)
        assert offset in {5, -7, 7, -5}

    def test_disabling_search_returns_raw_offset_zero(self):
        c = _c_major_triad()
        shifted = np.roll(c, 7)
        score, offset = harmonic_score(c, shifted, search_transposes=False)
        assert offset == 0
        # Raw match is poor without search.
        assert score < 0.5

    def test_zero_vector_safe(self):
        zero = np.zeros(12)
        c = _c_major_triad()
        score, offset = harmonic_score(c, zero, search_transposes=True)
        assert score == 0.0


# ── rhythmic_score ───────────────────────────────────────────────────────────

class TestRhythmic:
    def test_identical_bpm(self):
        assert rhythmic_score(120, 120) == pytest.approx(1.0)

    def test_octave_equivalent_bpm(self):
        assert rhythmic_score(120, 240) == pytest.approx(1.0)
        assert rhythmic_score(60, 120) == pytest.approx(1.0)

    def test_close_bpm(self):
        # 120 vs 140: 120/140 = ~0.857
        assert rhythmic_score(120, 140) == pytest.approx(120 / 140, rel=1e-4)

    def test_zero_or_negative(self):
        assert rhythmic_score(0, 120) == 0.0
        assert rhythmic_score(120, 0) == 0.0
        assert rhythmic_score(-1, 120) == 0.0

    def test_extreme_ratio(self):
        # 60 vs 200 → halve 200 to 100, 60/100 = 0.6
        assert rhythmic_score(60, 200) == pytest.approx(0.6, rel=1e-4)


# ── spectral_score ───────────────────────────────────────────────────────────

class TestSpectral:
    def test_identical_profiles_score_zero(self):
        p = _bass_heavy_profile()
        assert spectral_score(p, p) == pytest.approx(0.0)

    def test_complementary_profiles_score_high(self):
        bass = _bass_heavy_profile()
        treble = _treble_heavy_profile()
        score = spectral_score(bass, treble)
        assert score > 0.8

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            spectral_score(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))

    def test_zero_profile_safe(self):
        assert spectral_score(np.zeros(5), _bass_heavy_profile()) == 0.0


# ── score_pair ───────────────────────────────────────────────────────────────

class TestScorePair:
    def test_returns_mashability_score(self):
        c = _c_major_triad()
        feats_a = {"chroma_mean": c, "spectral_profile": _bass_heavy_profile()}
        feats_b = {"chroma_mean": np.roll(c, 7), "spectral_profile": _treble_heavy_profile()}
        result = score_pair(feats_a, feats_b, 120, 128)
        assert isinstance(result, MashabilityScore)
        assert 0.0 <= result.total <= 1.0
        assert 0.0 <= result.harmonic <= 1.0
        assert 0.0 <= result.rhythmic <= 1.0
        assert 0.0 <= result.spectral <= 1.0

    def test_weighted_sum_is_algebraic(self):
        c = _c_major_triad()
        feats_a = {"chroma_mean": c, "spectral_profile": _bass_heavy_profile()}
        feats_b = {"chroma_mean": np.roll(c, 7), "spectral_profile": _treble_heavy_profile()}
        weights = (0.5, 0.3, 0.2)
        r = score_pair(feats_a, feats_b, 120, 128, weights=weights)
        expected = (
            weights[0] * r.harmonic
            + weights[1] * r.rhythmic
            + weights[2] * r.spectral
        )
        assert r.total == pytest.approx(expected, abs=1e-9)

    def test_weights_must_sum_to_one(self):
        c = _c_major_triad()
        feats_a = {"chroma_mean": c, "spectral_profile": _bass_heavy_profile()}
        feats_b = {"chroma_mean": c, "spectral_profile": _treble_heavy_profile()}
        with pytest.raises(ValueError, match="weights must sum to 1"):
            score_pair(feats_a, feats_b, 120, 120, weights=(0.5, 0.5, 0.5))

    def test_missing_feature_key_raises(self):
        c = _c_major_triad()
        feats_a = {"chroma_mean": c}  # no spectral_profile
        feats_b = {"chroma_mean": c, "spectral_profile": _bass_heavy_profile()}
        with pytest.raises(ValueError, match="missing required key"):
            score_pair(feats_a, feats_b, 120, 120)
