"""Tests for the section-detection module.

Section detection is hard to unit-test thoroughly with synthetic
audio (the algorithms are tuned for real music), so this file tests
the API surface and edge cases. Empirical validation against real
songs lives in the example scripts and the Librarian's notes.
"""

import numpy as np
import pytest
import soundfile as sf


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def two_part_audio(tmp_path):
    """A 6-second audio file with a clear feature change at 3.0s:
    first half is a 220 Hz sine wave, second half is white noise.
    The boundary at 3.0s should be detectable."""
    sr = 22050
    duration = 6.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    sine = 0.3 * np.sin(2 * np.pi * 220 * t[:int(3 * sr)])
    noise = 0.1 * np.random.randn(int(3 * sr)).astype(np.float32)
    audio = np.concatenate([sine.astype(np.float32), noise])
    audio_stereo = np.stack([audio, audio], axis=-1)
    path = tmp_path / "two_part.wav"
    sf.write(str(path), audio_stereo, sr, subtype="FLOAT")
    return path


@pytest.fixture
def loud_quiet_loud_audio(tmp_path):
    """A 9-second audio file: loud (3s), quiet (3s), loud (3s).
    Each loud section is rich-spectrum noise; the quiet section is
    near silent."""
    sr = 22050
    rng = np.random.default_rng(42)
    loud = 0.3 * rng.standard_normal(3 * sr).astype(np.float32)
    quiet = 0.01 * rng.standard_normal(3 * sr).astype(np.float32)
    audio = np.concatenate([loud, quiet, loud])
    audio_stereo = np.stack([audio, audio], axis=-1)
    path = tmp_path / "lql.wav"
    sf.write(str(path), audio_stereo, sr, subtype="FLOAT")
    return path


# ── find_section_boundaries ──────────────────────────────────────────────────

class TestFindSectionBoundaries:
    def test_returns_n_plus_one_boundaries(self, two_part_audio):
        from stemflow.sections import find_section_boundaries

        bounds = find_section_boundaries(two_part_audio, n_segments=3)
        # n_segments=3 should yield 4 boundaries (incl. 0 and end)
        assert len(bounds) == 4
        assert bounds[0] == 0.0
        assert bounds[-1] == pytest.approx(6.0, abs=0.1)

    def test_boundaries_are_sorted(self, loud_quiet_loud_audio):
        from stemflow.sections import find_section_boundaries

        bounds = find_section_boundaries(loud_quiet_loud_audio, n_segments=4)
        assert np.all(np.diff(bounds) >= 0), "boundaries should be monotonic non-decreasing"

    def test_boundaries_inside_audio_duration(self, two_part_audio):
        from stemflow.sections import find_section_boundaries

        bounds = find_section_boundaries(two_part_audio, n_segments=3)
        assert bounds[0] >= 0.0
        assert bounds[-1] <= 6.01  # allow small float slop


# ── section_features ─────────────────────────────────────────────────────────

class TestSectionFeatures:
    def test_loud_section_has_higher_rms_than_quiet(self, loud_quiet_loud_audio):
        from stemflow.sections import section_features

        loud_stats = section_features(loud_quiet_loud_audio, 0.0, 3.0)
        quiet_stats = section_features(loud_quiet_loud_audio, 3.0, 6.0)
        assert loud_stats.rms_mean > quiet_stats.rms_mean * 5

    def test_returns_correct_duration(self, two_part_audio):
        from stemflow.sections import section_features

        stats = section_features(two_part_audio, 1.0, 3.5)
        assert stats.duration_s == pytest.approx(2.5)
        assert stats.start_s == 1.0
        assert stats.end_s == 3.5

    def test_inverted_range_raises(self, two_part_audio):
        from stemflow.sections import section_features

        with pytest.raises(ValueError, match="must be greater"):
            section_features(two_part_audio, 3.0, 1.0)

    def test_all_fields_populated(self, two_part_audio):
        from stemflow.sections import section_features

        stats = section_features(two_part_audio, 0.5, 2.5)
        # Every field should be a finite number
        for field in ("rms_mean", "spectral_centroid_mean", "spectral_flatness_mean",
                      "chroma_strength", "timbral_variance", "density"):
            value = getattr(stats, field)
            assert np.isfinite(value), f"{field} is not finite: {value}"
            assert value >= 0, f"{field} is negative: {value}"


# ── find_riff_candidates ─────────────────────────────────────────────────────

class TestFindRiffCandidates:
    def test_filters_out_short_sections(self, loud_quiet_loud_audio):
        from stemflow.sections import find_riff_candidates

        # min_duration_s = 5s should reject 3-second segments
        candidates = find_riff_candidates(
            loud_quiet_loud_audio, n_segments=3, min_duration_s=5.0,
        )
        assert all(c.duration_s >= 5.0 for c in candidates)

    def test_returns_sorted_by_start(self, loud_quiet_loud_audio):
        from stemflow.sections import find_riff_candidates

        candidates = find_riff_candidates(
            loud_quiet_loud_audio, n_segments=4, min_duration_s=0.5,
            min_rms_percentile=0.0, max_flatness=1.0,  # accept everything
        )
        starts = [c.start_s for c in candidates]
        assert starts == sorted(starts)

    def test_empty_when_all_filtered_out(self, loud_quiet_loud_audio):
        from stemflow.sections import find_riff_candidates

        # Impossible threshold: nothing passes
        candidates = find_riff_candidates(
            loud_quiet_loud_audio, n_segments=4, min_duration_s=100.0,
        )
        assert candidates == []
