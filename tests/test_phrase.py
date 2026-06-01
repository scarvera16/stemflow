"""Tests for the phrase-alignment module.

Uses synthetic audio with known periodicity rather than real song
fixtures, so the tests are deterministic and run in milliseconds.
"""

import numpy as np
import pytest


# ── Synthetic audio helpers ──────────────────────────────────────────────────

def _click(sr: int, duration_s: float, click_times_s: list[float], amp: float = 0.7) -> np.ndarray:
    """Build a mono click track: short impulses at the requested times."""
    n = int(duration_s * sr)
    y = np.zeros(n, dtype=np.float32)
    click_len = int(0.005 * sr)  # 5ms click
    for t in click_times_s:
        s = int(t * sr)
        if 0 <= s < n - click_len:
            y[s:s + click_len] += amp
    return y


def _periodic_pattern(sr: int, n_bars: int, bar_period: float,
                      pattern_offsets_in_bar: list[float],
                      pre_pickup_offsets: list[float] | None = None) -> tuple[np.ndarray, float]:
    """Build a click track that repeats `pattern_offsets_in_bar` for N bars,
    optionally with an artifact (pickup notes) BEFORE bar 1.

    Returns (audio, bar1_downbeat_time_in_seconds).
    """
    pre_pickup_offsets = pre_pickup_offsets or []
    # Build clicks: pickup notes negative of bar-1, then N bars of pattern
    times = []
    bar1_t = 1.0  # offset bar 1 from t=0 so pickups have room
    for p in pre_pickup_offsets:
        times.append(bar1_t + p)  # p is negative
    for bar in range(n_bars):
        for off in pattern_offsets_in_bar:
            times.append(bar1_t + bar * bar_period + off)
    duration = bar1_t + n_bars * bar_period + bar_period
    return _click(sr, duration, times), bar1_t


# ── bar_period_from_onsets ───────────────────────────────────────────────────

class TestBarPeriod:
    def test_recovers_known_period(self):
        from stemflow.phrase import bar_period_from_onsets

        sr = 22050
        # Build 8 bars of clicks on beats 1 and 3 of every bar at 80 BPM
        # 80 BPM => beat = 0.75s, bar = 3.0s
        true_period = 3.0
        audio, _ = _periodic_pattern(
            sr, n_bars=8, bar_period=true_period,
            pattern_offsets_in_bar=[0.0, 1.5],  # beats 1 and 3
        )
        detected = bar_period_from_onsets(audio, sr, search_seconds=(2.0, 4.0))
        # Allow ~3% tolerance because the onset envelope has finite resolution
        assert detected == pytest.approx(true_period, rel=0.03)

    def test_search_range_rejected_when_empty(self):
        from stemflow.phrase import bar_period_from_onsets

        with pytest.raises(ValueError, match="empty lag window"):
            bar_period_from_onsets(
                np.zeros(22050, dtype=np.float32), 22050,
                search_seconds=(3.0, 1.0),  # inverted
            )


# ── find_phrase_downbeat ─────────────────────────────────────────────────────

class TestFindPhraseDownbeat:
    def test_finds_true_downbeat_past_pickup(self):
        """Construct audio with a pickup hit BEFORE bar 1, then a clean
        bar pattern. find_phrase_downbeat should locate bar 1's
        downbeat, not the pickup."""
        from stemflow.phrase import find_phrase_downbeat

        sr = 22050
        true_period = 3.0
        # Pattern: beats 1 and 3 of every bar; PLUS a pickup at -0.5s
        # (i.e., 0.5s before bar 1) that's the same character as the
        # pattern clicks — a "false start" the algorithm needs to skip.
        audio, true_db = _periodic_pattern(
            sr, n_bars=6, bar_period=true_period,
            pattern_offsets_in_bar=[0.0, 1.5],
            pre_pickup_offsets=[-0.5],
        )
        # Candidate is the pickup time (the user's misleading guess)
        candidate = true_db - 0.5
        # Reference is bar 3's downbeat (definitely on-bar)
        reference = true_db + 2 * true_period

        found = find_phrase_downbeat(
            audio, sr, candidate_time=candidate,
            bar_period=true_period, reference_time=reference,
            search_radius=1.5,
        )
        # Should land within ~50ms of the true downbeat
        assert abs(found - true_db) < 0.05, (
            f"expected ~{true_db}, got {found} (delta {found - true_db:+.3f}s)"
        )

    def test_reference_default_is_three_bars_past(self):
        """When reference_time is None, default should land at
        candidate + 3*bar_period — past most pickup artifacts."""
        from stemflow.phrase import find_phrase_downbeat

        sr = 22050
        true_period = 3.0
        audio, true_db = _periodic_pattern(
            sr, n_bars=8, bar_period=true_period,
            pattern_offsets_in_bar=[0.0, 1.5],
        )
        # Candidate exactly on the true downbeat — should stay there.
        found = find_phrase_downbeat(
            audio, sr, candidate_time=true_db,
            bar_period=true_period,
            search_radius=0.5,
        )
        assert abs(found - true_db) < 0.05


# ── beat_aligned_stretch ─────────────────────────────────────────────────────

class TestBeatAlignedStretch:
    def test_exact_output_duration(self, tmp_path):
        """Output clip should be exactly n_beats * target_beat_period
        seconds long, regardless of source tempo wobble."""
        import soundfile as sf
        from stemflow.phrase import beat_aligned_stretch

        sr = 22050
        # Construct a 10-second source with simulated beats at irregular
        # spacing (mimicking real-world tempo drift).
        duration = 10.0
        source = np.random.randn(int(duration * sr)).astype(np.float32) * 0.1
        source_stereo = np.stack([source, source], axis=-1)

        path = tmp_path / "src.wav"
        sf.write(str(path), source_stereo, sr, subtype="FLOAT")

        # Beats at irregular intervals (tempo wobble around ~88 BPM)
        beats = np.array([0.50, 1.18, 1.88, 2.55, 3.23, 3.92, 4.60, 5.28,
                          5.95, 6.65, 7.32, 8.00, 8.68, 9.35])

        # Extract beats [2..10] (8 beats = 2 bars at 4/4), stretch to 80 BPM
        target_beat_period = 60.0 / 80.0  # 0.75s
        out, out_sr = beat_aligned_stretch(
            path, beats, start_beat_idx=2, n_beats=8,
            target_beat_period=target_beat_period,
        )

        expected_dur = 8 * target_beat_period  # 6.0s
        actual_dur = len(out) / out_sr
        # Beat-aligned stretch should hit exact target duration
        # (we trim/pad to exact target_samples in the implementation)
        assert actual_dur == pytest.approx(expected_dur, abs=1.0 / sr)
        assert out.shape[1] == 2  # stereo

    def test_out_of_range_raises(self, tmp_path):
        import soundfile as sf
        from stemflow.phrase import beat_aligned_stretch

        sr = 22050
        sf.write(str(tmp_path / "s.wav"),
                 np.zeros((sr, 2), dtype=np.float32), sr, subtype="FLOAT")
        with pytest.raises(ValueError, match="out of source_beats"):
            beat_aligned_stretch(
                tmp_path / "s.wav",
                np.array([0.5, 1.0]),
                start_beat_idx=0, n_beats=10,  # 10 > length-1
                target_beat_period=0.75,
            )
