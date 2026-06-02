"""Phrase alignment: find true bar-1 downbeats and produce beat-locked clips.

Solves the problem surfaced during the Sad But Levee mashup iterations
(v5 through v10): wall-clock timestamp trims and average-BPM stretches
produce clips that drift out of phase over the course of a mashup, even
when the start point looks right.

Two ideas this module provides:

1. **find_phrase_downbeat**. Given a candidate time in a source where you
   *think* a riff (or any repeating phrase) begins, find the actual
   phrase-bar-1 downbeat by cross-correlating against a stable reference
   later in the same recording. Works because a riff is, by construction,
   periodic: any properly-aligned bar is a statistical copy of any other,
   so the lag where correlation peaks tells you the true phase regardless
   of pickups, pre-roll artifacts, or detection noise.

2. **beat_aligned_stretch**. Extract exactly N source beats and time-stretch
   the result to exactly N target beats at a chosen tempo. This is
   independent of average song BPM, so it corrects for local tempo drift
   that an average-BPM ratio would accumulate as phase error.

Reference: the Sad But Levee mashup (May 2026) where v9b's riff entered at
beat 3 of bar 1 (not bar 1's downbeat), audible as 2-beat phrase shift.
Cross-correlation of one riff bar against later bars revealed that the
true downbeat was 1.36s earlier than the timestamp I had picked.

See `notes/the-mashup-engine.md` in the Librarian repo for the full arc.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


# ── Bar-period detection ─────────────────────────────────────────────────────

def bar_period_from_onsets(
    audio: np.ndarray,
    sr: int,
    search_seconds: tuple[float, float] = (1.5, 4.0),
    n_beats_per_bar: int = 4,
    expected_bpm: float | None = None,
    expected_bpm_tolerance: float = 0.15,
) -> float:
    """Detect the bar period (in seconds) via onset-envelope autocorrelation.

    Computes librosa's onset-strength envelope, autocorrelates it, and
    returns the lag (in seconds) of the strongest peak within the
    expected range. For a 4/4 song at typical metal tempos (70-120 BPM)
    the bar period is 2.0-3.5 seconds.

    When `expected_bpm` is provided, the search range is narrowed to a
    tight window around the expected bar period — this prevents the
    autocorrelation from picking up sub-bar periodicities (a riff with
    chord stabs on beats 1 and 3 will have a strong 2-beat autocorrelation
    peak that can outrank the 4-beat bar peak; the SBT riff is the
    canonical case).

    Args:
        audio: 1-D mono or 2-D stereo float audio.
        sr: Sample rate.
        search_seconds: (min, max) range for the bar period in seconds.
            Default (1.5, 4.0) covers ~60-160 BPM at n_beats_per_bar=4.
            Ignored if `expected_bpm` is provided.
        n_beats_per_bar: Time signature numerator. Default 4 for 4/4.
        expected_bpm: If given, narrow the search to bar periods within
            `expected_bpm_tolerance` of the implied target.
        expected_bpm_tolerance: Fractional tolerance around the expected
            BPM. Default 0.15 (±15%). Only used when `expected_bpm`
            is given.

    Returns:
        Bar period in seconds.
    """
    import librosa

    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio

    if expected_bpm is not None:
        expected_period = 60.0 / expected_bpm * n_beats_per_bar
        search_seconds = (
            expected_period * (1 - expected_bpm_tolerance),
            expected_period * (1 + expected_bpm_tolerance),
        )

    hop = 512
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    env_sr = sr / hop

    ac = librosa.autocorrelate(env, max_size=int(search_seconds[1] * env_sr) + 1)
    lag_min = int(search_seconds[0] * env_sr)
    lag_max = min(int(search_seconds[1] * env_sr), len(ac))
    if lag_max <= lag_min:
        raise ValueError(f"search range {search_seconds} produced empty lag window")

    best_lag_frames = lag_min + int(np.argmax(ac[lag_min:lag_max]))
    period = best_lag_frames / env_sr
    log.info("bar_period_from_onsets: %.4fs (= %.2f BPM at %d beats/bar)",
             period, 60.0 / period * n_beats_per_bar, n_beats_per_bar)
    return float(period)


# ── Phrase-downbeat refinement ───────────────────────────────────────────────

def find_phrase_downbeat(
    audio: np.ndarray,
    sr: int,
    candidate_time: float,
    bar_period: float,
    reference_time: float | None = None,
    search_radius: float = 2.0,
) -> float:
    """Find the true phrase-bar-1 downbeat near a candidate time.

    Cross-correlates a one-bar reference template (taken from
    `reference_time` later in the same audio, presumed to be on-bar)
    against the `search_radius`-seconds window around `candidate_time`.
    The lag where correlation peaks gives the true phrase-1 downbeat
    in source time.

    Args:
        audio: 1-D mono or 2-D stereo float audio.
        sr: Sample rate.
        candidate_time: Best-guess riff start time in seconds.
        bar_period: Bar period in seconds (use `bar_period_from_onsets`
            if you don't already have it).
        reference_time: Source time of a known-aligned bar to use as the
            reference template. If None, defaults to `candidate_time +
            3 * bar_period` (i.e., bar 4 of the assumed riff section,
            which is usually past any pickup artifacts).
        search_radius: How far in seconds (each side of candidate_time)
            to search for the true downbeat. Should be at least one bar
            period so the true downbeat is in range even if your
            candidate is a full bar off.

    Returns:
        True phrase-bar-1 downbeat in source seconds.
    """
    from scipy.signal import correlate, find_peaks

    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio

    if reference_time is None:
        reference_time = candidate_time + 3.0 * bar_period

    ref_s = int(reference_time * sr)
    ref_e = ref_s + int(bar_period * sr)
    if ref_e > len(mono):
        raise ValueError(
            f"reference window [{reference_time:.2f}..{reference_time + bar_period:.2f}]s "
            f"extends past audio (length {len(mono)/sr:.2f}s)"
        )

    # The target window must be long enough for the reference to slide across
    # the full ±search_radius range, so:
    #     target_length >= 2 * search_radius + bar_period
    # Output index k of `correlate(target, ref, mode="valid")` corresponds to
    # the reference placed at offset k from the target window's start, so the
    # last valid k is (target_len - ref_len) = 2 * search_radius * sr.
    search_s = max(0, int((candidate_time - search_radius) * sr))
    search_e = min(
        len(mono),
        search_s + int((2 * search_radius + bar_period) * sr),
    )
    if search_e - search_s < int(bar_period * sr) + 1:
        raise ValueError(
            f"search window [{search_s/sr:.2f}..{search_e/sr:.2f}]s too small for "
            f"reference of {bar_period:.2f}s (try larger search_radius or check audio length)"
        )

    ref = mono[ref_s:ref_e].astype(np.float64)
    target = mono[search_s:search_e].astype(np.float64)
    ref -= ref.mean()
    target -= target.mean()

    corr = correlate(target, ref, mode="valid")
    # Normalize so the result is in [-1, 1]
    norm = np.linalg.norm(ref) * np.sqrt(
        np.convolve(target ** 2, np.ones(len(ref)), mode="valid")
    )
    norm = np.where(norm > 0, norm, 1.0)
    corr_n = corr / norm

    # The best lag in the search window — and we restrict to local maxima
    # to avoid spurious raw-correlation peaks from slow-varying mean shifts.
    peaks, _ = find_peaks(corr_n, distance=int(0.1 * sr))
    if len(peaks) == 0:
        best_idx = int(np.argmax(corr_n))
    else:
        best_idx = int(peaks[np.argmax(corr_n[peaks])])

    downbeat_time = (search_s + best_idx) / sr
    log.info("find_phrase_downbeat: candidate=%.4fs, ref=%.4fs, best=%.4fs (delta %+.3fs)",
             candidate_time, reference_time, downbeat_time, downbeat_time - candidate_time)
    return float(downbeat_time)


# ── Beat-aligned trim and stretch ────────────────────────────────────────────

def beat_aligned_stretch(
    audio_file: Path,
    source_beats: np.ndarray,
    start_beat_idx: int,
    n_beats: int,
    target_beat_period: float,
) -> tuple[np.ndarray, int]:
    """Extract N source beats and stretch to N target beats at given tempo.

    The fundamental beat-aligned primitive. Given an array of beat times
    from beat-this (or any beat detector), this function:

      1. Trims the source audio from beat `start_beat_idx` to beat
         `start_beat_idx + n_beats` — exactly N beats wide in source time,
         independent of any tempo wobble in that section.
      2. Stretches that clip so it is exactly `n_beats * target_beat_period`
         seconds long in the output.

    The output has exactly `target_beat_period` seconds per beat, so any
    two clips produced with the same `target_beat_period` will be
    phase-lockable when layered.

    Args:
        audio_file: Path to source audio.
        source_beats: Beat times in seconds (e.g., from File2Beats).
        start_beat_idx: Index into `source_beats` of the start.
        n_beats: Number of beats to extract.
        target_beat_period: Desired beat period in the output (e.g., 0.75s
            for 80 BPM).

    Returns:
        (audio, sr) — stretched stereo float32 clip, exact target length.
    """
    from pedalboard import time_stretch

    audio_file = Path(audio_file)
    source_beats = np.asarray(source_beats)
    if start_beat_idx < 0 or start_beat_idx + n_beats >= len(source_beats):
        raise ValueError(
            f"beat range [{start_beat_idx}..{start_beat_idx + n_beats}] "
            f"out of source_beats (length {len(source_beats)})"
        )

    native_start = float(source_beats[start_beat_idx])
    native_end = float(source_beats[start_beat_idx + n_beats])
    native_dur = native_end - native_start
    target_dur = n_beats * target_beat_period

    # Pedalboard's stretch_factor is a SPEED multiplier:
    #   factor = native_dur / target_dur
    # If native is longer than target, factor > 1 (play faster, shorter output).
    # If native is shorter than target, factor < 1 (play slower, longer output).
    factor = native_dur / target_dur
    log.info(
        "beat_aligned_stretch: %s beats [%d..%d] = %.3fs native -> %.3fs target (factor %.4f)",
        audio_file.name, start_beat_idx, start_beat_idx + n_beats, native_dur, target_dur, factor,
    )

    y, sr = sf.read(str(audio_file), dtype="float32")
    if y.ndim == 1:
        y = np.stack([y, y], axis=-1)
    y = y[int(native_start * sr): int(native_end * sr)]

    out = time_stretch(
        y.T, float(sr), stretch_factor=factor,
        high_quality=True, transient_mode="crisp", preserve_formants=True,
    ).T

    # Trim or zero-pad to exact target length so downstream layering is
    # sample-precise (Pedalboard may differ by a frame or two).
    target_samples = int(target_dur * sr)
    if len(out) >= target_samples:
        out = out[:target_samples]
    else:
        pad = target_samples - len(out)
        out = np.pad(out, ((0, pad), (0, 0)))

    return out, sr
