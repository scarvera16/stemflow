"""Float32 numpy stereo mixer with equal-power crossfades."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import DEFAULT_CROSSFADE_MS, DEFAULT_SAMPLE_RATE

log = logging.getLogger(__name__)


def equal_power_fade(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (fade_in, fade_out) equal-power curves of length n_samples.

    Cos/sin curves maintain constant perceived loudness at transitions,
    unlike linear fades which create a perceptual dip at the midpoint.
    """
    t = np.linspace(0, np.pi / 2, n_samples)
    return np.sin(t), np.cos(t)


def build_mix(
    stems: dict[str, dict[str, Path]],
    structure: list[dict],
    output_dir: Path,
    output_name: str = "mix.wav",
    total_seconds: float | None = None,
    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
    sample_rate: int | None = None,
) -> Path:
    """
    Layer stems into a stereo WAV using float32 numpy mixing.

    No pydub — everything stays float32 end-to-end for zero quantization noise.
    Uses equal-power (cos/sin) crossfades for constant perceived loudness.

    Args:
        stems: {track_name: {stem_name: Path}} mapping.
        structure: List of structure entries, each with label, track, stem,
                   start_ms, end_ms, and optional vol_db.
        output_dir: Directory for the output file.
        output_name: Output filename.
        total_seconds: Total mix duration. If None, derived from structure.
        crossfade_ms: Crossfade duration in milliseconds.
        sample_rate: Sample rate. If None, detected from first stem.

    Returns:
        Path to the mixed WAV file.
    """
    output_dir = Path(output_dir)
    log.info("Assembling mix (float32 numpy mixer)...")

    # Detect sample rate from first available stem
    if sample_rate is None:
        for track_stems in stems.values():
            for stem_path in track_stems.values():
                info = sf.info(str(stem_path))
                sample_rate = info.samplerate
                break
            if sample_rate:
                break
    if sample_rate is None:
        sample_rate = DEFAULT_SAMPLE_RATE

    # Derive total duration from structure if not specified
    if total_seconds is None:
        max_end_ms = max(e["end_ms"] for e in structure)
        total_seconds = max_end_ms / 1000.0

    total_samples = int(total_seconds * sample_rate)
    crossfade_samples = int((crossfade_ms / 1000.0) * sample_rate)

    # Master bus: stereo float32
    mix = np.zeros((total_samples, 2), dtype=np.float32)

    for entry in structure:
        label = entry["label"]
        track = entry["track"]
        stem = entry["stem"]
        start = entry["start_ms"]
        end = entry["end_ms"]
        vol_db = entry.get("vol_db", 0)

        if track not in stems:
            log.warning("  [SKIP] Track '%s' not found — [%s]", track, label)
            continue
        if stem not in stems[track]:
            log.warning("  [SKIP] Stem '%s' not in '%s' — [%s]", stem, track, label)
            continue
        stem_path = stems[track][stem]

        # Convert ms to samples
        start_samp = int((start / 1000.0) * sample_rate)
        end_samp = int((end / 1000.0) * sample_rate)
        duration_samples = end_samp - start_samp

        log.info("  + [%s] %s @ %ds–%ds (%+ddB)", label, Path(stem_path).name, start // 1000, end // 1000, vol_db)

        # Read audio as float32
        y, sr = sf.read(str(stem_path), dtype="float32")
        if y.ndim == 1:
            y = np.stack([y, y], axis=-1)

        # Loop to fill window if shorter than needed
        if len(y) < duration_samples:
            repeats = (duration_samples // len(y)) + 1
            y = np.tile(y, (repeats, 1))
        y = y[:duration_samples]

        # Apply volume
        gain = 10.0 ** (vol_db / 20.0)
        y = y * gain

        # Apply equal-power crossfades
        fade_len = min(crossfade_samples, duration_samples // 4)
        if fade_len > 1:
            fade_in_curve, fade_out_curve = equal_power_fade(fade_len)
            for ch in range(y.shape[1]):
                y[:fade_len, ch] *= fade_in_curve
                y[-fade_len:, ch] *= fade_out_curve

        # Sum into master bus (float32 addition — no integer saturation)
        actual_end = min(start_samp + len(y), total_samples)
        usable = actual_end - start_samp
        mix[start_samp:actual_end] += y[:usable]

    # Peak-normalize with -1 dBTP headroom
    peak = np.max(np.abs(mix))
    if peak > 0:
        target_peak = 10.0 ** (-1.0 / 20.0)
        mix = mix * (target_peak / peak)

    out_path = output_dir / output_name
    sf.write(str(out_path), mix, sample_rate, subtype="FLOAT")
    log.info("  → Mix: %s (%.1fs, float32)", out_path, total_seconds)
    return out_path
