"""Two-stage mastering chain: Pedalboard DSP + pyloudnorm LUFS normalization."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import DEFAULT_TARGET_LUFS

log = logging.getLogger(__name__)


def master(
    mix_path: Path,
    output_dir: Path,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    output_name: str | None = None,
) -> Path:
    """
    Two-stage mastering chain optimized for multi-source mixes.

    Stage 1 — Pedalboard DSP:
        - 30 Hz high-pass (subsonic cleanup)
        - -3 dB @ 280 Hz (low-mid mud cut — critical when layering mastered sources)
        - +1.5 dB low shelf @ 100 Hz (warmth)
        - +1 dB @ 3 kHz (presence)
        - +1 dB high shelf @ 10 kHz (air)
        - Compressor: -18 dB threshold, 2.5:1, 10 ms attack, 100 ms release
        - +3 dB makeup gain
        - Brick-wall limiter @ -1.0 dBTP

    Stage 2 — pyloudnorm:
        - ITU-R BS.1770-4 integrated loudness normalization
        - True peak protection (ceiling at 0.99 if exceeded)

    Args:
        mix_path: Path to the raw mix WAV.
        output_dir: Directory for the mastered output.
        target_lufs: Target loudness in LUFS (default -14 for streaming).
        output_name: Output filename. If None, appends "_mastered".

    Returns:
        Path to the mastered WAV file.
    """
    import pedalboard
    import pyloudnorm

    mix_path = Path(mix_path)
    output_dir = Path(output_dir)

    log.info("Mastering: %s", mix_path.name)

    # ── Stage 1: Pedalboard effects chain ────────────────────────────────
    y, sr = sf.read(str(mix_path), dtype="float32")
    if y.ndim == 1:
        y = np.stack([y, y], axis=-1)

    board = pedalboard.Pedalboard([
        pedalboard.HighpassFilter(cutoff_frequency_hz=30.0),
        pedalboard.PeakFilter(cutoff_frequency_hz=280.0, gain_db=-3.0, q=0.8),
        pedalboard.LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=1.5),
        pedalboard.PeakFilter(cutoff_frequency_hz=3000.0, gain_db=1.0, q=1.0),
        pedalboard.HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.0),
        pedalboard.Compressor(threshold_db=-18.0, ratio=2.5, attack_ms=10.0, release_ms=100.0),
        pedalboard.Gain(gain_db=3.0),
        pedalboard.Limiter(threshold_db=-1.0, release_ms=100.0),
    ])

    y_proc = board(y.T.astype(np.float32), sr).T

    log.info("  EQ: -3dB@280Hz, +1.5dB@100Hz, +1dB@3kHz, +1dB@10kHz")
    log.info("  Compression: -18dB threshold, 2.5:1, 10ms attack, 100ms release")
    log.info("  Limiter: -1.0 dBTP ceiling")

    # ── Stage 2: pyloudnorm LUFS normalization ───────────────────────────
    meter = pyloudnorm.Meter(sr)
    current_lufs = meter.integrated_loudness(y_proc)
    log.info("  Pre-norm: %.1f LUFS → target %.1f LUFS", current_lufs, target_lufs)

    if np.isinf(current_lufs):
        log.warning("  Could not measure loudness — skipping normalization")
        y_final = y_proc
    else:
        y_final = pyloudnorm.normalize.loudness(y_proc, current_lufs, target_lufs)
        peak = np.max(np.abs(y_final))
        if peak > 1.0:
            y_final = y_final / peak * 0.99
            log.info("  True peak exceeded (%.3f) — applied ceiling", peak)

    if output_name is None:
        output_name = mix_path.stem + "_mastered.wav"
    final_path = output_dir / output_name
    sf.write(str(final_path), y_final, sr, subtype="FLOAT")

    final_lufs = meter.integrated_loudness(y_final)
    final_peak = 20.0 * np.log10(np.max(np.abs(y_final)) + 1e-10)
    log.info("  Final: %.1f LUFS, %.1f dBTP", final_lufs, final_peak)
    log.info("  → %s (%.1fs, float32)", final_path, len(y_final) / sr)
    return final_path
