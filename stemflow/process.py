"""Per-stem cleanup (noise gate + EQ) and time-stretching."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import HAS_PEDALBOARD_STRETCH, HAS_RUBBERBAND

log = logging.getLogger(__name__)


def clean_stem(
    audio_file: Path,
    stem_type: str,
    output_dir: Path,
) -> Path:
    """
    Per-stem cleanup using Pedalboard noise gate + EQ.

    Reduces bleed artifacts, removes noise in quiet parts, and carves
    frequency space so stems sit together cleanly.

    Args:
        audio_file: Path to the stem audio file.
        stem_type: One of "drums", "bass", "guitar", "vocals", "piano", "other".
        output_dir: Base output directory. Cleaned files go to output_dir/cleaned/.

    Returns:
        Path to the cleaned audio file.
    """
    import pedalboard

    audio_file = Path(audio_file)
    y, sr = sf.read(str(audio_file), dtype="float32")
    if y.ndim == 1:
        y = np.stack([y, y], axis=-1)

    effects = []

    if stem_type == "drums":
        effects = [
            pedalboard.NoiseGate(threshold_db=-30.0, ratio=4.0, attack_ms=1.0, release_ms=50.0),
            pedalboard.HighpassFilter(cutoff_frequency_hz=40.0),
            pedalboard.PeakFilter(cutoff_frequency_hz=800.0, gain_db=-2.0, q=1.0),
        ]
    elif stem_type == "bass":
        effects = [
            pedalboard.NoiseGate(threshold_db=-35.0, ratio=3.0, attack_ms=5.0, release_ms=80.0),
            pedalboard.LowpassFilter(cutoff_frequency_hz=5000.0),
            pedalboard.HighpassFilter(cutoff_frequency_hz=30.0),
        ]
    elif stem_type in ("guitar", "piano"):
        effects = [
            pedalboard.NoiseGate(threshold_db=-35.0, ratio=3.0, attack_ms=5.0, release_ms=100.0),
            pedalboard.HighpassFilter(cutoff_frequency_hz=80.0),
            pedalboard.PeakFilter(cutoff_frequency_hz=4000.0, gain_db=-1.5, q=1.5),
        ]
    elif stem_type == "vocals":
        effects = [
            pedalboard.NoiseGate(threshold_db=-32.0, ratio=4.0, attack_ms=2.0, release_ms=60.0),
            pedalboard.HighpassFilter(cutoff_frequency_hz=100.0),
            pedalboard.LowpassFilter(cutoff_frequency_hz=12000.0),
        ]
    else:  # "other" or unknown
        effects = [
            pedalboard.NoiseGate(threshold_db=-35.0, ratio=2.0, attack_ms=5.0, release_ms=100.0),
        ]

    if not effects:
        return audio_file

    board = pedalboard.Pedalboard(effects)
    y_clean = board(y.T.astype(np.float32), sr).T

    clean_dir = Path(output_dir) / "cleaned"
    clean_dir.mkdir(parents=True, exist_ok=True)
    out_path = clean_dir / f"{audio_file.stem}_clean.wav"
    sf.write(str(out_path), y_clean, sr, subtype="FLOAT")

    log.info("  [CLEAN] %s: %s → %s", stem_type, audio_file.name, out_path.name)
    return out_path


def stretch_to_bpm(
    audio_file: Path,
    source_bpm: float,
    target_bpm: float,
    output_dir: Path,
    track_name: str = "",
) -> Path:
    """
    Time-stretch without pitch shift.

    Engine priority: Pedalboard (Rubber Band, zero-copy) → pyrubberband → librosa.

    Args:
        audio_file: Path to audio file.
        source_bpm: Detected BPM of the source.
        target_bpm: Target BPM to stretch to.
        output_dir: Base output directory. Stretched files go to output_dir/stretched/.
        track_name: Optional prefix for output filename.

    Returns:
        Path to the stretched audio file.
    """
    import librosa

    audio_file = Path(audio_file)
    ratio = target_bpm / source_bpm
    log.info("Stretching %s: %.1f→%.1f BPM (×%.3f)", audio_file.name, source_bpm, target_bpm, ratio)

    y, sr = sf.read(str(audio_file), dtype="float32")
    if y.ndim == 1:
        y = np.stack([y, y], axis=-1)

    if HAS_PEDALBOARD_STRETCH:
        from pedalboard import time_stretch
        y_out = time_stretch(
            y.T, float(sr), stretch_factor=ratio,
            high_quality=True, transient_mode="crisp", preserve_formants=True,
        ).T
        engine = "Pedalboard"
    elif HAS_RUBBERBAND:
        import pyrubberband as pyrb
        y_out = np.stack([pyrb.time_stretch(y[:, ch], sr, ratio) for ch in range(y.shape[1])], axis=-1)
        engine = "pyrubberband"
    else:
        y_mono = np.mean(y, axis=1)
        y_s = librosa.effects.time_stretch(y_mono, rate=ratio)
        y_out = np.stack([y_s, y_s], axis=-1)
        engine = "librosa (mono fallback)"

    prefix = f"{track_name}__" if track_name else ""
    out_dir = Path(output_dir) / "stretched"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}{audio_file.stem}__{target_bpm:.0f}bpm.wav"
    sf.write(str(out_path), y_out, sr, subtype="FLOAT")

    log.info("  [%s] → %s", engine, out_path.name)
    return out_path
