"""BPM/beat grid detection and musical key estimation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from .config import HAS_BEAT_THIS, HAS_ESSENTIA

log = logging.getLogger(__name__)


@dataclass
class BeatAnalysis:
    """Result of beat/downbeat detection."""
    bpm: float
    beats: np.ndarray = field(default_factory=lambda: np.array([]))
    downbeats: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TrackAnalysis:
    """Combined BPM + key analysis for a track."""
    bpm: float
    beats: np.ndarray
    downbeats: np.ndarray
    key: str


def detect_bpm(audio_file: Path, device: str = "mps") -> BeatAnalysis:
    """
    Beat and downbeat detection.

    Uses beat-this! (ISMIR 2024 SOTA) with librosa fallback.
    beat-this! provides both beat positions and downbeat (bar boundary) positions.

    Args:
        audio_file: Path to audio file.
        device: Inference device for beat-this! ("mps", "cuda", "cpu").

    Returns:
        BeatAnalysis with bpm, beat times, and downbeat times.
    """
    audio_file = Path(audio_file)
    log.info("BPM + beat grid: %s", audio_file.name)

    if HAS_BEAT_THIS:
        try:
            from beat_this.inference import File2Beats
            file2beats = File2Beats(device=device, dbn=False)
            beats, downbeats = file2beats(str(audio_file))
            if len(beats) > 1:
                bpm = 60.0 / float(np.median(np.diff(beats)))
                log.info("  beat-this!: %.1f BPM, %d beats, %d downbeats", bpm, len(beats), len(downbeats))
                return BeatAnalysis(bpm=bpm, beats=beats, downbeats=downbeats)
        except Exception as e:
            log.warning("  beat-this! error: %s", e)

    # Fallback: librosa
    y, sr = librosa.load(str(audio_file), mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    # librosa 0.10+ returns tempo as a 1-d array (shape (1,)) rather
    # than a scalar; older versions returned a Python float. Coerce
    # via atleast_1d().item() so both shapes work.
    bpm = float(np.atleast_1d(tempo).item(0))
    log.info("  librosa: %.1f BPM, %d beats (no downbeat detection)", bpm, len(beats))
    return BeatAnalysis(bpm=bpm, beats=beats)


def detect_key(audio_file: Path) -> str:
    """
    Estimate musical key.

    Uses Essentia KeyExtractor with edma profile (70-80% accuracy),
    falls back to librosa chromagram (55-65%).

    Args:
        audio_file: Path to audio file.

    Returns:
        Key string like "F# minor" or "C major".
    """
    audio_file = Path(audio_file)
    log.info("Key: %s", audio_file.name)

    if HAS_ESSENTIA:
        try:
            import essentia.standard as es
            audio = es.MonoLoader(filename=str(audio_file), sampleRate=44100)()
            key, scale, strength = es.KeyExtractor(profileType="edma")(audio)
            key_str = f"{key} {scale}"
            log.info("  Essentia (edma): %s (confidence: %.2f)", key_str, strength)
            return key_str
        except Exception as e:
            log.warning("  Essentia error: %s", e)

    # Fallback: librosa chromagram
    y, sr = librosa.load(str(audio_file), mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    root = int(np.argmax(chroma))
    mode = "minor" if chroma[(root + 3) % 12] > chroma[(root + 4) % 12] else "major"
    key_str = f"{keys[root]} {mode}"
    log.info("  librosa: %s", key_str)
    return key_str


def analyze_track(audio_file: Path, device: str = "mps") -> TrackAnalysis:
    """Run full analysis (BPM + key) on a single track."""
    beat_result = detect_bpm(audio_file, device=device)
    key = detect_key(audio_file)
    return TrackAnalysis(
        bpm=beat_result.bpm,
        beats=beat_result.beats,
        downbeats=beat_result.downbeats,
        key=key,
    )


def compute_features(audio_file: Path, sr: int = 22050) -> dict:
    """
    Extra acoustic features for mashability scoring.

    Computes a chroma profile (12-dim, time-averaged), summary spectral
    statistics, and a coarse five-band energy profile. These complement
    the BPM/key from `analyze_track` and feed `mashability.score_pair`.

    Args:
        audio_file: Path to audio file.
        sr: Sample rate for loading. Default 22050 (librosa-typical).

    Returns:
        Dict with keys:
            chroma_mean (np.ndarray, shape (12,)):
                Mean chroma over the segment. Indices 0..11 correspond
                to C..B.
            spectral_centroid_mean (float):
                Mean spectral centroid in Hz.
            spectral_bandwidth_mean (float):
                Mean spectral bandwidth in Hz.
            spectral_rolloff_mean (float):
                Mean rolloff at 85% energy in Hz.
            spectral_profile (np.ndarray, shape (5,)):
                Normalized energy in five bands: sub (0-60 Hz), bass
                (60-250), low-mid (250-2000), mid (2000-6000), high
                (6000-Nyquist).
    """
    audio_file = Path(audio_file)
    log.info("Features: %s", audio_file.name)

    y, sr = librosa.load(str(audio_file), sr=sr, mono=True)

    # Chroma profile (12-dim mean over time).
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Summary spectral statistics.
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())

    # Coarse band-energy profile (5 bands).
    stft_mag = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bands = [(0, 60), (60, 250), (250, 2000), (2000, 6000), (6000, sr / 2)]
    band_energies = np.zeros(len(bands), dtype=float)
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band_energies[i] = stft_mag[mask].mean()
    total = band_energies.sum()
    spectral_profile = band_energies / total if total > 0 else band_energies

    log.info(
        "  chroma peak=%s, centroid=%.0f Hz, profile=%s",
        ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][int(np.argmax(chroma_mean))],
        centroid,
        np.round(spectral_profile, 2).tolist(),
    )

    return {
        "chroma_mean": chroma_mean,
        "spectral_centroid_mean": centroid,
        "spectral_bandwidth_mean": bandwidth,
        "spectral_rolloff_mean": rolloff,
        "spectral_profile": spectral_profile,
    }
