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
    bpm = float(tempo)
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
