"""stemflow — AI-powered audio deconstruction and reconstruction pipeline.

Deconstructs songs into stems, analyzes tempo and key, time-stretches
to a unified BPM, and reassembles selected layers into a mixed and
mastered track.

Modules:
    separate  — Stem separation (Demucs, audio-separator, BS-RoFormer)
    analyze   — BPM/beat/downbeat detection (beat-this!), key estimation (Essentia)
    process   — Per-stem cleanup (Pedalboard), time-stretching (Rubber Band)
    mix       — Float32 numpy stereo mixer with equal-power crossfades
    master    — Two-stage mastering (Pedalboard DSP + pyloudnorm LUFS)
    config    — Constants, feature flags, structure file loading
    cli       — Command-line interface
"""

from stemflow.analyze import BeatAnalysis, TrackAnalysis, analyze_track, detect_bpm, detect_key
from stemflow.config import load_structure
from stemflow.master import master
from stemflow.mix import build_mix, equal_power_fade
from stemflow.process import clean_stem, stretch_to_bpm
from stemflow.separate import separate_stems

__version__ = "0.1.0"

__all__ = [
    "separate_stems",
    "detect_bpm",
    "detect_key",
    "analyze_track",
    "BeatAnalysis",
    "TrackAnalysis",
    "clean_stem",
    "stretch_to_bpm",
    "build_mix",
    "equal_power_fade",
    "master",
    "load_structure",
]
