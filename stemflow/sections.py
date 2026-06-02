"""Section detection and characterization.

A song is not a single musical object; it is a sequence of sections
(intro, verse, chorus, bridge, riff, drum break, outro). The phrase
alignment work makes it possible to lock any clip to a target tempo,
but it cannot answer the question "which clip should I extract?"
That question is what this module begins to answer.

The Music Structure Analysis (MSA) field has been working on this
for two decades. State-of-the-art approaches use self-similarity
matrices of harmonic + timbral features, then cluster or label the
segments. This module uses the librosa-native version of that
recipe — practical, dependency-light, good enough to flag candidate
riff sections in a song; not as sophisticated as MSAF, which would
be the next step up.

Provides:

- `find_section_boundaries`: detect N segment boundaries in a track
  using librosa's agglomerative clustering on MFCC + chroma features.
  Returns boundary times in seconds.

- `section_features`: characterize a section's energy, spectral
  brightness, harmonic content, and timbral consistency. Used to
  decide what kind of section we're looking at (sparse vs dense,
  sustained vs varying, low-end-heavy vs full-spectrum).

- `find_riff_candidates`: high-level helper that runs the two above
  and returns sections that match riff-like criteria (sustained,
  high-energy, harmonically active).

This is v1 of section labeling — boundaries plus feature stats, no
semantic labels (intro/verse/chorus). Semantic labeling requires
either training data or a richer external tool (MSAF). The current
output is enough to bootstrap automatic riff-section selection,
which is what the Librarian's auto-novelty path needs first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ── Public types ─────────────────────────────────────────────────────────────

@dataclass
class SectionStats:
    """Feature characterization of one section of audio.

    All values are in the conventional MIR ranges:
        rms_mean: 0..~1 (clipped float32)
        spectral_centroid_mean: Hz
        spectral_flatness_mean: 0..1 (closer to 1 = noise-like)
        chroma_strength: 0..1 (sum of mean chroma vector, normalized)
        timbral_variance: variance of MFCC across frames (smaller = more
            consistent, larger = more varied)
        density: onsets per second
    """
    start_s: float
    end_s: float
    duration_s: float
    rms_mean: float
    spectral_centroid_mean: float
    spectral_flatness_mean: float
    chroma_strength: float
    timbral_variance: float
    density: float


# ── find_section_boundaries ──────────────────────────────────────────────────

def find_section_boundaries(
    audio_file: Path,
    sr: int = 22050,
    n_segments: int = 8,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Detect N section boundaries in a song.

    Loads `audio_file` at `sr`, builds a feature stack (MFCC + chroma),
    and runs librosa's agglomerative clustering to find N-1 boundary
    points (yielding N sections). Returns the boundary times in
    seconds, including the start (0.0) and end of the audio.

    Args:
        audio_file: Path to audio.
        sr: Sample rate to load at. 22050 is librosa's typical default.
        n_segments: Target number of sections. Typical pop/rock songs
            have 6-12 distinguishable sections; 8 is a sensible default.
        n_mfcc: Number of MFCC coefficients. 13 is the librosa default
            and captures most timbral variation.

    Returns:
        Array of boundary times in seconds. Length is n_segments + 1
        (start, N-1 internal boundaries, end).
    """
    import librosa

    audio_file = Path(audio_file)
    log.info("find_section_boundaries: loading %s", audio_file.name)
    y, sr = librosa.load(str(audio_file), sr=sr, mono=True)

    hop = 512
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)

    # Stack features and normalize. MFCC captures timbre, chroma
    # captures harmony; together they describe the perceptual change
    # listeners feel at section boundaries.
    features = np.vstack([
        librosa.util.normalize(mfcc),
        librosa.util.normalize(chroma),
    ])

    # Agglomerative clustering of feature frames into n_segments groups
    boundary_frames = librosa.segment.agglomerative(features, k=n_segments)
    boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=hop)

    # Ensure boundaries include 0.0 and the end of audio
    duration = len(y) / sr
    if boundary_times[0] > 0.0:
        boundary_times = np.concatenate([[0.0], boundary_times])
    if boundary_times[-1] < duration:
        boundary_times = np.concatenate([boundary_times, [duration]])

    log.info("find_section_boundaries: %d boundaries: %s",
             len(boundary_times), [f"{t:.1f}" for t in boundary_times])
    return boundary_times


# ── section_features ─────────────────────────────────────────────────────────

def section_features(
    audio_file: Path,
    start_s: float,
    end_s: float,
    sr: int = 22050,
) -> SectionStats:
    """Compute a feature characterization for one section of audio.

    Used to decide what kind of section we're looking at: sparse vs
    dense, sustained vs varied, harmonic vs percussive. The returned
    stats let a caller answer questions like "is this the high-energy
    section?" or "is this the section where the riff is locked-in?"
    without ever reading the audio directly.

    Args:
        audio_file: Path to audio.
        start_s: Section start time in seconds.
        end_s: Section end time in seconds.
        sr: Sample rate to load at.

    Returns:
        SectionStats with the section's RMS, spectral, chroma, timbral,
        and onset-density statistics.
    """
    import librosa

    audio_file = Path(audio_file)
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be greater than start_s ({start_s})")

    y_full, sr = librosa.load(str(audio_file), sr=sr, mono=True)
    s, e = int(start_s * sr), int(end_s * sr)
    y = y_full[s:e]
    if len(y) == 0:
        raise ValueError(f"empty section: [{start_s}, {end_s}]s")

    hop = 512

    # Energy
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_mean = float(rms.mean())

    # Spectral
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop).mean())
    flatness = float(librosa.feature.spectral_flatness(y=y, hop_length=hop).mean())

    # Harmonic content
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma_strength = float(np.linalg.norm(chroma.mean(axis=1)) / np.sqrt(12))

    # Timbral consistency: variance of MFCC vectors across frames
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    # Normalize each MFCC dimension to z-scores, then average frame-to-frame variance
    mfcc_z = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)
    timbral_variance = float(mfcc_z.std(axis=1).mean())

    # Onset density (onsets / second)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
    density = float(len(onset_frames) / (end_s - start_s)) if end_s > start_s else 0.0

    return SectionStats(
        start_s=float(start_s),
        end_s=float(end_s),
        duration_s=float(end_s - start_s),
        rms_mean=rms_mean,
        spectral_centroid_mean=centroid,
        spectral_flatness_mean=flatness,
        chroma_strength=chroma_strength,
        timbral_variance=timbral_variance,
        density=density,
    )


# ── find_riff_candidates ─────────────────────────────────────────────────────

def find_riff_candidates(
    audio_file: Path,
    sr: int = 22050,
    n_segments: int = 8,
    min_duration_s: float = 10.0,
    min_rms_percentile: float = 60.0,
    max_flatness: float = 0.5,
) -> list[SectionStats]:
    """Detect sections likely to be "riff" or "hook" sections.

    Runs `find_section_boundaries` followed by `section_features` on
    each section, then filters for sections that are:

    - At least `min_duration_s` long (typically 10s+; eliminates fills
      and transitions)
    - Above the `min_rms_percentile`-th percentile of song-wide RMS
      (eliminates intros and breakdowns)
    - Below `max_flatness` in spectral flatness (eliminates purely
      percussive sections; a riff has tonal content)

    Returns sorted by start time. This is a heuristic filter; tune the
    thresholds per genre.

    Args:
        audio_file: Path to audio.
        sr: Sample rate for analysis.
        n_segments: How many segments to consider in the song. More
            = finer-grained candidates.
        min_duration_s: Minimum section duration to count as a riff
            candidate.
        min_rms_percentile: Minimum RMS percentile (0-100). 60 means
            the section must be louder than 60% of all sections.
        max_flatness: Maximum spectral flatness. 0.5 is generous;
            tighter (0.3) excludes more drum-heavy sections.

    Returns:
        Filtered list of SectionStats objects, sorted by start_s.
    """
    audio_file = Path(audio_file)
    boundaries = find_section_boundaries(audio_file, sr=sr, n_segments=n_segments)

    sections = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < 0.5:  # tiny segments aren't useful even pre-filter
            continue
        stats = section_features(audio_file, s, e, sr=sr)
        sections.append(stats)

    if not sections:
        return []

    rms_threshold = np.percentile([s.rms_mean for s in sections], min_rms_percentile)

    candidates = [
        s for s in sections
        if s.duration_s >= min_duration_s
        and s.rms_mean >= rms_threshold
        and s.spectral_flatness_mean <= max_flatness
    ]

    log.info("find_riff_candidates: %d candidates from %d sections",
             len(candidates), len(sections))
    return candidates
