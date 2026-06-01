"""Mashability scoring (AutoMashUpper tradition, Davies et al. 2014).

A computable measure of how well two musical segments will combine
into a coherent mashup. Returns scalars in [0, 1]; higher is more
compatible.

The total score is a weighted sum of three hand-engineered components:

- harmonic: cosine similarity of the two segments' mean chroma
  vectors, with an optional search over the twelve semitone
  transpositions of one segment to find the best alignment.
- rhythmic: similarity of the two segments' BPMs after up-to-octave
  matching (the faster BPM is halved while it remains more than 1.5x
  the slower, then the ratio is reported).
- spectral: complementarity of the two segments' band-energy profiles.
  Pairs whose energy lives in the same band (two bass-heavy segments,
  say) score low because they would mash muddy.

This is the v1 hand-engineered scorer described as the Librarian
project's first Invention build target. A future v2 may add neural-
embedding-based scoring (MERT, CLAP) as an additional component;
that would slot in as a fourth weight in `score_pair`.

Reference:
    Davies, M. E. P., Hamel, P., Yoshii, K., & Goto, M. (2014).
    AutoMashUpper: Automatic Creation of Multi-Song Music Mashups.
    IEEE Transactions on Audio, Speech, and Language Processing,
    22(12), 1726-1737.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MashabilityScore:
    """Result of mashability scoring between two segments.

    Attributes:
        total: Weighted sum of the three components, in [0, 1].
        harmonic: Chroma cosine similarity in [0, 1] (best after
            optional transposition search).
        rhythmic: Tempo-similarity score in (0, 1] after octave matching.
        spectral: Spectral complementarity score in [0, 1].
        best_transpose_semitones: Signed semitone offset in -6..+5 that
            produced the best harmonic match. Zero when transposition
            search is disabled.
    """
    total: float
    harmonic: float
    rhythmic: float
    spectral: float
    best_transpose_semitones: int


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two equal-length vectors, clipped to [0, 1].

    Returns 0.0 if either vector is all zeros.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def harmonic_score(
    chroma_mean_a: np.ndarray,
    chroma_mean_b: np.ndarray,
    search_transposes: bool = True,
) -> tuple[float, int]:
    """Cosine similarity of two mean-chroma vectors, with transposition.

    If `search_transposes` is True (default), tests all twelve circular
    shifts of `chroma_mean_b` (one per semitone) and returns the best
    score plus the signed semitone offset that produced it. Otherwise
    returns the raw similarity at offset 0.

    Args:
        chroma_mean_a: 12-dim mean-chroma profile of segment A.
        chroma_mean_b: 12-dim mean-chroma profile of segment B.
        search_transposes: Whether to search over semitone shifts.

    Returns:
        (score, offset) where score is in [0, 1] and offset is signed
        semitones in -6..+5 (positive means transpose B up).
    """
    if not search_transposes:
        return _cosine_similarity(chroma_mean_a, chroma_mean_b), 0

    best_score = -1.0
    best_k = 0
    for k in range(12):
        shifted = np.roll(np.asarray(chroma_mean_b, dtype=float), k)
        score = _cosine_similarity(chroma_mean_a, shifted)
        if score > best_score:
            best_score = score
            best_k = k

    # Map 0..11 to a signed offset in -6..+5 (closest octave-equivalent).
    offset = best_k if best_k <= 6 else best_k - 12
    return best_score, offset


def rhythmic_score(bpm_a: float, bpm_b: float) -> float:
    """Score how close two tempos are after up-to-octave matching.

    The faster BPM is halved while it remains more than 1.5x the
    slower AND halving would not push it below the slower. The result
    is the ratio of (matched) slower to faster. A perfect match
    returns 1.0. Returns 0.0 if either BPM is non-positive.

    The second clause matters when the BPMs are far apart. For example
    60 vs 200: halving 200 to 100 leaves a 1.67x ratio, but halving
    again to 50 would put it below 60, so the function stops at 100
    and returns 60/100 = 0.6 (rather than over-halving to 50 and
    returning a misleadingly high 0.83).

    Args:
        bpm_a, bpm_b: BPMs of the two segments.

    Returns:
        Score in (0, 1].
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.0
    lo, hi = sorted([float(bpm_a), float(bpm_b)])
    # Halve the faster tempo while the ratio is wide AND halving would
    # not put it below the slower.
    while hi / lo > 1.5 and hi / 2 >= lo:
        hi = hi / 2
    return float(lo / hi)


def spectral_score(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Score the spectral complementarity of pairing two segments.

    Returns 1 minus the Bhattacharyya-like overlap of the two
    normalized band-energy profiles. Two identical profiles return
    0.0 (would mash muddy); two disjoint profiles return 1.0
    (complementary).

    Args:
        profile_a, profile_b: 1D band-energy arrays of the same length
            (typically the `spectral_profile` from
            `analyze.compute_features`).

    Returns:
        Score in [0, 1].

    Raises:
        ValueError: If the two profiles have different shapes.
    """
    a = np.asarray(profile_a, dtype=float)
    b = np.asarray(profile_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"spectral profiles must have the same shape, got {a.shape} vs {b.shape}"
        )
    sa, sb = a.sum(), b.sum()
    if sa == 0 or sb == 0:
        return 0.0
    a = a / sa
    b = b / sb
    overlap = float(np.minimum(a, b).sum())
    return 1.0 - overlap


def score_pair(
    features_a: dict,
    features_b: dict,
    bpm_a: float,
    bpm_b: float,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    search_transposes: bool = True,
) -> MashabilityScore:
    """Score the mashability of two segments.

    Default weights (0.5, 0.3, 0.2) emphasize harmonic compatibility
    as the dominant factor, then rhythmic alignment, then spectral
    complementarity. This roughly matches Davies 2014's weighting.

    Args:
        features_a, features_b: Output of `analyze.compute_features()`
            for each segment. Must contain `chroma_mean` and
            `spectral_profile` keys.
        bpm_a, bpm_b: BPM of each segment.
        weights: (harmonic, rhythmic, spectral) weights, must sum to 1.
        search_transposes: If True (default), searches all twelve
            semitone shifts of segment B's chroma for the best
            harmonic match. The chosen offset is recorded in the
            returned score.

    Returns:
        MashabilityScore with total and component breakdowns.

    Raises:
        ValueError: If weights do not sum to 1.0, or if required keys
            are missing from the feature dicts.
    """
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")
    for name, feats in [("features_a", features_a), ("features_b", features_b)]:
        for key in ("chroma_mean", "spectral_profile"):
            if key not in feats:
                raise ValueError(f"{name} missing required key {key!r}")

    h, offset = harmonic_score(
        features_a["chroma_mean"],
        features_b["chroma_mean"],
        search_transposes=search_transposes,
    )
    r = rhythmic_score(bpm_a, bpm_b)
    s = spectral_score(
        features_a["spectral_profile"],
        features_b["spectral_profile"],
    )

    wh, wr, ws = weights
    total = wh * h + wr * r + ws * s

    log.info(
        "Mashability: total=%.3f (harmonic=%.3f @ %+d st, rhythmic=%.3f, spectral=%.3f)",
        total, h, offset, r, s,
    )

    return MashabilityScore(
        total=float(total),
        harmonic=float(h),
        rhythmic=float(r),
        spectral=float(s),
        best_transpose_semitones=int(offset),
    )
