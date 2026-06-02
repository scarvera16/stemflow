"""Auto-composition: the first end-to-end auto-mashup function.

Takes two source tracks and produces a layered mashup without
requiring the listener to specify timestamps. Picks a drum-prominent
section from one track and a riff-prominent section from the other,
extracts both via beat-aligned stretching, and renders a mastered
output.

This is v1 — heuristic section picking, no listener-state input,
no frisson-target conditioning. It is the proof of concept that the
Librarian's full stack (sections + phrase + stretch + mix + master)
can run end-to-end without human-curated timestamps. It will pick
*something* for any pair of tracks; whether it picks what a human
would pick is a separate, harder question (the semantic-section
labeling problem from decisions.md).

The output transparently records what was picked, so a listener can
see *why* the system chose what it did and supply different
heuristics or manual overrides for the next round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .master import master
from .phrase import beat_aligned_stretch
from .sections import SectionStats, find_section_boundaries, section_features

log = logging.getLogger(__name__)


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass
class AutoMashupResult:
    """What the auto-composer produced and the choices it made."""
    output_path: Path
    drum_track: Path
    riff_track: Path
    drum_section: SectionStats
    riff_section: SectionStats
    drum_start_beat: int
    riff_start_beat: int
    target_bpm: float

    def explain(self) -> str:
        """Human-readable summary of the picks."""
        return (
            f"Auto-mashup at {self.target_bpm} BPM\n"
            f"  drums   : {self.drum_track.name}  section {self.drum_section.start_s:.1f}-{self.drum_section.end_s:.1f}s "
            f"(rms={self.drum_section.rms_mean:.3f}, chroma={self.drum_section.chroma_strength:.3f}, "
            f"density={self.drum_section.density:.2f}/s)  starting beat {self.drum_start_beat}\n"
            f"  riff    : {self.riff_track.name}  section {self.riff_section.start_s:.1f}-{self.riff_section.end_s:.1f}s "
            f"(rms={self.riff_section.rms_mean:.3f}, chroma={self.riff_section.chroma_strength:.3f}, "
            f"density={self.riff_section.density:.2f}/s)  starting beat {self.riff_start_beat}\n"
            f"  output  : {self.output_path}"
        )


# ── Section pickers (heuristics) ─────────────────────────────────────────────

def pick_drum_section(
    sections: list[SectionStats],
    min_duration: float = 5.0,
) -> Optional[SectionStats]:
    """Pick the section most likely to be a drum break or drum-prominent
    intro.

    Heuristic: among sections at least `min_duration` seconds long,
    rank by lowest chroma strength (= least tonal content), breaking
    ties by highest onset density (= most active percussion).

    Returns None if no section meets the duration threshold.
    """
    candidates = [s for s in sections if s.duration_s >= min_duration]
    if not candidates:
        return None
    candidates.sort(key=lambda s: (s.chroma_strength, -s.density))
    log.info("pick_drum_section: chose %.1f-%.1fs (chroma=%.3f, density=%.2f)",
             candidates[0].start_s, candidates[0].end_s,
             candidates[0].chroma_strength, candidates[0].density)
    return candidates[0]


def pick_riff_section(
    sections: list[SectionStats],
    min_duration: float = 20.0,
    min_chroma: float = 0.2,
    position_weight: float = 0.4,
) -> Optional[SectionStats]:
    """Pick the section most likely to be a primary riff section.

    Heuristic: among sections at least `min_duration` seconds long
    and with at least `min_chroma` chroma strength (i.e., enough
    tonal content to be a riff and not just drums), rank by a
    composite score that combines RMS energy with section position.
    Earlier sections get a position bonus because the iconic riff
    of a song is typically established near the song's beginning
    (after a short intro), not in late repetitions.

    Args:
        sections: List of section stats to choose from.
        min_duration: Minimum acceptable section duration.
        min_chroma: Minimum tonal content (rules out drum-only).
        position_weight: How much to bias toward earlier sections,
            in [0, 1]. 0 = pure RMS (v1/v2 behavior). 1 = pure
            earliest-meeting-criteria. 0.4 (default) gives the
            iconic-riff bias while still preferring loud sections.

    Returns None if no section meets the duration and chroma criteria.
    """
    candidates = [
        s for s in sections
        if s.duration_s >= min_duration and s.chroma_strength >= min_chroma
    ]
    if not candidates:
        return None

    # Normalize position by song duration (using the latest end as a proxy)
    total = max(s.end_s for s in candidates)
    max_rms = max(s.rms_mean for s in candidates) or 1.0

    def score(s: SectionStats) -> float:
        rms_score = s.rms_mean / max_rms
        position_score = 1.0 - (s.start_s / total)  # 1 at earliest, 0 at latest
        return (1 - position_weight) * rms_score + position_weight * position_score

    candidates.sort(key=lambda s: -score(s))
    pick = candidates[0]
    log.info(
        "pick_riff_section: chose %.1f-%.1fs (rms=%.3f, chroma=%.3f, position_bias=%.2f)",
        pick.start_s, pick.end_s, pick.rms_mean, pick.chroma_strength, position_weight,
    )
    return pick


# ── Auto tempo selection ─────────────────────────────────────────────────────

def auto_target_bpm(
    drum_bpm: float,
    riff_bpm: float,
    max_ratio: float = 2.0,
) -> float:
    """Choose a mashup target tempo from two source BPMs.

    Returns the arithmetic midpoint after light octave-matching. If
    one source is more than `max_ratio` times the other (e.g., Levee
    Breaks at 71 BPM versus Master of Puppets at 214), the higher
    is halved until the ratio is within range. This compromise
    stretches both sources by roughly equal amounts and avoids
    extreme single-source distortion that destroys recognizability.

    Use the default unless you have a reason to override. For pairs
    with very different native tempos, the midpoint still requires
    significant stretching of both sources; some pairs simply don't
    tempo-match cleanly and may sound better at a target near one
    source's native BPM (set `target_bpm` explicitly in that case).

    Args:
        drum_bpm: BPM of the drum source.
        riff_bpm: BPM of the riff source.
        max_ratio: Maximum allowable BPM ratio before octave-matching.
            Default 2.0 (one octave).

    Returns:
        Target BPM for the mashup.
    """
    a, b = float(drum_bpm), float(riff_bpm)
    # Octave-match the larger toward the smaller
    while b / a > max_ratio:
        b /= 2
    while a / b > max_ratio:
        a /= 2
    return (a + b) / 2


# ── Section discovery helper ─────────────────────────────────────────────────

def _all_sections(audio_file: Path, n_segments: int, min_section: float = 0.5) -> list[SectionStats]:
    """Run find_section_boundaries + section_features and return the list."""
    boundaries = find_section_boundaries(audio_file, n_segments=n_segments)
    out: list[SectionStats] = []
    for i in range(len(boundaries) - 1):
        s, e = float(boundaries[i]), float(boundaries[i + 1])
        if e - s < min_section:
            continue
        out.append(section_features(audio_file, s, e))
    return out


# ── compose_section_mashup ───────────────────────────────────────────────────

def compose_section_mashup(
    drum_track: Path,
    riff_track: Path,
    output_dir: Path,
    output_name: str = "auto_mashup.wav",
    *,
    drum_extract_from: Optional[Path] = None,
    riff_extract_from: Optional[list[Path]] = None,
    balance_layers: bool = True,
    layer_target_lufs: float = -18.0,
    target_bpm: Optional[float] = None,
    drum_intro_bars: int = 2,
    riff_bars: int = 8,
    n_segments: int = 10,
    drum_gain_db: float = 0.0,
    riff_gain_db: float = 0.0,
    fade_ms: int = 50,
    device: str = "mps",
) -> AutoMashupResult:
    """Compose an auto-mashup: drums from one track, riff from another.

    Pipeline:
      1. Find sections in each track via librosa segmentation (on the
         full mix, where section boundaries are most pronounced).
      2. Pick the most drum-prominent section from drum_track and the
         most riff-prominent section from riff_track.
      3. Run beat-this on both tracks.
      4. Extract the first `drum_intro_bars + riff_bars` of beats from
         drum_track's chosen section, and `riff_bars` of beats from
         riff_track's chosen section.
      5. Beat-aligned stretch each clip to the target BPM. If
         `drum_extract_from` / `riff_extract_from` are provided,
         audio is extracted from those stem paths instead of the
         full tracks (vocals get cleanly excluded).
      6. Optionally LUFS-balance the two layers to `layer_target_lufs`
         so they reach the master at perceptually equal loudness.
      7. Layer: drums alone for the intro bars, then drums + riff for
         the rest. Master to -14 LUFS.

    Args:
        drum_track: Path to the track providing drums (used for
            section analysis + beat detection).
        riff_track: Path to the track providing the riff (used for
            section analysis + beat detection).
        output_dir: Where to write the rendered file.
        output_name: Output filename for the mastered output.
        drum_extract_from: If provided, extract drum audio from this
            stem path instead of `drum_track`. Use the Demucs drum
            stem to keep vocals/other instruments out of the drum
            layer. Section boundaries and beats are still computed
            from the full track.
        riff_extract_from: If provided, extract riff audio from these
            stem paths (summed) instead of `riff_track`. Pass `[other,
            bass]` to get a full instrumental riff without vocals.
        balance_layers: LUFS-balance each layer to `layer_target_lufs`
            before mixing. Strongly recommended; without it, the layer
            with hotter mastering dominates the mix. Default True.
        layer_target_lufs: Target loudness for each layer when
            `balance_layers=True`. Default -18 LUFS (gives headroom
            for the final mix and master).
        target_bpm: Common tempo for the mashup. If None (default),
            computed automatically via `auto_target_bpm()` from the
            two source BPMs (lightly octave-matched midpoint). Pass
            an explicit value to override.
        drum_intro_bars: Bars of drums alone before the riff drops in.
        riff_bars: Bars of riff layered with drums.
        n_segments: Sections to consider per track.
        drum_gain_db: Per-layer gain on drums *after* LUFS balancing.
            Default 0.
        riff_gain_db: Per-layer gain on riff *after* LUFS balancing.
            Default 0. When `balance_layers=False`, the historical
            behavior was -3 dB to make drums prominent; you may want
            to set that explicitly.
        fade_ms: Equal-power fade-in on the riff entry.
        device: beat-this inference device.

    Returns:
        AutoMashupResult with the picks recorded for transparency.

    Raises:
        ValueError: If no suitable section is found in either track,
            or if `drum_extract_from` / `riff_extract_from` paths are
            unreadable.
    """
    from beat_this.inference import File2Beats

    drum_track = Path(drum_track)
    riff_track = Path(riff_track)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("compose_section_mashup: drum=%s, riff=%s", drum_track.name, riff_track.name)

    # Compute target_bpm from sources if not explicitly provided. We need the
    # source BPMs first, so run beat-this up front (we'd run it later anyway).
    from beat_this.inference import File2Beats
    f2b = File2Beats(device=device, dbn=False)
    drum_beats, _ = f2b(str(drum_track))
    riff_beats, _ = f2b(str(riff_track))
    drum_beats = np.asarray(drum_beats)
    riff_beats = np.asarray(riff_beats)

    drum_bpm = 60.0 / float(np.median(np.diff(drum_beats))) if len(drum_beats) > 1 else 120.0
    riff_bpm = 60.0 / float(np.median(np.diff(riff_beats))) if len(riff_beats) > 1 else 120.0

    if target_bpm is None:
        target_bpm = auto_target_bpm(drum_bpm, riff_bpm)
        log.info("auto target_bpm: drum %.1f + riff %.1f -> %.1f BPM",
                 drum_bpm, riff_bpm, target_bpm)

    # Required section durations (the auto-composer must pick sections long
    # enough to actually supply the requested bars; otherwise it would silently
    # extract beats past the chosen section's end).
    target_beat_period = 60.0 / target_bpm
    total_bars = drum_intro_bars + riff_bars
    drum_required_s = total_bars * 4 * target_beat_period
    riff_required_s = riff_bars * 4 * target_beat_period

    # 1-2: Section discovery + pick (filtered by required duration)
    drum_sections = _all_sections(drum_track, n_segments)
    drum_pick = pick_drum_section(drum_sections, min_duration=drum_required_s)
    if drum_pick is None:
        raise ValueError(
            f"no drum-prominent section >= {drum_required_s:.0f}s in {drum_track.name} "
            f"(longest section: {max((s.duration_s for s in drum_sections), default=0):.1f}s). "
            "Try fewer drum_intro_bars + riff_bars, or n_segments=15+ for finer boundaries."
        )

    riff_sections = _all_sections(riff_track, n_segments)
    riff_pick = pick_riff_section(riff_sections, min_duration=riff_required_s)
    if riff_pick is None:
        raise ValueError(
            f"no riff-prominent section >= {riff_required_s:.0f}s in {riff_track.name}. "
            "Try fewer riff_bars or n_segments=15+ for finer boundaries."
        )

    # 3: Beats already computed above (needed for auto_target_bpm).

    # 4: First beat at or after each section start
    drum_start_idx = int(np.searchsorted(drum_beats, drum_pick.start_s))
    riff_start_idx = int(np.searchsorted(riff_beats, riff_pick.start_s))

    drum_n_beats = total_bars * 4
    riff_n_beats = riff_bars * 4

    if drum_start_idx + drum_n_beats >= len(drum_beats):
        raise ValueError(
            f"drum section ends before {drum_n_beats} beats are available "
            f"(have {len(drum_beats) - drum_start_idx})"
        )
    if riff_start_idx + riff_n_beats >= len(riff_beats):
        raise ValueError(
            f"riff section ends before {riff_n_beats} beats are available "
            f"(have {len(riff_beats) - riff_start_idx})"
        )

    # 5: Beat-aligned stretch.
    # Extract from a stem path if provided, else from the full track.
    # Beats and section boundaries were computed from the full track
    # (more reliable), but the actual audio comes from the stem.
    drum_source = Path(drum_extract_from) if drum_extract_from is not None else drum_track
    drum_clip, sr = beat_aligned_stretch(
        drum_source, drum_beats, drum_start_idx, drum_n_beats, target_beat_period,
    )

    if riff_extract_from is not None:
        riff_paths = [Path(p) for p in riff_extract_from]
        if not riff_paths:
            raise ValueError("riff_extract_from must be a non-empty list of stem paths")
        riff_layers = []
        for p in riff_paths:
            clip, _ = beat_aligned_stretch(
                p, riff_beats, riff_start_idx, riff_n_beats, target_beat_period,
            )
            riff_layers.append(clip)
        # Sum the stems element-wise (same shape since same beat range + sr)
        riff_clip = np.sum(riff_layers, axis=0).astype(np.float32)
    else:
        riff_clip, _ = beat_aligned_stretch(
            riff_track, riff_beats, riff_start_idx, riff_n_beats, target_beat_period,
        )

    # 6: LUFS balance so both layers reach the master at perceptually equal
    # loudness. Without this step, the layer with hotter source mastering
    # dominates the mix regardless of structural intent.
    if balance_layers:
        import pyloudnorm
        meter = pyloudnorm.Meter(sr)
        # pyloudnorm wants mono; use the mean of stereo channels
        drum_lufs = float(meter.integrated_loudness(drum_clip.mean(axis=1)))
        riff_lufs = float(meter.integrated_loudness(riff_clip.mean(axis=1)))
        drum_balance_gain = 10 ** ((layer_target_lufs - drum_lufs) / 20.0)
        riff_balance_gain = 10 ** ((layer_target_lufs - riff_lufs) / 20.0)
        log.info(
            "LUFS balance: drums %.1f -> %.1f LUFS (%+0.2f dB), riff %.1f -> %.1f LUFS (%+0.2f dB)",
            drum_lufs, layer_target_lufs, layer_target_lufs - drum_lufs,
            riff_lufs, layer_target_lufs, layer_target_lufs - riff_lufs,
        )
        drum_clip = (drum_clip * drum_balance_gain).astype(np.float32)
        riff_clip = (riff_clip * riff_balance_gain).astype(np.float32)

    # 7: Apply per-layer gain (artistic tilt on top of the LUFS balance),
    # then layer + master.
    drum_clip = drum_clip * (10 ** (drum_gain_db / 20.0))
    riff_clip = riff_clip * (10 ** (riff_gain_db / 20.0))

    total_samples = int(total_bars * 4 * target_beat_period * sr)
    mix = np.zeros((total_samples, 2), dtype=np.float32)
    mix[:len(drum_clip)] += drum_clip

    riff_start_sample = int(drum_intro_bars * 4 * target_beat_period * sr)
    riff_layer = riff_clip.copy()

    fade_samples = min(int(fade_ms / 1000.0 * sr), len(riff_layer))
    fade_in = np.sin(np.linspace(0, np.pi / 2, fade_samples)).astype(np.float32)
    for ch in range(2):
        riff_layer[:fade_samples, ch] *= fade_in

    end_idx = min(riff_start_sample + len(riff_layer), total_samples)
    mix[riff_start_sample:end_idx] += riff_layer[:end_idx - riff_start_sample]

    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = mix * (10 ** (-1.0 / 20.0) / peak)

    raw_path = output_dir / output_name.replace(".wav", " raw.wav")
    sf.write(str(raw_path), mix, sr, subtype="FLOAT")
    output_path = master(raw_path, output_dir, target_lufs=-14.0, output_name=output_name)

    return AutoMashupResult(
        output_path=output_path,
        drum_track=drum_track,
        riff_track=riff_track,
        drum_section=drum_pick,
        riff_section=riff_pick,
        drum_start_beat=drum_start_idx,
        riff_start_beat=riff_start_idx,
        target_bpm=target_bpm,
    )
