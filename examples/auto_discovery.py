"""Auto-discovery: score the whole corpus and render the most interesting pairs.

Five rounds of iteration on three specified pairings produced the v5
auto-composer. This script flips the direction: instead of "compose this
pair," it asks "which pairs from my corpus are worth composing?" and
renders the answers.

Method:
  1. Score every unordered pair via mashability.score_pair on full tracks.
  2. Filter for pairs we have stems for on both sides (need stems for
     clean extraction).
  3. Filter for pairs we have NOT already rendered (Levee x SBT/MoP/EnterS
     have been exhaustively iterated).
  4. Bias toward cross-genre / different-era pairings (the mashup thesis
     calls this "contextual incongruity" and the predictive-coding sweet
     spot lives there).
  5. Pick a handful with a varied set of "drum role" sources — not just
     Levee — so the listener hears different rhythmic foundations.
  6. Render each with the v5 auto-composer (full auto on section pick).

Run from any cwd:
    python -m examples.auto_discovery
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf

from stemflow.analyze import compute_features, detect_bpm
from stemflow.auto import compose_section_mashup
from stemflow.mashability import score_pair


HOME = Path.home()
CORPUS = HOME / "Music/stemflow-corpus/sources"
STEMS_LEGACY = HOME / "Documents/Development/scholls_workspace/pipeline_output/stems_v2"
STEMS_NEW = HOME / "Music/stemflow-corpus/stems/htdemucs_ft"
OUT = HOME / "Music/stemflow-corpus/mashups/auto"


@dataclass
class Track:
    name: str
    source: Path
    drums: Path
    other: Path
    bass: Path
    genre: str  # for incongruity scoring

    def stems_ready(self) -> bool:
        return all(p.exists() for p in (self.drums, self.other, self.bass))


def legacy(base: str, stem: str) -> Path:
    return STEMS_LEGACY / base / f"{base.split('/')[-1] if '/' in base else base}_({stem})_htdemucs_ft.wav"


# Per-track configuration. Genre tags from the competitive-landscape
# perspective: hard-rock-blues / metal-thrash / electronic-disco.
TRACKS = [
    Track(
        name="Levee Breaks",
        source=CORPUS / "Led Zeppelin/When the Levee Breaks.mp3",
        drums=STEMS_LEGACY / "levee/Led Zeppelin - 08. When the Levee Breaks (Remaster) (Rem_(Drums)_htdemucs_ft.wav",
        other=STEMS_LEGACY / "levee/Led Zeppelin - 08. When the Levee Breaks (Remaster) (Rem_(Other)_htdemucs_ft.wav",
        bass=STEMS_LEGACY / "levee/Led Zeppelin - 08. When the Levee Breaks (Remaster) (Rem_(Bass)_htdemucs_ft.wav",
        genre="hard-rock-blues",
    ),
    Track(
        name="Sad But True",
        source=CORPUS / "Metallica/Sad But True.mp3",
        drums=STEMS_LEGACY / "sad_but_true/Metallica - 02. Sad But True_(Drums)_htdemucs_ft.wav",
        other=STEMS_LEGACY / "sad_but_true/Metallica - 02. Sad But True_(Other)_htdemucs_ft.wav",
        bass=STEMS_LEGACY / "sad_but_true/Metallica - 02. Sad But True_(Bass)_htdemucs_ft.wav",
        genre="metal-groove",
    ),
    Track(
        name="Master of Puppets",
        source=CORPUS / "Metallica/Master of Puppets.mp3",
        drums=STEMS_NEW / "Master of Puppets/drums.wav",
        other=STEMS_NEW / "Master of Puppets/other.wav",
        bass=STEMS_NEW / "Master of Puppets/bass.wav",
        genre="metal-thrash",
    ),
    Track(
        name="Enter Sandman",
        source=CORPUS / "Metallica/Enter Sandman.mp3",
        drums=STEMS_NEW / "Enter Sandman/drums.wav",
        other=STEMS_NEW / "Enter Sandman/other.wav",
        bass=STEMS_NEW / "Enter Sandman/bass.wav",
        genre="metal-arena",
    ),
    Track(
        name="Harder Better Faster Stronger",
        source=CORPUS / "Daft Punk/Harder Better Faster Stronger.mp3",
        drums=STEMS_NEW / "Harder Better Faster Stronger/drums.wav",
        other=STEMS_NEW / "Harder Better Faster Stronger/other.wav",
        bass=STEMS_NEW / "Harder Better Faster Stronger/bass.wav",
        genre="electronic-disco",
    ),
    Track(
        name="Harvester of Sorrow",
        source=CORPUS / "Metallica/Harvester of Sorrow.mp3",
        drums=STEMS_NEW / "Harvester of Sorrow/drums.wav",
        other=STEMS_NEW / "Harvester of Sorrow/other.wav",
        bass=STEMS_NEW / "Harvester of Sorrow/bass.wav",
        genre="metal-doom",
    ),
    Track(
        name="One",
        source=CORPUS / "Metallica/One.mp3",
        drums=STEMS_NEW / "One/drums.wav",
        other=STEMS_NEW / "One/other.wav",
        bass=STEMS_NEW / "One/bass.wav",
        genre="metal-ballad",
    ),
    Track(
        name="Orion",
        source=CORPUS / "Metallica/Orion.m4a",
        drums=STEMS_NEW / "Orion/drums.wav",
        other=STEMS_NEW / "Orion/other.wav",
        bass=STEMS_NEW / "Orion/bass.wav",
        genre="metal-instrumental",
    ),
]


# Pairs already exhaustively iterated (v1 through v5). Skip these.
EXHAUSTED_PAIRS = {
    frozenset(("Levee Breaks", "Sad But True")),
    frozenset(("Levee Breaks", "Master of Puppets")),
    frozenset(("Levee Breaks", "Enter Sandman")),
}


def score_pairings(tracks: list[Track]) -> list[dict]:
    """Compute mashability for every pair-with-stems, return ranked by total."""
    bpm_cache = {}
    feat_cache = {}
    results = []
    for a, b in combinations(tracks, 2):
        if not (a.stems_ready() and b.stems_ready()):
            continue
        if frozenset((a.name, b.name)) in EXHAUSTED_PAIRS:
            continue

        for t in (a, b):
            if t.name not in bpm_cache:
                print(f"  analyzing {t.name}...")
                bpm_cache[t.name] = detect_bpm(t.source).bpm
                feat_cache[t.name] = compute_features(t.source)

        score = score_pair(
            feat_cache[a.name], feat_cache[b.name],
            bpm_cache[a.name], bpm_cache[b.name],
        )

        cross_genre = 1.0 if a.genre != b.genre else 0.6
        novelty_score = score.total * cross_genre

        results.append({
            "a": a, "b": b,
            "score": score,
            "bpm_a": bpm_cache[a.name],
            "bpm_b": bpm_cache[b.name],
            "novelty_score": novelty_score,
            "cross_genre": a.genre != b.genre,
        })

    results.sort(key=lambda r: -r["novelty_score"])
    return results


def pick_render_set(ranked: list[dict], max_renders: int = 4) -> list[dict]:
    """Pick a varied set of pairs to render.

    Aim for: variety in the drum-source track (so we hear non-Levee
    foundations), at least one cross-genre pair, and high mashability.
    """
    chosen: list[dict] = []
    drum_track_names_used: set[str] = set()

    # Pass 1: highest-scoring with variety in drum source
    for r in ranked:
        if len(chosen) >= max_renders:
            break
        drum_name = r["a"].name  # we'll assign drums to A by default
        if drum_name not in drum_track_names_used:
            chosen.append(r)
            drum_track_names_used.add(drum_name)

    # Pass 2: fill remaining slots with the next highest scores
    for r in ranked:
        if len(chosen) >= max_renders:
            break
        if r not in chosen:
            chosen.append(r)

    return chosen


def render_one(r: dict, out_dir: Path) -> Path | None:
    a, b = r["a"], r["b"]
    # Assign drums to the LOWER-BPM source by default (it has more flexibility
    # to be stretched up; the riff source provides character).
    if r["bpm_a"] <= r["bpm_b"]:
        drum, riff = a, b
    else:
        drum, riff = b, a

    output_name = f"Auto discovery - {drum.name} drums x {riff.name} riff.wav"
    print(f"\n{'=' * 72}")
    print(f"  {output_name}")
    print(f"  mashability {r['score'].total:.3f}  (h={r['score'].harmonic:.2f} {r['score'].best_transpose_semitones:+d}st, "
          f"r={r['score'].rhythmic:.2f}, s={r['score'].spectral:.2f})  "
          f"cross-genre={r['cross_genre']}")
    print(f"{'=' * 72}")

    try:
        result = compose_section_mashup(
            drum_track=drum.source,
            riff_track=riff.source,
            output_dir=out_dir,
            output_name=output_name,
            drum_extract_from=drum.drums,
            riff_extract_from=[riff.other, riff.bass],
            balance_layers=True,
            # No timestamp overrides — pure auto-discovery, lean on v5's
            # position-biased picker
        )
        print()
        print(result.explain())
        return result.output_path
    except ValueError as e:
        print(f"  RENDER FAILED: {e}")
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    ready = [t for t in TRACKS if t.stems_ready()]
    skipped = [t.name for t in TRACKS if not t.stems_ready()]
    print(f"Tracks ready (stems present): {len(ready)}")
    for t in ready:
        print(f"  {t.name} ({t.genre})")
    if skipped:
        print(f"Tracks skipped (stems missing):")
        for n in skipped:
            print(f"  {n}")
    print()

    print("Scoring all pair candidates...")
    ranked = score_pairings(ready)

    print(f"\n{'=' * 72}")
    print(f"  Mashability ranking (excluding already-exhausted Levee × Metallica trio)")
    print(f"{'=' * 72}")
    print(f"  {'novelty':>8}  {'mash':>6}  {'BPMs':>13}  {'cross-genre':>11}  pair")
    for r in ranked[:15]:
        print(
            f"  {r['novelty_score']:>8.3f}  {r['score'].total:>6.3f}  "
            f"{r['bpm_a']:>5.0f}/{r['bpm_b']:>5.0f}  {str(r['cross_genre']):>11}  "
            f"{r['a'].name} × {r['b'].name}"
        )

    chosen = pick_render_set(ranked, max_renders=4)
    print(f"\nPicked {len(chosen)} for rendering (variety in drum source + high novelty score)")

    rendered = []
    for r in chosen:
        path = render_one(r, OUT)
        if path is not None:
            rendered.append(path)

    print(f"\n\n{'=' * 72}")
    print(f"  RENDERED {len(rendered)} mashups")
    print(f"{'=' * 72}")
    for p in rendered:
        print(f"  {p}")


if __name__ == "__main__":
    main()
