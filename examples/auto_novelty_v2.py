"""Auto-novelty v2: stem-based extraction + LUFS-balanced layers.

Addresses two real problems v1 surfaced when the rendered outputs
were listened to:

  1. v1 picked a "drum-prominent" section based on full-track chroma,
     which for Led Zeppelin's When the Levee Breaks meant the 80.9-
     130.3s window — drums + harmonica + Robert Plant vocals. The
     "drum layer" of every v1 mashup contained vocals.

  2. v1 peak-normalized only the final sum, which let the layer with
     hotter source mastering (Levee Breaks in this case) dominate
     the mix. The Metallica riffs got buried.

v2 fixes both:

  - Section boundaries + beat detection still run on the full track
    (most reliable signal). But audio extraction comes from
    Demucs-separated stems: the drum stem for the drum layer, and
    the (other + bass) stems summed for the riff layer. Vocals are
    cleanly excluded from both.

  - Each layer is LUFS-balanced to -18 LUFS before summing, so they
    reach the master at perceptually equal loudness. The hotter
    source no longer wins by default.

Run from any cwd:
    python -m examples.auto_novelty_v2
"""
from __future__ import annotations

import logging
from pathlib import Path

from stemflow.auto import compose_section_mashup


HOME   = Path.home()
CORPUS = HOME / "Music/stemflow-corpus/sources"
OUT    = HOME / "Music/stemflow-corpus/mashups/auto"

# Stem locations:
#   Pre-existing stems from the March 2026 Demucs runs.
LEGACY_STEMS = HOME / "Documents/Development/scholls_workspace/pipeline_output/stems_v2"
LEVEE_LEGACY_BASE = (
    LEGACY_STEMS
    / "levee/Led Zeppelin - 08. When the Levee Breaks (Remaster) (Rem"
)
SBT_LEGACY_BASE = (
    LEGACY_STEMS / "sad_but_true/Metallica - 02. Sad But True"
)

#   New stems from the June 2026 Demucs run for the other Metallica tracks.
NEW_STEMS = HOME / "Music/stemflow-corpus/stems/htdemucs_ft"


def legacy_stem(base: Path, stem_name: str) -> Path:
    """Stems from the March 2026 pipeline are named '<base>_(Stem)_htdemucs_ft.wav'."""
    return Path(f"{base}_({stem_name})_htdemucs_ft.wav")


def new_stem(track_dirname: str, stem: str) -> Path:
    """Stems from the new Demucs run live at NEW_STEMS/<track>/<stem>.wav."""
    return NEW_STEMS / track_dirname / f"{stem}.wav"


# Source tracks (used for analysis + beat detection)
LEVEE     = CORPUS / "Led Zeppelin/When the Levee Breaks.mp3"
SBT       = CORPUS / "Metallica/Sad But True.mp3"
MOP       = CORPUS / "Metallica/Master of Puppets.mp3"
ENTER_S   = CORPUS / "Metallica/Enter Sandman.mp3"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    # Levee drum stem — used as the drum source for all three mashups
    levee_drum_stem = legacy_stem(LEVEE_LEGACY_BASE, "Drums")

    pairings = [
        {
            "name": "Auto v2 - Levee drums x Sad But True riff.wav",
            "drum_track": LEVEE,
            "riff_track": SBT,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                legacy_stem(SBT_LEGACY_BASE, "Other"),
                legacy_stem(SBT_LEGACY_BASE, "Bass"),
            ],
        },
        {
            "name": "Auto v2 - Levee drums x Master of Puppets riff.wav",
            "drum_track": LEVEE,
            "riff_track": MOP,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                new_stem("Master of Puppets", "other"),
                new_stem("Master of Puppets", "bass"),
            ],
        },
        {
            "name": "Auto v2 - Levee drums x Enter Sandman riff.wav",
            "drum_track": LEVEE,
            "riff_track": ENTER_S,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                new_stem("Enter Sandman", "other"),
                new_stem("Enter Sandman", "bass"),
            ],
        },
    ]

    for cfg in pairings:
        print(f"\n{'=' * 72}")
        print(f"  {cfg['name']}")
        print(f"{'=' * 72}")

        # Verify stems exist
        missing = []
        if not cfg["drum_extract_from"].exists():
            missing.append(str(cfg["drum_extract_from"]))
        for p in cfg["riff_extract_from"]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            print("  SKIPPED — missing stems:")
            for m in missing:
                print(f"    {m}")
            continue

        try:
            result = compose_section_mashup(
                drum_track=cfg["drum_track"],
                riff_track=cfg["riff_track"],
                output_dir=OUT,
                output_name=cfg["name"],
                drum_extract_from=cfg["drum_extract_from"],
                riff_extract_from=cfg["riff_extract_from"],
                balance_layers=True,
                layer_target_lufs=-18.0,
                target_bpm=80.0,
            )
            print()
            print(result.explain())
        except ValueError as e:
            print(f"  SKIPPED: {e}")


if __name__ == "__main__":
    main()
