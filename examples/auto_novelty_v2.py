"""Auto-novelty v2 / v3: stems, LUFS balance, riff position, auto tempo.

Three iteration generations of fixes from listening to the outputs:

  v1 -> v2: stem-based extraction (cleanly excludes vocals) and
  LUFS-balanced layers (so the hotter source doesn't dominate).

  v2 -> v3: bias the riff picker toward EARLIER sections (the iconic
  riff lives near the beginning of the song, not in late repetitions
  where pure RMS would land), and auto-compute target_bpm as the
  midpoint of the two source BPMs (so Enter Sandman's 125 BPM doesn't
  get stretched to 80, killing its recognizability).

The script outputs files prefixed "Auto v3" so they sit alongside the
v1 and v2 versions for A/B comparison.

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
            "name": "Auto v3 - Levee drums x Sad But True riff.wav",
            "drum_track": LEVEE,
            "riff_track": SBT,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                legacy_stem(SBT_LEGACY_BASE, "Other"),
                legacy_stem(SBT_LEGACY_BASE, "Bass"),
            ],
        },
        {
            "name": "Auto v3 - Levee drums x Master of Puppets riff.wav",
            "drum_track": LEVEE,
            "riff_track": MOP,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                new_stem("Master of Puppets", "other"),
                new_stem("Master of Puppets", "bass"),
            ],
        },
        {
            "name": "Auto v3 - Levee drums x Enter Sandman riff.wav",
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
                # target_bpm omitted -> auto-computed midpoint
                # position_weight inside the picker defaults to 0.4
            )
            print()
            print(result.explain())
        except ValueError as e:
            print(f"  SKIPPED: {e}")


if __name__ == "__main__":
    main()
