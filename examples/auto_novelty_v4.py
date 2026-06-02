"""Auto-novelty v4: human-in-the-loop timestamp overrides.

After three rounds of listening (v1 → v2 → v3), the auto picker is
still missing the iconic riffs the listener has in mind. v4 accepts
that auto-discovery has limits and uses explicit timestamp overrides:

  - Enter Sandman: iconic riff at 0:55 in source
  - Master of Puppets: iconic section starts ~0:21 (post-intro)
  - Sad But True: iconic riff entry at ~0:21

The drum source is still Levee Breaks' 80.9-130.3s window (the
longest drum-prominent section meeting the 30s minimum), but the
start time is shifted ~1 beat past the section boundary to skip
the cymbal crash that lands at the section start.

Output prefix: "Auto v4".

Run from any cwd:
    python -m examples.auto_novelty_v4
"""
from __future__ import annotations

import logging
from pathlib import Path

from stemflow.auto import compose_section_mashup


HOME   = Path.home()
CORPUS = HOME / "Music/stemflow-corpus/sources"
OUT    = HOME / "Music/stemflow-corpus/mashups/auto"

LEGACY_STEMS = HOME / "Documents/Development/scholls_workspace/pipeline_output/stems_v2"
LEVEE_LEGACY_BASE = LEGACY_STEMS / "levee/Led Zeppelin - 08. When the Levee Breaks (Remaster) (Rem"
SBT_LEGACY_BASE   = LEGACY_STEMS / "sad_but_true/Metallica - 02. Sad But True"
NEW_STEMS = HOME / "Music/stemflow-corpus/stems/htdemucs_ft"


def legacy_stem(base: Path, stem: str) -> Path:
    return Path(f"{base}_({stem})_htdemucs_ft.wav")


def new_stem(track_dirname: str, stem: str) -> Path:
    return NEW_STEMS / track_dirname / f"{stem}.wav"


LEVEE   = CORPUS / "Led Zeppelin/When the Levee Breaks.mp3"
SBT     = CORPUS / "Metallica/Sad But True.mp3"
MOP     = CORPUS / "Metallica/Master of Puppets.mp3"
ENTER_S = CORPUS / "Metallica/Enter Sandman.mp3"


# Drum lead-skip: section starts at 80.9s with a cymbal crash. Levee BPM is
# 71.4, so 1 beat = 60/71.4 = 0.84s. 80.9 + 0.84 = 81.74s, past the cymbal.
DRUM_START = 81.75


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    levee_drum_stem = legacy_stem(LEVEE_LEGACY_BASE, "Drums")

    pairings = [
        {
            "name": "Auto v4 - Levee drums x Sad But True riff.wav",
            "drum_track": LEVEE,
            "riff_track": SBT,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                legacy_stem(SBT_LEGACY_BASE, "Other"),
                legacy_stem(SBT_LEGACY_BASE, "Bass"),
            ],
            "riff_start_time": 21.0,  # user-specified iconic-riff start
        },
        {
            "name": "Auto v4 - Levee drums x Master of Puppets riff.wav",
            "drum_track": LEVEE,
            "riff_track": MOP,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                new_stem("Master of Puppets", "other"),
                new_stem("Master of Puppets", "bass"),
            ],
            "riff_start_time": 21.0,
        },
        {
            "name": "Auto v4 - Levee drums x Enter Sandman riff.wav",
            "drum_track": LEVEE,
            "riff_track": ENTER_S,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                new_stem("Enter Sandman", "other"),
                new_stem("Enter Sandman", "bass"),
            ],
            "riff_start_time": 55.0,
        },
    ]

    for cfg in pairings:
        print(f"\n{'=' * 72}")
        print(f"  {cfg['name']}")
        print(f"{'=' * 72}")

        missing = [p for p in [cfg["drum_extract_from"], *cfg["riff_extract_from"]] if not p.exists()]
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
                drum_start_time=DRUM_START,
                riff_start_time=cfg["riff_start_time"],
                balance_layers=True,
                layer_target_lufs=-18.0,
                # target_bpm omitted -> auto-computed midpoint
            )
            print()
            print(result.explain())
        except ValueError as e:
            print(f"  SKIPPED: {e}")


if __name__ == "__main__":
    main()
