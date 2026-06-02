"""Auto-novelty v5: octave-multiplier on extreme-tempo riffs + downbeat snap.

Fixes two of three issues v4 still had:

  - Master of Puppets riff sounded slow (was being stretched 0.42x, i.e.
    half speed). v5 detects the extreme BPM ratio and applies an octave
    multiplier at the extraction layer: take 2x as many source beats over
    the same time window. The riff now plays close to its native speed
    while still fitting the mashup's time slot.

  - Sad But True riff sounded "not lined up." The 21.0s timestamp snapped
    to beat 32, which is NOT the iconic riff downbeat (beat 34 at 22.38s
    per beat-this). v5 snaps timestamp overrides to the nearest downbeat
    when one is within half a bar, ensuring phrase alignment.

What v5 does not fix: Enter Sandman at 125 BPM mashed with Levee at 71 has
a fundamental single-tempo gap. The midpoint (98) hurts both directions.
The polyrhythmic option (drum + riff at different effective tempos) is a
bigger redesign deferred to v6.

Run from any cwd:
    python -m examples.auto_novelty_v5
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

DRUM_START = 81.75  # one beat past Levee's section boundary at 80.9s


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    levee_drum_stem = legacy_stem(LEVEE_LEGACY_BASE, "Drums")

    pairings = [
        {
            "name": "Auto v5 - Levee drums x Sad But True riff.wav",
            "drum_track": LEVEE, "riff_track": SBT,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                legacy_stem(SBT_LEGACY_BASE, "Other"),
                legacy_stem(SBT_LEGACY_BASE, "Bass"),
            ],
            "riff_start_time": 21.0,
        },
        {
            "name": "Auto v5 - Levee drums x Master of Puppets riff.wav",
            "drum_track": LEVEE, "riff_track": MOP,
            "drum_extract_from": levee_drum_stem,
            "riff_extract_from": [
                new_stem("Master of Puppets", "other"),
                new_stem("Master of Puppets", "bass"),
            ],
            "riff_start_time": 21.0,
        },
        {
            "name": "Auto v5 - Levee drums x Enter Sandman riff.wav",
            "drum_track": LEVEE, "riff_track": ENTER_S,
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
            print("  SKIPPED — missing stems")
            continue
        try:
            result = compose_section_mashup(
                drum_track=cfg["drum_track"], riff_track=cfg["riff_track"],
                output_dir=OUT, output_name=cfg["name"],
                drum_extract_from=cfg["drum_extract_from"],
                riff_extract_from=cfg["riff_extract_from"],
                drum_start_time=DRUM_START,
                riff_start_time=cfg["riff_start_time"],
                balance_layers=True,
            )
            print()
            print(result.explain())
        except ValueError as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
