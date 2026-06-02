"""End-to-end auto-novelty demo.

Picks sections automatically and composes a mashup with no human
timestamp input. Prints the transparency report so you can see
exactly which sections the system chose and why.

Tries three pairs from the corpus to demonstrate variability:
    - Levee × Sad But True (compare against the manual v11)
    - Levee × Master of Puppets (novelty pair)
    - Levee × Enter Sandman (novelty pair)

Run from any cwd:
    python -m examples.auto_novelty
"""
from __future__ import annotations

import logging
from pathlib import Path

from stemflow.auto import compose_section_mashup


HOME   = Path.home()
CORPUS = HOME / "Music/stemflow-corpus/sources"
OUT    = HOME / "Music/stemflow-corpus/mashups/auto"

LEVEE     = CORPUS / "Led Zeppelin/When the Levee Breaks.mp3"
SBT       = CORPUS / "Metallica/Sad But True.mp3"
MOP       = CORPUS / "Metallica/Master of Puppets.mp3"
ENTER_S   = CORPUS / "Metallica/Enter Sandman.mp3"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    pairings = [
        (LEVEE, SBT,     "Auto - Levee drums x Sad But True riff.wav"),
        (LEVEE, MOP,     "Auto - Levee drums x Master of Puppets riff.wav"),
        (LEVEE, ENTER_S, "Auto - Levee drums x Enter Sandman riff.wav"),
    ]

    for drum_track, riff_track, output_name in pairings:
        print(f"\n{'=' * 72}")
        print(f"  Auto-composing: {drum_track.stem} drums x {riff_track.stem} riff")
        print(f"{'=' * 72}")
        try:
            result = compose_section_mashup(
                drum_track=drum_track,
                riff_track=riff_track,
                output_dir=OUT,
                output_name=output_name,
                target_bpm=80.0,
            )
            print()
            print(result.explain())
        except ValueError as e:
            print(f"  SKIPPED: {e}")


if __name__ == "__main__":
    main()
