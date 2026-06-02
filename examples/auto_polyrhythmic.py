"""Polyrhythmic mashup test (v6) — drum-locked tempo for tempo-distant pairs.

The v5 arc surfaced a limitation: Levee (71.4 BPM) and Orion (150 BPM)
have a 2.1× tempo ratio that no single-tempo midpoint satisfies.
v3's auto_target_bpm picks 73.2 (midpoint after octave-matching);
v5's octave multiplier prevents catastrophic stretch but introduces
a polyrhythmic feel that doesn't lock cleanly because the ratio
(150 / 73.2 = 2.05) isn't exactly 2.

This example demonstrates a clean polyrhythmic mode that's already
achievable through the existing API — by setting target_bpm
EXPLICITLY to the drum's native BPM:

  target_bpm = drum_bpm                            (= 71.4 for Levee)
  riff_octave = ceil(riff_bpm / target_bpm)        (= 2 for Orion at 150)
  effective riff tempo in output = target_bpm * 2  (= 142.8 BPM perceived)

The drum plays at exactly native speed (zero stretch). The riff is
slightly stretched (150 → 142.8, ~5% slowdown). The result is an
EXACT 2:1 polyrhythm where every drum beat has exactly 2 riff beats
on top, no drift over the whole mashup.

For the Levee × Orion case Carver flagged as "not sequenced right,"
this should be the better behavior: the polyrhythm locks at every
drum beat instead of drifting through.

Run from any cwd:
    python -m examples.auto_polyrhythmic
"""
from __future__ import annotations

import logging
from pathlib import Path

from stemflow.auto import compose_section_mashup


HOME   = Path.home()
CORPUS = HOME / "Music/stemflow-corpus/sources"
OUT    = HOME / "Music/stemflow-corpus/mashups/auto"

STEMS_LEGACY = HOME / "Documents/Development/scholls_workspace/pipeline_output/stems_v2"
STEMS_NEW    = HOME / "Music/stemflow-corpus/stems/htdemucs_ft"


def legacy(prefix: str, stem: str) -> Path:
    return Path(f"{prefix}_({stem})_htdemucs_ft.wav")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    LEVEE_BASE = STEMS_LEGACY / "levee/Led Zeppelin - 08. When the Levee Breaks (Remaster) (Rem"

    result = compose_section_mashup(
        drum_track=CORPUS / "Led Zeppelin/When the Levee Breaks.mp3",
        riff_track=CORPUS / "Metallica/Orion.m4a",
        output_dir=OUT,
        output_name="Auto v6 polyrhythmic - Levee drums x Orion riff.wav",
        drum_extract_from=legacy(LEVEE_BASE, "Drums"),
        riff_extract_from=[
            STEMS_NEW / "Orion/other.wav",
            STEMS_NEW / "Orion/bass.wav",
        ],
        balance_layers=True,
        # The key change vs auto-discovery: target_bpm pinned to Levee's
        # native BPM. The riff_octave logic in compose_section_mashup
        # auto-detects that Orion at 150 is too fast for target 71.4 and
        # uses octave-2 (half-time) interpretation, giving the
        # 2:1 polyrhythm.
        target_bpm=71.4,
    )
    print()
    print(result.explain())


if __name__ == "__main__":
    main()
