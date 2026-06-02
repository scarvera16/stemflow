"""Rebuild the Sad But Levee mashup using only stemflow's public API.

Where the May 2026 iteration scripts (v5 through v10) hard-coded the
timestamp trims and ratios inline, this script uses the new phrase
module — find_phrase_downbeat for true bar-1 detection, and
beat_aligned_stretch as the trim-and-stretch primitive. The output
should match v10's musical structure but with all the cross-
correlation phase-finding and beat-locked stretching done through
the library, not ad-hoc.

Run from any cwd:
    python -m examples.sad_but_levee_v11

Output:
    ~/Music/stemflow-corpus/mashups/regenerated/Sad But Levee v11.wav
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from beat_this.inference import File2Beats

from stemflow.master import master
from stemflow.phrase import (
    bar_period_from_onsets,
    beat_aligned_stretch,
    find_phrase_downbeat,
)


# ── Paths ────────────────────────────────────────────────────────────────────
HOME       = Path.home()
CORPUS     = HOME / "Music/stemflow-corpus/sources"
LEVEE_SRC  = CORPUS / "Led Zeppelin/When the Levee Breaks.mp3"
SBT_SRC    = CORPUS / "Metallica/Sad But True.mp3"

SW         = HOME / "Documents/Development/scholls_workspace/pipeline_output"
LEVEE_DRUMS = SW / "stems_v2/levee/Led Zeppelin - 08. When the Levee Breaks (Remaster) (Rem_(Drums)_htdemucs_ft.wav"
SBT_OTHER   = SW / "stems_v2/sad_but_true/Metallica - 02. Sad But True_(Other)_htdemucs_ft.wav"
SBT_BASS    = SW / "stems_v2/sad_but_true/Metallica - 02. Sad But True_(Bass)_htdemucs_ft.wav"

OUT_DIR    = HOME / "Music/stemflow-corpus/mashups/regenerated"


# ── Spec ─────────────────────────────────────────────────────────────────────
# 80 BPM mashup. 2 measures of Bonzo's drum intro, then 8 measures of
# Sad But True's primary riff layered on top of continuing drums.
TARGET_BPM       = 80.0
TARGET_BEAT_T    = 60.0 / TARGET_BPM     # 0.75 s
BONZO_TOTAL_BEATS = 40                    # 10 measures of drums
RIFF_BEATS        = 32                    # 8 measures of riff
BONZO_INTRO_LEN_S = 8 * TARGET_BEAT_T     # 6.0 s before riff enters


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    f2b = File2Beats(device="mps", dbn=False)

    # ── Stage 1: Locate Levee's first beat ───────────────────────────────────
    print("\n[1/5] Locating Bonzo's first beat")
    beats_levee, _ = f2b(str(LEVEE_SRC))
    beats_levee = np.asarray(beats_levee)
    print(f"  first beat at source {beats_levee[0]:.4f}s")

    # ── Stage 2: Locate the true SBT riff downbeat ───────────────────────────
    print("\n[2/5] Locating SBT primary-riff downbeat")
    beats_sbt, downbeats_sbt = f2b(str(SBT_SRC))
    beats_sbt = np.asarray(beats_sbt)
    downbeats_sbt = np.asarray(downbeats_sbt)
    sbt_audio, sr = sf.read(str(SBT_SRC), dtype="float32")

    # The bar period varies across the song; in the riff section
    # (post 25s) it's ~2.7s at ~89 BPM. Compute the local period
    # from beat-this's beats array: four consecutive beats inside
    # the riff section sum to one bar.
    riff_beat_idx = int(np.argmin(np.abs(beats_sbt - 30.0)))  # any beat in the riff
    local_bar_period = float(beats_sbt[riff_beat_idx + 4] - beats_sbt[riff_beat_idx])
    print(f"  local bar period (from beat-this 4-beat span at riff): {local_bar_period:.4f}s "
          f"({60/local_bar_period*4:.2f} BPM)")

    # Reference: pick a beat-this downbeat well into the riff section.
    # These are reliable per-bar boundaries even when downbeat *labeling*
    # is wrong (i.e., we don't trust which one is bar 1 of the riff
    # phrase, but each downbeat IS on a measure boundary).
    ref_db_idx = int(np.argmin(np.abs(downbeats_sbt - 33.0)))
    ref_time = float(downbeats_sbt[ref_db_idx])
    print(f"  reference downbeat (on-bar by construction): {ref_time:.4f}s")

    # Now cross-correlate to find the true phrase-1 downbeat near 24s.
    true_downbeat_s = find_phrase_downbeat(
        sbt_audio, sr,
        candidate_time=24.0,
        bar_period=local_bar_period,
        reference_time=ref_time,
        search_radius=3.0,
    )
    print(f"  cross-correlated true downbeat: {true_downbeat_s:.4f}s")

    # Snap to the nearest beat in beats_sbt to ensure beat-grid alignment
    sbt_start_idx = int(np.argmin(np.abs(beats_sbt - true_downbeat_s)))
    print(f"  snapped to beat index {sbt_start_idx} at {beats_sbt[sbt_start_idx]:.4f}s")

    # ── Stage 3: Stretch each layer using the library primitive ──────────────
    print("\n[3/5] Beat-aligned stretching")
    bonzo, sr = beat_aligned_stretch(
        LEVEE_DRUMS, beats_levee,
        start_beat_idx=0, n_beats=BONZO_TOTAL_BEATS,
        target_beat_period=TARGET_BEAT_T,
    )
    print(f"  bonzo: {len(bonzo)/sr:.4f}s (target {BONZO_TOTAL_BEATS * TARGET_BEAT_T:.4f}s)")

    guitar, _ = beat_aligned_stretch(
        SBT_OTHER, beats_sbt,
        start_beat_idx=sbt_start_idx, n_beats=RIFF_BEATS,
        target_beat_period=TARGET_BEAT_T,
    )
    print(f"  guitar: {len(guitar)/sr:.4f}s")

    bass, _ = beat_aligned_stretch(
        SBT_BASS, beats_sbt,
        start_beat_idx=sbt_start_idx, n_beats=RIFF_BEATS,
        target_beat_period=TARGET_BEAT_T,
    )
    print(f"  bass: {len(bass)/sr:.4f}s")

    # ── Stage 4: Mix ─────────────────────────────────────────────────────────
    print("\n[4/5] Mixing")
    total_samples = int(BONZO_TOTAL_BEATS * TARGET_BEAT_T * sr)
    mix_buf = np.zeros((total_samples, 2), dtype=np.float32)
    mix_buf[:len(bonzo)] += bonzo

    riff_layer = guitar * (10 ** (-3 / 20)) + bass * (10 ** (-3 / 20))
    fade_samp = int(0.05 * sr)
    fade = np.sin(np.linspace(0, np.pi / 2, fade_samp)).astype(np.float32)
    for ch in range(2):
        riff_layer[:fade_samp, ch] *= fade

    riff_start = int(BONZO_INTRO_LEN_S * sr)
    mix_buf[riff_start:riff_start + len(riff_layer)] += riff_layer

    peak = float(np.max(np.abs(mix_buf)))
    if peak > 0:
        mix_buf = mix_buf * (10 ** (-1 / 20) / peak)

    raw_path = OUT_DIR / "Sad But Levee v11 raw.wav"
    sf.write(str(raw_path), mix_buf, sr, subtype="FLOAT")

    # ── Stage 5: Master ──────────────────────────────────────────────────────
    print("\n[5/5] Mastering")
    mastered = master(raw_path, OUT_DIR, target_lufs=-14.0, output_name="Sad But Levee v11.wav")
    print(f"\n✓ {mastered}")


if __name__ == "__main__":
    main()
