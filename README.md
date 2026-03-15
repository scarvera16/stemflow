# stemflow

AI-powered audio deconstruction and reconstruction pipeline. Deconstructs songs into stems, analyzes their tempo and key, time-stretches everything to a unified BPM, and reassembles selected layers into a mixed and mastered track.

## What it does

```
Source tracks → Stem separation → BPM/key analysis → Cleanup → Time-stretch → Assembly → Mastering
```

1. **Separate** each song into isolated stems (drums, bass, vocals, other) using AI source separation
2. **Analyze** tempo with joint beat + downbeat detection, and estimate musical key
3. **Clean** each stem with per-instrument noise gates and EQ to reduce separation artifacts
4. **Time-stretch** all stems to a common target BPM without pitch shift
5. **Layer** stems onto a stereo mix bus according to a declarative timeline structure
6. **Master** with EQ, compression, limiting, and LUFS loudness normalization

## Stack

Best-in-class tools for each stage:

| Stage | Tool | Why |
|-------|------|-----|
| Stem separation | [Demucs](https://github.com/facebookresearch/demucs) / [audio-separator](https://github.com/karaokenerds/python-audio-separator) | htdemucs_ft for multi-stem, BS-RoFormer (SDR 12.97) for vocals |
| Beat detection | [beat-this!](https://github.com/CPJKU/beat_this) (ISMIR 2024) | SOTA joint beat + downbeat detection — enables bar-aligned transitions |
| Key detection | [Essentia](https://essentia.upf.edu/) KeyExtractor | edma profile, 70-80% accuracy (vs librosa's 55-65%) |
| Time-stretch | [Pedalboard](https://github.com/spotify/pedalboard) | Rubber Band engine (same as Ableton Warp), zero-copy, releases GIL |
| Stem cleanup | [Pedalboard](https://github.com/spotify/pedalboard) | Per-instrument noise gate + EQ chains |
| Mastering | Pedalboard + [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | EQ → compression → limiting → ITU-R BS.1770-4 LUFS normalization |
| Mixing | numpy float32 | No pydub — float32 end-to-end, equal-power crossfades |

## Install

```bash
# Core (required)
pip install numpy soundfile librosa pedalboard pyloudnorm

# Full stack (recommended)
pip install demucs pyrubberband essentia
pip install "beat-this @ git+https://github.com/CPJKU/beat_this.git"
pip install audio-separator onnxruntime

# System (macOS)
brew install ffmpeg rubberband

# Install as a package
pip install -e .
```

Requires Python 3.11+.

## Usage

### CLI

```bash
# Process tracks (separate → analyze → clean → stretch):
stemflow --input-dir ./tracks --output-dir ./output --target-bpm 120

# Full pipeline with assembly and mastering:
stemflow --input-dir ./tracks --output-dir ./output --target-bpm 120 --structure structure.json

# Options:
stemflow --help
```

### As a library

Each module is independently importable:

```python
from stemflow import separate_stems, detect_bpm, detect_key, clean_stem, stretch_to_bpm
from stemflow import build_mix, equal_power_fade, master, load_structure

# Analyze a track
result = detect_bpm("song.flac")
print(f"{result.bpm:.0f} BPM, {len(result.downbeats)} bar boundaries")

key = detect_key("song.flac")
print(f"Key: {key}")  # e.g. "F# minor"

# Clean and stretch a stem
cleaned = clean_stem("drums.wav", stem_type="drums", output_dir="./output")
stretched = stretch_to_bpm(cleaned, source_bpm=98, target_bpm=120, output_dir="./output")

# Master a mix
mastered = master("raw_mix.wav", output_dir="./output", target_lufs=-14.0)
```

### Structure file

Assembly is driven by a JSON file that maps stems to timeline positions:

```json
{
  "total_seconds": 60,
  "crossfade_ms": 1500,
  "entries": [
    {"label": "intro_guitar", "track": "song_a", "stem": "guitar", "start_ms": 0, "end_ms": 16000, "vol_db": -2},
    {"label": "main_drums",   "track": "song_b", "stem": "drums",  "start_ms": 14000, "end_ms": 45000, "vol_db": 0},
    {"label": "peak_synth",   "track": "song_c", "stem": "other",  "start_ms": 30000, "end_ms": 45000, "vol_db": -3}
  ]
}
```

The `track` field matches the filename stem of a source audio file in your input directory.

See [example_structure.json](example_structure.json) for a complete example.

## Design decisions

- **No pydub in the mix path** — pydub quantizes to 16-bit integers and uses integer saturation on overlay, introducing audible noise. All mixing is done with numpy float32 array addition.
- **No matchering for mastering** — matchering applies a single global EQ curve from a reference track, which fights multi-genre content. Manual Pedalboard chain + pyloudnorm is more appropriate.
- **200-400 Hz subtractive EQ before compression** — layering already-mastered sources accumulates low-mid energy. Without a mud cut, compression pumps on accumulated low-mid buildup.
- **Equal-power crossfades** — linear fades create a perceptual dip at the midpoint. Cos/sin curves maintain constant power.
- **beat-this! for bar alignment** — knowing where downbeats land (not just the BPM) enables bar-boundary-aligned transitions.

## Mastering chain

Two-stage chain optimized for multi-source mixes:

**Stage 1 — Pedalboard DSP:**
- 30 Hz high-pass (subsonic cleanup)
- -3 dB peak @ 280 Hz, Q=0.8 (low-mid mud cut)
- +1.5 dB low shelf @ 100 Hz (warmth)
- +1 dB peak @ 3 kHz (presence)
- +1 dB high shelf @ 10 kHz (air)
- Compressor: -18 dB threshold, 2.5:1 ratio, 10 ms attack, 100 ms release
- +3 dB makeup gain
- Brick-wall limiter @ -1.0 dBTP

**Stage 2 — pyloudnorm:**
- ITU-R BS.1770-4 integrated loudness normalization to -14 LUFS
- True peak protection

## Per-stem cleanup profiles

| Stem | Noise Gate | EQ |
|------|-----------|-----|
| Drums | -30 dB threshold, fast attack | HPF @ 40 Hz, -2 dB @ 800 Hz (cut guitar bleed) |
| Bass | -35 dB threshold | LPF @ 5 kHz, HPF @ 30 Hz |
| Guitar | -35 dB threshold | HPF @ 80 Hz, -1.5 dB @ 4 kHz (tame harsh artifacts) |
| Vocals | -32 dB threshold | HPF @ 100 Hz, LPF @ 12 kHz |

## License

AGPL-3.0-or-later
