"""CLI entry point for the stemflow pipeline."""

import argparse
import json
import logging
from pathlib import Path

from .analyze import detect_bpm, detect_key
from .config import AUDIO_EXTENSIONS, DEFAULT_TARGET_LUFS, load_structure
from .master import master
from .mix import build_mix
from .process import clean_stem, stretch_to_bpm
from .separate import separate_stems

log = logging.getLogger(__name__)


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    target_bpm: float,
    structure_file: Path | None = None,
    clean: bool = True,
    demucs_model: str = "htdemucs_6s",
    target_lufs: float = DEFAULT_TARGET_LUFS,
    device: str = "mps",
) -> Path | None:
    """
    Run the full stemflow pipeline.

    Steps:
        1. Stem separation (Demucs)
        2. BPM + beat/downbeat detection (beat-this! / librosa)
        3. Key detection (Essentia / librosa)
        3.5. Per-stem cleanup (Pedalboard noise gate + EQ)
        4. Time-stretch to target BPM (Pedalboard / pyrubberband)
        5. Assembly (float32 numpy mixer) — only if structure file provided
        6. Mastering (Pedalboard + pyloudnorm) — only if assembly ran

    Args:
        input_dir: Folder of source audio files.
        output_dir: Output folder for stems, analysis, stretched files, and mix.
        target_bpm: Unified BPM for all stems.
        structure_file: Path to JSON/YAML structure file. If None, skips assembly.
        clean: Whether to run per-stem cleanup.
        demucs_model: Demucs model name.
        target_lufs: Target loudness for mastering.
        device: Inference device for beat-this! ("mps", "cuda", "cpu").

    Returns:
        Path to the final mastered output, or None if no assembly was requested.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  stemflow")
    print(f"  Target BPM : {target_bpm}")
    print(f"  Input dir  : {input_path}")
    print(f"  Output dir : {output_path}")
    print(f"  Model      : {demucs_model}")
    print(f"  Cleanup    : {'on' if clean else 'off'}")
    print(f"  Structure  : {structure_file or 'none (process only)'}")
    print(f"{'=' * 60}")

    # Find source files
    source_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not source_files:
        print(f"\n[WARN] No audio files found in {input_dir}")
        return None

    analysis = {}
    stretched_stems = {}

    for audio_file in source_files:
        print(f"\n{'─' * 50}")
        print(f"  {audio_file.name}")
        print(f"{'─' * 50}")

        # Step 1: Separate
        stems = separate_stems(audio_file, output_path, model=demucs_model)

        # Step 2: BPM
        beat_result = detect_bpm(audio_file, device=device)

        # Step 3: Key
        key = detect_key(audio_file)

        analysis[audio_file.stem] = {
            "bpm": beat_result.bpm,
            "beats": beat_result.beats.tolist(),
            "downbeats": beat_result.downbeats.tolist(),
            "key": key,
        }

        # Step 3.5: Cleanup
        if clean:
            cleaned_stems = {}
            for stem_name, stem_path in stems.items():
                cleaned_stems[stem_name] = clean_stem(stem_path, stem_name, output_path)
            stems = cleaned_stems

        # Step 4: Time-stretch
        stretched_stems[audio_file.stem] = {}
        for stem_name, stem_path in stems.items():
            stretched_stems[audio_file.stem][stem_name] = stretch_to_bpm(
                stem_path, beat_result.bpm, target_bpm, output_path,
                track_name=audio_file.stem,
            )

    # Save analysis
    analysis_path = output_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"\n[INFO] Analysis → {analysis_path}")

    # Step 5 + 6: Assembly and mastering (if structure provided)
    final_path = None
    if structure_file:
        struct_data = load_structure(structure_file)
        entries = struct_data["entries"]
        total_seconds = struct_data.get("total_seconds")
        crossfade_ms = struct_data.get("crossfade_ms", 1500)

        raw_mix = build_mix(
            stretched_stems, entries, output_path,
            output_name="mix_raw.wav",
            total_seconds=total_seconds,
            crossfade_ms=crossfade_ms,
        )

        final_path = master(raw_mix, output_path, target_lufs=target_lufs, output_name="mix_mastered.wav")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Done.")
    print(f"  Analysis   : {analysis_path}")
    print(f"  Stems      : {output_path / 'stretched'}")
    if final_path:
        print(f"  Mastered   : {final_path}")
    else:
        print(f"  No structure file — stems processed, ready for assembly.")
        print(f"  Provide --structure structure.json to assemble and master.")
    print(f"{'=' * 60}\n")

    return final_path


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Audio stemflow production pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process tracks only (separate, analyze, clean, stretch):
  stemflow --input-dir ./tracks --output-dir ./output --target-bpm 120

  # Full pipeline with assembly and mastering:
  stemflow --input-dir ./tracks --output-dir ./output --target-bpm 120 --structure structure.json
        """,
    )
    parser.add_argument("--input-dir", default="./source_tracks",
                        help="Folder of source audio files (WAV, MP3, FLAC, M4A)")
    parser.add_argument("--output-dir", default="./pipeline_output",
                        help="Output folder for all pipeline artifacts")
    parser.add_argument("--target-bpm", type=float, default=120.0,
                        help="Unified BPM for all stems (default: 120)")
    parser.add_argument("--structure", type=str, default=None,
                        help="Path to JSON/YAML structure file for assembly")
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip per-stem cleanup (noise gate + EQ)")
    parser.add_argument("--model", default="htdemucs_6s",
                        help="Demucs model (htdemucs_6s, htdemucs_ft, etc.)")
    parser.add_argument("--target-lufs", type=float, default=DEFAULT_TARGET_LUFS,
                        help="Target loudness in LUFS (default: -14)")
    parser.add_argument("--device", default="mps",
                        help="Inference device for beat detection (mps, cuda, cpu)")
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_bpm=args.target_bpm,
        structure_file=Path(args.structure) if args.structure else None,
        clean=not args.no_clean,
        demucs_model=args.model,
        target_lufs=args.target_lufs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
