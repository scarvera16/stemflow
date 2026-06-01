"""CLI entry point for stemflow.

The default invocation (no subcommand) runs the full source-to-mastered
pipeline. Subcommands provide narrower operations:

    stemflow score FILE_A FILE_B   pairwise mashability score
    stemflow index PATH [PATH...]  add files or directories to the corpus
    stemflow query [filters]       list corpus rows or find mashup matches
"""

import argparse
import json
import logging
from pathlib import Path

from .analyze import analyze_track, compute_features, detect_bpm, detect_key
from .config import AUDIO_EXTENSIONS, DEFAULT_TARGET_LUFS, load_structure
from .master import master
from .mashability import score_pair
from .mix import build_mix
from .process import clean_stem, stretch_to_bpm
from .separate import separate_stems

log = logging.getLogger(__name__)


# ── Default pipeline ─────────────────────────────────────────────────────────

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


# ── Subcommand handlers ──────────────────────────────────────────────────────

def cmd_score(args: argparse.Namespace) -> None:
    """Score mashability between two audio files."""
    path_a = Path(args.file_a).expanduser().resolve()
    path_b = Path(args.file_b).expanduser().resolve()

    feats_a = compute_features(path_a)
    feats_b = compute_features(path_b)
    bpm_a = detect_bpm(path_a, device=args.device).bpm
    bpm_b = detect_bpm(path_b, device=args.device).bpm

    weights = tuple(float(w) for w in args.weights.split(",")) if args.weights else (0.5, 0.3, 0.2)
    result = score_pair(
        feats_a, feats_b, bpm_a, bpm_b,
        weights=weights,
        search_transposes=not args.no_transpose,
    )

    if args.json:
        print(json.dumps({
            "file_a": str(path_a),
            "file_b": str(path_b),
            "bpm_a": bpm_a,
            "bpm_b": bpm_b,
            "total": result.total,
            "harmonic": result.harmonic,
            "rhythmic": result.rhythmic,
            "spectral": result.spectral,
            "best_transpose_semitones": result.best_transpose_semitones,
        }, indent=2))
        return

    print(f"\n  A: {path_a.name}  ({bpm_a:.1f} BPM)")
    print(f"  B: {path_b.name}  ({bpm_b:.1f} BPM)")
    print(f"\n  Total mashability : {result.total:.3f}")
    print(f"    harmonic        : {result.harmonic:.3f}  (best transpose {result.best_transpose_semitones:+d} st)")
    print(f"    rhythmic        : {result.rhythmic:.3f}")
    print(f"    spectral        : {result.spectral:.3f}")
    print(f"    weights         : {weights}")
    print()


def cmd_index(args: argparse.Namespace) -> None:
    """Index files or directories into the corpus database."""
    from .corpus import default_db_path, index_directory, index_track

    db_path = Path(args.db).expanduser().resolve() if args.db else default_db_path()
    total = 0

    for raw in args.paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            n = index_directory(
                p, db_path=db_path,
                recursive=args.recursive,
                force=args.force,
                device=args.device,
            )
            print(f"  indexed {n} files from {p}")
            total += n
        elif p.is_file():
            row = index_track(p, db_path=db_path, force=args.force, device=args.device)
            if row is not None:
                print(f"  indexed {p}")
                total += 1
            else:
                print(f"  skipped (unreadable) {p}")
        else:
            print(f"  skipped (not found) {p}")

    print(f"\n  total: {total} tracks now in {db_path}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query the corpus or find mashup candidates for a target file."""
    from .corpus import default_db_path, find_mashups, query

    db_path = Path(args.db).expanduser().resolve() if args.db else default_db_path()

    if args.mash_with:
        target = Path(args.mash_with).expanduser().resolve()
        results = find_mashups(
            target, db_path=db_path,
            top=args.top,
            search_transposes=not args.no_transpose,
        )
        print(f"\n  Mashup candidates against {target.name} (top {args.top}):\n")
        if not results:
            print("  (corpus empty or no matches)")
            return
        for score, track in results:
            print(
                f"  {score.total:.3f}  {Path(track['file_path']).name}  "
                f"(h={score.harmonic:.2f} {score.best_transpose_semitones:+d}st, "
                f"r={score.rhythmic:.2f}, s={score.spectral:.2f})"
            )
        print()
        return

    rows = query(
        db_path=db_path,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        key=args.key,
        limit=args.top,
    )
    print(f"\n  {len(rows)} track(s) in {db_path}\n")
    for r in rows:
        bpm = f"{r['bpm']:.1f}" if r["bpm"] else "    "
        key = r["key"] or "       "
        print(f"  {bpm:>5}  {key:<10}  {Path(r['file_path']).name}")
    print()


# ── argparse setup ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stemflow",
        description="Audio stemflow production pipeline and mashability tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pipeline (default subcommand, runs separate/analyze/clean/stretch):
  stemflow --input-dir ./tracks --output-dir ./output --target-bpm 120
  stemflow --input-dir ./tracks --output-dir ./output --target-bpm 120 --structure structure.json

  # Score mashability between two files:
  stemflow score a.wav b.wav
  stemflow score a.wav b.wav --weights 0.6,0.2,0.2 --json

  # Index a music library into the corpus:
  stemflow index ~/Music/sources --recursive
  stemflow index ~/Music/sources --reindex   # force re-analyze

  # Query the corpus:
  stemflow query --bpm-min 100 --bpm-max 130 --top 25
  stemflow query --mash-with seed.wav --top 10
        """,
    )

    # Default pipeline args (used when no subcommand is given).
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

    # Subcommands.
    subparsers = parser.add_subparsers(dest="cmd", metavar="SUBCOMMAND")

    # score
    score = subparsers.add_parser(
        "score", help="Score mashability between two audio files")
    score.add_argument("file_a", help="First audio file")
    score.add_argument("file_b", help="Second audio file")
    score.add_argument("--weights", default=None,
                       help="Comma-separated weights H,R,S (default 0.5,0.3,0.2)")
    score.add_argument("--no-transpose", action="store_true",
                       help="Skip the semitone-shift search for harmonic")
    score.add_argument("--json", action="store_true",
                       help="Print machine-readable JSON instead of formatted output")

    # index
    index = subparsers.add_parser(
        "index", help="Index files or directories into the corpus DB")
    index.add_argument("paths", nargs="+",
                       help="One or more files or directories")
    index.add_argument("--db", default=None,
                       help="Override the corpus DB path (default: ~/.stemflow/corpus.db)")
    index.add_argument("--reindex", action="store_true", dest="force",
                       help="Re-analyze files even if their mtime is unchanged")
    index.add_argument("--no-recursive", action="store_false", dest="recursive",
                       help="When indexing a directory, do not descend into subdirectories")

    # query
    query_p = subparsers.add_parser(
        "query", help="Query the corpus or find mashup candidates")
    query_p.add_argument("--db", default=None,
                         help="Override the corpus DB path")
    query_p.add_argument("--bpm-min", type=float, default=None,
                         help="Filter: minimum BPM")
    query_p.add_argument("--bpm-max", type=float, default=None,
                         help="Filter: maximum BPM")
    query_p.add_argument("--key", default=None,
                         help="Filter: exact key match (e.g., 'F# minor')")
    query_p.add_argument("--mash-with", default=None,
                         help="Score every corpus track against this file and return top matches")
    query_p.add_argument("--no-transpose", action="store_true",
                         help="With --mash-with, skip the semitone-shift search")
    query_p.add_argument("--top", type=int, default=25,
                         help="Maximum number of rows to return (default 25)")

    return parser


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "score":
        return cmd_score(args)
    if args.cmd == "index":
        return cmd_index(args)
    if args.cmd == "query":
        return cmd_query(args)

    # No subcommand: fall through to the pipeline.
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
