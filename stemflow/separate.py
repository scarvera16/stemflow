"""Stem separation using Demucs or audio-separator."""

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def separate_stems(
    input_file: Path,
    output_dir: Path,
    model: str = "htdemucs_6s",
) -> dict[str, Path]:
    """
    Separate audio into stems using Demucs.

    Args:
        input_file: Path to source audio file.
        output_dir: Base output directory. Stems go to output_dir/stems/<model>/<track>/.
        model: Demucs model name (htdemucs_6s, htdemucs_ft, etc.).

    Returns:
        Dict mapping stem name to file path, e.g. {"drums": Path(...), "bass": Path(...)}.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)

    log.info("Separating stems: %s (model=%s)", input_file.name, model)

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "--out", str(output_dir / "stems"),
        str(input_file),
    ]
    subprocess.run(cmd, check=True)

    stem_dir = output_dir / "stems" / model / input_file.stem
    stems = {f.stem: f for f in stem_dir.glob("*.wav")}

    log.info("  → %s", stem_dir)
    for name in stems:
        log.info("    %s", name)

    return stems
