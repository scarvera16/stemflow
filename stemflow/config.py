"""Shared constants, feature flags, and structure loading."""

import json
import logging
from pathlib import Path
from typing import TypedDict

log = logging.getLogger(__name__)

# ── Audio defaults ───────────────────────────────────────────────────────────

DEFAULT_SAMPLE_RATE: int = 44100
DEFAULT_TARGET_BPM: float = 120.0
DEFAULT_TARGET_LUFS: float = -14.0
DEFAULT_TRUE_PEAK_CEILING_DB: float = -1.0
DEFAULT_CROSSFADE_MS: int = 1500
AUDIO_EXTENSIONS: set = {".wav", ".mp3", ".flac", ".aiff", ".m4a"}

# ── Feature detection ────────────────────────────────────────────────────────

HAS_BEAT_THIS = False
try:
    from beat_this.inference import File2Beats as _  # noqa: F401
    HAS_BEAT_THIS = True
except (ImportError, Exception):
    pass

HAS_ESSENTIA = False
try:
    import essentia.standard as _es  # noqa: F401
    HAS_ESSENTIA = True
except (ImportError, Exception):
    pass

HAS_PEDALBOARD_STRETCH = False
try:
    from pedalboard import time_stretch as _ts  # noqa: F401
    HAS_PEDALBOARD_STRETCH = True
except (ImportError, Exception):
    pass

HAS_RUBBERBAND = False
try:
    import pyrubberband as _pyrb  # noqa: F401
    HAS_RUBBERBAND = True
except (ImportError, Exception):
    pass


# ── Structure file ───────────────────────────────────────────────────────────

class StructureEntry(TypedDict, total=False):
    label: str
    track: str
    stem: str
    start_ms: int
    end_ms: int
    vol_db: float


def load_structure(path: Path) -> dict:
    """
    Load a structure file (JSON or YAML).

    Expected format:
    {
        "total_seconds": 60,
        "crossfade_ms": 1500,
        "entries": [
            {"label": "...", "track": "...", "stem": "drums", "start_ms": 0, "end_ms": 18000, "vol_db": -2}
        ]
    }
    """
    path = Path(path)
    text = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text)
        except ImportError:
            raise ImportError("PyYAML required for .yaml structure files: pip install pyyaml")
    else:
        data = json.loads(text)

    if "entries" not in data:
        raise ValueError("Structure file must contain an 'entries' key")

    return data
