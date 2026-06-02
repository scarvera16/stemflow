"""Tests for the auto-composition section pickers.

The compose_section_mashup integration is exercised by the example
script and validated against real audio (manual listening). These
unit tests cover the section-picker heuristics, which are the
deterministic part of auto-composition.
"""

import pytest

from stemflow.auto import pick_drum_section, pick_riff_section
from stemflow.sections import SectionStats


def _stats(start, end, rms=0.1, centroid=3000, flatness=0.1,
           chroma=0.5, timbral=0.5, density=2.0):
    """Build a SectionStats with reasonable defaults."""
    return SectionStats(
        start_s=start, end_s=end, duration_s=end - start,
        rms_mean=rms, spectral_centroid_mean=centroid,
        spectral_flatness_mean=flatness, chroma_strength=chroma,
        timbral_variance=timbral, density=density,
    )


# ── pick_drum_section ────────────────────────────────────────────────────────

class TestPickDrumSection:
    def test_prefers_low_chroma(self):
        """Among long-enough sections, picks the one with lowest chroma."""
        sections = [
            _stats(0, 30, chroma=0.6),    # tonal section
            _stats(30, 60, chroma=0.2),   # drum-prominent section (low chroma)
            _stats(60, 90, chroma=0.5),   # tonal section
        ]
        pick = pick_drum_section(sections)
        assert pick is not None
        assert pick.start_s == 30  # the low-chroma section

    def test_breaks_ties_by_density(self):
        """When chroma is equal, prefers higher onset density."""
        sections = [
            _stats(0, 30, chroma=0.2, density=1.0),
            _stats(30, 60, chroma=0.2, density=4.0),  # busier
            _stats(60, 90, chroma=0.2, density=2.0),
        ]
        pick = pick_drum_section(sections)
        assert pick.start_s == 30

    def test_rejects_short_sections(self):
        """Sections shorter than min_duration are skipped."""
        sections = [
            _stats(0, 2, chroma=0.05),   # very low chroma but too short
            _stats(2, 20, chroma=0.3),   # acceptable
        ]
        pick = pick_drum_section(sections, min_duration=5.0)
        assert pick.start_s == 2

    def test_returns_none_if_nothing_passes(self):
        sections = [_stats(0, 2, chroma=0.05)]
        assert pick_drum_section(sections, min_duration=10.0) is None


# ── pick_riff_section ────────────────────────────────────────────────────────

class TestPickRiffSection:
    def test_prefers_highest_rms(self):
        sections = [
            _stats(0, 30, rms=0.10, chroma=0.5),
            _stats(30, 60, rms=0.25, chroma=0.5),   # loudest, tonal
            _stats(60, 90, rms=0.15, chroma=0.5),
        ]
        pick = pick_riff_section(sections)
        assert pick.start_s == 30

    def test_filters_short_sections(self):
        """Sections shorter than min_duration are skipped."""
        sections = [
            _stats(0, 10, rms=0.50, chroma=0.5),    # loud but too short
            _stats(10, 40, rms=0.20, chroma=0.5),
        ]
        pick = pick_riff_section(sections, min_duration=20.0)
        assert pick.start_s == 10  # 10 to 40 is 30s long
        # The 0-10 section is rejected as too short

    def test_filters_low_chroma(self):
        """Sections with too-low chroma (drum-only) are skipped."""
        sections = [
            _stats(0, 30, rms=0.50, chroma=0.05),   # very loud but purely drum
            _stats(30, 60, rms=0.20, chroma=0.4),   # quieter but tonal
        ]
        pick = pick_riff_section(sections, min_chroma=0.2)
        assert pick.start_s == 30

    def test_returns_none_if_nothing_passes(self):
        sections = [_stats(0, 10, rms=0.5, chroma=0.05)]
        assert pick_riff_section(sections, min_duration=20.0) is None
