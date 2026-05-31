# Claude Code context for stemflow

stemflow is the audio engineering library for **The Librarian**, a personal music project at `github.com/scarvera16/the-librarian` (private). The Librarian's working thesis is **mashup-as-Invention**: when the listener's existing library cannot satisfy the current moment, novel content is produced by recombining stems from the listener's own corpus into personalized mashups, rather than generating music from scratch via diffusion models.

stemflow implements the audio-engineering half of that thesis. It takes source audio, separates stems, analyzes beats and key, applies cleanup and time-stretching, assembles a declarative timeline, and masters the result. The pipeline is the renderer. The Invention layer's inference logic (mashability scoring, candidate generation, listener-state inference) sits above stemflow and is not yet built.

## What you should know

- The library is intended to be called from multiple surfaces: a CLI (`stemflow.cli:main`), a Tauri desktop app (`github.com/scarvera16/stemflow-app`, which depends on this library via editable pip install), and eventually the Librarian's runtime. The Python API is the contract; any new feature should be exposed through the package API rather than through the CLI alone.

- The **structure file** (JSON or YAML, see `stemflow/example_structure.json`) is the seam between authoring and rendering. Any system that can produce a valid structure file can drive stemflow. The schema lives in `stemflow/config.py`. Existing producers (the desktop app's `buildStructureFile()` in `src/lib/stemflow.ts`, the example file, future Librarian inference code) all need to stay valid. Be cautious about breaking changes.

- **Audio is float32 end-to-end through the mix bus.** pydub is deliberately not in the mix path because it would force 16-bit integer saturation at every mixing operation. Keep new audio code in float32.

- **Crossfades are equal-power cos/sin, not linear.** Linear crossfades introduce a 6 dB dip at the center; equal-power preserves perceived loudness.

- The **mastering chain** in `master.py` is tuned for layered already-mastered source material: a 200-400 Hz subtractive EQ before compression addresses the low-mid buildup endemic to mashup-style overlays. The two-stage pattern (Pedalboard DSP followed by pyloudnorm BS.1770 normalization to -14 LUFS) is intentional; pyloudnorm runs second because it is the standard for streaming-target loudness.

- **Stem separation shells out to demucs as a subprocess.** The library does not import demucs directly. This keeps the library's import surface small and lets Demucs maintain its own dependency tree.

## Coding conventions

- Keep modules independently importable. The pipeline is a series of pure functions over numpy arrays and dataclasses; each step should remain usable in isolation.
- Prefer extending `TrackAnalysis` over adding parallel data structures when new track-level features get cached.
- New features that would belong in the Invention inference layer (a mashability scorer, candidate generation, listener-state inference) should live in their own modules so they can be imported by both the CLI and any future Librarian runtime.

## Where to read more

- `~/Projects/the-librarian/notes/the-mashup-engine.md` — the as-built documentation of stemflow + stemflow-app and the gap roadmap to the mashup thesis. Read this first if you need context fast.
- `~/Projects/the-librarian/notes/the-mashup-thesis.md` — the architectural thesis arguing for mashup-from-corpus over generation-from-scratch.
- `~/Projects/the-librarian/notes/the-soundtrack-vision.md` — the Librarian's vision statement.
- `~/Projects/the-librarian/glossary.md` — working vocabulary; relevant entries include mashup-as-Invention, mashability, musical congruity and contextual incongruity, predictive coding of music, frisson crack, extended-peak architecture, twin-peak architecture, inverted-U sweet spot.
