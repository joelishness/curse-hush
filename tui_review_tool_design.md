# Design Doc: Terminal Review Tool for Censor Candidates (Rev. 2)

Status: **vision / groundwork only** — not implemented. Rev. 2 after reading the actual `profanity-hush` source (`hush.sh`, `pipeline.py`, `utils.py`, `steps/review.py`, `steps/mute.py`, `config/`, `autocensor-design.md`). Every mechanism below now cites the real function/file/field it hooks into rather than inventing a parallel one.

## Changes from Rev. 1

- Reviewing candidates is opt-in, one/a-few/all — never a forced full pass.
- Dropped the `keep_correction_artifacts` open question — confirmed retained, no longer a dependency to flag.
- Transcript words are now positioned by real timestamp under the waveform (not left-to-right), and an SRT row is added when a subtitle file is supplied.
- "Position" and "Span" are just the terminology now, no longer under consideration.
- **Biggest change:** dropped the Audacity-style draggable edit-mode entirely. Mute boundaries are always *derived*, never hand-dragged (see §5).
- Position is the only required input; Span has a default. Left/Right nudge Position, Up/Down adjust Span.
- The standalone-script and hush.sh-flag launch paths are now fully specified, including how job lookup works from just a directory, and — the thing that was genuinely unresolved in Rev. 1 — why the tool never needs to know the output directory at all (§7).
- Dropped the independently-zoomable timeline pane; the ruler is just a strip attached to the waveform that re-centers on Position.

---

## 1. Problem, goals, non-goals

Verifying a flagged (or missed) candidate today means cross-referencing `matches.json`/`censor_log.json`/`transcript.json` by hand, converting seconds to timestamps mentally, or opening the actual media in a player. This tool lets you type a rough position and see waveform + transcript + (optionally) subtitle text together, for as many or as few candidates as you actually want to look at.

**Goals**
- Jump to a timestamp, see waveform + time-aligned transcript instantly.
- See existing candidates (`matches.json`) overlaid on the waveform.
- Reject a false positive or stage a missed word/phrase, for one candidate, a handful, or the whole list — your call every time, never a forced walkthrough.
- Produce output that plugs directly into the correction workflow that already exists (`review.json`, `--skip-index`, `--add-interval`) — no parallel data model.

**Non-goals**
- Not a general audio editor. No draggable boundaries, no destructive editing.
- Not a video player.
- Not a mandatory review queue. A 400-candidate film and a 40-candidate film are handled identically — you look at what you want and stop whenever.
- Doesn't replace listening entirely for genuinely ambiguous cases.

## 2. How this fits what already exists

This isn't a new feature bolted on sideways — it's the "companion tool" `autocensor-design.md` §13.4 already names under **Remaining future work**: *"extending the interactive loop (or a companion tool) to play the audio snippet around each flagged word directly, so deciding doesn't require re-watching the film."* This design doc is that companion tool (audio *preview* is still deferred — see §10 — but the visual half of "deciding without re-watching" is the whole point here).

Everything this tool produces is exactly what `steps/review.py` already consumes:

- `review.json`: `{"overrides": [...]}`, sparse — a match with no override is implicitly approved. `{"action": "skip", "word_index": N, "text": ...}` to reject, `{"action": "add", "word_index": N|null, "text": ..., "start": ..., "end": ...}` to add. This tool writes this exact schema — it does not invent a `corrections.json` or any parallel format.
- Applying it is `steps/review.py:apply_corrections(job_dir, skip_indices, add_intervals, log)` — already implemented, already what `--skip-index`/`--add-interval` call. This tool calls the same function, or produces the equivalent `hush.sh` invocation for the user to run (see §7 for which, and why).
- Confidence display already matches `interactive.min_confidence_for_prompt`'s existing meaning (§13.6, already implemented in `review()`'s loop): high confidence = probably right, low confidence = worth a second look. This tool's "browse low-confidence words" action (§6) is the same idea applied outside the sequential Y/N/A/S/Q loop.

Nothing here touches `matches.json` or re-runs `find_matches()` — same rule Step 5 already follows (design doc §4): the flag phase is the single source of truth for what counts as a candidate.

## 3. Layout

```
┌─ Independence Day (1996) — 20260712_182333_independence-day-1996_f1b7b839 ───────────────┐
│ dialog.wav        09:03.0                                                      09:07.0    │
│  ▁▂▃▅▇█▇▅▃▂▁▁▂▄▆█▇▅▃▂▁▁▁▂▃▅▆▇█▇▆▅▄▃▂▁▁▁▁▂▃▄▅▆▇██████▇▆▅▄▃▂▁▁▁▂▃▅▆▇█▇▅▃▂▁▁▂▃▄▅▆▇█▇▅▃▂▁    │
│  ASR   David!  What the [hell's] the  point  of having a       beeper                    │
│  SRT   ─────── "What's the point of having a beeper if you're not gonna turn it on?" ──  │
├───────────────────────────────────────────────────────────────────────────────────────── ┤
│ Position: 9:03        Span: 4s        [f] false-positive list   [m] missed-word list      │
└───────────────────────────────────────────────────────────────────────────────────────── ┘
```

- **Waveform** — rendered from `dialog.wav` (the already-separated stem, confirmed retained via `output.keep_correction_artifacts: true`, independent of `output.keep_intermediates` — `steps/mute.py`'s docstring and §13.4 are explicit about this, no further confirmation needed).
- **ASR row** — every `transcript.json` word whose span intersects the visible window, positioned at its *real* proportional column (word.start mapped to the same x-axis as the waveform above it — the same idea as an Audacity label track, just ASR words instead of manual labels), not evenly spaced. The word under Position is bracketed. A word that's a currently-flagged candidate is tinted; a word that's a coverage-gap-shaped near-miss (see §6) gets its own marker.
- **SRT row** — shown only if a subtitle file was supplied to the tool (independent of the pipeline's own Step 4/`align_srt.py`, which isn't implemented yet — see §6 for why this row doesn't wait on that). Rendered as a bar spanning each cue's own `[start, end]`, with the cue text, since a cue is usually several words wide rather than one.
- **Input row** — Position, Span, and single-key entry points into the two action lists (§6).

## 4. Position & Span

Both accept `utils.parse_timestamp()`'s formats already — `9:03`, `543`, `0:09:03.000` — verbatim, since that's the same parser `--add-interval` and the interactive loop's manual-entry fallback already use. Display uses `utils.fmt_timestamp()`, so anything shown on screen can be pasted straight into `--add-interval`/`--skip-index` workflows elsewhere with no reformatting.

**Only Position is required.** Span defaults (a few seconds — enough to read 5–10 surrounding words) and is never a mandatory field.

- **Left/Right** — nudge Position.
- **Up/Down** — widen/narrow Span.
- Typing an explicit number into either field always works, for a big jump or an exact value.

There's no separate zoomable timeline pane. The ruler is a thin strip glued to the top of the waveform, always at the waveform's own Span — it re-centers on Position as you move, but doesn't get its own independent zoom level to manage. One less control to think about, matching "fast and simple" over an Audacity-parity feature set.

## 5. Boundary derivation, not boundary editing

This is the main simplification from Rev. 1. **The user never drags a mute boundary.** Position/Span control only what's on screen; the actual `start`/`end` written to a `review.json` override is always computed by one of two rules, matching exactly how `steps/review.py`'s existing `_prompt_add()` already works:

1. **Anchored to a transcript word** — if the word/phrase you're flagging (a false positive to reject, or a miss you're adding) corresponds to an actual `transcript.json` word — whether it's an existing `matches.json` candidate, or a real word Whisper transcribed but the word list didn't catch (like `hell's` — see the flagging-threshold analysis) — the override uses that word's own `start`/`end` verbatim. No adjustment, because none is needed: that's already the correct span.
2. **Unanchored (a true dropout)** — if Position falls somewhere Whisper produced no word at all (the common case per the missing-dialogue investigation — 5 of your 6 original examples), there's nothing to anchor to. The tool proposes `[Position − d/2, Position + d/2]`, where `d` is this *job's own* mean matched-word duration — computed live from `censor_log.json`'s `end − start` across its entries (for the Independence Day job we've been using throughout, that's ≈0.26s) rather than a hardcoded constant, so it naturally adapts to each film's pacing. If `matches.json` is empty (nothing to average), it falls back to a fixed default (e.g. 0.3s) and just leans on Span/eyeballing.

Either way, the value the tool writes into an `add` override is **unpadded** — exactly like every `add` override already is today. `steps/mute.py`'s `_resolve_intervals()` applies `censoring.padding_ms` on top uniformly for matched and added entries alike; the tool doesn't need to know or duplicate that logic.

Net effect: the only numbers a person ever types are Position and Span. Confirming a flag is one keystroke; there's no modal "now drag the edges" state at all.

## 6. Actions

Two entry points, both usable as "browse a list" or "just tell me about the one at Position" — your choice each time, and stopping after reviewing one is always fine (see §1).

**Remove false positive**
- List mode: every `matches.json` candidate, with confidence and context (reusing the same context-window logic `steps/review.py`'s `_context_str()` already renders — "...N words before MATCH N words after..."), sorted however's useful (by time, or lowest-confidence-first). Pick one, stage a `skip`.
- Position mode: land on a candidate while browsing and reject it directly.

**Add missed word/phrase**
- List mode: every transcript word below a confidence cutoff (reusing `interactive.min_confidence_for_prompt`'s existing meaning, just as a browsing filter here rather than a sequential prompt gate) — a low-confidence word is worth a glance regardless of whether it happened to match the word list, since Whisper might have misheard actual profanity as something else entirely.
- Position mode: the manual waveform lookup from §5.
- **Future work, not this pass:** the missing-dialogue investigation's method — per-SRT-cue transcript-coverage checking — is exactly the mechanism for auto-*suggesting* candidates here when an SRT is supplied, rather than only displaying it for reference (§3's SRT row). Flagging this explicitly since you mentioned wanting it eventually; it reuses a method already validated on this exact film, not a new one.

## 7. Launch modes

Two ways in, per your spec, both landing in the same tool:

**1. Standalone script**
```
rehush.sh "/nas/media/arm/movies/Independence Day (1996)/"
rehush.sh --job-id f1b7b839bdcf
rehush.sh --job-dir ~/.local/share/profanity-hush/jobs/20260712_182333_independence-day-1996_f1b7b839
```
A sibling to `hush.sh`, same path-resolution helpers (`abs_path`/`resolve_path`), same `--jobs DIR` default.

Given a directory (your example), job lookup can't use `pipeline.py`'s `compute_job_id()` directly — that needs a specific file + its exact mtime, and a directory doesn't pin either down. Instead: scan `jobs_dir/*/job.json` for one whose recorded `input_path` (already stored there, directory-form, exactly as `job.json` already shows) matches — a sibling to `utils.find_job_dir()`, same "warn on unreadable job.json rather than silently skipping it" behavior, just matching on `input_path` instead of `job_id`. If more than one job matches (re-run history), default to the most recently completed one and list the rest. Given a specific file instead of a directory, or `--job-id`/`--job-dir` directly, this reverse lookup isn't needed at all — the existing `find_job_dir()` handles it as-is.

**2. `hush.sh` flag**
```
hush.sh --review "Independence Day (1996).1080p.hevc.mkv"
```
Forwarded to `pipeline.py` as a new mode flag: compute `compute_job_id()` normally, `find_job_dir()` normally, and if found, launch the tool instead of the step pipeline (refusing if no job exists yet, same as `--redo-step`'s existing behavior — nothing to review before Step 4b has run). `hush.sh`'s TTY allocation (`-it`) already covers `--interactive`/`--redo-review`; add `--review` to that same condition.

**The output-directory question, resolved:** the tool never needs one, because it never re-mutes/re-muxes anything itself. Its entire job is reading `transcript.json`/`matches.json`/`dialog.wav`/`censor_log.json` and writing `review.json` — all inside the job directory, all already covered by mounting just `jobs_dir`. `steps/review.py:apply_corrections(job_dir, ...)` itself only takes `job_dir` — confirmed from the actual signature, not assumed. Producing the actual corrected video is a separate, already-solved problem: the normal `--skip-index`/`--add-interval` correction workflow, which already knows how to resolve an output directory (defaulting to same-as-input, same as every other `hush.sh` run). So a review session ends by either printing the exact ready-to-run command:
```
hush.sh --skip-index 4133 --add-interval "hell's" 543.98 544.18 "/nas/media/arm/movies/Independence Day (1996)/Independence Day (1996).1080p.hevc.mkv"
```
(built entirely from what's already in `job.json` — `input_path` + `input_filename` — regardless of which launch mode found the job), or — if you'd rather not copy-paste — directly shelling out to run it. Either way, zero new mux/encode code, and zero need to ask "but what's the output dir" at review time.

One inherited caveat, not a new one this design introduces: `compute_job_id()` is path+mtime based, so that hand-off command only lands on the same job if the input file's path and mtime haven't changed since the job completed. If they have, `hush.sh` fails with its existing "job hasn't completed Step 4b's flag phase" error rather than silently doing something wrong — safe, just worth knowing.

## 8. Mount footprint

Given §7, the tool's own process needs only:
- `jobs_dir` (read/write) — everything it touches lives there.
- `config_dir` (read-only, optional) — only to display `censoring.padding_ms` as a preview and, for §6's low-confidence browsing, `interactive.min_confidence_for_prompt`'s configured value as a sensible starting cutoff.

No `/input`, no `/output`, no `/cache` (no models are ever invoked). Notably smaller than `hush.sh`'s own mount set — a nice side effect of the split in §7, not something separately engineered.

## 9. Rendering approach

- **Waveform**: precompute a min/max peak envelope (same idea as Audacity's own waveform thumbnail) at 2–3 zoom levels, cached per job so re-opening is instant. Map peaks to a small glyph ramp (`▁▂▃▄▅▆▇█`, optionally half-blocks for extra vertical resolution).
- **Timing caveat carried over from the missing-dialogue investigation**: WhisperX's forced-alignment can stretch a word's *end* timestamp to swallow a following silence when nothing else was recognized nearby. Treat a displayed word's span as approximate near a known dropout, not authoritative.
- **Stack**: Python, consistent with the rest of the codebase. [`textual`](https://textual.textualize.io/) for layout/keybindings (`curses` as a zero-dependency fallback); `soundfile`/numpy for the peak decimation. Lives as a new top-level `src/review_tui.py` (sibling to `pipeline.py`, not under `steps/` — it's a companion tool operating on a finished job, not one of the 1a–7 pipeline steps), importing `utils.fmt_timestamp`/`parse_timestamp`/`find_job_dir`/`cfg_get`/`unmark_step_done` and `steps.review.apply_corrections` directly rather than re-implementing any of them.

## 10. Roadmap

1. **Phase 0** — non-interactive: given a job + Position (Span defaulted), print one static rendering to stdout. Validates rendering with the least code.
2. **Phase 1** — persistent interactive session: live re-render on Position/Span changes, Left/Right/Up/Down.
3. **Phase 2** — candidate awareness: overlay `matches.json`, the two list-browse actions (§6), write `review.json` via `apply_corrections()` directly.
4. **Phase 3** — SRT row (§3) when a subtitle file is supplied; the auto-suggestion future-work item from §6 once wanted.
5. **Phase 4 (stretch, and still the one item explicitly named in §13.4's existing future-work list)** — inline snippet playback (`aplay`/`ffplay` on just the current window, still never leaving this program) and an automatic "low trust" queue for the anomalous-token hallucination-loop pattern found in the missing-dialogue investigation.

Phases dropped entirely from Rev. 1: the draggable edit-mode UI (§5 removed the need), and forced queue-walkthrough (§1/§6 made it opt-in from the start instead of a phase to add later).
