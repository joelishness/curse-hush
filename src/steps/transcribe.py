"""
profanity-hush — Step 3: transcription with word-level timestamps

Runs WhisperX on each dialog stem, then one or more numbered ALIGNMENT
STAGES against the recognized text, and produces per-segment transcript
JSON files with segment-local (0-based) word timestamps. Steps/merge.py
consumes these and produces the canonical, film-absolute-timestamped
files this module's own per-segment ones are named after.

Pipeline (run in sequence, per segment):
  1. model.transcribe() — batched Whisper inference → recognized text +
     rough (WhisperX-internal) segment-level timestamps. Always runs;
     this is the source of the TEXT every alignment stage below works
     from — no stage changes what was recognized, only when it occurs.
  2. One or more alignment stages, each producing its own word-level
     timing for the SAME recognized text — see _ALIGNMENT_STAGES below.

── Alignment stages ──────────────────────────────────────────────────────

  _ALIGNMENT_STAGES is this pipeline's registry of alignment methods, in
  a fixed order:

    1. WhisperX — wav2vec2/CTC alignment within the ~30s decode-chunk
       boundaries step 1 already committed to. ALWAYS runs, for every
       segment with any recognized text at all, regardless of
       alignment.backend or alignment.dual_output — see "Why stage 1
       always runs" below.
    2. MFA (steps/align_mfa.py) — a GMM-HMM aligner that searches the
       *whole* segment's audio against the recognized text in one pass,
       independent of stage 1's own chunk boundaries. Found to fix a
       multi-second timing error stage 1 alone silently inherits when
       WhisperX's own segment timing is wrong — see
       docs/timestamp-drift-investigation.md for the validated before/
       after numbers, not just the original case. Runs when it's the
       authoritative stage (alignment.backend: mfa) or purely for
       comparison (alignment.dual_output: true, alignment.backend:
       whisperx) — see "Which stages actually run" below.

  Deliberately never tied to a specific tool's name in any FILENAME this
  pipeline writes (transcript_1_NN.json, transcript_2_NN.json, ... — see
  "Per-segment output files" below) — only the NUMBER is structural.
  job.json's own "alignment_stages" block (written once, at the end of
  this function) records which tool a given number actually was on a
  given job; every other step (steps/merge.py, steps/transcript_srt.py,
  steps/mux.py) reads that rather than importing this registry, so a
  future stage 3 — a different aligner entirely — only ever requires
  changes in this one file: one more entry in _ALIGNMENT_STAGES, plus
  the actual code to run it, appended to the per-segment loop below.
  Nothing about job.json's shape, merge.py's merging, transcript_srt.py's
  SRT export, or mux.py's embedding needs to change to pick it up.

  Neither stage changes what text was recognized — both take Whisper's
  own output as given and only refine *when* each word occurs. Fixing
  what gets recognized in the first place (WhisperX's decoder skipping a
  stretch of dense, repetitive dialogue outright — see the same
  investigation doc) is a separate, still-open recall problem, not
  something any alignment stage can address.

── Which stages actually run ─────────────────────────────────────────────

  alignment.backend (config.yaml) selects the AUTHORITATIVE stage number
  — the one whose result becomes transcript_NN.json (this segment's
  contribution to the canonical, censoring-relevant transcript.json) —
  via the tool→number lookup below: "mfa" → 2, "whisperx" → 1.

  A stage runs when EITHER of these holds:
    - its number is <= the authoritative stage number (needed to resolve
      the authoritative result at all, with cascade-down — see below), or
    - alignment.dual_output is true (run every registered stage
      regardless, purely so steps/transcript_srt.py can offer a
      side-by-side comparison track for each — see that module and
      config.yaml's own documentation of this setting).

  Resolving the authoritative transcript_NN.json for a segment: walk
  DOWN from the authoritative stage number, use the first one that
  actually produced data for THIS segment. Exactly today's "MFA,
  falling back to WhisperX on failure" rule (alignment.mfa.
  fallback_to_whisperx) — generalized so it isn't hardcoded to exactly
  two named stages, and so a future stage 3 slots into the same cascade
  automatically, highest-registered-number first. Stage 1 always having
  already run (see below) is what makes this cascade unconditionally
  available rather than a reactive, only-on-failure fallback path.

  A stage that fails for a segment it wasn't required for (e.g. MFA
  purely for comparison, backend: whisperx) never raises — logged, and
  that segment just has no data for that stage's own transcript. A
  stage that fails where it WAS required for the authoritative result
  (MFA, backend: mfa) raises only if alignment.mfa.fallback_to_whisperx
  is false; otherwise it cascades to stage 1, same as always.

── Why stage 1 always runs ───────────────────────────────────────────────

  Stage 1 (WhisperX's own alignment) is the fallback source the
  authoritative resolution above depends on whenever alignment.backend
  names a later stage — so it was never truly "optional" even before
  this pipeline had a name for that concept, just reactively triggered
  (only on MFA failure) instead of run proactively. Running it
  unconditionally, for every segment, is what lets the resolution above
  be a plain cascade over already-computed results instead of a
  branch that decides whether to compute stage 1 at all partway through
  handling a segment. It also means transcript_1.json (steps/merge.py)
  is always available as a comparison track (steps/transcript_srt.py),
  not only when alignment.dual_output happens to be on.

  This is a real, if modest, cost change from an earlier version of this
  pipeline, worth knowing rather than glossing over: a job with
  alignment.backend: mfa and MFA succeeding for nearly every segment
  previously never ran whisperx.align() at all outside of an actual MFA
  failure. It now runs on every segment, unconditionally. WhisperX's own
  alignment is a single wav2vec2/CTC forward pass, not a search the way
  MFA's beam-search alignment is — meaningfully cheaper than MFA either
  way — so this adds a bounded, comparatively small amount of time per
  segment rather than approaching MFA's own cost.

── Per-segment output files ──────────────────────────────────────────────

  transcript_NN.json          — authoritative (see resolution above);
                                 unchanged in name and role from every
                                 earlier version of this pipeline.
  transcript_{stage}_NN.json  — one per stage that actually produced
                                 words for this segment (see "Which
                                 stages actually run" above) — e.g.
                                 transcript_1_01.json, transcript_2_01.json.
  All three (when all exist) share the same {"language",
  "segment_index", "segment_start_offset", "words"} shape — see
  _write_transcript_variant().

The Whisper model (and the whisperx align model, loaded for stage 1
every run now that it's unconditional) are loaded once for the whole
job, not per segment, to avoid repeated multi-minute load times.

Casing policy:
  Word casing is preserved exactly as WhisperX produces it.  Do NOT
  lowercase.  Original casing is required for case-sensitive (=) word list
  entries, compared in steps/matching.py.  WhisperX capitalises proper
  nouns and sentence-initial words; this is the signal used to distinguish
  e.g. "Dick" (name) from "dick" (profanity).

Punctuation policy:
  Punctuation attached to words (e.g. "shit,", "warning.") is preserved
  here; stripping happens at match time in steps/matching.py.

Unaligned words:
  Some tokens cannot be aligned (numerals, currency symbols, punctuation-
  only tokens).  These appear in the transcript with start/end/score set to
  null.  Downstream steps skip null-timestamped words at mute time.

VAD:
  Uses Silero VAD (vad_method="silero"), which is free and requires no
  HuggingFace token.  The default WhisperX VAD backend (pyannote) requires
  token-authenticated model downloads, which conflicts with the project's
  CPU-only, no-cloud-dependency constraint.  The dialog stems from Demucs
  are already relatively clean, so VAD quality is a secondary concern.

Resume support:
  If transcript_NN.json already exists for a segment it is skipped (its
  per-stage sibling files are checked directly rather than assumed
  present — see the per-segment resume block below).
  If '3_transcribe' is already marked done in job.json, the step returns
  immediately with paths recovered from job.json.

Marks '3_transcribe' done once all segments complete.
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Optional

from utils import (
    cfg_get,
    fmt_duration,
    mark_step_done,
    read_job,
    step_logger,
    write_job,
)
from steps.align_mfa import align_with_mfa, MFAError


# ── Alignment stage registry ─────────────────────────────────────────────────
# See this module's own docstring ("Alignment stages") for the full
# rationale. Adding a stage 3 means one more entry here, plus the actual
# code to run it in the per-segment loop below — nothing else in this
# pipeline needs to change to pick it up (see that same docstring section).
_ALIGNMENT_STAGES = (
    {"number": 1, "tool": "whisperx", "label": "WhisperX"},
    {"number": 2, "tool": "mfa",      "label": "MFA"},
)
_STAGE_BY_TOOL = {s["tool"]: s for s in _ALIGNMENT_STAGES}


def transcribe(
    job_dir: Path,
    segments: list[tuple[Path, float]],   # (audio_stereo_NN.wav, start_offset_sec)
    stem_pairs: list[tuple[Path, Path]],  # (dialog_NN.wav, score_sfx_NN.wav)
    cfg: dict,
    log: Optional[logging.LoggerAdapter] = None,
) -> list[Path]:
    """
    Step 3: transcribe each dialog stem with WhisperX, then run every
    applicable alignment stage against the result (see module docstring).

    Returns a list of transcript_NN.json paths (one per segment, in
    order) — the AUTHORITATIVE per-segment files; unchanged in shape from
    every earlier version of this pipeline. Each stage's own
    transcript_{stage}_NN.json (steps/merge.py, steps/transcript_srt.py)
    is a side effect, not part of this return value — callers that want
    them look for job_dir / f"transcript_{n}.json" directly, the same
    fixed-name discoverability transcript.json/dialog.wav/score_sfx.wav
    already have.

    segments   — (audio_stereo_NN.wav, start_offset_sec) from Step 1c.
                 Provides the segment start offsets stored in the JSON output
                 so that Step 3b can apply them as global-timestamp offsets.
    stem_pairs — (dialog_NN.wav, score_sfx_NN.wav) from Step 2.
                 Only the dialog stem (first element) is used here.
    """
    if log is None:
        log = step_logger("transcribe")

    # ── Resume check ──────────────────────────────────────────────────────────
    state = read_job(job_dir)
    if "3_transcribe" in state.get("steps_completed", []):
        log.info("Step 3 — ↩  already complete; loading transcript paths from job.json.")
        return _transcripts_from_state(job_dir, state)

    # ── Config ────────────────────────────────────────────────────────────────
    model_name   = cfg_get(cfg, "whisperx", "model")
    # allow_null=True: config.yaml documents `language: null` as meaning
    # "auto-detect" (distinct from the key being absent entirely, which is
    # a ConfigError like any other missing required setting). See
    # utils.cfg_get()'s docstring.
    language     = cfg_get(cfg, "whisperx", "language", allow_null=True)
    batch_size   = int(cfg_get(cfg, "whisperx", "batch_size"))
    beam_size    = int(cfg_get(cfg, "whisperx", "beam_size"))
    device       = cfg_get(cfg, "whisperx", "device")
    # compute_type: int8 for CPU (faster inference, lower RAM); float16 for
    # GPU -- see config.yaml's own comment for the full tradeoff. Read
    # directly from config.yaml rather than derived from `device`: if you
    # change device, remember to set compute_type to match (config.yaml
    # ships with int8 + cpu, its matching pair, out of the box).
    compute_type = cfg_get(cfg, "whisperx", "compute_type")

    align_backend        = cfg_get(cfg, "alignment", "backend")
    mfa_fallback_allowed = bool(cfg_get(cfg, "alignment", "mfa", "fallback_to_whisperx"))
    dual_output          = bool(cfg_get(cfg, "alignment", "dual_output"))
    if align_backend not in _STAGE_BY_TOOL:
        raise ValueError(
            f"alignment.backend must be one of {sorted(_STAGE_BY_TOOL)}, "
            f"got {align_backend!r}"
        )
    target_stage = _STAGE_BY_TOOL[align_backend]["number"]

    n = len(stem_pairs)
    log.info("Step 3 — WhisperX transcription")
    log.info(
        "  model=%s  language=%s  batch_size=%d  beam_size=%d"
        "  device=%s  compute_type=%s  segments=%d  align_backend=%s"
        "  (stage %d)  dual_output=%s",
        model_name, language or "auto",
        batch_size, beam_size, device, compute_type, n, align_backend,
        target_stage, dual_output,
    )

    # ── Import whisperx ───────────────────────────────────────────────────────
    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "whisperx is not installed inside the container.  "
            "Ensure the Dockerfile pip-installs whisperx."
        ) from exc

    # ── Load Whisper model (once for all segments) ────────────────────────────
    log.info("  Loading Whisper model '%s' ...", model_name)
    t_load = time.monotonic()
    wx_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
        asr_options={"beam_size": beam_size},
        # Silero VAD: no HuggingFace token required.  See module docstring.
        vad_method="silero",
    )
    log.info("  ✓  Model loaded in %.1f s.", time.monotonic() - t_load)

    # ── Alignment model cache (per language) ──────────────────────────────────
    # Reload only if detected language changes across segments (rare in practice
    # for a single film, but handle it gracefully).
    align_model:    object = None
    align_metadata: object = None
    loaded_lang:    str    = ""

    transcript_paths: list[Path] = []
    segment_results:  list[dict] = []

    for i, ((dialog, _score_sfx), (seg_wav, start_offset)) in enumerate(
        zip(stem_pairs, segments)
    ):
        seg_idx = i + 1   # 1-based; transcript files always use _NN suffix
        t_path  = job_dir / f"transcript_{seg_idx:02d}.json"

        # ── Per-segment resume ────────────────────────────────────────────────
        if t_path.exists():
            log.info(
                "  [%d/%d] ↩  %s already exists — skipping.",
                seg_idx, n, t_path.name,
            )
            transcript_paths.append(t_path)
            # Checked directly rather than assumed -- a segment can be
            # "already done" for its authoritative transcript while still
            # lacking one or more stages' own data, e.g. if
            # alignment.dual_output was turned on after this segment was
            # originally transcribed (see config.yaml's own note on that
            # setting only applying going forward). Reflecting the true,
            # current state here rather than a stale guess keeps this
            # run's per-stage aggregate below honest even when a skipped
            # segment is involved.
            stages_with_data = [
                s["number"] for s in _ALIGNMENT_STAGES
                if (job_dir / f"transcript_{s['number']}_{seg_idx:02d}.json").exists()
            ]
            try:
                existing = json.loads(t_path.read_text())
                word_count = len(existing.get("words", []))
            except (OSError, json.JSONDecodeError):
                word_count = None
            segment_results.append({
                "index":      seg_idx,
                "transcript": t_path.name,
                "word_count": word_count,
                "skipped":    True,
                # Not tracked for a skip-recovered segment -- whether
                # *this* segment's original run fell back away from the
                # authoritative stage isn't recoverable from
                # transcript_NN.json alone, only from the aggregate write
                # at the end of a run that completed this segment fresh
                # (see below). Explicit None, not an omitted key, so
                # every segment_results entry has the same shape either
                # way.
                "mfa_fallback_reason": None,
                "stages_with_data": stages_with_data,
            })
            continue

        dur_sec = _seg_duration(state, seg_wav.name)
        log.info(
            "  [%d/%d] Transcribing %s  (%.0f s, global offset %.1f s) ...",
            seg_idx, n, dialog.name, dur_sec, start_offset,
        )

        t0 = time.monotonic()
        # Set below if stage 2 (MFA) fails while it's the authoritative
        # stage -- None otherwise. Carried into this segment's own
        # segment_results entry further down, and summed across all
        # segments into transcription.mfa_fallback_segments at the end of
        # this function -- the same "field present only when notable"
        # shape as encode.py's own fallback_reason, surfaced the same way
        # in pipeline.py's end-of-run summary.
        mfa_fallback_reason: Optional[str] = None

        # ── Load audio ────────────────────────────────────────────────────────
        # whisperx.load_audio() handles stereo→mono and 44.1kHz→16kHz
        # conversion internally; the 44.1kHz PCM WAV from Demucs is fine as-is.
        audio = whisperx.load_audio(str(dialog))

        # ── Transcribe ────────────────────────────────────────────────────────
        result = wx_model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,
        )

        detected_lang = result.get("language") or language or "en"
        segs_out = result.get("segments", [])
        log.debug(
            "    Whisper pass: %d segments, detected language=%s",
            len(segs_out), detected_lang,
        )

        # ── Alignment stages ─────────────────────────────────────────────────
        # stage_words[N] holds stage N's own words for this segment, only
        # for a stage that actually ran and produced something -- see
        # "Which stages actually run" in the module docstring.
        stage_words: dict[int, list[dict]] = {}

        if not segs_out:
            # No speech detected (silent or noise-only segment). Nothing
            # for any stage to align -- write an empty authoritative
            # transcript below and skip every stage's own file entirely
            # (there's nothing to compare; every stage would trivially
            # agree "nothing here").
            log.warning(
                "  [%d/%d] WhisperX found no speech in %s — writing empty transcript.",
                seg_idx, n, dialog.name,
            )
        else:
            # Stage 1 (WhisperX) -- always runs; see "Why stage 1 always
            # runs" in the module docstring.
            align_model, align_metadata, loaded_lang = _ensure_align_model(
                align_model, align_metadata, loaded_lang, detected_lang,
                device, whisperx, log,
            )
            t_stage = time.monotonic()
            aligned = whisperx.align(
                segs_out, align_model, align_metadata, audio, device,
                return_char_alignments=False,
            )
            stage_words[1] = _collect_whisperx_words(aligned)
            log.debug(
                "    Stage 1 (WhisperX) alignment: %d words in %.1fs for %s.",
                len(stage_words[1]), time.monotonic() - t_stage, dialog.name,
            )

            # Stage 2 (MFA) -- runs when it's the authoritative stage
            # (needed to resolve transcript_NN.json below) or purely for
            # comparison (alignment.dual_output).
            if target_stage >= 2 or dual_output:
                try:
                    t_stage = time.monotonic()
                    stage_words[2] = align_with_mfa(dialog, segs_out, cfg, log)
                    log.debug(
                        "    Stage 2 (MFA) alignment: %d words in %.1fs for %s.",
                        len(stage_words[2]), time.monotonic() - t_stage, dialog.name,
                    )
                except MFAError as exc:
                    # Only fatal when MFA was actually required for the
                    # authoritative result and the config says not to
                    # cascade away from it -- a failure during a
                    # comparison-only attempt (target_stage < 2) is never
                    # fatal regardless of this setting, since nothing
                    # censoring-relevant depended on it succeeding.
                    if target_stage >= 2 and not mfa_fallback_allowed:
                        raise RuntimeError(
                            f"MFA alignment failed for {dialog.name} and "
                            "alignment.mfa.fallback_to_whisperx is false "
                            "(config.yaml) -- not falling back."
                        ) from exc
                    mfa_fallback_reason = str(exc)
                    log.warning(
                        "  [%d/%d] Stage 2 (MFA) alignment failed for %s%s.  "
                        "Reason: %s",
                        seg_idx, n, dialog.name,
                        " -- cascading to stage 1 for the authoritative result"
                        if target_stage >= 2 else " -- no stage 2 data for "
                        "this segment's comparison transcript",
                        exc,
                    )

        # Release the numpy audio array before the next segment loads its own.
        del audio
        gc.collect()

        elapsed = time.monotonic() - t0

        # ── Resolve the authoritative result for this segment ───────────────
        # Walk down from target_stage, use the first stage that actually
        # has data -- see "Which stages actually run" in the module
        # docstring for why this cascade is always available rather than
        # a reactive, only-on-failure fallback. [] (not an error) when
        # segs_out was empty to begin with -- stage_words is empty too in
        # that case, so the loop below simply never finds anything.
        words: list[dict] = []
        for stage in reversed(_ALIGNMENT_STAGES):
            if stage["number"] > target_stage:
                continue
            if stage["number"] in stage_words:
                words = stage_words[stage["number"]]
                break

        # ── Write JSON: authoritative, then each stage that produced data ───
        t_path = _write_transcript_variant(
            job_dir, seg_idx, "", words, detected_lang, start_offset,
        )
        for number, w in stage_words.items():
            _write_transcript_variant(
                job_dir, seg_idx, f"_{number}", w, detected_lang, start_offset,
            )

        log.info(
            "  [%d/%d] ✓  %s  words=%d  elapsed=%s  [%d/%d segments transcribed]",
            seg_idx, n, t_path.name, len(words),
            fmt_duration(elapsed), seg_idx, n,
        )

        transcript_paths.append(t_path)
        segment_results.append({
            "index":       seg_idx,
            "transcript":  t_path.name,
            "word_count":  len(words),
            "elapsed_sec": round(elapsed, 1),
            "mfa_fallback_reason": mfa_fallback_reason,
            "stages_with_data": sorted(stage_words.keys()),
        })

    # ── Cleanup models ────────────────────────────────────────────────────────
    del wx_model
    if align_model is not None:
        del align_model, align_metadata
    gc.collect()

    # ── Persist metadata and mark done ────────────────────────────────────────
    state = read_job(job_dir)
    state["alignment_stages"] = [
        {
            "number": s["number"],
            "tool":   s["tool"],
            "label":  s["label"],
            # How many of the n segments this job actually has this
            # stage's own data for -- checked live above rather than
            # assumed, so this is accurate whether dual_output was on
            # for the whole job, part of it, or this stage's data only
            # exists because it was the authoritative one. Directly what
            # steps/merge.py checks to decide whether transcript_{number}.json
            # gets produced at all, and what steps/transcript_srt.py /
            # steps/mux.py use for this stage's SRT/track label.
            "segments_with_data": sum(
                1 for r in segment_results if s["number"] in r.get("stages_with_data", [])
            ),
        }
        for s in _ALIGNMENT_STAGES
    ]
    state["transcription"] = {
        "model":         model_name,
        "language":      language or "auto",
        "segments":      segment_results,
        "align_backend": align_backend,
        "target_stage":  target_stage,
        "dual_output":   dual_output,
        # Aggregate, not just per-segment detail -- so callers that only
        # care about "did anything notable happen" (pipeline.py's own
        # end-of-run summary, and via that, hush.sh's --batch log) can
        # check one int instead of scanning segment_results themselves.
        # Undercounts only for segments recovered via the skip-existing
        # path above, whose original fallback status isn't recoverable --
        # see that branch's comment.
        "mfa_fallback_segments": sum(
            1 for s in segment_results if s.get("mfa_fallback_reason")
        ),
    }
    write_job(job_dir, state)
    mark_step_done(job_dir, "3_transcribe")

    total_words = sum(s.get("word_count") or 0 for s in segment_results)
    log.info("  ✓  All segments transcribed.  Total words: %d", total_words)
    return transcript_paths


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_align_model(
    align_model: object,
    align_metadata: object,
    loaded_lang: str,
    detected_lang: str,
    device: str,
    whisperx,
    log: logging.LoggerAdapter,
) -> tuple[object, object, str]:
    """
    Lazily (re)load whisperx's own alignment model for detected_lang,
    reusing the already-loaded one when the language hasn't changed.
    Stage 1 (see _ALIGNMENT_STAGES) is this function's only caller as of
    this version, but kept separate from the per-segment loop above on
    its own merits: a self-contained, independently testable unit rather
    than inline state-juggling. Returns
    (align_model, align_metadata, loaded_lang).
    """
    if align_model is None or loaded_lang != detected_lang:
        if align_model is not None:
            log.debug(
                "    Language changed %s→%s; reloading alignment model.",
                loaded_lang, detected_lang,
            )
            del align_model, align_metadata
            gc.collect()
        log.debug(
            "    Loading alignment model for language '%s' ...", detected_lang
        )
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=device,
        )
        loaded_lang = detected_lang
    return align_model, align_metadata, loaded_lang


def _collect_whisperx_words(aligned: dict) -> list[dict]:
    """
    Extract this pipeline's {"word","start","end","score"} shape from
    whisperx.align()'s own return value (stage 1 -- see
    _ALIGNMENT_STAGES).

    Timestamps are segment-local (0-based); Step 3b applies the global
    start_offset to produce film-absolute timestamps. Words that couldn't
    be aligned have start/end/score = None; included anyway so the full
    word count is preserved in the JSON.
    """
    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            word_text = w.get("word", "")
            if not word_text:
                continue   # skip empty tokens (defensive)
            words.append({
                "word":  word_text,          # original casing + punctuation
                "start": w.get("start"),     # None if alignment failed
                "end":   w.get("end"),
                "score": w.get("score"),
            })
    return words


def _write_transcript_variant(
    job_dir: Path,
    seg_idx: int,
    suffix: str,
    words: list[dict],
    detected_lang: str,
    start_offset: float,
) -> Path:
    """
    Write one transcript variant for one segment -- the single writer
    every call site in this module goes through, whether for the
    authoritative result (suffix="", → transcript_NN.json) or for one
    alignment stage's own output (suffix=f"_{stage_number}", →
    transcript_{stage_number}_NN.json — see _ALIGNMENT_STAGES). Same
    {"language", "segment_index", "segment_start_offset", "words"} shape
    either way -- steps/merge.py applies the same offset-and-concatenate
    logic to any of these, regardless of which one it's assembling into
    its own canonical, film-absolute-timestamped file.

    Returns the path written.
    """
    path = job_dir / f"transcript{suffix}_{seg_idx:02d}.json"
    path.write_text(json.dumps({
        "language":             detected_lang,
        "segment_index":        seg_idx,
        "segment_start_offset": start_offset,
        "words":                words,
    }, indent=2, ensure_ascii=False))
    return path


def _seg_duration(state: dict, seg_wav_name: str) -> float:
    """
    Segment duration in seconds, from job.json's own segments list
    (written by Step 1c) rather than re-probing the audio file directly.
    Avoids a second ffprobe call — the information is already on disk from
    Step 1c.  Returns 0.0 if the segment isn't found (should not occur in
    normal operation but handled gracefully).
    """
    segs      = state.get("segments", [])
    total_sec = float(state.get("total_duration_sec", 0.0))

    for j, seg in enumerate(segs):
        if seg.get("path") == seg_wav_name:
            if j + 1 < len(segs):
                return float(segs[j + 1]["start_sec"]) - float(seg["start_sec"])
            return total_sec - float(seg["start_sec"])

    # Single-segment passthrough: job.json has path="audio_stereo.wav" but
    # the lookup above should always find it.  Fall back to total duration.
    return total_sec if len(segs) == 1 else 0.0


def _transcripts_from_state(job_dir: Path, state: dict) -> list[Path]:
    """
    Recover the ordered list of (authoritative) transcript paths from
    job.json.

    Used on resume when '3_transcribe' is already marked complete.
    Raises RuntimeError if any listed file is missing.
    """
    paths: list[Path] = []
    for seg in state.get("transcription", {}).get("segments", []):
        p = job_dir / seg["transcript"]
        if not p.exists():
            raise RuntimeError(
                f"Step 3 is marked complete but transcript file is missing: {p}\n"
                "Delete the job directory and re-run from scratch."
            )
        paths.append(p)
    return paths
