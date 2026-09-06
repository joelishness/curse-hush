"""
profanity-hush — Step 3: transcription with word-level timestamps

Runs whichever transcription/alignment engines are configured
(alignment.engines.* in config.yaml) against each dialog stem, and
produces per-segment transcript JSON files with segment-local (0-based)
word timestamps. steps/merge.py consumes these and produces the
canonical, film-absolute-timestamped files this module's own per-segment
ones are named after.

── Engines, and how this differs from earlier versions of this module ───

Earlier versions of this pipeline picked a single "alignment.backend"
(one of a small, hardcoded set of numbered "stages") and cascaded down
through the others on failure. That's gone. Every engine in
utils.ALIGNMENT_ENGINE_NAMES ("whisperx", "mfa", "crisperwhisper") is now
fully independent and separately toggled via its own
alignment.engines.<name> block:

    enabled          -- does this engine run at all this job.
    debug_subtitle    -- export this engine's own, unedited transcript as
                        a per-engine comparison subtitle (steps/
                        transcript_srt.py).
    final             -- is this engine's transcript THE authoritative one
                        -- the one Step 4b flags words against and Step 5
                        mutes/beeps from. Exactly one enabled engine must
                        have this true (checked once, at startup, by
                        utils.validate_alignment_engines() -- this module
                        assumes it's already true by the time it runs).
    final_subtitle    -- only consulted on whichever engine is final:
                        whether to also export that transcript as the
                        plain, non-karaoke "final" subtitle.
    embed_subtitle    -- whether to mux this engine's own subtitle
                        output(s) into the delivered output video.

There is no cross-engine cascade any more: the authoritative transcript
for a segment is engine_words[final_engine] for that segment, full stop,
with exactly ONE documented exception below (MFA's own
fallback_to_whisperx) -- an engine that produces nothing for a segment
contributes nothing to transcript.json for that segment's span, even if
a different engine happens to be enabled and has data there. Multiple
engines running side by side is for comparison (debug_subtitle), not
redundancy.

── The one structural dependency: MFA needs WhisperX's own pass ─────────

MFA (steps/align_mfa.py) never performs its own speech recognition -- it
re-times WhisperX's own recognized text by searching for it across the
audio it's handed. So whenever alignment.engines.mfa.enabled is true,
WhisperX's own recognition (model.transcribe()) AND its own wav2vec2/CTC
alignment pass (whisperx.align()) both run internally too, REGARDLESS of
alignment.engines.whisperx.enabled -- there would be nothing for MFA to
re-time otherwise. What alignment.engines.whisperx.enabled actually
controls is narrower than "does the recognition run": it's "is that
recognition ALSO exposed" -- written to its own
transcript_whisperx_NN.json, eligible for debug_subtitle, eligible to be
final. A job with whisperx.enabled: false and mfa.enabled: true still
pays WhisperX's recognition+alignment cost every segment (there's no way
around that -- it's what MFA re-times), it just never writes
transcript_whisperx_NN.json or offers WhisperX's own result anywhere.
Logged once, plainly, at the start of a run where this applies (see
below), so it's never a silent surprise.

One consequence worth knowing: with the default config (crisperwhisper
only), this module never imports the whisperx package at all -- `import
whisperx` happens lazily, inside this function, only when
need_whisperx_pass is true. CrisperWhisper does its own audio loading
from a file path and has no dependency on whisperx's own load_audio()/
align() machinery.

── MFA's own fallback_to_whisperx, and why it's not a cascade ───────────

alignment.engines.mfa.fallback_to_whisperx (default true) is preserved
exactly as before, but it's scoped entirely to MFA's own per-segment
result, not a generic multi-engine mechanism: if align_with_mfa() raises
MFAError for a WHOLE segment (an environment problem every chunk in that
segment would hit identically -- see steps/align_mfa.py), this module
does two DIFFERENT things depending on what's being asked:

  - MFA's own transcript_mfa_NN.json for that segment stays genuinely
    empty/absent -- an honest, unmodified record of what MFA itself
    produced (nothing), matching every other per-engine debug transcript's
    own "gap where the engine didn't cover something" convention.
  - IF mfa is the final engine, the AUTHORITATIVE transcript.json for
    that segment's span uses WhisperX's own raw alignment instead (already
    computed as MFA's own required input, per the dependency above) --
    this is the one and only case in this module where a segment's
    authoritative words don't come directly from engine_words[final_engine].

This mirrors, at whole-segment granularity, the identical per-CHUNK
fallback steps/align_mfa.py always applies internally regardless of this
setting -- both exist because MFA structurally requires WhisperX's own
timing as an interpolation/fallback basis, not because engines cascade
into each other generically.

── Casing/punctuation policy (applies to every engine) ──────────────────

Word casing is preserved exactly as each engine produces it. Do NOT
lowercase. Original casing is required for case-sensitive (=) word list
entries, compared in steps/matching.py. WhisperX (and, by inheritance,
MFA, which never changes WhisperX's own text) capitalises proper nouns
and sentence-initial words; this is the signal used to distinguish e.g.
"Dick" (name) from "dick" (profanity). CrisperWhisper's own casing
conventions may differ (see steps/transcribe_crisperwhisper.py) -- this
is one of the real, if narrow, ways engines can disagree in kind, not
just in timing, when crisperwhisper is authoritative.

Punctuation attached to words (e.g. "shit,", "warning.") is preserved
here; stripping happens at match time in steps/matching.py.

── Unaligned words ───────────────────────────────────────────────────────

whisperx: some tokens can't be aligned to a real character boundary
  (rare with the wildcard-column handling recent whisperx versions use
  for unknown characters, but not eliminated by it). These appear with
  start/end/score set to null. Downstream steps skip null-timestamped
  words at mute time.

mfa: MFA's own dictionary can't place a token containing a character
  outside its G2P model's alphabet (numerals, foreign scripts, ...) at
  all, so those tokens are never sent to it in the first place (see
  steps/align_mfa.py's _sanitize_for_mfa()). Rather than appearing null
  the way whisperx's unalignable tokens do, these get an INTERPOLATED
  timestamp instead -- WhisperX's own relative timing between the
  nearest words MFA did confidently place, affine-warped to fit MFA's
  corrected span (see _interpolate_stage2_words() in steps/align_mfa.py).
  Word TEXT is unaffected either way -- always WhisperX's original token.

crisperwhisper: every word it returns gets score=None (see steps/
  transcribe_crisperwhisper.py); it has no separate "couldn't align this
  one" null-timestamp convention of its own in this pipeline's usage.

── Resume support ────────────────────────────────────────────────────────

If transcript_NN.json already exists for a segment it is skipped -- its
per-engine sibling files (transcript_<engine>_NN.json) are checked
directly against utils.ALIGNMENT_ENGINE_NAMES rather than assumed
present, since a segment can be "already done" for its authoritative
transcript while lacking one or more engines' own data (e.g. if
debug_subtitle was turned on for an engine after this segment was
originally transcribed).

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
    ALIGNMENT_ENGINE_NAMES as _ENGINES,
    cfg_get,
    fmt_duration,
    mark_step_done,
    read_job,
    step_logger,
    write_job,
)
from steps.align_mfa import align_with_mfa, MFAError
from steps.transcribe_crisperwhisper import load_crisperwhisper_model, transcribe_with_crisperwhisper


# Display labels for job.json's own "alignment_engines" block and for log
# lines -- purely cosmetic, never used to decide behaviour (that's always
# _ENGINES / each engine's own config toggles). Adding a new engine to
# _ENGINES (utils.ALIGNMENT_ENGINE_NAMES) without adding a label here
# still works -- _engine_label() below falls back to the raw name.
_ENGINE_LABELS = {
    "whisperx":       "WhisperX",
    "mfa":            "MFA",
    "crisperwhisper": "CrisperWhisper",
}


def _engine_label(name: str) -> str:
    return _ENGINE_LABELS.get(name, name)


def _engine_toggles(cfg: dict, name: str) -> dict:
    """The five settings common to every engine -- see this module's own
    docstring, and config.yaml's alignment.engines comment, for what each
    means."""
    return {
        "enabled":        bool(cfg_get(cfg, "alignment", "engines", name, "enabled")),
        "debug_subtitle": bool(cfg_get(cfg, "alignment", "engines", name, "debug_subtitle")),
        "final":          bool(cfg_get(cfg, "alignment", "engines", name, "final")),
        "final_subtitle": bool(cfg_get(cfg, "alignment", "engines", name, "final_subtitle")),
        "embed_subtitle": bool(cfg_get(cfg, "alignment", "engines", name, "embed_subtitle")),
    }


def transcribe(
    job_dir: Path,
    segments: list[tuple[Path, float]],   # (audio_stereo_NN.wav, start_offset_sec)
    stem_pairs: list[tuple[Path, Path]],  # (dialog_NN.wav, score_sfx_NN.wav)
    cfg: dict,
    log: Optional[logging.LoggerAdapter] = None,
) -> list[Path]:
    """
    Step 3: run every configured engine against each dialog stem (see
    module docstring).

    Returns a list of transcript_NN.json paths (one per segment, in
    order) -- the AUTHORITATIVE per-segment files; unchanged in shape
    from every earlier version of this pipeline. Each engine's own
    transcript_<engine>_NN.json (steps/merge.py, steps/transcript_srt.py)
    is a side effect, not part of this return value -- callers that want
    them look for job_dir / f"transcript_{name}.json" directly, once
    steps/merge.py has produced it, the same fixed-name discoverability
    transcript.json/dialog.wav/score_sfx.wav already have.

    segments   -- (audio_stereo_NN.wav, start_offset_sec) from Step 1c.
    stem_pairs -- (dialog_NN.wav, score_sfx_NN.wav) from Step 2. Only the
                  dialog stem (first element) is used here.
    """
    if log is None:
        log = step_logger("transcribe")

    # ── Resume check ──────────────────────────────────────────────────────────
    state = read_job(job_dir)
    if "3_transcribe" in state.get("steps_completed", []):
        log.info("Step 3 — ↩  already complete; loading transcript paths from job.json.")
        return _transcripts_from_state(job_dir, state)

    # ── Resolve engine config ────────────────────────────────────────────────
    # Assumed already validated (exactly one enabled engine has final:
    # true, debug_subtitle/final both imply enabled) by utils.
    # validate_alignment_engines(), called once at pipeline.py startup --
    # this module re-derives final_engine defensively below rather than
    # trusting that blindly, in case it's ever called some other way.
    toggles = {name: _engine_toggles(cfg, name) for name in _ENGINES}

    final_candidates = [name for name in _ENGINES if toggles[name]["final"]]
    if len(final_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one alignment.engines.*.final: true, found "
            f"{len(final_candidates)} ({final_candidates!r}). This should "
            "have been caught by utils.validate_alignment_engines() at "
            "startup -- see pipeline.py's main()."
        )
    final_engine = final_candidates[0]

    whisperx_enabled       = toggles["whisperx"]["enabled"]
    mfa_enabled             = toggles["mfa"]["enabled"]
    crisperwhisper_enabled  = toggles["crisperwhisper"]["enabled"]
    need_whisperx_pass      = whisperx_enabled or mfa_enabled

    mfa_fallback_allowed = (
        bool(cfg_get(cfg, "alignment", "engines", "mfa", "fallback_to_whisperx"))
        if mfa_enabled else False
    )

    n = len(stem_pairs)
    log.info("Step 3 — transcription")
    log.info(
        "  engines enabled: %s  |  final (authoritative): %s  |  segments=%d",
        ", ".join(name for name in _ENGINES if toggles[name]["enabled"]) or "(none)",
        final_engine, n,
    )
    if mfa_enabled and not whisperx_enabled:
        log.info(
            "  alignment.engines.mfa.enabled is true and alignment.engines."
            "whisperx.enabled is false -- WhisperX's own recognition + "
            "alignment will still run every segment as MFA's required "
            "input (and interpolation/fallback basis), just not be "
            "written out or exposed as its own transcript/subtitle -- see "
            "this module's own docstring and config.yaml's comment on "
            "alignment.engines.whisperx for why."
        )

    # ── Config for whichever engines are actually in play ────────────────────
    wx_model_name   = cfg_get(cfg, "alignment", "engines", "whisperx", "model")
    wx_language     = cfg_get(cfg, "alignment", "engines", "whisperx", "language", allow_null=True)
    wx_batch_size   = int(cfg_get(cfg, "alignment", "engines", "whisperx", "batch_size"))
    wx_beam_size    = int(cfg_get(cfg, "alignment", "engines", "whisperx", "beam_size"))
    wx_device       = cfg_get(cfg, "alignment", "engines", "whisperx", "device")
    wx_compute_type = cfg_get(cfg, "alignment", "engines", "whisperx", "compute_type")

    cw_language = cfg_get(cfg, "alignment", "engines", "crisperwhisper", "language", allow_null=True)

    # ── Load models (only what's actually needed) ────────────────────────────
    whisperx = None
    wx_model = None
    if need_whisperx_pass:
        try:
            import whisperx  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "whisperx is not installed inside the container, but it's "
                "needed this run -- alignment.engines.whisperx.enabled "
                "and/or alignment.engines.mfa.enabled is true (MFA re-times "
                "WhisperX's own recognized text; see this module's own "
                "docstring). Ensure the Dockerfile pip-installs whisperx."
            ) from exc

        log.info(
            "  Loading Whisper model '%s'  (language=%s batch_size=%d "
            "beam_size=%d device=%s compute_type=%s) ...",
            wx_model_name, wx_language or "auto", wx_batch_size, wx_beam_size,
            wx_device, wx_compute_type,
        )
        t_load = time.monotonic()
        wx_model = whisperx.load_model(
            wx_model_name,
            wx_device,
            compute_type=wx_compute_type,
            language=wx_language,
            asr_options={"beam_size": wx_beam_size},
            # Silero VAD: no HuggingFace token required.
            vad_method="silero",
        )
        log.info("  ✓  Model loaded in %.1f s.", time.monotonic() - t_load)

    cw_model = None
    if crisperwhisper_enabled:
        cw_model = load_crisperwhisper_model(cfg, log)

    # ── Alignment model cache (per language, whisperx only) ──────────────────
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
            engines_with_data = [
                name for name in _ENGINES
                if (job_dir / f"transcript_{name}_{seg_idx:02d}.json").exists()
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
                "mfa_fallback_reason": None,
                "engines_with_data": engines_with_data,
            })
            continue

        dur_sec = _seg_duration(state, seg_wav.name)
        log.info(
            "  [%d/%d] Transcribing %s  (%.0f s, global offset %.1f s) ...",
            seg_idx, n, dialog.name, dur_sec, start_offset,
        )

        t0 = time.monotonic()
        mfa_fallback_reason: Optional[str] = None
        engine_words: dict[str, list[dict]] = {}
        whisperx_words: list[dict] = []
        whisperx_lang = wx_language or "en"

        # ── WhisperX recognition + its own alignment (shared by whisperx
        #    and mfa -- see module docstring) ────────────────────────────────
        if need_whisperx_pass:
            audio = whisperx.load_audio(str(dialog))
            result = wx_model.transcribe(audio, batch_size=wx_batch_size, language=wx_language)
            whisperx_lang = result.get("language") or wx_language or "en"
            segs_out = result.get("segments", [])
            log.debug(
                "    Whisper pass: %d segments, detected language=%s",
                len(segs_out), whisperx_lang,
            )

            if not segs_out:
                log.warning(
                    "  [%d/%d] WhisperX found no speech in %s — no "
                    "whisperx/mfa data for this segment.",
                    seg_idx, n, dialog.name,
                )
            else:
                align_model, align_metadata, loaded_lang = _ensure_align_model(
                    align_model, align_metadata, loaded_lang, whisperx_lang,
                    wx_device, whisperx, log,
                )
                t_stage = time.monotonic()
                aligned = whisperx.align(
                    segs_out, align_model, align_metadata, audio, wx_device,
                    return_char_alignments=False,
                )
                whisperx_words = _collect_whisperx_words(aligned)
                log.debug(
                    "    WhisperX alignment: %d words in %.1fs for %s.",
                    len(whisperx_words), time.monotonic() - t_stage, dialog.name,
                )

                if whisperx_enabled:
                    engine_words["whisperx"] = whisperx_words

                if mfa_enabled:
                    try:
                        t_stage = time.monotonic()
                        mfa_words = align_with_mfa(dialog, dur_sec, segs_out, whisperx_words, cfg, log)
                        engine_words["mfa"] = mfa_words
                        log.debug(
                            "    MFA alignment: %d words in %.1fs for %s.",
                            len(mfa_words), time.monotonic() - t_stage, dialog.name,
                        )
                    except MFAError as exc:
                        # Only fatal when MFA was actually required for the
                        # authoritative result and config says not to
                        # degrade away from it.
                        if final_engine == "mfa" and not mfa_fallback_allowed:
                            raise RuntimeError(
                                f"MFA alignment failed for {dialog.name} and "
                                "alignment.engines.mfa.fallback_to_whisperx "
                                "is false (config.yaml) -- not falling back."
                            ) from exc
                        mfa_fallback_reason = str(exc)
                        log.warning(
                            "  [%d/%d] MFA alignment failed for %s%s.  "
                            "Reason: %s",
                            seg_idx, n, dialog.name,
                            " -- the authoritative transcript will use "
                            "WhisperX's own timing for this segment instead"
                            if final_engine == "mfa" else
                            " -- no MFA data for this segment's comparison "
                            "transcript",
                            exc,
                        )

            del audio
            gc.collect()

        # ── CrisperWhisper -- fully independent, runs regardless of
        #    whether WhisperX found anything (it never shares that text) ──────
        if crisperwhisper_enabled:
            try:
                t_stage = time.monotonic()
                cw_words = transcribe_with_crisperwhisper(cw_model, dialog, cw_language, cfg, log)
                engine_words["crisperwhisper"] = cw_words
                log.debug(
                    "    CrisperWhisper transcription: %d words in %.1fs for %s.",
                    len(cw_words), time.monotonic() - t_stage, dialog.name,
                )
            except Exception as exc:  # noqa: BLE001 -- never fatal, see below
                # Unlike MFA, never raises even when crisperwhisper IS the
                # final engine -- there is no other engine sharing its
                # text to fall back to (see module docstring), so a
                # failure here just means this segment's span is empty in
                # transcript.json, the same as a genuinely silent segment
                # already is today.
                log.warning(
                    "  [%d/%d] CrisperWhisper failed for %s%s.  Reason: %s",
                    seg_idx, n, dialog.name,
                    " -- the authoritative transcript has no data for "
                    "this segment" if final_engine == "crisperwhisper" else
                    " -- no CrisperWhisper data for this segment's "
                    "comparison transcript",
                    exc,
                )

        elapsed = time.monotonic() - t0

        # ── Resolve the authoritative words for this segment ─────────────────
        # No generic cascade -- engine_words[final_engine], full stop,
        # with exactly one documented exception (see module docstring):
        # a whole-segment MFA failure degrades the AUTHORITATIVE result
        # to WhisperX's own timing when mfa is final and fallback is
        # allowed, while MFA's own transcript_mfa_NN.json (below) stays
        # honestly empty for this segment either way.
        if final_engine in engine_words:
            words = engine_words[final_engine]
        elif final_engine == "mfa" and mfa_fallback_reason is not None and mfa_fallback_allowed:
            words = whisperx_words
        else:
            words = []

        lang_by_engine = {
            "whisperx":       whisperx_lang,
            "mfa":            whisperx_lang,
            "crisperwhisper": cw_language or "en",
        }
        final_lang = lang_by_engine.get(final_engine, whisperx_lang)

        # ── Write JSON: authoritative, then each enabled engine's own ────────
        t_path = _write_transcript_variant(
            job_dir, seg_idx, "", words, final_lang, start_offset,
        )
        for name, w in engine_words.items():
            if toggles[name]["enabled"]:
                _write_transcript_variant(
                    job_dir, seg_idx, f"_{name}", w,
                    lang_by_engine.get(name, final_lang), start_offset,
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
            "engines_with_data": sorted(engine_words.keys()),
        })

    # ── Cleanup models ────────────────────────────────────────────────────────
    if wx_model is not None:
        del wx_model
    if align_model is not None:
        del align_model, align_metadata
    if cw_model is not None:
        del cw_model
    gc.collect()

    # ── Persist metadata and mark done ────────────────────────────────────────
    state = read_job(job_dir)
    state["alignment_engines"] = [
        {
            "engine": name,
            "label":  _engine_label(name),
            **toggles[name],
            # How many of the n segments this job actually has THIS
            # engine's own data for -- checked live above rather than
            # assumed, so this is accurate whether debug_subtitle was on
            # for the whole job, part of it, or an engine's data only
            # exists because it was the final one. Directly what
            # steps/merge.py checks to decide whether transcript_<name>.
            # json gets produced at all, and what steps/transcript_srt.py
            # / steps/mux.py use for that engine's own SRT/track label.
            "segments_with_data": sum(
                1 for r in segment_results if name in r.get("engines_with_data", [])
            ),
        }
        for name in _ENGINES
    ]
    state["transcription"] = {
        "final_engine": final_engine,
        "segments":     segment_results,
        # Aggregate, not just per-segment detail -- so callers that only
        # care about "did anything notable happen" (pipeline.py's own
        # end-of-run summary, and via that, hush.sh's --batch log) can
        # check one int instead of scanning segment_results themselves.
        # Undercounts only for segments recovered via the skip-existing
        # path above, whose original fallback status isn't recoverable.
        # 0 whenever alignment.engines.mfa.enabled is false, since the
        # whole branch that could set mfa_fallback_reason never runs.
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
    Kept separate from the per-segment loop above on its own merits: a
    self-contained, independently testable unit rather than inline
    state-juggling. Returns (align_model, align_metadata, loaded_lang).
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
    whisperx.align()'s own return value.

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
    engine's own output (suffix=f"_{engine_name}", →
    transcript_{engine_name}_NN.json). Same {"language", "segment_index",
    "segment_start_offset", "words"} shape either way -- steps/merge.py
    applies the same offset-and-concatenate logic to any of these,
    regardless of which one it's assembling into its own canonical,
    film-absolute-timestamped file.

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
