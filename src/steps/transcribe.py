"""
profanity-hush — Step 3: transcription with word-level timestamps

Runs WhisperX on each dialog stem and produces per-segment transcript JSON
files (transcript_01.json, transcript_02.json, …) with segment-local
(0-based) word timestamps.  Steps/merge.py consumes these and produces the
canonical transcript.json with global timestamps.

Pipeline (run in sequence, per-segment):
  1. model.transcribe() — batched Whisper inference → recognized text +
     rough (WhisperX-internal) segment-level timestamps.
  2. Word-level alignment, backend selected by config.yaml's
     alignment.backend:
       - "mfa" (default) — Montreal Forced Aligner (steps/align_mfa.py), a
         GMM-HMM aligner that searches the *whole* segment's audio against
         the recognized text in one pass, independent of WhisperX's own
         ~30s decode-chunk boundaries. Default as of this version, after
         whisperx.align() was found to silently inherit a several-second
         timing error from step 1 when WhisperX's own segment timing is
         wrong -- see docs/timestamp-drift-investigation.md, which also
         has the validated before/after numbers, not just the original
         case. Falls back to whisperx.align() per-segment on MFA failure
         if alignment.mfa.fallback_to_whisperx is true (default); raises
         otherwise.
       - "whisperx" — wav2vec2/CTC alignment within the segment boundaries
         step 1 already committed to. Set this directly only to avoid the
         conda/MFA install entirely and accept the drift risk above.
       MFA fixes *timing* of words WhisperX already recognized. It does
       not fix WhisperX failing to recognize a stretch of dialogue as
       text at all in the first place -- that's a separate, still-open
       recall problem, not an alignment problem (see the same doc for the
       one specific case this traces down: dense phrase repetition
       confusing WhisperX's own decoder, independent of alignment
       backend or beam width).

The Whisper model (and the whisperx align model, if that backend is ever
used this run) are loaded once for the whole job, not per segment, to
avoid repeated multi-minute load times.

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
  If transcript_NN.json already exists for a segment it is skipped.
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

# align_mfa's own top-level imports are stdlib + utils only (praatio is
# imported lazily inside the function that needs it) -- so this import is
# safe and cheap even when alignment.backend == "whisperx" and MFA is never
# actually invoked. Matches this codebase's existing convention of
# unconditional top-level `from steps.X import ...` (see pipeline.py).


def transcribe(
    job_dir: Path,
    segments: list[tuple[Path, float]],   # (audio_stereo_NN.wav, start_offset_sec)
    stem_pairs: list[tuple[Path, Path]],  # (dialog_NN.wav, score_sfx_NN.wav)
    cfg: dict,
    log: Optional[logging.LoggerAdapter] = None,
) -> list[Path]:
    """
    Step 3: transcribe each dialog stem with WhisperX.

    Returns a list of transcript_NN.json paths (one per segment, in order).

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
    if align_backend not in ("whisperx", "mfa"):
        raise ValueError(
            f"alignment.backend must be 'whisperx' or 'mfa', got {align_backend!r}"
        )

    n = len(stem_pairs)
    log.info("Step 3 — WhisperX transcription")
    log.info(
        "  model=%s  language=%s  batch_size=%d  beam_size=%d"
        "  device=%s  compute_type=%s  segments=%d  align_backend=%s",
        model_name, language or "auto",
        batch_size, beam_size, device, compute_type, n, align_backend,
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
            # "already done" for its primary transcript while still
            # lacking comparison data, e.g. if alignment.dual_output was
            # turned on after this segment was originally transcribed (see
            # config.yaml's own note on that setting only applying going
            # forward). Reflecting the true, current state here rather
            # than a stale guess keeps this run's has_mfa_comparison /
            # has_whisperx_comparison aggregate below honest even when a
            # skipped segment is involved.
            has_mfa       = (job_dir / f"transcript_mfa_{seg_idx:02d}.json").exists()
            has_whisperx  = (job_dir / f"transcript_whisperx_{seg_idx:02d}.json").exists()
            try:
                existing = json.loads(t_path.read_text())
                segment_results.append({
                    "index":      seg_idx,
                    "transcript": t_path.name,
                    "word_count": len(existing.get("words", [])),
                    "skipped":    True,
                    # Not tracked for a skip-recovered segment -- whether
                    # *this* segment's original run fell back to whisperx
                    # isn't recoverable from transcript_NN.json alone, only
                    # from the aggregate write at the end of a run that
                    # completed this segment fresh (see below). Explicit
                    # None, not an omitted key, so every segment_results
                    # entry has the same shape either way.
                    "mfa_fallback_reason": None,
                    "has_mfa_comparison":      has_mfa,
                    "has_whisperx_comparison": has_whisperx,
                })
            except (OSError, json.JSONDecodeError):
                segment_results.append({
                    "index":      seg_idx,
                    "transcript": t_path.name,
                    "skipped":    True,
                    "mfa_fallback_reason": None,
                    "has_mfa_comparison":      has_mfa,
                    "has_whisperx_comparison": has_whisperx,
                })
            continue

        dur_sec = _seg_duration(state, seg_wav.name)
        log.info(
            "  [%d/%d] Transcribing %s  (%.0f s, global offset %.1f s) ...",
            seg_idx, n, dialog.name, dur_sec, start_offset,
        )

        t0 = time.monotonic()
        # Set inside the MFA except-branch below when this segment falls
        # back to whisperx.align() -- None otherwise. Carried into this
        # segment's own segment_results entry further down, and summed
        # across all segments into transcription.mfa_fallback_segments at
        # the end of this function -- the same "field present only when
        # notable" shape as encode.py's own fallback_reason, surfaced the
        # same way in pipeline.py's end-of-run summary.
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

        # ── Forced alignment ──────────────────────────────────────────────────
        if not segs_out:
            # No speech detected (silent or noise-only segment).  Write an
            # empty transcript rather than calling align() on an empty list.
            log.warning(
                "  [%d/%d] WhisperX found no speech in %s — writing empty transcript.",
                seg_idx, n, dialog.name,
            )
            words: list[dict] = []
        else:
            # use_whisperx_align starts true only for the default backend;
            # an MFA failure can also flip it on mid-loop (fallback), which
            # is why this is a mutable flag rather than a one-shot branch.
            use_whisperx_align = (align_backend == "whisperx")

            if align_backend == "mfa":
                try:
                    t_mfa = time.monotonic()
                    words = align_with_mfa(dialog, segs_out, cfg, log)
                    log.debug(
                        "    MFA alignment: %d words in %.1fs for %s.",
                        len(words), time.monotonic() - t_mfa, dialog.name,
                    )
                except MFAError as exc:
                    if not mfa_fallback_allowed:
                        raise RuntimeError(
                            f"MFA alignment failed for {dialog.name} and "
                            "alignment.mfa.fallback_to_whisperx is false "
                            "(config.yaml) -- not falling back."
                        ) from exc
                    log.warning(
                        "  [%d/%d] MFA alignment failed for %s -- falling back "
                        "to whisperx.align() for this segment.  Reason: %s",
                        seg_idx, n, dialog.name, exc,
                    )
                    use_whisperx_align = True
                    mfa_fallback_reason = str(exc)

            if use_whisperx_align:
                align_model, align_metadata, loaded_lang = _ensure_align_model(
                    align_model, align_metadata, loaded_lang, detected_lang,
                    device, whisperx, log,
                )
                aligned = whisperx.align(
                    segs_out,
                    align_model,
                    align_metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                words = _collect_whisperx_words(aligned)
            # else: words was already populated by align_with_mfa() above,
            # in the same {"word","start","end","score"} shape.

        # ── Comparison alignment (alignment.dual_output) ────────────────────────
        # Independent of, and never allowed to affect, `words` above -- see
        # config.yaml's own documentation of alignment.dual_output. Tracks
        # which backend actually produced words for THIS segment, whether
        # as the primary result just above or as the comparison pass here,
        # so the per-segment file written for each backend below is always
        # an honest reflection of what happened, not just "whichever was
        # configured as primary."
        mfa_words:      Optional[list[dict]] = None
        whisperx_words: Optional[list[dict]] = None
        if segs_out:
            if use_whisperx_align:
                whisperx_words = words
            else:
                mfa_words = words

            if dual_output:
                if use_whisperx_align:
                    # Primary was whisperx -- either configured directly, or
                    # MFA already failed and fell back for THIS segment. In
                    # the fallback case, retrying the exact same MFA call
                    # would just fail again (MFA's own failures here are
                    # deterministic, not flaky), so only attempt a fresh one
                    # when MFA was never tried at all for this segment.
                    if mfa_fallback_reason is None:
                        try:
                            t_cmp = time.monotonic()
                            mfa_words = align_with_mfa(dialog, segs_out, cfg, log)
                            log.debug(
                                "    MFA comparison alignment: %d words in "
                                "%.1fs for %s.",
                                len(mfa_words), time.monotonic() - t_cmp, dialog.name,
                            )
                        except MFAError as exc:
                            log.warning(
                                "  [%d/%d] MFA comparison alignment failed for "
                                "%s -- no MFA data for this segment in "
                                "transcript_mfa.json.  Reason: %s",
                                seg_idx, n, dialog.name, exc,
                            )
                    else:
                        log.debug(
                            "  [%d/%d] Skipping MFA comparison for %s -- MFA "
                            "already failed as primary for this segment (%s).",
                            seg_idx, n, dialog.name, mfa_fallback_reason,
                        )
                else:
                    # Primary was MFA and it succeeded -- still need a fresh
                    # whisperx.align() pass for the comparison; nothing
                    # above already computed it. Broad except (not just a
                    # specific whisperx exception type) deliberately: this
                    # is a best-effort comparison pass, and no failure mode
                    # of it should be allowed to take the segment down --
                    # KeyboardInterrupt is a BaseException, not caught here,
                    # so Ctrl-C still stops the run immediately regardless.
                    try:
                        t_cmp = time.monotonic()
                        align_model, align_metadata, loaded_lang = _ensure_align_model(
                            align_model, align_metadata, loaded_lang, detected_lang,
                            device, whisperx, log,
                        )
                        aligned_cmp = whisperx.align(
                            segs_out, align_model, align_metadata, audio, device,
                            return_char_alignments=False,
                        )
                        whisperx_words = _collect_whisperx_words(aligned_cmp)
                        log.debug(
                            "    WhisperX comparison alignment: %d words in "
                            "%.1fs for %s.",
                            len(whisperx_words), time.monotonic() - t_cmp, dialog.name,
                        )
                    except Exception as exc:
                        log.warning(
                            "  [%d/%d] WhisperX comparison alignment failed for "
                            "%s -- no WhisperX data for this segment in "
                            "transcript_whisperx.json.  Reason: %s",
                            seg_idx, n, dialog.name, exc,
                        )

        # Release the numpy audio array before the next segment loads its own.
        del audio
        gc.collect()

        elapsed = time.monotonic() - t0

        # ── Write JSON ────────────────────────────────────────────────────────
        transcript: dict = {
            "language":             detected_lang,
            "segment_index":        seg_idx,
            "segment_start_offset": start_offset,
            "words":                words,
        }
        t_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))

        # Backend-labeled copies (alignment.dual_output's comparison SRTs --
        # see steps/transcript_srt.py and steps/merge.py) -- written
        # whenever that backend actually produced words for this segment,
        # regardless of whether it did so as the primary result above or as
        # the comparison pass above. Same shape as transcript_NN.json;
        # merge.py concatenates whichever of these exist, per backend, into
        # transcript_mfa.json / transcript_whisperx.json. Absent (not an
        # empty file) for a segment where that backend was never attempted
        # (dual_output off) or was attempted and failed -- merge.py treats
        # a missing segment as a real gap in that backend's transcript, not
        # an error.
        mfa_path, whisperx_path = None, None
        if mfa_words is not None:
            mfa_path = job_dir / f"transcript_mfa_{seg_idx:02d}.json"
            mfa_path.write_text(json.dumps(
                {**transcript, "words": mfa_words}, indent=2, ensure_ascii=False,
            ))
        if whisperx_words is not None:
            whisperx_path = job_dir / f"transcript_whisperx_{seg_idx:02d}.json"
            whisperx_path.write_text(json.dumps(
                {**transcript, "words": whisperx_words}, indent=2, ensure_ascii=False,
            ))

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
            "has_mfa_comparison":      mfa_path is not None,
            "has_whisperx_comparison": whisperx_path is not None,
        })

    # ── Cleanup models ────────────────────────────────────────────────────────
    del wx_model
    if align_model is not None:
        del align_model, align_metadata
    gc.collect()

    # ── Persist metadata and mark done ────────────────────────────────────────
    state = read_job(job_dir)
    state["transcription"] = {
        "model":       model_name,
        "language":    language or "auto",
        "segments":    segment_results,
        "dual_output": dual_output,
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
        # How many of the n segments this job has an MFA-aligned /
        # WhisperX-aligned transcript for, out of n -- checked live above
        # rather than assumed, so this is accurate whether dual_output was
        # on for the whole job, part of it, or produced by
        # alignment.backend alone with dual_output off entirely. Directly
        # what steps/merge.py checks to decide whether transcript_mfa.json
        # / transcript_whisperx.json get produced at all.
        "segments_with_mfa_data":      sum(
            1 for s in segment_results if s.get("has_mfa_comparison")
        ),
        "segments_with_whisperx_data": sum(
            1 for s in segment_results if s.get("has_whisperx_comparison")
        ),
    }
    write_job(job_dir, state)
    mark_step_done(job_dir, "3_transcribe")

    total_words = sum(s.get("word_count", 0) for s in segment_results)
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
    Shared by the primary alignment.backend: whisperx path and the
    alignment.dual_output comparison path above, so the two can't drift
    into loading/reloading logic that behaves differently from each
    other. Returns (align_model, align_metadata, loaded_lang).
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
    whisperx.align()'s own return value. Shared by the primary path and
    the alignment.dual_output comparison path above so both interpret
    whisperx's output identically.

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


def _seg_duration(state: dict, seg_wav_name: str) -> float:
    """
    Return the audio duration of a segment from job.json metadata.

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
    Recover the ordered list of transcript paths from job.json.

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
