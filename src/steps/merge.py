"""
profanity-hush — Step 3b: merge per-segment transcripts and audio stems

Consolidates per-segment outputs from Steps 2 and 3 into three canonical
files consumed by all downstream steps:

  transcript.json  — all words with film-absolute timestamps
  dialog.wav       — full-duration dialog stem (lossless PCM concat)
  score_sfx.wav    — full-duration score+SFX stem (lossless PCM concat)

Transcript merge:
  For each transcript_NN.json, adds segment_start_offset to every word's
  start and end, then concatenates the adjusted word lists in segment order.
  The output transcript.json has no segment_index or segment_start_offset
  at the top level — only the flat words array with global timestamps.

Alignment-stage transcript merge (alignment.dual_output — see config.yaml):
  Same offset-and-concatenate logic (_merge_transcript_source(), shared
  with the primary merge above), applied independently to whichever
  transcript_{N}_NN.json files steps/transcribe.py actually produced, for
  each stage number in job.json's own "alignment_stages" legend (also
  written by that step), into transcript_{N}.json — present only when at
  least one segment has that stage's data. Reacts to per-segment file
  existence rather than reading alignment.dual_output itself, so it's
  correct whether that setting was on for the whole job, part of it, or
  off entirely (in which case whichever stage matches alignment.backend
  still gets its own numbered file here — redundant with transcript.json
  in that case, but deliberately so; see steps/transcript_srt.py, the
  only consumer of these files, which only ever looks for them by
  stage-numbered name and never reads the generic transcript.json at
  all). A future stage 3 needs no changes here: it appears in
  alignment_stages the moment steps/transcribe.py starts writing one, and
  is merged the same way as stages 1 and 2 today. Never affects
  transcript.json itself or what Steps 4–7 do with it.

Audio stem merge (multi-segment only):
  Uses the ffmpeg concat demuxer with a temporary file list for reliable
  lossless PCM concatenation.  All input stems are 44.1 kHz stereo PCM
  (the same format throughout the pipeline), so stream-copy is valid.

Single-segment passthrough:
  dialog.wav and score_sfx.wav already exist with canonical names from
  Step 2 (separate.py creates them without a numeric suffix for single-
  segment jobs).  No audio concat is performed.  transcript_01.json is
  read, its single-segment words are adjusted (offset=0, so no numeric
  change), and transcript.json is written.  This step still runs so that
  all downstream steps can unconditionally depend on the canonical names.

Intermediate cleanup:
  - audio_stereo.wav and audio_stereo_NN.wav — always deleted here unless
    keep_intermediates (they are no longer needed; audio_raw.{ext} is the
    resume artifact for future per-channel reprocessing, §13.3)
  - dialog_NN.wav, score_sfx_NN.wav — deleted only for multi-segment runs,
    only if not keep_intermediates (canonical versions now exist)
  - transcript_NN.json, and every alignment stage's own
    transcript_{N}_NN.json (whichever numbers job.json's
    "alignment_stages" lists) — deleted under the same keep_intermediates
    condition as the files above (see "Correction" note below)
  See utils.keep_intermediate() — the single source of truth for this
  policy, shared with steps/mute.py, steps/recombine.py, and steps/mux.py
  so it can't drift between steps the way it could when each one
  re-implemented its own condition.

Correction (transcript_NN.json retention): an earlier version of this
module kept transcript_NN.json unconditionally, regardless of
keep_intermediates, on the theory that the correction workflow (§13.4)
needed it. It doesn't: apply_corrections() in steps/review.py edits
review.json against matches.json only, and never opens a
transcript_NN.json; the interactive review() phase's word-search only
ever receives the merged transcript.json (see pipeline.py). Their one
real purpose is letting Step 3 (transcribe.py) skip already-finished
segments if it's interrupted partway through a run -- fully served the
moment this merge succeeds -- so they now follow the same
keep_intermediates policy as the other per-segment intermediates above,
rather than being kept forever.

Marks '3b_merge' done.  Writes the canonical filenames into job.json's
"merge" block ("files": {"transcript", "dialog", "score_sfx"}, plus
"transcript_{N}" for each alignment stage number actually produced),
alongside the segment/word_count stats it already recorded.
Returns (transcript.json, dialog.wav, score_sfx.wav) as a 3-tuple --
unchanged in shape from every earlier version of this pipeline.
transcript_{N}.json files are NOT part of this return value; callers
(pipeline.py) discover them the same way they're discovered here --
checking for job_dir / f"transcript_{N}.json" directly -- since they're
always at those exact, fixed names when present at all, the same
discoverability transcript.json/dialog.wav/score_sfx.wav already have on
pipeline.py's own "Steps 1a-3b already complete" fast path.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from utils import (
    finalize_output,
    fmt_duration,
    fmt_size,
    keep_intermediate,
    mark_step_done,
    read_job,
    run_cmd,
    step_logger,
    tmp_output_path,
    verify_and_hash_before_publish,
    write_job,
)


def merge(
    job_dir: Path,
    segments: list[tuple[Path, float]],    # (audio_stereo_NN.wav, start_offset_sec)
    stem_pairs: list[tuple[Path, Path]],   # (dialog_NN.wav, score_sfx_NN.wav)
    transcript_paths: list[Path],           # transcript_NN.json from Step 3
    cfg: dict,
    log: Optional[logging.LoggerAdapter] = None,
) -> tuple[Path, Path, Path]:
    """
    Step 3b: merge per-segment outputs into canonical single files.

    Returns (transcript.json, dialog.wav, score_sfx.wav).
    These are the only audio and transcript inputs used by Steps 4–7.
    """
    if log is None:
        log = step_logger("merge")

    # ── Resume check ──────────────────────────────────────────────────────────
    state = read_job(job_dir)
    if "3b_merge" in state.get("steps_completed", []):
        log.info("Step 3b — ↩  already complete.")
        t_out = job_dir / "transcript.json"
        d_out = job_dir / "dialog.wav"
        s_out = job_dir / "score_sfx.wav"
        if not t_out.exists() or not d_out.exists() or not s_out.exists():
            raise RuntimeError(
                "Step 3b is marked complete but canonical output files are missing. "
                "Delete the job directory and re-run from scratch."
            )
        return t_out, d_out, s_out

    n    = len(segments)

    log.info("Step 3b — merging %d segment(s) into canonical outputs.", n)

    transcript_out = job_dir / "transcript.json"
    dialog_out     = job_dir / "dialog.wav"
    score_sfx_out  = job_dir / "score_sfx.wav"

    # ── 1. Merge transcripts ───────────────────────────────────────────────────
    total_words = _merge_transcript_source(transcript_paths, segments, transcript_out, state, n, log)

    # ── 1b. Merge each alignment stage's own transcript ─────────────────────────
    # Reacts to job.json's own "alignment_stages" legend (written by Step 3
    # -- see steps/transcribe.py) for WHICH stage numbers to look for, and
    # to whatever per-segment transcript_{N}_NN.json files actually exist
    # for WHETHER a given stage has anything to merge -- both correct
    # whether alignment.dual_output was on for the whole job, part of it,
    # or a stage's data only exists because it was the authoritative one
    # (see steps/transcribe.py's own docstring). A future stage 3 needs no
    # changes here at all: it shows up in alignment_stages the moment
    # steps/transcribe.py starts writing one, and is merged the same way.
    #
    # transcript_{N}.json for the authoritative stage therefore ends up
    # produced even with dual_output off (redundant with transcript.json's
    # own content in that case, but deliberately so: steps/transcript_srt.py
    # only ever looks for these stage-numbered files, never the generic
    # transcript.json, so it doesn't need its own separate "which stage was
    # authoritative" logic -- see that module's docstring).
    #
    # The existence check against each stage's OUTPUT path first, before
    # looking at per-segment sources, matters for the same reason it
    # already does for transcript.json above: this job's own per-segment
    # transcript_{N}_NN.json sources are deleted once this merge succeeds
    # (see cleanup below), so a resumed run must recognize "already merged"
    # without depending on those sources still existing.
    stage_outputs: dict[int, Path] = {}
    for stage in state.get("alignment_stages", []):
        number   = stage["number"]
        out_path = job_dir / f"transcript_{number}.json"
        seg_paths = [job_dir / f"transcript_{number}_{i+1:02d}.json" for i in range(n)]
        seg_paths = [p if p.exists() else None for p in seg_paths]
        if out_path.exists() or any(p is not None for p in seg_paths):
            _merge_transcript_source(seg_paths, segments, out_path, state, n, log)
            stage_outputs[number] = out_path

    # ── 2. Merge audio stems ───────────────────────────────────────────────────
    dialogs    = [d for (d, _) in stem_pairs]
    score_sfxs = [s for (_, s) in stem_pairs]

    if n == 1:
        # Single-segment: dialog.wav and score_sfx.wav are already the canonical
        # names produced by separate.py.  No concat needed — just verify.
        if not dialog_out.exists():
            raise RuntimeError(
                f"Single-segment merge: {dialog_out} not found.  "
                "Did Step 2 (separate) complete successfully?"
            )
        if not score_sfx_out.exists():
            raise RuntimeError(
                f"Single-segment merge: {score_sfx_out} not found.  "
                "Did Step 2 (separate) complete successfully?"
            )
        log.info(
            "  ✓  Single-segment passthrough — dialog.wav (%s)  score_sfx.wav (%s)",
            fmt_size(dialog_out), fmt_size(score_sfx_out),
        )
    else:
        # Multi-segment: lossless PCM concatenation via ffmpeg concat demuxer.
        #
        # Guarded by an existence check (not just the top-level '3b_merge'
        # resume check) because cleanup below deletes the per-segment
        # dialog_NN.wav / score_sfx_NN.wav sources *before* mark_step_done is
        # called.  A crash between "concat succeeded" and "mark_step_done"
        # would otherwise re-enter this branch on the next run with the
        # canonical files already correct but their per-segment sources
        # already gone, and fail outright on a redo that wasn't needed.
        #
        # That existence check is only trustworthy because _ffmpeg_concat()
        # writes to a temp path and is only published under dialog_out /
        # score_fx_out via finalize_output() once ffmpeg has actually
        # succeeded -- a concat interrupted mid-write leaves nothing under
        # the final name at all, rather than a truncated file this check
        # would otherwise have no way to tell apart from a real one. The
        # duration check right after (either branch) is the second,
        # independent layer: it's what would catch a file that predates
        # that fix, or any other way a "complete" file might not actually
        # be one -- see steps/extract.py's _validate_audio_raw() for the
        # fuller version of this same reasoning at Step 1a.
        if dialog_out.exists() and score_sfx_out.exists():
            log.info(
                "  ↩  dialog.wav + score_sfx.wav already exist — verifying "
                "(resumed after a prior interrupted run) ..."
            )
        else:
            log.info("  Concatenating %d dialog stems ...", n)
            _ffmpeg_concat(dialogs, dialog_out, log)

            log.info("  Concatenating %d score/SFX stems ...", n)
            _ffmpeg_concat(score_sfxs, score_sfx_out, log)

            log.info(
                "  ✓  dialog.wav (%s)  score_sfx.wav (%s)",
                fmt_size(dialog_out), fmt_size(score_sfx_out),
            )

    # ── 2b. Integrity check + provenance hash ─────────────────────────────────
    # Runs for both branches above (single-segment passthrough and
    # multi-segment concat) and regardless of whether this merge just ran
    # fresh or is resuming after an interruption -- dialog.wav and
    # score_sfx.wav are the two artifacts a correction re-run
    # (pipeline.py's --skip-index/--add-interval/--redo-review) depends
    # on, possibly much later and in an entirely separate invocation, so
    # this is the one place to both confirm they're good *now* and record
    # what "good" looks like for steps/mute.py and steps/recombine.py to
    # check again immediately before they actually consume these files --
    # which may not be this run at all. Passing here only proves these
    # files were fine at merge time; it says nothing about what might
    # happen to them in the meantime, which is exactly the gap that later
    # check exists to close.
    total_sec      = float(state.get("total_duration_sec", 0.0))
    dialog_hash    = verify_and_hash_before_publish(dialog_out, "dialog.wav", total_sec, log)
    score_sfx_hash = verify_and_hash_before_publish(score_sfx_out, "score_sfx.wav", total_sec, log)
    log.info("  ✓  dialog.wav + score_sfx.wav passed integrity check.")

    # ── 3. Cleanup intermediates ──────────────────────────────────────────────
    # Delete audio_stereo_NN.wav per-segment files (multi-segment) or
    # audio_stereo.wav (single-segment).  These are no longer needed;
    # audio_raw.{ext} in the job store is the resume artifact (§13.3).
    if not keep_intermediate(cfg, correction_artifact=False):
        for seg_wav, _ in segments:
            _unlink_if(seg_wav, log)
        # Also delete the full unsegmented audio_stereo.wav if it still exists
        # (multi-segment: was split into _NN files at Step 1c, which then fed
        # Step 2, so the full file may already be gone — unlink_if is harmless).
        full_stereo = job_dir / "audio_stereo.wav"
        _unlink_if(full_stereo, log)

        if n > 1:
            # Delete per-segment dialog and score_sfx stems now that the
            # canonical concatenated files exist.
            for d, s in stem_pairs:
                _unlink_if(d, log)
                _unlink_if(s, log)

        # Delete per-segment transcript_NN.json now that the canonical,
        # globally-offset transcript.json exists (built in step 1, above).
        # Their one real purpose was letting Step 3 (transcribe.py) skip
        # already-finished segments if it was interrupted partway through --
        # fully served the moment this merge succeeds. Nothing downstream
        # (Step 4b's flag/review phases, Step 5, or the correction
        # workflow's apply_corrections() in steps/review.py, §13.4) ever
        # reads a transcript_NN.json again; only the merged transcript.json.
        # Logged per-file at debug level by _unlink_if, same as the WAV
        # intermediates above -- there's no longer a special case here to
        # call out separately (see the module docstring's "Correction" note).
        for t_path in transcript_paths:
            _unlink_if(t_path, log)

        # Same reasoning, same policy, for each alignment stage's own
        # per-segment files -- transcript_{N}.json above (for whichever
        # numbers stage_outputs has) is the merged, canonical version;
        # nothing downstream ever reads the _NN per-segment sources again
        # either. Harmless no-ops for any segment/stage combination that
        # never had one (dual_output off, or that stage wasn't
        # attempted/failed for that specific segment).
        for stage in state.get("alignment_stages", []):
            for i in range(n):
                _unlink_if(job_dir / f"transcript_{stage['number']}_{i+1:02d}.json", log)

    # ── 4. Persist metadata and mark done ─────────────────────────────────────
    state = read_job(job_dir)
    files = {
        "transcript": transcript_out.name,
        "dialog":     dialog_out.name,
        "score_sfx":  score_sfx_out.name,
    }
    for number, path in stage_outputs.items():
        files[f"transcript_{number}"] = path.name
    # Present only when produced -- same "field present only when notable"
    # shape used elsewhere in this pipeline (e.g. transcribe.py's own
    # mfa_fallback_segments) -- so a job with no comparison data at all
    # (dual_output never on for this job) doesn't carry keys pointing at
    # files that don't exist.
    state["merge"] = {
        "segments":   n,
        "word_count": total_words,
        "files": files,
        "dialog_sha256":    dialog_hash,
        "score_sfx_sha256": score_sfx_hash,
    }
    write_job(job_dir, state)
    mark_step_done(job_dir, "3b_merge")

    log.info(
        "  ✓  Step 3b complete.  Total words: %d  Duration: %s",
        total_words,
        fmt_duration(float(state.get("total_duration_sec", 0.0))),
    )
    return transcript_out, dialog_out, score_sfx_out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_transcript_source(
    per_segment_paths: list[Optional[Path]],
    segments: list[tuple[Path, float]],
    out_path: Path,
    state: dict,
    n: int,
    log: logging.LoggerAdapter,
) -> int:
    """
    Merge one alignment source's per-segment transcript files (offset-
    adjusted, concatenated in segment order) into one canonical file at
    out_path. Shared by the primary transcript.json merge and the
    alignment.dual_output comparison merges (transcript_mfa.json /
    transcript_whisperx.json) in merge() above -- identical offset-
    application logic either way, just parameterized by which
    per-segment files to read and where to write the result.

    A None entry in per_segment_paths means this segment has no data for
    this particular source (see steps/transcribe.py's dual_output
    comparison logic, which can leave a segment's comparison file
    genuinely absent when that backend wasn't attempted, or was attempted
    and failed, rather than writing an empty one) -- contributes zero
    words for that segment's span: a real, correctly-reported gap in that
    source's transcript, not a merge failure.

    Guarded by an out_path existence check for the same reason the
    original (pre-multi-source) version of this logic always was: Step
    3b's cleanup deletes per_segment_paths once a merge succeeds, so a
    crash between "this write succeeded" and mark_step_done() would
    otherwise need to redo the merge on the next run with its sources
    already gone.

    Returns the total word count merged.
    """
    if out_path.exists():
        log.info(
            "  ↩  %s already exists — verifying (resumed after a prior "
            "interrupted run) ...", out_path.name,
        )
        return len(json.loads(out_path.read_text()).get("words", []))

    all_words:     list[dict] = []
    detected_lang: str        = "en"
    total_words = 0

    for i, (t_path, (seg_wav, start_offset)) in enumerate(zip(per_segment_paths, segments)):
        seg_idx = i + 1
        if t_path is None:
            log.info(
                "  [%d/%d] %s — no data for this segment.",
                seg_idx, n, out_path.name,
            )
            continue

        data = json.loads(t_path.read_text())
        detected_lang = data.get("language", detected_lang)
        seg_words     = data.get("words", [])

        # Apply global offset to each word's timestamps.
        # Words with null timestamps (alignment failures) are preserved as-is.
        adjusted: list[dict] = []
        for w in seg_words:
            aw = dict(w)
            if w.get("start") is not None:
                aw["start"] = round(float(w["start"]) + start_offset, 4)
            if w.get("end") is not None:
                aw["end"]   = round(float(w["end"])   + start_offset, 4)
            adjusted.append(aw)

        seg_end_sec = start_offset + _seg_duration(state, seg_wav.name, i)
        log.info(
            "  [%d/%d] %s  offset=%s  words=%d  (%s → %s)",
            seg_idx, n, t_path.name,
            f"{start_offset:.1f}s",
            len(seg_words),
            fmt_duration(start_offset),
            fmt_duration(seg_end_sec),
        )
        all_words.extend(adjusted)
        total_words += len(seg_words)

    transcript_data: dict = {
        "language": detected_lang,
        "words":    all_words,
    }
    out_path.write_text(
        json.dumps(transcript_data, indent=2, ensure_ascii=False)
    )
    log.info("  ✓  %s  words=%d", out_path.name, total_words)
    return total_words


def _ffmpeg_concat(sources: list[Path], dest: Path, log: logging.LoggerAdapter) -> None:
    """
    Concatenate PCM WAV files using the ffmpeg concat demuxer (stream copy).

    All sources must have identical format (sample rate, bit depth, channels).
    The 44.1 kHz 16-bit stereo PCM constraint throughout the pipeline
    guarantees this.

    Uses a temporary file list rather than the 'concat:' protocol because
    the demuxer handles WAV header size fields correctly for all lengths
    and is the ffmpeg-recommended approach for concatenating file streams.
    """
    list_path = dest.parent / f".concat_{dest.stem}.txt"
    tmp = tmp_output_path(dest)
    try:
        list_path.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in sources) + "\n"
        )
        tmp.unlink(missing_ok=True)   # clear a partial attempt from an interrupted prior run
        run_cmd(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                str(tmp),
            ],
            log,
        )
        finalize_output(tmp, dest)
    finally:
        if list_path.exists():
            list_path.unlink()


def _seg_duration(state: dict, seg_wav_name: str, fallback_index: int) -> float:
    """Compute segment duration from job.json start_sec offsets."""
    segs      = state.get("segments", [])
    total_sec = float(state.get("total_duration_sec", 0.0))

    for j, seg in enumerate(segs):
        if seg.get("path") == seg_wav_name:
            if j + 1 < len(segs):
                return float(segs[j + 1]["start_sec"]) - float(seg["start_sec"])
            return total_sec - float(seg["start_sec"])

    return total_sec if len(segs) == 1 else 0.0


def _unlink_if(path: Path, log: logging.LoggerAdapter) -> None:
    """Delete a file if it exists; no-op and no error if absent."""
    if path.exists():
        path.unlink()
        log.debug("  Removed intermediate: %s", path.name)
