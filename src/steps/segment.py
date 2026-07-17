"""
profanity-hush — Step 1c: segment audio_stereo.wav

Splits the stereo WAV into fixed-length chunks for per-segment Demucs
processing (Step 2).  Short files that fit within one segment are returned
as a single-item list pointing to audio_stereo.wav itself — no splitting.

Returns:
  list[tuple[Path, float]]  —  (segment_wav_path, start_offset_sec)

The start offsets are used at Step 3b to convert segment-local word timestamps
back to film-absolute timestamps.
"""
import json
import logging
import math
from pathlib import Path
from typing import Optional

from utils import (
    cfg_get,
    check_duration_matches,
    finalize_output,
    fmt_duration,
    fmt_size,
    mark_step_done,
    read_job,
    run_cmd,
    step_logger,
    tmp_output_path,
    write_job,
)


def segment(
    job_dir: Path,
    cfg: dict,
    log: Optional[logging.LoggerAdapter] = None,
) -> list[tuple[Path, float]]:
    """
    Step 1c: split audio_stereo.wav into fixed-size segments.

    Behaviour:
      segment_size_sec == 0 or duration <= segment_size_sec
        → single-segment passthrough; returns [(audio_stereo.wav, 0.0)]
        → no files are created

      duration > segment_size_sec
        → splits into N = ceil(duration / segment_size_sec) chunks
        → returns [(audio_stereo_01.wav, 0.0), (audio_stereo_02.wav, 1800.0), ...]

    Segment WAV files use stream-copy (-c copy) so splitting is near-instant
    regardless of file size.  The final segment omits -t to capture any
    sub-second remainder precisely. Each is written to a temp path and
    atomically published under its final name only once ffmpeg has actually
    finished, so a run interrupted mid-split never leaves a truncated
    audio_stereo_NN.wav sitting under a name Step 2 would later trust.

    Segment files are large intermediates and are deleted after Step 3b
    unless keep_intermediates is set.

    Resume support:
      If '1c_segment' is already marked done, the recorded segments in
      job.json are re-derived and each one re-validated (duration vs. its
      own start/end arithmetic) before being trusted -- see
      _segments_from_state()'s docstring for what happens if one of them
      is missing rather than merely invalid; those are handled
      differently.

    Marks '1c_segment' done and writes segment metadata to job.json.
    """
    if log is None:
        log = step_logger("segment")

    stereo = job_dir / "audio_stereo.wav"

    state = read_job(job_dir)
    if "1c_segment" in state.get("steps_completed", []):
        segs = _segments_from_state(job_dir, state, log)
        if segs is not None:
            log.info("Step 1c — ↩  already complete; %d segment(s) verified.", len(segs))
            return segs
        log.info(
            "Step 1c — marked complete, but the recorded segment(s) are no "
            "longer all present — regenerating from audio_stereo.wav."
        )

    if not stereo.exists():
        raise RuntimeError(
            f"Step 1c: audio_stereo.wav not found in {job_dir} — did Step 1b complete?"
        )

    size_sec = int(cfg_get(cfg, "audio", "segment_size_sec"))

    log.info("Step 1c — probing audio_stereo.wav ...")
    duration = _probe_duration(stereo, log)
    log.info(
        "  Duration: %.1f s  (%s)  |  segment_size: %s s",
        duration, fmt_duration(duration), size_sec,
    )

    if size_sec == 0 or duration <= size_sec:
        reason = "segment_size_sec=0 (disabled)" if size_sec == 0 else "duration ≤ segment_size"
        log.info("  Single-segment passthrough (%s).", reason)
        segs: list[tuple[Path, float]] = [(stereo, 0.0)]
    else:
        segs = _split(stereo, job_dir, duration, size_sec, log)

    _persist(job_dir, segs, duration)
    mark_step_done(job_dir, "1c_segment")
    return segs


# ── Internal ──────────────────────────────────────────────────────────────────

def _segments_from_state(
    job_dir: Path,
    state: dict,
    log: logging.LoggerAdapter,
) -> Optional[list[tuple[Path, float]]]:
    """
    Re-derive (path, start_sec) pairs from job.json's recorded 'segments'
    block for a resumed run, validating each one still exists and (for a
    multi-segment job) still has the duration its own start/end arithmetic
    says it should.

    Returns None -- rather than raising -- if a recorded file is simply
    missing. That's not necessarily corruption: it's the expected shape of
    Step 3b's own cleanup (steps/merge.py deletes these per-segment files
    once its canonical dialog.wav/score_sfx.wav exist) having run on some
    earlier attempt that was then interrupted before Step 3b itself
    reached mark_step_done('3b_merge') -- see that module's own comment on
    the identical race for its own outputs. segment() treats None as "fall
    through and regenerate from audio_stereo.wav," exactly as on a first
    run; _split() below only recreates whichever specific segments are
    actually absent.

    Still raises, though, if a recorded file *exists* but fails its
    duration check via _validate_segment() -- that's not an expected
    cleanup-ordering artifact, it's the truncation this whole mechanism
    exists to catch. "Missing" and "corrupt" warrant different responses;
    see steps/extract.py's _validate_audio_raw() for the same distinction
    one step earlier in the pipeline.
    """
    segs_state = state.get("segments", [])
    if not segs_state:
        return None

    total_sec = float(state.get("total_duration_sec", 0.0))
    multi     = len(segs_state) > 1
    segs: list[tuple[Path, float]] = []

    for i, seg in enumerate(segs_state):
        p     = job_dir / seg["path"]
        start = float(seg["start_sec"])
        if not p.exists():
            return None

        if multi:
            end = (
                float(segs_state[i + 1]["start_sec"])
                if i + 1 < len(segs_state) else total_sec
            )
            _validate_segment(p, end - start, log)

        segs.append((p, start))

    return segs


def _validate_segment(path: Path, expected_duration: float, log: logging.LoggerAdapter) -> None:
    """
    Integrity check for one split segment file -- same rationale as
    steps/extract.py's audio_raw.* validation: _split()'s ffmpeg
    stream-copy writes each audio_stereo_NN.wav the same way Step 1a's
    ffmpeg writes audio_raw.*, so a run interrupted mid-split would,
    without this check, leave a truncated segment sitting under the exact
    filename Step 2 (separate.py) later trusts on resume.

    expected_duration comes from this same module's own start/end
    arithmetic (job.json's recorded segment boundaries on a resume, or
    the same computation freshly on a first run) rather than a second
    independent source -- so, unlike extract.py's source-video
    cross-check, this is really confirming "the split landed where the
    math says it should," and a tight tolerance is appropriate.

    On failure: deletes path (so a subsequent run doesn't get stuck
    re-validating the same bad file -- it just regenerates that one
    segment) and raises.
    """
    actual = _probe_duration(path, log)
    try:
        check_duration_matches(
            actual, expected_duration, log=log,
            label=f"{path.name} (split segment)", tolerance_sec=2.0,
        )
    except RuntimeError:
        path.unlink(missing_ok=True)
        log.error("  Deleted incomplete %s — re-run to split it fresh.", path.name)
        raise


def _probe_duration(wav: Path, log: logging.LoggerAdapter) -> float:
    """Return duration of wav in seconds via ffprobe."""
    result = run_cmd(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "json",
            str(wav),
        ],
        log,
    )
    data    = json.loads(result.stdout)
    streams = data.get("streams", [])

    if streams and "duration" in streams[0]:
        return float(streams[0]["duration"])

    # WAV sometimes omits stream-level duration; fall back to format level
    result2 = run_cmd(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            str(wav),
        ],
        log,
    )
    fmt = json.loads(result2.stdout)
    dur = fmt.get("format", {}).get("duration")
    if dur is None:
        raise RuntimeError(f"Could not determine duration of {wav.name}")
    return float(dur)


def _split(
    stereo: Path,
    job_dir: Path,
    duration: float,
    size_sec: int,
    log: logging.LoggerAdapter,
) -> list[tuple[Path, float]]:
    """Split stereo WAV into N fixed-size chunks; return (path, start_sec) list."""
    n    = math.ceil(duration / size_sec)
    segs: list[tuple[Path, float]] = []

    log.info("  Splitting into %d segment(s) × ≤ %s each ...", n, fmt_duration(size_sec))

    for i in range(n):
        start = float(i * size_sec)
        end   = min(start + size_sec, duration)
        out   = job_dir / f"audio_stereo_{i + 1:02d}.wav"

        if out.exists():
            log.info(
                "  [%d/%d] ↩  %s already exists (%s) — verifying ...",
                i + 1, n, out.name, fmt_size(out),
            )
            _validate_segment(out, end - start, log)
        else:
            # -ss before -i → input seeking (direct byte offset into PCM WAV).
            # Placing -ss after -i uses output/decoder seeking, which must scan
            # from the start of the file for every segment — O(file size) rather
            # than O(1).  For PCM WAV, input seeking is both faster and exact
            # (no frame-boundary alignment needed for uncompressed audio).
            tmp = tmp_output_path(out)
            tmp.unlink(missing_ok=True)   # clear a partial attempt from an interrupted prior run
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y",
                "-ss", str(start),   # input seek — must be before -i
                "-i", str(stereo),
            ]
            if i < n - 1:
                cmd += ["-t", str(size_sec)]   # final segment: omit -t to capture remainder
            cmd += ["-c", "copy", str(tmp)]

            run_cmd(cmd, log)
            finalize_output(tmp, out)
            _validate_segment(out, end - start, log)
            log.info(
                "  [%d/%d] %s  %s → %s  (%.1f s, %s)",
                i + 1, n,
                out.name,
                fmt_duration(start),
                fmt_duration(end),
                end - start,
                fmt_size(out),
            )

        segs.append((out, start))

    log.info("  ✓  %d segment(s) ready.", n)
    return segs


def _persist(job_dir: Path, segs: list[tuple[Path, float]], total_sec: float) -> None:
    """Write segment list and total duration into job.json."""
    state = read_job(job_dir)
    state["total_duration_sec"] = total_sec
    state["segments"] = [
        {"index": i + 1, "path": p.name, "start_sec": s}
        for i, (p, s) in enumerate(segs)
    ]
    write_job(job_dir, state)
