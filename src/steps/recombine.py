"""
profanity-hush — Step 6: recombine dialog + score/SFX stems

Mixes the censored dialog stem back together with the (untouched)
score/SFX stem from Step 3b's merge, restoring a single full audio track —
now with the flagged words silenced and the music/sound effects playing
through uninterrupted underneath. This is the whole reason muting happens
on the isolated dialog stem instead of the full mix (see design doc §4).

Input  : dialog_censored.wav (Step 5), score_sfx.wav (Step 3b)
Output : audio_censored.wav

score_sfx.wav may have been sitting untouched since Step 3b wrote it --
see steps/mute.py's module docstring for the identical reasoning applied
to dialog.wav -- so it's re-verified against the duration and hash
steps/merge.py recorded at that time before this step actually reads it
(utils.verify_stem_before_reuse()). A mismatch raises rather than
silently regenerating anything, for the same reason: there's no cheap
fix, since score_sfx.wav's only source is Step 2's Demucs separation.

dialog_censored.wav gets the same treatment against the duration/hash
steps/mute.py recorded, but for a different reason: --skip-index/
--add-interval/--redo-review always redo Steps 5 and 6 together, so
dialog_censored.wav is never stale by the time this step reads it in
that workflow -- but pipeline.py's --redo-step can name 6_recombine
(or 6b_encode/7_mux) without also naming 5_mute, in which case this
step runs fresh while dialog_censored.wav is left over, unverified,
from however long ago 5_mute last actually ran. Unlike score_sfx.wav, a
mismatch here is cheap to fix (--redo-step 5_mute cascades forward
through this step automatically), so its regenerate_hint says that
instead of pointing at a from-scratch re-run.

Tool (ffmpeg's amix filter):
  ffmpeg -i dialog_censored.wav -i score_sfx.wav \
      -filter_complex amix=inputs=2:duration=first:normalize=0 \
      -c:a pcm_s16le \
      audio_censored.wav

  duration=first  — output length follows dialog_censored.wav. The two
                    inputs come from the same source audio via Demucs and
                    should already be the same length; this just pins the
                    behavior explicitly rather than leaving it to amix's
                    "longest" default in case they ever differ by a
                    sample or two.
  normalize=0     — amix's default behavior scales every input down by
                    1/N to leave headroom for summing; with two
                    already-mixed, full-range stems that would quietly
                    halve both the dialog and the score/SFX levels
                    relative to the original mix. normalize=0 preserves
                    the source levels as recombined.
  -c:a pcm_s16le  — explicit, even though the design doc's reference
                    command (§8) omits it: amix negotiates its own
                    internal sample format (often float), and left
                    unspecified, the .wav muxer would just take whatever
                    that happens to be. Every other WAV in this pipeline
                    is 44.1kHz/16-bit PCM (see steps/merge.py's
                    concatenation, which depends on that being uniform);
                    pinning the format here keeps that invariant instead
                    of silently widening audio_censored.wav for Step 6b
                    (encode) to deal with later.

Intermediate cleanup:
  dialog_censored.wav is fully consumed once audio_censored.wav exists —
  nothing downstream needs it again, and it's also cheap to regenerate
  from dialog.wav + matches.json/review.json if ever needed (no
  Demucs re-run required) — so it's deleted whenever output.keep_intermediates
  is false, same as steps/merge.py's own now-superseded intermediates.

  score_sfx.wav, however, is governed differently: it's the *other* half
  of what a future correction needs (alongside dialog.wav, kept by
  steps/mute.py — see that module's docstring) to redo Steps 5-6-7
  without re-running Step 2's Demucs separation. So score_sfx.wav is
  deleted only if *both* output.keep_intermediates and
  output.keep_correction_artifacts (default true) are false — not just
  keep_intermediates alone. pipeline.py's "Steps 1a-3b already complete"
  resume shortcut accounts for score_sfx.wav's absence not being an error
  once Step 6 has run, regardless of which setting caused the deletion.

Marks '6_recombine' done.
Returns the path to audio_censored.wav.
"""

import logging
from pathlib import Path
from typing import Optional

from utils import (
    fmt_size,
    keep_intermediate,
    mark_step_done,
    read_job,
    run_cmd,
    step_logger,
    verify_and_hash_before_publish,
    verify_stem_before_reuse,
    write_job,
)


def recombine(
    job_dir: Path,
    dialog_censored_path: Path,
    score_sfx_path: Path,
    cfg: dict,
    log: Optional[logging.LoggerAdapter] = None,
) -> Path:
    """
    Step 6: mix dialog_censored.wav + score_sfx.wav into audio_censored.wav.

    Returns the path to audio_censored.wav.
    """
    if log is None:
        log = step_logger("recombine")

    state               = read_job(job_dir)
    done                = state.get("steps_completed", [])
    audio_censored_out  = job_dir / "audio_censored.wav"

    if "6_recombine" in done:
        log.info("Step 6 — ↩  already complete; re-using %s.", audio_censored_out.name)
        # audio_censored.wav is a large WAV intermediate that Step 6b
        # (encode) deletes once audio_encoded.mka exists (unless
        # keep_intermediates) -- so it's only *required* to still be on
        # disk if Step 6b hasn't run yet. Same reasoning as
        # steps/mute.py's resume-check applies one step later here.
        if "6b_encode" not in done and not audio_censored_out.exists():
            raise RuntimeError(
                f"Step 6 is marked complete but {audio_censored_out} is missing, "
                "and Step 6b (encode) hasn't run yet to explain its absence.  "
                "Delete the job directory and re-run from scratch."
            )
        return audio_censored_out

    if not dialog_censored_path.exists():
        raise RuntimeError(
            f"Step 6: censored dialog stem not found at {dialog_censored_path} — "
            "did Step 5 (mute) complete?"
        )
    if not score_sfx_path.exists():
        raise RuntimeError(
            f"Step 6: score/SFX stem not found at {score_sfx_path} — did Step 3b "
            "(merge) complete?"
        )
    merge_info = state.get("merge", {})
    total_sec  = float(state.get("total_duration_sec", 0.0))
    mute_info  = state.get("mute", {})
    verify_stem_before_reuse(
        dialog_censored_path,
        total_sec,
        mute_info.get("dialog_censored_sha256"),
        log,
        label="dialog_censored.wav",
        written_by="Step 5 (mute)",
        regenerate_hint=(
            "Unlike score_sfx.wav, this is cheap to fix: re-run with "
            "--redo-step 5_mute, which will cascade forward through this "
            "step (and 6b_encode/7_mux) automatically -- see pipeline.py's "
            "--redo-step --help."
        ),
    )
    verify_stem_before_reuse(
        score_sfx_path,
        total_sec,
        merge_info.get("score_sfx_sha256"),
        log,
        label="score_sfx.wav",
    )

    log.info("Step 6 — recombine dialog + score/SFX stems")

    run_cmd(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y",
            "-i", str(dialog_censored_path),
            "-i", str(score_sfx_path),
            "-filter_complex", "amix=inputs=2:duration=first:normalize=0",
            "-c:a", "pcm_s16le",
            str(audio_censored_out),
        ],
        log,
    )
    log.info("  ✓  audio_censored.wav  (%s)", fmt_size(audio_censored_out))

    # Write-time half of the integrity check steps/encode.py runs
    # immediately before it actually consumes audio_censored.wav.
    audio_censored_hash = verify_and_hash_before_publish(
        audio_censored_out, "audio_censored.wav", total_sec, log,
    )

    # dialog_censored.wav: trivially regenerable from dialog.wav (no
    # Demucs re-run needed) -- governed by keep_intermediates alone.
    if not keep_intermediate(cfg, correction_artifact=False):
        _unlink_if(dialog_censored_path, log)
    # score_sfx.wav: the *other* half of what a future correction needs
    # (alongside dialog.wav -- see steps/mute.py) to redo Steps 5-6-7
    # without re-running Demucs.
    if not keep_intermediate(cfg, correction_artifact=True):
        _unlink_if(score_sfx_path, log)

    state = read_job(job_dir)
    state["recombine"] = {
        "output": audio_censored_out.name,
        "audio_censored_sha256": audio_censored_hash,
    }
    write_job(job_dir, state)
    mark_step_done(job_dir, "6_recombine")

    log.info("  ✓  Step 6 complete.")
    return audio_censored_out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unlink_if(path: Path, log: logging.LoggerAdapter) -> None:
    """Delete a file if it exists; no-op and no error if absent."""
    if path.exists():
        path.unlink()
        log.debug("  Removed intermediate: %s", path.name)
