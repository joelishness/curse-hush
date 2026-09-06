"""
profanity-hush — Step 7: mux censored audio into the original video

Combines the original video's bitstream-exact video stream with the
already-encoded censored audio (audio_encoded.mka, Step 6b) into the final
output file, written to /output. This is the last step of v1's core
pipeline (§4) — once this succeeds, the job is done.

Input  : original video file, audio_encoded.mka (Step 6b)
Output : /output/{filename per output.naming_style} -- see _output_path
         for the two supported styles (plex_edition, the default, and the
         original v1 suffix style). For a TV episode under plex_edition,
         /output is expected to already be the show's sibling edition
         directory by the time this runs -- see hush.sh's
         redirect_for_tv_edition() for where that decision is actually
         made and why it can't be made in here.

**The actual muxing tool depends on output.format** — this is the one step
in the pipeline that doesn't use ffmpeg for its primary job:

  format: mkv (the default, and the one config.yaml recommends) → mkvmerge
  format: mp4                                                   → ffmpeg

Why mkvmerge for mkv, when every other step in this pipeline is ffmpeg:
splitting the audio re-encode out of this step (steps/encode.py, Step 6b)
fixed one real bug (a cross-input timestamp-origin mismatch — see that
module's docstring) but, against a real production file, did *not* fix
the actual symptom: a subset of long, multi-subtitle-track Blu-ray rips
still played back "audio mostly silent, present only in scattered spots"
in VLC and multiple mpv-based players even with both streams now pure
copies on both sides — and Plex didn't merely play it wrong, it refused
to load the file at all (stuck on its loading spinner indefinitely). Four
independent player codebases struggling on the same file pointed at a
genuine structural defect, not a timestamp nuance and not a per-player
quirk. The isolating test: re-muxing the exact same (already-confirmed-
correct) audio_encoded.mka against the original video with subtitles,
chapters, and attachments stripped out (`-map 0:v:0 -map 1:a:0` only, no
`-map 0:s?`/`-map 0:t?`/`-map_chapters`) played back correctly everywhere
tested. The original video's PGS subtitle tracks are exactly the ones
ffmpeg's own demuxer already warns it can't fully analyze on input
(`Could not find codec parameters ... unspecified size`) — and copying
tracks ffmpeg itself admits it couldn't fully parse, via ffmpeg's own
matroska muxer, is exactly the kind of operation likely to produce a
malformed result. mkvmerge — a different, Matroska-specific muxer
implementation, not built on the same generic libavformat probing ffmpeg
uses — was tested against the *identical* source file with subtitles and
chapters fully included (the exact tracks ffmpeg's demuxer warned about),
produced no analogous warning at all, and played back correctly in every
player tested, including Plex. That's a clean enough A/B (same input
bytes, same audio track, only the muxer implementation differs) to make
the muxer itself, not the subtitle data, the confirmed fault — so Step 7
uses mkvmerge for the format (mkv) that's actually affected, rather than
ffmpeg with the subtitle/chapter/attachment copying removed (which would
"fix" this by silently dropping content from every future output, not by
fixing the actual defect).

mp4 output stays on ffmpeg because mkvmerge can only produce Matroska or
WebM — there's no mkvmerge equivalent to ask for here. This isn't a gap in
practice: mp4 output already never attempts subtitle/attachment copying
in the first place (see below), which is the exact category of content
implicated above, so the mkvmerge fix's motivating case doesn't apply to
the mp4 path to begin with.

**mkvmerge command** (mkv path):
```
mkvmerge -o output.part.mkv --no-audio video.mkv audio_encoded.mka
```
mkvmerge's default behaviour, absent any flag saying otherwise, is to
copy *everything* from each input file — video, every subtitle track,
chapters, attachments, tags. `--no-audio` applies to the file named
immediately after it (`video.mkv`) and only suppresses that file's own
(now-uncensored, soon to be replaced) audio track[s] — it does not affect
`audio_encoded.mka`, named with no flag of its own, which contributes its
one audio track unmodified. This single line is therefore already
"video + every subtitle + chapters + attachments from the original, audio
from Step 6b" with no further flags needed — confirmed by hand against
this exact file, with subtitles and chapters both included, before this
module was switched over to it.

mkvmerge's exit codes are not the universal 0=success/nonzero=failure
convention every other tool in this pipeline follows: 0 is a clean run,
1 means it completed successfully but logged at least one warning (e.g.
a track it couldn't fully identify some metadata for, muxed correctly
regardless), and only 2 is an actual failure. `run_cmd(..., ok_exit_codes=
frozenset({0, 1}))` is what keeps a warning-only run from being treated
as a Step 7 failure — see utils.run_cmd's docstring.

`-avoid_negative_ts make_zero`, used on the ffmpeg/mp4 path below for the
same defensive reasons as steps/encode.py, has no mkvmerge equivalent
(and no evidence from testing that it's needed there) — mkvmerge computes
its own track timing from the source files' own block timestamps and was
confirmed correct as-is.

**Subtitle/chapter/attachment preservation for mp4 output:** unlike the
mkvmerge/mkv path above (which preserves all of this by default with no
extra flags), mp4 output does not attempt subtitle or attachment
passthrough at all — PGS/VOBSUB bitmap subtitles in particular generally
aren't valid in MP4, and attempting the copy would make ffmpeg fail
outright rather than just producing a censored file without subtitles.
Chapters are carried forward (`-map_chapters 0`; MP4 supports them
natively via a different mechanism than Matroska, but ffmpeg already
does this by default for a single input — kept explicit here rather than
relying on that default).

**Each configured transcript SRT (Step 6c — steps/transcript_srt.py's
export_srt(), zero to (number of debug_subtitle-enabled engines, plus one
more if the final engine has final_subtitle: true) of them) whose own
alignment.engines.<name>.embed_subtitle is true gets embedded as an
ADDITIONAL subtitle track — separate from, and unaffected by, the
mp4-path limitation just above.** Each SrtSource steps/transcript_srt.py
returns carries its own `embed` flag already resolved from that setting
(see that module's own docstring) — this step never reads
alignment.engines.* itself, it just respects whichever sources arrive
with embed=True and leaves the rest as job-directory/sidecar files only.
That limitation concerns the *original* video's own subtitle tracks
(arbitrary formats, some of them the exact PGS/bitmap tracks ffmpeg can't
reliably parse); these are always plain UTF-8 SRT text
steps/transcript_srt.py wrote itself, so they hit neither problem that
motivated skipping subtitle passthrough for mp4 in the first place.

  mkv: each source added as one more mkvmerge input file, with
    --language / --track-name / --default-track-flag scoped to it via
    the same "options apply to the next-named file" convention already
    used above for audio_encoded.mka's implicit (no-flag) inclusion.
    Whatever style each was written in (transcript_srt.karaoke) survives
    byte-for-byte — confirmed directly, by round-tripping a tagged file
    through mkvmerge and back out — including the <font color> tags a
    karaoke rendering depends on, since mkvmerge stores SRT as SRT (codec
    S_TEXT/UTF8), no re-encoding of the text.

    Replace, don't duplicate: before adding these, video_path's own
    existing subtitle tracks are probed (_probe_subtitle_tracks(), via
    `mkvmerge -J`) for any whose name exactly matches one of THIS run's
    subtitle_sources' own track_name (steps/transcript_srt.py — built
    from transcript_srt.track_name_prefix plus each source's label) —
    i.e. this pipeline's own tracks from a previous run, if video_path
    happens to be pointed back at this job's own prior output rather
    than the original source. Matches are excluded from passthrough
    (--subtitle-tracks with the surviving IDs, or --no-subtitles if every
    existing subtitle track matched) so re-muxing replaces them rather
    than piling up duplicates every run. Matched by EXACT current name
    only: change track_name_prefix between runs (or the set of alignment
    stages that ran) and a track already embedded under the OLD name is
    no longer recognized as "ours" and is left in place rather than
    replaced — the new name just gets added alongside it going forward.
    The probe is best-effort (_probe_subtitle_tracks() returns "found
    nothing to replace" rather than raising if video_path can't be
    identified this way at all) —
    this is a nice-to-have in service of a debug track staying tidy on
    repeat runs, never a reason the actual censored video fails to mux.

  mp4: each source converted via ffmpeg's mov_text codec — the only way
    to place a text subtitle track inside an MP4 container at all, and
    NOT the same tags-survive story mkv gets: confirmed directly (the
    same round-trip test) that mov_text strips every <font>/<b>/<i>/<u>
    tag outright on the way through. This module therefore always embeds
    each source's mp4_fallback_text — a plain, one-cue-per-group
    rendering of the exact same grouping the job-directory/sidecar SRT
    used (see steps/transcript_srt.py's docstring) — rather than that SRT
    itself, regardless of transcript_srt.karaoke. Embedding the karaoke
    file as-is would still technically succeed, but every one-cue-per-word
    group would decode as several back-to-back, visually IDENTICAL
    redraws of the same plain line once the color is silently gone — at
    best pointless, at worst looking like a player bug. No replace-by-name
    detection needed here the way mkv has: video_path for the mp4 path is
    always the pristine original (mp4 never passes ANY of the original's
    own subtitle tracks through — see above — so it never carries a track
    of ours from a previous run to begin with; every mp4 mux starts from
    scratch and only ever contains what's explicitly added here).

  Either way each added track is written non-default (--default-track-flag
  0:no for mkv; -disposition:s:0 0 for mp4, the latter best-effort only —
  confirmed directly that ffmpeg's mp4 muxer, unlike matroska's, has no
  equivalent "don't infer a default track" override, so a player may
  still auto-select the first one on open despite the flag; every player
  tested still lets a viewer turn it back off regardless) so none of them
  compete with any subtitle track a viewer actually opened the file for.
  Skipped entirely — no subtitle-related flags added to either command —
  whenever subtitle_sources is empty (transcript_srt.enabled is false,
  Step 6c failed, or neither transcript had any word with usable timing —
  see steps/transcript_srt.py) or none of the sources present have
  embed=True (i.e. every relevant alignment.engines.<name>.
  embed_subtitle is false).

Crash safety: the muxed file is written to a temp sibling inside /output
(utils.tmp_output_path() / finalize_output() — the same write-then-rename
idiom utils.write_job() uses for job.json) and only published under its
final name once the muxing tool exits with one of ok_exit_codes.
Applied here, rather than left to whichever tool's own behavior, because
/output, unlike /jobs, is the one place in this pipeline a half-written
file would be directly user-visible and easy to mistake for a finished
one.

audio_encoded.mka gets the same re-verify-before-consuming treatment
audio_censored.wav gets in steps/encode.py, against the duration/hash
steps/encode.py recorded: --skip-index/--add-interval/--redo-review
always redo Steps 5/6/6b/7 together, so this file is never stale in
that workflow, but pipeline.py's --redo-step can name 7_mux alone, in
which case this step runs fresh against whatever audio_encoded.mka
happened to be left over from a previous run.

Intermediate cleanup (conditional on keep_intermediates):
  audio_encoded.mka is fully consumed once the final muxed video exists —
  nothing in v1 needs it again — so it's deleted here unless
  keep_intermediates is set, matching every earlier step's cleanup
  pattern for its own now-superseded intermediates. audio_raw.{ext} is
  NOT touched here: it's always kept (§6), independent of this step, for
  future per-channel reprocessing (§13.3).

Marks '7_mux' done. Returns the path to the final output file in /output.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Optional
import logging

from utils import (
    cfg_get,
    finalize_output,
    fmt_dir,
    fmt_size,
    keep_intermediate,
    mark_step_done,
    read_job,
    run_cmd,
    step_logger,
    tmp_output_path,
    verify_stem_before_reuse,
    write_job,
)
from steps.transcript_srt import SrtSource


def mux(
    job_dir: Path,
    video_path: Path,
    audio_encoded_path: Path,
    output_dir: Path,
    cfg: dict,
    log: Optional[logging.LoggerAdapter] = None,
    *,
    subtitle_sources: Optional[list[SrtSource]] = None,
) -> Path:
    """
    Step 7: mux audio_encoded.mka (Step 6b) into the original video's
    container -- mkvmerge for mkv output, ffmpeg for mp4 (see module
    docstring for why these differ).

    subtitle_sources -- steps/transcript_srt.py's export_srt() return
    value, passed straight through from pipeline.py: zero or more
    SrtSource entries, each already carrying its own `embed` flag
    resolved from that engine's alignment.engines.<name>.embed_subtitle
    (see steps/transcript_srt.py's own docstring). None or an empty list
    (or a list where every entry has embed=False) embeds nothing and
    reproduces this function's exact pre-Step-6c behavior; see the
    module docstring above for how an embeddable entry is used
    differently for mkv vs mp4 output.

    Returns the path to the final output file in /output.

    Writes job.json's "mux" block with output_path (a host-navigable
    directory when AC_OUTPUT_HOST_DIR reached the container, else this
    container's own /output) and output_filename (bare filename),
    alongside the format/tool it already recorded.
    """
    if log is None:
        log = step_logger("mux")

    state      = read_job(job_dir)
    out_format = str(cfg_get(cfg, "output", "format")).lower()
    if out_format not in ("mkv", "mp4"):
        raise RuntimeError(
            f"Step 7: unknown output.format '{out_format}' (expected 'mkv' or 'mp4')."
        )
    out_path = _output_path(video_path, output_dir, cfg, out_format)

    # Compares real host paths (AC_INPUT_HOST_DIR / AC_OUTPUT_HOST_DIR,
    # merged into cfg by utils.load_config() -- see paths.input_host_dir /
    # paths.output_host_dir), not output_dir/video_path themselves: those
    # are always /output and /input, two distinct container mount points
    # that can never compare equal to each other regardless of what real
    # host directories they're actually bound to. (An earlier version of
    # this check compared those instead and could never have caught
    # anything as a result -- comparing container-side paths here was the
    # bug, not the idea of checking at all.) This is cheap insurance
    # against the failure mode that matters most: a bug in whatever
    # redirected the output directory (hush.sh's TV-editions redirection,
    # or a bad --output value) silently landing the final write on top of
    # the source file. Skipped, not raised, if either host path is
    # unknown -- e.g. this cfg came from a context that never set these
    # env vars at all -- since there's nothing meaningful to compare then.
    real_input_dir  = cfg_get(cfg, "paths", "input_host_dir", default=None)
    real_output_dir = cfg_get(cfg, "paths", "output_host_dir", default=None)
    if real_input_dir and real_output_dir:
        real_video = Path(real_input_dir) / video_path.name
        real_out   = Path(real_output_dir) / out_path.name
        if real_video == real_out:
            raise RuntimeError(
                f"Step 7: computed output path ({real_out}) is identical "
                f"to the input video -- refusing to overwrite the "
                "source. Check output.naming_style and, for a TV "
                "episode, whatever directory hush.sh redirected /output "
                "to for this file."
            )

    if "7_mux" in state.get("steps_completed", []):
        log.info("Step 7 — ↩  already complete; re-using %s.", out_path.name)
        if not out_path.exists():
            raise RuntimeError(
                f"Step 7 is marked complete but {out_path} is missing.  "
                "Delete the job directory and re-run from scratch."
            )
        return out_path

    if not audio_encoded_path.exists():
        raise RuntimeError(
            f"Step 7: encoded audio not found at {audio_encoded_path} — "
            "did Step 6b (encode) complete?"
        )
    if not video_path.exists():
        raise RuntimeError(f"Step 7: original video not found at {video_path}.")

    verify_stem_before_reuse(
        audio_encoded_path,
        float(state.get("total_duration_sec", 0.0)),
        state.get("encode", {}).get("audio_encoded_sha256"),
        log,
        label="audio_encoded.mka",
        written_by="Step 6b (encode)",
        regenerate_hint=(
            "This is cheap to fix: re-run with --redo-step 6b_encode "
            "(or 7_mux, if only the mux itself needs redoing) -- see "
            "pipeline.py's --redo-step --help."
        ),
    )

    tool = "mkvmerge" if out_format == "mkv" else "ffmpeg"
    log.info("Step 7 — mux encoded audio into video  (format=%s, tool=%s)", out_format, tool)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_output_path(out_path)
    tmp_path.unlink(missing_ok=True)  # leftover from a previous interrupted attempt, if any

    subtitle_sources = subtitle_sources or []
    # Per-source, not a single global gate any more -- each SrtSource
    # already carries its own resolved embed flag (that engine's own
    # alignment.engines.<name>.embed_subtitle -- see
    # steps/transcript_srt.py's own docstring). This step never reads
    # alignment.engines.* itself; it just respects whichever sources
    # arrive with embed=True.
    embeddable_sources = [s for s in subtitle_sources if s.embed]
    embed_subtitles = bool(embeddable_sources)
    track_lang = str(cfg_get(cfg, "transcript_srt", "track_language"))

    if out_format == "mkv":
        cmd = ["mkvmerge", "-o", str(tmp_path)]

        # Replace, don't duplicate: if a previous run of this pipeline
        # already embedded one of our own tracks into video_path (e.g. it
        # was pointed back at this job's own prior output -- see module
        # docstring), passing it through unfiltered alongside the fresh
        # copy added below would leave two tracks with the same name in
        # the result. Matched by exact name against each source's own
        # CURRENT track_name (steps/transcript_srt.py -- built from
        # transcript_srt.track_name_prefix plus that source's label) --
        # see _probe_subtitle_tracks()'s own docstring for what changing
        # that setting between runs does to this matching.
        if embed_subtitles:
            configured_names = {s.track_name for s in embeddable_sources}
            existing_subs = _probe_subtitle_tracks(video_path, log)
            all_sub_ids = [t["id"] for t in existing_subs]
            replace_ids = [
                t["id"] for t in existing_subs
                if t.get("properties", {}).get("track_name") in configured_names
            ]
            if replace_ids:
                log.info(
                    "  Replacing %d existing track(s) in %s matching this "
                    "pipeline's own subtitle track name(s) (track ID(s) %s) "
                    "rather than adding duplicates.",
                    len(replace_ids), video_path.name,
                    ", ".join(str(i) for i in replace_ids),
                )
                keep_ids = [i for i in all_sub_ids if i not in replace_ids]
                if keep_ids:
                    cmd += ["--subtitle-tracks", ",".join(str(i) for i in keep_ids)]
                else:
                    cmd += ["--no-subtitles"]

        cmd += ["--no-audio", str(video_path), str(audio_encoded_path)]

        for source in embeddable_sources:
            # source.job_dir_path as-is -- whatever style
            # transcript_srt.karaoke wrote, mkvmerge preserves it
            # byte-for-byte (S_TEXT/UTF8) -- see module docstring.
            cmd += [
                "--language", f"0:{track_lang}",
                "--track-name", f"0:{source.track_name}",
                "--default-track-flag", "0:no",
                str(source.job_dir_path),
            ]

        # 0 = clean, 1 = succeeded with warnings, 2 = real failure --
        # see utils.run_cmd's ok_exit_codes docstring.
        run_cmd(cmd, log, ok_exit_codes=frozenset({0, 1}))
    else:
        # mp4 never passes the original video's own subtitle tracks
        # through at all (see the "Subtitle/chapter/attachment
        # preservation for mp4 output" section above) -- video_path is
        # always the pristine original here, never this pipeline's own
        # prior mp4 output, so there is nothing of ours already inside it
        # to detect or replace. Every mux starts from scratch and only
        # ever contains what's explicitly added below.
        #
        # mov_text strips <font> styling outright (confirmed directly --
        # see module docstring), so mp4 always gets each source's plain
        # fallback text, written here to scratch files for ffmpeg's own
        # -i (it needs real paths, not strings) -- never
        # source.job_dir_path itself, even on the rare run where that
        # already happens to be plain (transcript_srt.karaoke: false).
        # One code path either way, correct in both cases, and nothing
        # left behind afterward.
        subtitle_tmps: list[Path] = []
        for source in embeddable_sources:
            if not source.mp4_fallback_text:
                continue
            tmp_srt = job_dir / f".transcript_mp4_embed_{source.key}.srt"
            tmp_srt.write_text(source.mp4_fallback_text, encoding="utf-8")
            subtitle_tmps.append(tmp_srt)

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path),
            "-i", str(audio_encoded_path),
        ]
        for tmp_srt in subtitle_tmps:
            cmd += ["-i", str(tmp_srt)]

        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        for i in range(len(subtitle_tmps)):
            cmd += ["-map", f"{i + 2}:s:0"]
        cmd += ["-map_chapters", "0", "-c:v", "copy", "-c:a", "copy"]
        if subtitle_tmps:
            cmd += ["-c:s", "mov_text"]
            for i, source in enumerate(
                s for s in embeddable_sources if s.mp4_fallback_text
            ):
                cmd += [
                    f"-metadata:s:s:{i}", f"language={track_lang}",
                    f"-metadata:s:s:{i}", f"handler_name={source.track_name}",
                    # Best-effort only -- see module docstring on why
                    # this isn't guaranteed to actually suppress
                    # auto-selection for mp4 the way its mkv counterpart is.
                    f"-disposition:s:{i}", "0",
                ]
        cmd += [
            "-avoid_negative_ts", "make_zero",
            "-f", "mp4",
            str(tmp_path),
        ]
        try:
            run_cmd(cmd, log)
        finally:
            for tmp_srt in subtitle_tmps:
                tmp_srt.unlink(missing_ok=True)

    finalize_output(tmp_path, out_path)
    log.info("  ✓  %s  (%s)", out_path.name, fmt_size(out_path))

    if not keep_intermediate(cfg, correction_artifact=False):
        _unlink_if(audio_encoded_path, log)

    state = read_job(job_dir)
    # output_path is a *directory*, not the full path to the file --
    # output_filename (bare, below) already carries the filename. Same
    # host-vs-container reasoning as pipeline.py's input_path: prefer
    # AC_OUTPUT_HOST_DIR (the real, host-navigable directory this file
    # was actually written to -- see utils.paths_banner()) and fall back
    # to this container's own view of it (output_dir, normally /output)
    # when that env var never reached the container.
    output_host_dir = cfg_get(cfg, "paths", "output_host_dir", default=None)
    output_dir_display = (
        fmt_dir(output_host_dir) if output_host_dir else fmt_dir(output_dir)
    )
    state["mux"] = {
        "output_path":     output_dir_display,
        "output_filename": out_path.name,
        "format": out_format,
        "tool":   tool,
        "subtitles_embedded": [s.key for s in embeddable_sources],
    }
    write_job(job_dir, state)
    mark_step_done(job_dir, "7_mux")

    log.info("  ✓  Step 7 complete.  Final output: %s", out_path)
    return out_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _probe_subtitle_tracks(video_path: Path, log: logging.LoggerAdapter) -> list[dict]:
    """
    List video_path's own subtitle tracks via `mkvmerge -J` (mkvmerge's
    machine-readable identify mode) -- used by mux()'s mkv branch to find
    and replace a previous run's own embedded track(s) rather than
    duplicating them (see that branch's comment, and this module's
    docstring's subtitle-embedding section).

    Returns [] -- "found nothing to replace," the same as before this
    detection existed -- rather than raising, whenever video_path can't
    be identified this way at all (not a Matroska file, mkvmerge itself
    unavailable, malformed/unexpected output, a timeout, ...). This is a
    best-effort probe in service of a nice-to-have (avoiding a duplicate
    debug subtitle track on re-mux); it should never be the reason the
    actual mux -- and with it, the censored video the user is waiting on
    -- fails to produce anything at all.
    """
    try:
        result = subprocess.run(
            ["mkvmerge", "-J", str(video_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            log.debug(
                "  Could not identify %s's existing subtitle tracks "
                "(mkvmerge -J exited %d) -- treating it as having none.",
                video_path.name, result.returncode,
            )
            return []
        info = json.loads(result.stdout)
        return [t for t in info.get("tracks", []) if t.get("type") == "subtitles"]
    except Exception as exc:
        log.debug(
            "  Could not identify %s's existing subtitle tracks (%s) -- "
            "treating it as having none.", video_path.name, exc,
        )
        return []


def _output_path(video_path: Path, output_dir: Path, cfg: dict, out_format: str) -> Path:
    """
    Imported directly by batch_plan.py, not just called from mux() below --
    the leading underscore here is a "not part of steps.mux's own public
    step-function API" marker (that's mux() alone), not "nothing outside
    this file may import it." Treat this signature as a two-caller
    contract when changing it.

    Build the final output filename, per output.naming_style:

    plex_edition (default) -- a Plex-friendly {edition-Name} tag (see
      https://support.plex.tv/articles/multiple-editions/). Movies and TV
      episodes follow *different Plex conventions entirely*, not just a
      different insertion point in the same filename:

      Movies -- the tag is inserted right after the "(YYYY)" release-year
        portion of the filename if one is present, so Plex shows the
        censored file as a selectable Edition of the same movie instead
        of an unrelated second item:
          "Movie (1986).sd.hevc.mkv" -> "Movie (1986) {edition-Hushed}.sd.hevc.mkv"
        Falls back to appending the tag at the very end -- still valid
        Plex syntax -- if no "(YYYY)" pattern is found at all.

      TV episodes -- Plex has no per-episode edition concept
        (https://support.plex.tv/articles/multiple-editions-tv-shows/
        says so outright). Instead, the whole *show* gets a sibling
        directory: "Show (Year)" -> "Show (Year) {edition-Hushed}", with
        the season/specials structure and episode filenames mirrored
        underneath completely unchanged. That redirection is decided and
        carried out by hush.sh, in bash, before this function is ever
        called (see hush.sh's redirect_for_tv_edition()) -- it depends on
        real, surrounding directory names this function has no way to
        see: `output_dir` here is always /output, the container's own
        opaque mount point name -- never the real host directory hush.sh
        actually pointed it at, regardless of what that is. Checking
        `output_dir` itself for the edition tag (an earlier bug) can
        never see it there even when hush.sh redirected correctly; the
        real host directory only reaches this function via
        cfg["paths"]["output_host_dir"] (populated from AC_OUTPUT_HOST_DIR
        -- see utils.load_config()'s docstring and mux()'s own
        output_host_dir lookup just below, for the same reasoning applied
        to logging). By the time this runs, whichever directory
        AC_OUTPUT_HOST_DIR names IS already the correct one either way --
        this function only needs to tell the two cases apart to decide
        the *filename*: if the edition tag appears anywhere in that real
        host path, hush.sh already redirected it here for a TV episode,
        so the filename needs no tag at all, just the out_format
        extension swap, same as any file. Otherwise this is a movie (or a
        TV episode under naming_style: suffix, which never redirects --
        see below) and gets the "insert after year" treatment above.

    suffix -- the original v1 behaviour: a plain suffix appended before
      the extension, no Plex Edition semantics and no TV redirection
      either -- there's no Plex Edition convention to follow for a style
      that isn't representing an Edition in the first place.
        "movie.mkv" -> "movie_censored.mkv"

    Path(name).stem strips only the final extension, so a filename like
    "movie.sd.hevc.mkv" keeps everything after the first dot intact in
    either style above.
    """
    naming_style = str(cfg_get(cfg, "output", "naming_style")).lower()
    stem = Path(video_path.name).stem

    if naming_style == "plex_edition":
        edition_name = str(cfg_get(cfg, "output", "edition_name"))
        tag = f"{{edition-{edition_name}}}"

        # See this function's docstring above: output_dir is always the
        # opaque /output mount point, never informative here -- the real
        # host directory (which hush.sh may have redirected for a TV
        # episode) only reaches us via AC_OUTPUT_HOST_DIR, already merged
        # into cfg by utils.load_config(). Empty/missing (e.g. this cfg
        # came from batch_plan.py's own planning pass, which never sets
        # AC_OUTPUT_HOST_DIR at all) correctly falls through to the movie
        # branch below -- batch_plan.py never calls this for a TV episode
        # in the first place, so that's the only case that reaches here.
        real_output_dir = cfg_get(cfg, "paths", "output_host_dir", default=None) or ""

        if re.search(r"\{edition-[^}]+\}", real_output_dir):
            # TV episode, already redirected to its show's sibling
            # edition directory by hush.sh -- the directory carries the
            # tag, so the filename itself is untouched.
            new_stem = stem
        else:
            year_match = re.search(r"\(\d{4}\)", stem)
            if year_match:
                new_stem = f"{stem[:year_match.end()]} {tag}{stem[year_match.end():]}"
            else:
                new_stem = f"{stem} {tag}"
    elif naming_style == "suffix":
        suffix = str(cfg_get(cfg, "output", "suffix"))
        new_stem = f"{stem}{suffix}"
    else:
        raise RuntimeError(
            f"Step 7: unknown output.naming_style '{naming_style}' "
            "(expected 'plex_edition' or 'suffix')."
        )

    return output_dir / f"{new_stem}.{out_format}"


def _unlink_if(path: Path, log: logging.LoggerAdapter) -> None:
    """Delete a file if it exists; no-op and no error if absent."""
    if path.exists():
        path.unlink()
        log.debug("  Removed intermediate: %s", path.name)
