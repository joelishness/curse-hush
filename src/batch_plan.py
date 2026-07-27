#!/usr/bin/env python3
"""
profanity-hush — batch planning helper (used by hush.sh --batch)

Not meant to be run by hand. Given a directory of video files, decides
which ones still need processing and prints their paths to stdout
(NUL-terminated, one per file) for hush.sh's batch loop to consume --
everything else (the human-readable summary, warnings) goes to stderr,
so hush.sh's `mapfile -d ''` only ever sees the file list on stdout.

"Already done" is judged by whether the file this pipeline would produce
for a given input already exists in the output directory -- not by the
local jobs store. Two reasons, both from the batch-processing request
this was built for:

  1. A library gets built up across machines (hush.sh's own --jobs flag
     already exists because job history isn't assumed to travel with the
     media) -- the job that produced a given episode's output may simply
     never have run on whatever machine is doing THIS batch. The output
     file sitting in the folder is the one signal that's actually
     portable.
  2. It's the same thing a person would do by eye, looking at the
     folder: "does the censored version already exist?" -- not "does
     job.json somewhere say it was built?"

The predicted output path is computed with the exact same
steps.mux._output_path() Step 7 itself uses (see the note on that
function) -- not a second implementation -- so this can never drift out
of sync with what Step 7 actually names a file, including the TV/movie-
aware tag placement and the ellipsis-safe dot-boundary logic it depends
on. The local jobs store IS still consulted, but only as a secondary,
informational check (_job_already_muxed()) for the one thing a pure
file-existence check can't catch on its own: a job that DID complete
Step 7 on *this* machine but whose output is no longer where Step 7 left
it (moved, renamed, deleted by hand). That's surfaced as a warning, not
acted on -- the file is queued either way, and pipeline.py's own Step 7
resume-check will either reuse the file correctly or fail with its own
actionable message ("Step 7 is marked complete but ... is missing.").

Caveat worth knowing about, not solved here: if output.edition_name (or
output.suffix) ever changes, files censored under the *old* name are
still correctly recognised as "an output, not a source" (see
_is_already_output()'s "{edition-...}" regex, which matches any edition
name) for plex_edition style — but a matching *suffix*-style output only
recognises the currently configured suffix, since an arbitrary suffix
string has no self-identifying marker the way "{edition-...}" does.
"""
import argparse
import re
import sys
from pathlib import Path

import utils
from utils import cfg_get, compute_job_id, find_job_dir, setup_logging, step_logger
from steps.mux import _output_path

# Matches pipeline.py's own OUTPUT_DIR / CONFIG_PATH constants -- batch mode
# mounts the same way a single-file run does, just at directory scope.
OUTPUT_DIR  = Path("/output")
CONFIG_PATH = Path("/config/config.yaml")

# Deliberately a fixed list rather than a new CLI flag -- easy to extend
# here if a format this library actually uses turns up missing.
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".m4v", ".avi", ".mov"}


def discover_videos(root: Path, recursive: bool) -> "list[Path]":
    """All video files directly under `root` (recursive if requested),
    sorted for deterministic, read-top-to-bottom processing order."""
    walker = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p for p in walker
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _is_already_output(path: Path, cfg: dict) -> bool:
    """
    True if `path` itself looks like something this pipeline already
    produced -- i.e. it must never be treated as a candidate *input*,
    regardless of whether ITS OWN predicted output happens to exist.

    Without this, scanning a folder that already has both the original
    and a previously-censored sibling sitting side by side (exactly what
    a batch re-run of a partially-done season directory looks like)
    would pick up the censored file too and try to censor it a second
    time.
    """
    stem = path.stem
    naming_style = str(cfg_get(cfg, "output", "naming_style")).lower()
    if naming_style == "plex_edition":
        # Any "{edition-...}" tag, not just the currently configured
        # edition_name -- so a file tagged under a since-changed name is
        # still recognised as an output. See module docstring's caveat
        # for the one direction this doesn't cover.
        return re.search(r"\{edition-[^}]+\}", stem) is not None
    if naming_style == "suffix":
        suffix = str(cfg_get(cfg, "output", "suffix"))
        return stem.endswith(suffix)
    return False


def _job_already_muxed(video: Path, jobs_dir: Path) -> bool:
    """
    Secondary, informational-only signal -- see module docstring. Never
    used to decide skip vs. queue by itself, only to warn when it
    disagrees with the primary signal (the output file's own existence).
    """
    try:
        job_id = compute_job_id(video)
    except OSError:
        return False
    job_dir = find_job_dir(jobs_dir, job_id)
    if job_dir is None:
        return False
    state = utils.read_job(job_dir)
    return "7_mux" in state.get("steps_completed", [])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="batch_plan.py",
        description=(
            "profanity-hush -- batch planning helper. Prints the "
            "container-side paths of video files under INPUT_DIR that "
            "still need processing (NUL-terminated, stdout only); "
            "everything else goes to stderr. Called by hush.sh --batch; "
            "not meant to be run directly."
        ),
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--config", default=str(CONFIG_PATH), metavar="PATH")
    args = parser.parse_args()

    cfg = utils.load_config(args.config)
    utils.validate_config(cfg)
    log_level = cfg_get(cfg, "output", "log_level")
    setup_logging(log_level)
    log = step_logger("batch-plan")

    out_format = str(cfg_get(cfg, "output", "format")).lower()
    if out_format not in ("mkv", "mp4"):
        log.error("Unknown output.format '%s' (expected 'mkv' or 'mp4').", out_format)
        sys.exit(1)

    jobs_dir = Path(cfg_get(cfg, "storage", "jobs_dir"))

    root = args.input_dir
    if not root.is_dir():
        log.error("Not a directory: %s", root)
        sys.exit(1)

    candidates = discover_videos(root, args.recursive)
    sources: "list[Path]" = []
    for p in candidates:
        if _is_already_output(p, cfg):
            log.debug("  (already an output, not a source) %s", p.relative_to(root))
            continue
        sources.append(p)
    skipped_as_output = len(candidates) - len(sources)

    queued: "list[Path]" = []
    already_done: "list[Path]" = []
    warnings = 0

    for video in sources:
        rel_dir = video.parent.relative_to(root)
        out_dir = OUTPUT_DIR if rel_dir == Path(".") else OUTPUT_DIR / rel_dir
        try:
            predicted = _output_path(video, out_dir, cfg, out_format)
        except RuntimeError as exc:
            # A bad naming_style affects every file identically -- fail
            # the whole plan once, up front, instead of repeating the
            # same error for each of potentially dozens of files.
            log.error("%s", exc)
            sys.exit(1)

        if predicted.exists():
            already_done.append(video)
            log.debug("  (already done) %s  ->  %s", video.relative_to(root), predicted)
            continue

        if _job_already_muxed(video, jobs_dir):
            warnings += 1
            log.warning(
                "  ⚠  %s — a job on this machine already completed Step 7 "
                "for this exact file, but %s is missing. Queuing it "
                "anyway; if the run stops with \"Step 7 is marked "
                "complete but ... is missing\", delete that job's "
                "directory under %s and re-run.",
                video.relative_to(root), predicted, jobs_dir,
            )

        log.debug("  (queued) %s", video.relative_to(root))
        queued.append(video)

    log.info(
        "Found %d video file(s) under %s%s.",
        len(candidates), root, " (recursive)" if args.recursive else "",
    )
    if skipped_as_output:
        log.info(
            "  %d already look like a censored output themselves — not treated as source.",
            skipped_as_output,
        )
    log.info("  %d already have a censored output in place — skipping.", len(already_done))
    log.info("  %d queued for processing.", len(queued))
    if warnings:
        log.info(
            "  %d warning(s) above — local job history disagrees with the output folder.",
            warnings,
        )

    for video in queued:
        sys.stdout.write(str(video))
        sys.stdout.write("\0")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
