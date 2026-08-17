"""
profanity-hush — Step 6c: export aligned transcripts as SRT subtitles

Why this exists: figuring out *why* a word got muted at the wrong moment,
or why two alignment stages disagree, normally means cross-referencing
the output video against a transcript against the original audio by
hand, across three separate tools — exactly the tedious process
docs/timestamp-drift-investigation.md's whole worked example walks
through. This step makes that a "turn a subtitle track on" problem
instead: each available transcript — every alignment stage's own, plus
the authoritative/censoring-relevant result — becomes its own subtitle
track showing every word recognized there, at exactly the timestamp it
gives it, so a mistimed mute — or a disagreement between stages — is
immediately visible as a mismatch between what the track says is being
spoken and what's actually audible.

Input  : transcript_{N}.json for each alignment stage number job.json's
         "alignment_stages" legend lists (steps/transcribe.py,
         steps/merge.py), plus transcript.json, the authoritative one.
Output : transcript_{N}.srt for each stage, plus transcript.srt for the
         authoritative result — saved in the job directory, and —
         transcript_srt.write_sidecar — a copy of each next to the output
         video too. Also returned in memory (see export_srt()'s
         docstring) for steps/mux.py to embed.
Marks '6c_transcript_srt' done.

Does this belong in the --skip-index/--add-interval/--redo-review
correction-mode unmark list (pipeline.py) alongside 5_mute/6_recombine/
6b_encode/7_mux? As of transcript_srt.hush (see below), yes, for the
authoritative source specifically: hushing needs censor_log.json, which
is exactly what those corrections change, so the authoritative SRT can
now go stale on one the same way 5_mute/6_recombine/6b_encode/7_mux's own
outputs can. (Before hush existed, this step read only transcript.json/
transcript_{N}.json — what was recognized, never what got muted — so it
genuinely couldn't go stale on a content correction; per-stage sources
still only ever read transcript_{N}.json and are still exactly as
stale-proof as before.) See pipeline.py's own docstring and
_cascade_steps() for the unmark list itself.

Failure handling is deliberately NOT the same as every other step's: an
unexpected failure here logs a warning and lets the pipeline continue
with no subtitle tracks this run, rather than failing the whole job (see
pipeline.py's call site). This is a debugging aid layered on top of the
actual censoring pipeline, not part of it — by the time this step runs,
every genuinely expensive step (1a through 6b) has already succeeded, and
there's no good reason a bug in grouping/rendering a subtitle file should
cost someone the censored video they've been waiting hours for. Config
validation failures are the one exception and stay fatal, same as every
other setting (see utils.validate_config(), and, specific to hush.mode/
hush.simple_style, validate_hush_config() below) — both are called, and
so both already have to have passed, before Step 1a even starts, not
here.

── Hushing the authoritative SRT ─────────────────────────────────────────

transcript_srt.hush (config.yaml) redacts words known to have been muted
in the actual audio — censor_log.json, Step 5's final, post-review/post-
correction record — from the authoritative ("final") source's own SRT
text, so a viewer with that track on doesn't get to read the exact word
they can't hear (autocensor-design.md §13.1.1 has the original problem
statement). Applied ONLY to the authoritative source, never to a
per-stage comparison one: censor_log.json's word timing is only ever
computed against transcript.json (steps/matching.py's find_matches()),
and this whole module exists because a per-stage transcript's timing for
nominally "the same" word can disagree with that by several real seconds
(docs/timestamp-drift-investigation.md) — reusing censor_log.json's
timing against a per-stage source risks a silently wrong redaction, with
no way to tell from the output alone that it happened. The per-stage
sources stay a truthful, unmodified debugging aid instead, exactly as
before hush existed. See _apply_hush() below, and config.yaml's own
transcript_srt.hush comment, for the rest.

── Does SRT actually support karaoke-style word highlighting? ───────────

Short answer: yes, but only by working around what the format doesn't
have, and it doesn't survive everywhere the resulting file might end up.
Confirmed directly (a tagged file round-tripped through each tool below),
not just read about:

  - SRT has no official specification and no native concept of timing
    *within* one cue — a cue is one fixed block of text for one fixed
    time range, full stop. What it does have, as a widely-implemented
    (if unofficial) extension, is a small set of HTML-like tags —
    <b>, <i>, <u>, and <font color="#RRGGBB"> — that most players and
    muxers pass through as-is.
  - There is therefore no way to change a word's color partway through a
    cue that's already on screen. The only way to get a "the line stays
    up, the highlight moves" effect is to fake it: emit one cue per
    WORD, each covering just that word's own timing, re-stating the
    group's full text every time with only the current word wrapped in
    <font color>. That's exactly what karaoke=True below does — it's a
    real, if brute-force, use of plain SRT, not a nonstandard extension
    of the format.
  - Confirmed: `ffmpeg -i tagged.srt out.ass` converts
    <font color="#FFD400">word</font> into the equivalent libass/ASS
    override, `{\\c&HD4FF&}word{\\c}` — the color survives, byte-exact,
    through the exact code path both ffmpeg's own `subtitles` burn-in
    filter and mpv use to render text. Any SRT-capable player pointed at
    a transcript SRT directly (VLC, mpv, and others — the Firecore/Infuse
    forum is one of many explicitly confirming this exact tag) should
    show real, correct karaoke highlighting.
  - Confirmed: muxing an SRT into an MKV with mkvmerge and extracting the
    resulting track back out again reproduces the <font color> tags
    byte-for-byte (codec S_TEXT/UTF8 — mkvmerge stores SRT as SRT, no
    lossy re-encoding of the text at all). This is why output.format: mkv
    (config.yaml's default) embeds a transcript SRT directly, as-is, with
    no fallback needed.
  - NOT confirmed to survive, and in fact confirmed NOT to: mov_text,
    the only way ffmpeg can put a text subtitle track inside an MP4
    container at all. Round-tripping the exact same tagged file through
    `-c:s mov_text` and back out strips every tag — <font>, <b>, <i>,
    <u>, all of it — leaving only the plain words behind. This is a
    structural limitation of mov_text itself, not a flag this module
    forgot to pass. Embedding the karaoke rendering as-is into an mp4
    would therefore still "work" in the sense that ffmpeg wouldn't
    error, but every one-cue-per-word group would render as several
    back-to-back, visually IDENTICAL redraws of the same plain line —
    at best pointless, at worst looking like a player bug. See
    steps/mux.py for how it avoids this (embedding the plain,
    one-cue-per-group rendering for mp4 output specifically, regardless
    of transcript_srt.karaoke).
  - Also confirmed: an empty .srt (zero cues) is accepted by mkvmerge
    but makes ffmpeg refuse the whole mux outright ("Invalid data found
    when processing input"). _export_one_source() below returns None
    whenever a transcript has no word with usable timing at all, so
    steps/mux.py never hands ffmpeg an empty file.

Burning the karaoke rendering permanently into the video's pixels (which
would sidestep mov_text entirely) was considered and rejected: it would
require re-encoding the video stream, breaking the bit-for-bit video copy
this whole pipeline is built around (README's "How it works" step 6;
steps/mux.py's own docstring) just to add a track most viewers of the
actual censored film would never want burned in permanently in the first
place. Soft-embedding (a selectable, non-default track) is the only
option consistent with that guarantee — see steps/mux.py.

── Why one SRT per stage, plus one more for the authoritative result ────

Earlier versions of this step read a single, generic transcript.json —
whatever alignment.backend happened to produce, per segment, including
any per-segment fallback baked invisibly into the same file (see
steps/transcribe.py's cascade). That's still exactly what transcript.json
is for matching/muting (steps/matching.py, steps/mute.py) — unchanged.
But it makes a poor comparison tool on its own: there's no way to tell,
from transcript.json alone, whether a given word's timing came from the
authoritative stage or a fallback, which is precisely the distinction
someone troubleshooting a disagreement between stages needs to see.
Stage-numbered sources (transcript_1.json, transcript_2.json, ... — see
steps/merge.py) fix that: each is a clean, honestly-gapped view of what
one specific stage actually produced, independent of which one was
authoritative for censoring purposes. Numbered, not named after a
specific tool, for the same reason steps/transcribe.py's own
_ALIGNMENT_STAGES registry is: a future stage 3 (a different aligner
entirely) needs no changes here — job.json's own "alignment_stages"
legend is read fresh each run, so a new stage number just starts showing
up as one more SRT the moment steps/transcribe.py starts producing it.

That same authoritative/per-stage split governs two more things below,
for related but distinct reasons documented at each: karaoke rendering
(transcript_srt.karaoke — a per-stage debugging aid, never applied to
the authoritative SRT, which stays traditional grouped subtitles
regardless of that setting; see _export_one_source()) and hush
redaction (transcript_srt.hush — the reverse: applied only to the
authoritative SRT's own text, never to a per-stage one; see
_apply_hush()). Both consistently treat the per-stage SRTs as the raw,
unedited debugging view and the authoritative one as the track shaped
for an actual viewer.
"""

import json
import logging
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils import (
    cfg_get,
    finalize_output,
    fmt_size,
    mark_step_done,
    read_job,
    step_logger,
    tmp_output_path,
    write_job,
)

# A word ending in one of these (optionally followed by a closing quote/
# paren/bracket) always ends its group, regardless of the word-count/
# char-count/gap/duration knobs in config.yaml -- this is what makes
# grouping actually read like traditional subtitle lines broken at
# sentence boundaries, rather than arbitrary fixed-size chunks. Not
# configurable on purpose: there's no real tradeoff to expose here the
# way there is for, say, max_words_per_group.
_SENTENCE_END_RE = re.compile(r"[.?!]+[\"'\u2019\u201d)\]]*$")

# Floor for a single karaoke cue's duration, in seconds. A stage's word
# list should always be strictly time-ordered with non-overlapping spans
# (a forced/CTC aligner can't place two words in the same instant) --
# this only guards against a same-timestamp or out-of-order anomaly
# producing a zero- or negative-duration SRT cue, which some parsers
# reject outright.
_MIN_CUE_SEC = 0.05


@dataclass
class SrtSource:
    """
    One transcript's SRT output for this job — everything steps/mux.py
    needs to embed it, and everything pipeline.py needs to log it as a
    kept output.

    key                 — "1", "2", ... for one alignment stage's own
                           transcript (see job.json's "alignment_stages"
                           legend, steps/transcribe.py), or "final" for
                           the authoritative/censoring-relevant one. Used
                           for logging and job.json bookkeeping only --
                           not part of any filename this module writes
                           (see job_dir_path/sidecar_path below).
    label               — human-readable, e.g. "Stage 2 (MFA)" or
                           "Final (authoritative)".
    job_dir_path        — job_dir/transcript_{N}.srt for a stage, or
                           job_dir/transcript.srt for the authoritative
                           one. Always written (in whichever style
                           transcript_srt.karaoke selects) whenever this
                           source exists at all.
    sidecar_path        — a copy of the same content next to the output
                           video, named per config.yaml's convention.
                           None when transcript_srt.write_sidecar is false.
    mp4_fallback_text   — a plain (no <font> tags) one-cue-per-group
                           rendering, always plain regardless of
                           transcript_srt.karaoke — steps/mux.py uses this
                           for output.format: mp4 specifically (see this
                           module's docstring for why mp4 can't show the
                           karaoke version at all).
    track_name          — transcript_srt.track_name_prefix plus this
                           source's own label, for steps/mux.py's
                           embedded-track metadata.
    """
    key:                str
    label:              str
    job_dir_path:       Path
    sidecar_path:       Optional[Path]
    mp4_fallback_text:  str
    track_name:         str


def export_srt(
    job_dir: Path,
    output_video_path: Path,
    cfg: dict,
    log: Optional[logging.LoggerAdapter] = None,
) -> list[SrtSource]:
    """
    Step 6c: render each available transcript into its own SRT — every
    alignment stage job.json's "alignment_stages" legend (steps/
    transcribe.py) lists a transcript_{N}.json for (steps/merge.py), plus
    transcript.json, the authoritative/censoring-relevant one.

    output_video_path — the final output video's path (e.g. from
    steps.mux._output_path(), called early by pipeline.py for exactly
    this purpose). Used only to derive sidecar filenames
    (<output_video_path.stem>.<language>.<N-or-nothing>.srt, written next
    to it) — the file itself need not exist yet when this runs, since
    Step 6c always runs before Step 7 actually produces it.

    Returns one SrtSource per stage that had a transcript_{N}.json with
    at least one word with usable alignment timing, in ascending stage
    order, followed by one more for transcript.json if it qualifies too
    — so anywhere from 0 to (number of registered stages + 1) entries.
    Empty whenever transcript_srt.enabled is false, or nothing at all
    had any usable timing.
    """
    if log is None:
        log = step_logger("srt")

    state = read_job(job_dir)
    done  = state.get("steps_completed", [])
    already_done = "6c_transcript_srt" in done

    if not bool(cfg_get(cfg, "transcript_srt", "enabled")):
        log.info("Step 6c — transcript_srt.enabled is false — skipping.")
        mark_step_done(job_dir, "6c_transcript_srt")
        return []

    # Discovered from job.json's own "alignment_stages" legend (written
    # by steps/transcribe.py), not a hardcoded list here -- see this
    # module's own docstring ("Why one SRT per stage...") for why: a
    # future stage 3 just starts appearing the moment that legend lists
    # it, with nothing in this function needing to change.
    candidates: list[tuple[str, str, Path, str]] = [
        (
            str(stage["number"]),
            f"Stage {stage['number']} ({stage['label']})",
            job_dir / f"transcript_{stage['number']}.json",
            f"_{stage['number']}",
        )
        for stage in state.get("alignment_stages", [])
    ]
    candidates.append(("final", "Final", job_dir / "transcript.json", ""))

    sources: list[SrtSource] = []
    for key, label, transcript_path, suffix in candidates:
        if not transcript_path.exists():
            continue
        source = _export_one_source(
            job_dir, output_video_path, key, label, suffix,
            transcript_path, already_done, cfg, log,
        )
        if source is not None:
            sources.append(source)

    mark_step_done(job_dir, "6c_transcript_srt")

    if not sources:
        log.warning(
            "Step 6c — no transcript (any alignment stage, or the "
            "authoritative transcript.json) had any word with usable "
            "alignment timing — nothing to export."
        )
    log.info("  ✓  Step 6c complete.")
    return sources


def _export_one_source(
    job_dir: Path,
    output_video_path: Path,
    key: str,
    label: str,
    suffix: str,
    transcript_path: Path,
    already_done: bool,
    cfg: dict,
    log: logging.LoggerAdapter,
) -> Optional[SrtSource]:
    """
    Process one transcript (one alignment stage's own, or the
    authoritative one) into an SrtSource — shared logic for every source
    export_srt() finds, called once per candidate.

    suffix is the job_dir_path filename tag: "" for the authoritative
    transcript.json (→ transcript.srt), or f"_{N}" for stage N (→
    transcript_N.srt). The sidecar filename (<video>.eng.N.srt /
    <video>.eng.srt) uses a dot-separated variant derived from `key`
    instead -- see config.yaml's own documentation of write_sidecar for
    the full naming convention and why the two differ.

    Returns None if transcript_path has no word with usable alignment
    timing at all (nothing to place on a timeline either way) — logged,
    not an error.

    Karaoke rendering (transcript_srt.karaoke) is only ever considered
    for a per-stage source (key != "final") — the authoritative SRT is
    always the plain, one-cue-per-group rendering, regardless of that
    setting's own value. See config.yaml's karaoke comment for why: in
    short, the authoritative track is the one meant for an actual viewer
    (and, per the hush section below, the one whose text can already
    read "s***" instead of the real word) — flickering per-word color on
    every line reads as a debugging aid there, not a viewing feature; the
    per-stage tracks are the debugging aid, and keep the precision.
    """
    srt_out = job_dir / f"transcript{suffix}.srt"

    data      = json.loads(transcript_path.read_text())
    all_words = data.get("words", [])
    # Same rule steps/matching.py already applies to transcript.json: a
    # word with no alignment timing has nothing to place on a subtitle
    # timeline, exactly as it has nothing to mute (see that module's
    # docstring).
    words = [w for w in all_words if w.get("start") is not None and w.get("end") is not None]

    if not words:
        log.warning(
            "Step 6c — %s has %d word(s), none with usable alignment "
            "timing — nothing to export for %s.",
            transcript_path.name, len(all_words), label,
        )
        return None

    # Only the authoritative source's own words are ever candidates for
    # hushing — censor_log.json's word timing is only ever computed
    # against transcript.json (steps/matching.py's find_matches()), so a
    # per-stage transcript_N.json (key != "final") is left exactly as
    # recognized. Done here, before grouping/wrapping below, so a
    # replacement's actual on-screen length (not the original word's)
    # is what line-wrapping and sentence-boundary detection see — see
    # config.yaml's transcript_srt.hush comment for the full reasoning.
    if key == "final":
        words = _apply_hush(words, job_dir, cfg, log)

    track_prefix = str(cfg_get(cfg, "transcript_srt", "track_name_prefix"))
    track_name   = f"{track_prefix} \u2014 {label}"

    sidecar_path: Optional[Path] = None
    if bool(cfg_get(cfg, "transcript_srt", "write_sidecar")):
        track_language = str(cfg_get(cfg, "transcript_srt", "track_language"))
        # Dot-separated, not the underscore job_dir_path uses -- Plex/
        # Kodi's own "<basename>.<language>.<flag>.ext" convention (see
        # config.yaml's write_sidecar comment) needs the stage number as
        # its own dot-delimited segment, e.g. ".eng.2.srt", not ".eng_2.srt".
        sidecar_suffix = f".{key}" if key != "final" else ""
        sidecar_path = output_video_path.parent / (
            f"{output_video_path.stem}.{track_language}{sidecar_suffix}.srt"
        )
        # Step 6c runs before Step 7 (steps/mux.py), which is normally
        # what creates OUTPUT_DIR -- ensure it exists here too, rather
        # than assuming it already does, so a sidecar write doesn't fail
        # on a job's very first run.
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    prepared   = _prepare_groups(words, cfg)
    color      = str(cfg_get(cfg, "transcript_srt", "karaoke_color"))
    # Always needed by the caller (steps/mux.py, for mp4 output) — cheap
    # enough to compute unconditionally rather than gating it behind the
    # resume check below, which only governs whether the *files* get
    # rewritten.
    plain_text = _render_srt(prepared, karaoke=False, color=color)

    if already_done:
        log.info("Step 6c — ↩  already complete; re-using %s.", srt_out.name)
        if not srt_out.exists():
            raise RuntimeError(
                f"Step 6c is marked complete but {srt_out} is missing.  "
                "Delete the job directory and re-run from scratch."
            )
        # A sidecar is a plain file copy next to the output video, not
        # something "6c_transcript_srt" being marked done otherwise
        # guarantees stays in sync with -- e.g. a user could delete just
        # the sidecar without touching the job directory, or
        # write_sidecar could have been turned on after this job's Step
        # 6c last ran fresh. (Re)written here every time regardless of
        # the overall step flag, since it's a cheap copy either way.
        if sidecar_path is not None:
            sidecar_path.write_text(srt_out.read_text(encoding="utf-8"), encoding="utf-8")
        return SrtSource(key, label, srt_out, sidecar_path, plain_text, track_name)

    # Karaoke only ever applies to a per-stage comparison source (key !=
    # "final") -- see config.yaml's karaoke comment for the full
    # rationale. The authoritative SRT is always plain_text, computed
    # above, regardless of transcript_srt.karaoke's own value.
    karaoke_enabled = key != "final" and bool(cfg_get(cfg, "transcript_srt", "karaoke"))
    chosen_text = (
        _render_srt(prepared, karaoke=True, color=color) if karaoke_enabled else plain_text
    )

    log.info(
        "Step 6c — exporting %s transcript to SRT  "
        "(%d word(s) in %d group(s), karaoke=%s)",
        label, len(words), len(prepared), karaoke_enabled,
    )

    tmp = tmp_output_path(srt_out)
    tmp.unlink(missing_ok=True)   # leftover from a previous interrupted attempt, if any
    tmp.write_text(chosen_text, encoding="utf-8")
    finalize_output(tmp, srt_out)

    n_cues = chosen_text.count(" --> ")
    log.info("  ✓  %s  (%s)  %d cue(s)", srt_out.name, fmt_size(srt_out), n_cues)

    if sidecar_path is not None:
        sidecar_path.write_text(chosen_text, encoding="utf-8")
        log.info("  ✓  %s", sidecar_path)

    state = read_job(job_dir)
    per_source = state.get("transcript_srt", {})
    per_source[key] = {
        "label":   label,
        "words":   len(words),
        "groups":  len(prepared),
        "cues":    n_cues,
        "karaoke": karaoke_enabled,
        "file":    srt_out.name,
        "sidecar": sidecar_path.name if sidecar_path is not None else None,
    }
    state["transcript_srt"] = per_source
    write_job(job_dir, state)

    return SrtSource(key, label, srt_out, sidecar_path, plain_text, track_name)


# ── Hushing (redacting censor_log.json's muted words in the authoritative
# source's own SRT text — see this module's docstring and config.yaml's
# transcript_srt.hush comment for why this applies only to that source) ──

# Same leading/trailing-only punctuation set steps/matching.py's
# strip_punct() strips (mid-word punctuation like the apostrophe in
# "don't" is kept) — duplicated rather than imported so _split_punct()
# below can recover the stripped runs themselves, not just the core
# strip_punct() itself returns.
_PUNCT = string.punctuation

_HUSH_MODES = ("keep", "simple", "substitute")


def validate_hush_config(cfg: dict) -> None:
    """
    Fail fast, before Step 1a starts, on a transcript_srt.hush.mode (or
    simple_style) this version can't actually do anything with. Called
    once from pipeline.py's main(), immediately after
    utils.validate_config(cfg) — and defensively again, every run, from
    _apply_hush() below, in case something ever calls export_srt()
    without going through pipeline.py's main() first.

    The same kind of check steps/mute.py makes for censoring.method (mute
    vs the not-yet-implemented beep) — including reusing that function's
    own RuntimeError-with-a-clear-message style — except done here,
    upfront, rather than at first use inside the step itself: unlike
    Step 5, this step (Step 6c) deliberately treats its own exceptions as
    non-fatal (see this module's docstring), so a bad value caught only
    inside _apply_hush() would surface as a quiet warning after Steps
    1a-6b already spent hours of CPU time, not a clear error before any
    of them started. Deliberately separate from utils.validate_config():
    that function only ever checks a setting is PRESENT (see its own
    docstring) — this checks that this one setting's small enum is
    something Step 6c can actually do.
    """
    mode = cfg_get(cfg, "transcript_srt", "hush", "mode")
    if mode == "substitute":
        raise RuntimeError(
            "transcript_srt.hush.mode: substitute is not implemented yet "
            "— see autocensor-design.md §13.1.1. Set transcript_srt.hush."
            "mode to 'simple' (or 'keep') in config.yaml to proceed."
        )
    if mode not in ("keep", "simple"):
        raise RuntimeError(
            f"Step 6c: unknown transcript_srt.hush.mode {mode!r} "
            "(expected 'keep', 'simple', or 'substitute')."
        )
    if mode == "simple":
        style = cfg_get(cfg, "transcript_srt", "hush", "simple_style")
        if style not in ("token", "mask"):
            raise RuntimeError(
                f"Step 6c: unknown transcript_srt.hush.simple_style "
                f"{style!r} (expected 'token' or 'mask')."
            )


def _load_muted_intervals(
    job_dir: Path, log: logging.LoggerAdapter,
) -> list[tuple[float, float]]:
    """
    Read censor_log.json — Step 5's definitive, post-review/post-
    correction record of exactly what got muted (steps/mute.py) — and
    return every entry's (start, end) as a start-sorted list.

    Deliberately NOT matches.json: that's Step 4b's pre-review candidate
    list, which still includes anything a human (or --skip-index) later
    rejected — hushing against it would redact words that are actually
    still audible in the output. censor_log.json is what Step 5 actually
    acted on, so it's the only list that can't disagree with the audio.

    Missing entirely is treated as "nothing muted" rather than an error
    (shouldn't happen given Step 6c always runs after Step 5 — see
    pipeline.py's STEP_ORDER — but this function has no reason to be the
    one place that's fatal about it, consistent with this whole step's
    own deliberately-non-fatal failure handling).
    """
    path = job_dir / "censor_log.json"
    if not path.exists():
        log.debug("  hush: %s not found — treating as no muted words.", path.name)
        return []
    entries = json.loads(path.read_text()).get("entries", [])
    return sorted(
        (float(e["start"]), float(e["end"]))
        for e in entries
        if e.get("start") is not None and e.get("end") is not None
    )


def _hushed_word_indices(
    words: list[dict], muted_intervals: list[tuple[float, float]],
) -> set[int]:
    """
    Which positions in `words` (already time-ordered, already filtered to
    words with real start/end — see _export_one_source()) overlap a
    muted_intervals span.

    A phrase match covers several consecutive words in one word_list.txt
    entry (steps/matching.py's Match.span), but only its first word's
    index survives into censor_log.json (steps/mute.py's own log entries
    keep word_index, start, and end, but not span) — checking [start,
    end] overlap here, rather than trusting word_index alone, is what
    catches every word in a multi-word match, not just the first.

    Standard sorted-interval-list intersection sweep — both `words` and
    muted_intervals are individually non-overlapping and start-sorted
    (words because real speech doesn't overlap itself; muted_intervals
    because every one of them derives from those same non-overlapping
    word timestamps) — so each pointer only ever advances: one O(n+m)
    pass over both lists together, never a rewind.

    Compares against censor_log.json's UNPADDED start/end — Step 5's
    censoring.padding_ms is an audio-click guard, not a claim about word
    boundaries; using the padded interval here could pull in a
    genuinely different, unmatched neighboring word if the gap between
    the two is smaller than 2x padding_ms.
    """
    hushed: set[int] = set()
    i = j = 0
    while i < len(words) and j < len(muted_intervals):
        ws, we = float(words[i]["start"]), float(words[i]["end"])
        ms, me = muted_intervals[j]
        if max(ws, ms) < min(we, me):
            hushed.add(i)
        if we < me:
            i += 1
        else:
            j += 1
    return hushed


def _split_punct(word: str) -> tuple[str, str, str]:
    """
    Split into (leading_punct, core, trailing_punct) — the same leading/
    trailing-only stripping steps/matching.strip_punct() does (mid-word
    punctuation like the apostrophe in "don't" stays put), but keeping
    both stripped runs instead of discarding them, so hushing can
    reattach them around the replacement text unchanged — e.g. "shit!"
    keeps its "!" after masking, so _SENTENCE_END_RE (grouping happens
    after hushing — see _export_one_source()) still sees it and makes
    the same line-break decision it would have for the unhushed word.
    """
    core = word.strip(_PUNCT)
    if not core:
        return word, "", ""
    start = word.index(core)
    return word[:start], core, word[start + len(core):]


def _mask_core(core: str, cfg: dict) -> str:
    """
    "simple_style: mask" rendering of one word's core (post-_split_punct)
    — mask_character repeated out to mask_fixed_length (or core's own
    length, when that's 0, the default), optionally keeping the real
    first and/or last letter per mask_preserve_first/mask_preserve_last.

    Always leaves at least one real mask character in the middle, even
    when that makes the result one character longer than requested —
    otherwise a short core, or preserve_first+preserve_last together on
    a two-letter word, could reveal the entire original word while still
    being labelled "hushed".
    """
    mask_char = (str(cfg_get(cfg, "transcript_srt", "hush", "mask_character")) or "*")[0]
    preserve_first = bool(cfg_get(cfg, "transcript_srt", "hush", "mask_preserve_first"))
    preserve_last  = bool(cfg_get(cfg, "transcript_srt", "hush", "mask_preserve_last"))
    fixed_length   = int(cfg_get(cfg, "transcript_srt", "hush", "mask_fixed_length"))

    first = core[0] if preserve_first else ""
    last  = core[-1] if (preserve_last and len(core) > 1) else ""
    target_len = fixed_length if fixed_length > 0 else len(core)
    middle_len = max(target_len - len(first) - len(last), 1)
    return f"{first}{mask_char * middle_len}{last}"


def _hush_word_text(word: str, cfg: dict) -> str:
    """
    One matched word's replacement text for hush.mode: simple — leading/
    trailing punctuation preserved exactly (_split_punct above), core
    replaced per simple_style (a fixed token, or a mask_core() mask).
    """
    prefix, core, suffix = _split_punct(word)
    if not core:
        return word   # nothing to hush -- matching.py never matches an
                       # all-punctuation token, so this shouldn't happen
                       # in practice; the word as-is is the safe fallback
                       # if it somehow does.

    style = cfg_get(cfg, "transcript_srt", "hush", "simple_style")
    if style == "token":
        replacement = str(cfg_get(cfg, "transcript_srt", "hush", "simple_token"))
    else:
        replacement = _mask_core(core, cfg)
    return f"{prefix}{replacement}{suffix}"


def _apply_hush(
    words: list[dict], job_dir: Path, cfg: dict, log: logging.LoggerAdapter,
) -> list[dict]:
    """
    Render transcript_srt.hush.mode's chosen treatment over `words` — the
    authoritative ("final") transcript's own word list. Called ONLY for
    that source (see _export_one_source()) — never for a per-stage
    comparison transcript; see config.yaml's transcript_srt.hush comment
    for why cross-referencing censor_log.json against those isn't safe.

    Returns a new list, same length and order as `words`. Only entries
    whose [start, end] overlaps something in censor_log.json get a
    different "word" value — every other key (start/end/score, ...) is
    untouched, and every entry that ISN'T hushed is the exact same dict
    object passed in, never copied, so _render_group_lines()'s identity-
    based highlight_word check (this module's docstring) keeps working
    unchanged downstream.

    validate_hush_config(cfg) is assumed to have already run once, at
    startup (pipeline.py's main()) — re-checked here anyway, cheaply, as
    a safety net for any other caller.
    """
    validate_hush_config(cfg)
    mode = cfg_get(cfg, "transcript_srt", "hush", "mode")

    if mode == "keep":
        return words

    muted_intervals = _load_muted_intervals(job_dir, log)
    if not muted_intervals:
        return words

    hushed_idx = _hushed_word_indices(words, muted_intervals)
    if not hushed_idx:
        return words

    result = list(words)
    for i in hushed_idx:
        w = result[i]
        result[i] = {**w, "word": _hush_word_text(str(w.get("word", "") or ""), cfg)}

    log.info(
        "  hush: %d word(s) replaced in the authoritative SRT "
        "(transcript_srt.hush.mode=%s).", len(hushed_idx), mode,
    )
    return result


# ── Grouping (bundling flat words into subtitle-line-sized chunks) ───────────

def _prepare_groups(
    words: list[dict], cfg: dict,
) -> list[tuple[list[dict], list[list[dict]]]]:
    """
    Group words into subtitle lines, then wrap each group's text into
    display lines — one pass, shared by both the karaoke and plain
    renderings below so the two can never disagree about where a group
    or line break falls.

    Returns a list of (flat_words, wrapped_lines) pairs: flat_words is
    the group's words in time order (used for overall timing — first
    word's start, last word's end); wrapped_lines is the same words
    re-bucketed (same dict objects, never copied) into up to
    max_lines_per_group display lines.
    """
    max_words          = int(cfg_get(cfg, "transcript_srt", "max_words_per_group"))
    max_chars_per_line = int(cfg_get(cfg, "transcript_srt", "max_chars_per_line"))
    max_lines          = int(cfg_get(cfg, "transcript_srt", "max_lines_per_group"))
    max_gap_sec        = float(cfg_get(cfg, "transcript_srt", "max_group_gap_sec"))
    max_duration_sec   = float(cfg_get(cfg, "transcript_srt", "max_group_duration_sec"))

    groups = _split_into_groups(
        words,
        max_words=max_words,
        max_chars_total=max_chars_per_line * max_lines,
        max_gap_sec=max_gap_sec,
        max_duration_sec=max_duration_sec,
    )
    return [(g, _wrap_group(g, max_chars_per_line, max_lines)) for g in groups]


def _split_into_groups(
    words: list[dict],
    *,
    max_words: int,
    max_chars_total: int,
    max_gap_sec: float,
    max_duration_sec: float,
) -> list[list[dict]]:
    """
    Bundle a flat, time-ordered word list into subtitle-line-sized groups.

    A new group starts as soon as adding the next word would push the
    current one past any ONE of: a word-count cap, a total-character cap
    (max_chars_total — a group's max_lines_per_group * max_chars_per_line
    budget, so a group never needs more wrapped lines than that), an
    on-screen-duration cap, or a pause since the previous word of at
    least max_gap_sec (the same kind of real silence
    docs/timestamp-drift-investigation.md's worked example discusses — a
    shot change, a beat in the dialogue) — plus one unconditional rule:
    a word ending in '.', '?', or '!' always ends its group (see
    _SENTENCE_END_RE above).

    Assumes every word already has usable start/end timing (export_srt()
    filters before calling this) and is in time order — each backend's
    own transcript.json invariant; see steps/merge.py, which only ever
    concatenates already-ordered per-segment word lists.
    """
    groups: list[list[dict]] = []
    current: list[dict] = []

    for w in words:
        if current:
            prev = current[-1]
            gap = float(w["start"]) - float(prev["end"])
            span_if_added = float(w["end"]) - float(current[0]["start"])
            prev_ends_sentence = bool(
                _SENTENCE_END_RE.search(str(prev.get("word", "") or ""))
            )
            text_if_added = (
                " ".join(str(x.get("word", "") or "") for x in current)
                + " " + str(w.get("word", "") or "")
            )

            if (
                prev_ends_sentence
                or gap > max_gap_sec
                or len(current) >= max_words
                or span_if_added > max_duration_sec
                or len(text_if_added) > max_chars_total
            ):
                groups.append(current)
                current = []

        current.append(w)

    if current:
        groups.append(current)

    return groups


def _wrap_group(
    group: list[dict], max_chars_per_line: int, max_lines: int,
) -> list[list[dict]]:
    """
    Break one group's words into up to max_lines display lines, wrapping
    at word boundaries once a line would exceed max_chars_per_line —
    ordinary greedy word-wrap, the same thing any subtitle editor does
    when a line is fuller than its target width.

    Never drops or splits a word: a single word longer than
    max_chars_per_line still gets its own (overflowing) line rather than
    being cut mid-word, and once max_lines is already reached, every
    remaining word is appended to the last line regardless of length
    rather than starting a line this group isn't allowed to have. Both
    are deliberate — an occasionally-long line reads fine; a silently
    truncated word doesn't.
    """
    lines: list[list[dict]] = [[]]
    for w in group:
        line = lines[-1]
        if line:
            candidate_text = (
                " ".join(str(x.get("word", "") or "") for x in line)
                + " " + str(w.get("word", "") or "")
            )
            if len(candidate_text) > max_chars_per_line and len(lines) < max_lines:
                lines.append([w])
                continue
        line.append(w)
    return lines


# ── Rendering (groups/lines -> SRT cue text) ─────────────────────────────────

def _escape(text: str) -> str:
    """
    Escape the handful of characters that would otherwise be misread as
    markup once <font> tags are involved. Deliberately not a general
    HTML-escape (no quote/apostrophe handling) — this text is never
    placed inside an HTML attribute here, only as tag-delimited running
    text, so & / < / > are the only characters that could actually be
    misinterpreted.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_group_lines(
    lines: list[list[dict]],
    highlight_word: Optional[dict] = None,
    color: str = "",
) -> str:
    """
    Render one group's wrapped lines as the literal text of one SRT cue.

    highlight_word, if given, is matched by identity (`is`, not `==`)
    against the word dicts inside `lines` — safe because _wrap_group()
    only ever re-buckets the exact dict objects _split_into_groups()
    produced, never copies them, so "is this the word this cue is
    highlighting" is always a single unambiguous object check, never a
    value comparison that could accidentally also match some other,
    unrelated occurrence of the same word elsewhere in the line (real
    film dialogue repeats short words constantly).
    """
    rendered_lines = []
    for line in lines:
        parts = []
        for w in line:
            text = _escape(str(w.get("word", "") or ""))
            if highlight_word is not None and w is highlight_word:
                text = f'<font color="{color}">{text}</font>'
            parts.append(text)
        rendered_lines.append(" ".join(parts))
    return "\n".join(rendered_lines)


def _render_srt(
    prepared_groups: list[tuple[list[dict], list[list[dict]]]],
    karaoke: bool,
    color: str,
) -> str:
    """
    Render already-grouped-and-wrapped words into complete SRT text.

    karaoke=False: one cue per group, spanning its first word's start to
      its last word's end, plain text throughout (no <font> tags at all)
      — traditional grouped subtitles, nothing more.

    karaoke=True: one cue per WORD. SRT has no way to change a color
      mid-cue (see module docstring), so the highlight is built by
      re-emitting the group's full (already-wrapped) text once per word,
      with only that word wrapped in <font color>, each cue timed to
      exactly when it — and only it — is the "current" word:
        cue[i].start = word[i].start
        cue[i].end   = word[i+1].start        (word[i].end for the last
                                                 word in the group)
      so consecutive cues within a group are exactly contiguous — no gap
      the whole line would otherwise vanish for, no overlap — and the
      moment the highlight moves from one word to the next lands exactly
      on that next word's own recognized start time: the same timing
      this feature exists to make visible in the first place.
    """
    cues: list[tuple[float, float, str]] = []

    for flat_words, lines in prepared_groups:
        if not karaoke:
            text = _render_group_lines(lines)
            cues.append((float(flat_words[0]["start"]), float(flat_words[-1]["end"]), text))
            continue

        n = len(flat_words)
        for i, w in enumerate(flat_words):
            start = float(w["start"])
            end   = float(flat_words[i + 1]["start"]) if i + 1 < n else float(w["end"])
            if end <= start:
                end = start + _MIN_CUE_SEC
            cues.append((start, end, _render_group_lines(lines, highlight_word=w, color=color)))

    return _cues_to_srt(cues)


def _fmt_srt_timestamp(seconds: float) -> str:
    """
    Format a position in seconds as SRT's own HH:MM:SS,mmm — zero-padded
    two-digit hours and a COMMA decimal separator, both load-bearing:
    confirmed directly that a period there is silently misread. This is
    deliberately separate from utils.fmt_timestamp(), which uses this
    codebase's own H:MM:SS.mmm convention (unpadded hour, period) for
    job.json/matches.json/censor_log.json/review.json and the
    interactive review prompts — the two formats look almost alike but
    are not interchangeable, and this one exists only for the SRTs this
    module writes.

    Computed via integer milliseconds (not repeated float division), the
    same carry-safe approach as utils.fmt_timestamp(), so a value like
    59.9996s rounds to the next whole second cleanly rather than
    producing an invalid "60" in the seconds field.
    """
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms, total_s = total_ms % 1000, total_ms // 1000
    s,  total_m = total_s % 60,    total_s // 60
    m,  h       = total_m % 60,    total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _cues_to_srt(cues: list[tuple[float, float, str]]) -> str:
    """Serialize (start, end, text) cues into SRT's numbered-block format."""
    blocks = [
        f"{i}\n{_fmt_srt_timestamp(start)} --> {_fmt_srt_timestamp(end)}\n{text}\n"
        for i, (start, end, text) in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)
