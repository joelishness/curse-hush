"""
profanity-hush — Step 6c: export aligned transcripts as SRT subtitles

Why this exists: figuring out *why* a word got muted at the wrong moment,
or why MFA and WhisperX disagree about a word, normally means cross-
referencing the output video against a transcript against the original
audio by hand, across three separate tools — exactly the tedious process
docs/timestamp-drift-investigation.md's whole worked example walks
through. This step makes that a "turn a subtitle track on" problem
instead: each available aligned transcript becomes its own subtitle
track showing every word that backend recognized, at exactly the
timestamp it gives it, so a mistimed mute — or a disagreement between
alignment methods — is immediately visible as a mismatch between what
the track says is being spoken and what's actually audible.

Input  : transcript_mfa.json and/or transcript_whisperx.json (Step 3b —
         see steps/merge.py's docstring for exactly when each exists:
         depends on alignment.backend and alignment.dual_output).
Output : transcript_mfa.srt and/or transcript_whisperx.srt, saved in the
         job directory, and — transcript_srt.write_sidecar — a copy of
         each next to the output video too. Also returned in memory (see
         export_srt()'s docstring) for steps/mux.py to embed.
Marks '6c_transcript_srt' done.

Does this belong in the --skip-index/--add-interval/--redo-review
correction-mode unmark list (pipeline.py) alongside 5_mute/6_recombine/
6b_encode/7_mux? No, deliberately not: these SRTs are derived purely from
transcript_mfa.json / transcript_whisperx.json (what was recognized),
never from matches.json/review.json/censor_log.json (what got muted) — a
correction changes the latter, never the former, so this step's own
output can't be stale after one. It still runs on a correction re-run
(pipeline.py always calls it, same as every run), but only to hit its own
"already complete" branch below and hand steps/mux.py back the same
files.

Failure handling is deliberately NOT the same as every other step's: an
unexpected failure here logs a warning and lets the pipeline continue
with no subtitle tracks this run, rather than failing the whole job (see
pipeline.py's call site). This is a debugging aid layered on top of the
actual censoring pipeline, not part of it — by the time this step runs,
every genuinely expensive step (1a through 6b) has already succeeded, and
there's no good reason a bug in grouping/rendering a subtitle file should
cost someone the censored video they've been waiting hours for. Config
validation failures are the one exception and stay fatal, same as every
other setting (see utils.validate_config()) — those are caught before
Step 1a even starts, not here.

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

── Why two SRTs instead of one ───────────────────────────────────────────

Earlier versions of this step read a single, generic transcript.json —
whatever alignment.backend happened to produce, per segment, including
any per-segment whisperx fallback baked invisibly into the same file (see
steps/align_mfa.py's fallback handling). That's still exactly what
transcript.json is for matching/muting (steps/matching.py, steps/mute.py)
— unchanged. But it makes a poor comparison tool: there's no way to tell,
from transcript.json alone, whether a given word's timing came from MFA
or from a whisperx fallback, which is precisely the distinction someone
troubleshooting a disagreement between the two needs to see. Backend-
labeled sources (transcript_mfa.json / transcript_whisperx.json — see
steps/merge.py) fix that: each is a clean, honestly-gapped view of what
one specific backend actually produced, independent of which one was
"primary" for censoring purposes.
"""

import json
import logging
import re
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

# Floor for a single karaoke cue's duration, in seconds. A backend's word
# list should always be strictly time-ordered with non-overlapping spans
# (a forced/CTC aligner can't place two words in the same instant) --
# this only guards against a same-timestamp or out-of-order anomaly
# producing a zero- or negative-duration SRT cue, which some parsers
# reject outright.
_MIN_CUE_SEC = 0.05

# The two alignment backends this step knows how to source a transcript
# from -- also the config-key suffix (transcript_srt.track_name_mfa, ...)
# and the filename tag (transcript_mfa.srt, <video>.eng.mfa.srt, ...) for
# each. Order here is the order sources are processed and returned in.
_BACKENDS = ("mfa", "whisperx")


@dataclass
class SrtSource:
    """
    One alignment backend's SRT output for this job — everything
    steps/mux.py needs to embed it, and everything pipeline.py needs to
    log it as a kept output.

    backend            — "mfa" | "whisperx".
    job_dir_path        — job_dir/transcript_{backend}.srt. Always written
                           (in whichever style transcript_srt.karaoke
                           selects) whenever this source exists at all.
    sidecar_path         — a copy of the same content next to the output
                           video, named per config.yaml's convention.
                           None when transcript_srt.write_sidecar is false.
    mp4_fallback_text   — a plain (no <font> tags) one-cue-per-group
                           rendering, always plain regardless of
                           transcript_srt.karaoke — steps/mux.py uses this
                           for output.format: mp4 specifically (see this
                           module's docstring for why mp4 can't show the
                           karaoke version at all).
    track_name          — transcript_srt.track_name_{backend}, for
                           steps/mux.py's embedded-track metadata.
    """
    backend:            str
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
    Step 6c: render each available aligned transcript into its own SRT.

    output_video_path — the final output video's path (e.g. from
    steps.mux._output_path(), called early by pipeline.py for exactly
    this purpose). Used only to derive sidecar filenames
    (<output_video_path.stem>.<language>.<backend>.srt, written next to
    it) — the file itself need not exist yet when this runs, since Step
    6c always runs before Step 7 actually produces it.

    Returns one SrtSource per backend that had a transcript_{backend}.json
    (steps/merge.py) with at least one word with usable alignment timing
    — so 0, 1, or 2 entries, in _BACKENDS order. Empty whenever
    transcript_srt.enabled is false, or neither backend produced any
    usable timing at all.
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

    sources: list[SrtSource] = []
    for backend in _BACKENDS:
        transcript_path = job_dir / f"transcript_{backend}.json"
        if not transcript_path.exists():
            continue
        source = _export_one_source(
            job_dir, output_video_path, backend, transcript_path,
            already_done, cfg, log,
        )
        if source is not None:
            sources.append(source)

    mark_step_done(job_dir, "6c_transcript_srt")

    if not sources:
        log.warning(
            "Step 6c — neither transcript_mfa.json nor "
            "transcript_whisperx.json exists (or neither has any word "
            "with usable alignment timing) — nothing to export."
        )
    log.info("  ✓  Step 6c complete.")
    return sources


def _export_one_source(
    job_dir: Path,
    output_video_path: Path,
    backend: str,
    transcript_path: Path,
    already_done: bool,
    cfg: dict,
    log: logging.LoggerAdapter,
) -> Optional[SrtSource]:
    """
    Process one alignment backend's transcript into an SrtSource — shared
    logic for both "mfa" and "whisperx", called once per available
    backend by export_srt() above.

    Returns None if transcript_path has no word with usable alignment
    timing at all (nothing to place on a timeline either way) — logged,
    not an error.
    """
    srt_out = job_dir / f"transcript_{backend}.srt"

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
            transcript_path.name, len(all_words), backend,
        )
        return None

    track_name = str(cfg_get(cfg, "transcript_srt", f"track_name_{backend}"))

    sidecar_path: Optional[Path] = None
    if bool(cfg_get(cfg, "transcript_srt", "write_sidecar")):
        track_language = str(cfg_get(cfg, "transcript_srt", "track_language"))
        sidecar_path = output_video_path.parent / (
            f"{output_video_path.stem}.{track_language}.{backend}.srt"
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
        return SrtSource(backend, srt_out, sidecar_path, plain_text, track_name)

    karaoke_enabled = bool(cfg_get(cfg, "transcript_srt", "karaoke"))
    chosen_text = (
        _render_srt(prepared, karaoke=True, color=color) if karaoke_enabled else plain_text
    )

    log.info(
        "Step 6c — exporting %s-aligned transcript to SRT  "
        "(%d word(s) in %d group(s), karaoke=%s)",
        backend, len(words), len(prepared), karaoke_enabled,
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
    per_backend = state.get("transcript_srt", {})
    per_backend[backend] = {
        "words":   len(words),
        "groups":  len(prepared),
        "cues":    n_cues,
        "karaoke": karaoke_enabled,
        "file":    srt_out.name,
        "sidecar": sidecar_path.name if sidecar_path is not None else None,
    }
    state["transcript_srt"] = per_backend
    write_job(job_dir, state)

    return SrtSource(backend, srt_out, sidecar_path, plain_text, track_name)


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
