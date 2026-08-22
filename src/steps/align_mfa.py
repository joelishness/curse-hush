"""
Word-level alignment via Montreal Forced Aligner (MFA) — the default
alignment engine as of this version (config.yaml's alignment.backend: mfa).
Falls back to whisperx.align()'s wav2vec2/CTC alignment per-segment (and,
as of this version, per-chunk within a segment — see CHUNKING below) on
failure; see that file for the reasoning behind the switch. Short version:
whisperx.align() aligns *within* whatever segment boundary WhisperX's own
transcribe() pass already committed to, so a chunk-boundary timing error
upstream (see docs/timestamp-drift-investigation.md) propagates straight
through it uncorrected. MFA's forced-alignment search is given known text
and told to find it somewhere in the audio it's handed — it never *trusts*
a WhisperX-reported timestamp to decide where to look, so it has nothing to
inherit a chunk-boundary error from. Validated directly, not just reasoned
about — see the same doc for the before/after numbers on the original,
whole-segment-per-call design.

Imported unconditionally at the top of transcribe.py (not behind a
conditional "only if backend == mfa" guard) — this module's own top-level
imports are stdlib + utils only, so there's no cost to importing it even on
a job that ends up using whisperx for every segment.

CHUNKING — why, and why it doesn't undo the paragraph above:
The original version of this function handed align_one one entire ~30-
*minute* job-segment as a single utterance, on the reasoning that MFA's
whole-file search is what makes it drift-immune in the first place, so
splitting it seemed like exactly the wrong direction. That held for
correctness, but not for runtime: align_one's beam search cost scales with
how long the given audio is, not just with beam width, and a real job hit
this directly — a single segment's align_one call ran past a 1800s subprocess
timeout without finishing (confirmed reproducible: two independent attempts,
different temp dirs, both ran ~2600s total before being killed, agreeing to
within 5s). MFA's own docs are explicit that individual alignment units
should generally stay under ~30s, and that align_one/single-file alignment
specifically isn't really intended for long files at all.

So this now runs align_one once per short chunk (target
alignment.mfa.chunk_target_sec, built from consecutive whisper_segments —
see _build_alignment_chunks()) instead of once for the whole job-segment.
The risk that raises: whisper_segments' own start/end timestamps are
*exactly* the untrusted signal this whole module exists to not depend on
(see the paragraph above, and docs/timestamp-drift-investigation.md's own
worked example — a skipped line leaves every subsequent WhisperX segment
timestamped several seconds to tens of seconds early, growing further
across a repetitive stretch before a real silence resets it). Naively
slicing dialog_wav at a chunk's claimed [start, end] and asking MFA to
search *only* that narrow slice would silently reintroduce that same
dependency one level down.

FIRST ATTEMPT AT A FIX, AND WHY IT WAS WRONG:
The first version of this padded each chunk's slice well past its claimed
boundaries (more room after than before — every confirmed drift case in
docs/timestamp-drift-investigation.md ran the same direction, offset
always negative, WhisperX's claimed time consistently earlier than real
time) and handed align_one that whole padded window directly, on the
assumption that forced alignment fits the *given* text to its best span
and doesn't spread it across whatever extra room it's offered. That
assumption was wrong, and re-validating against Independence Day (which
already has 17 hand-confirmed drift cases with known-correct timestamps
from the original investigation) caught it directly rather than leaving it
undiscovered: MFA does not reliably stop at the true end of speech when
the given audio has a lot of trailing silence in it. A single word was
observed smeared to over 25 seconds. Mechanically, this isn't MFA
misbehaving so much as it is Viterbi decoding doing what it does when
asked to explain an entire given audio span using only the given phones —
if there's no strong reason to transition into (and pay the cost of
sitting in) a silence model trained on ordinary pause lengths, extending
the last phone's own self-loop across the padding is often the
higher-likelihood path. A cross-chunk plausibility check (comparing a
chunk's anchors against where the previous chunk's own accepted anchors
ended) was added to catch the fallout, but it was compensating for the
smearing's downstream effects — inflated chunk-boundary times, cascading
false rejections of otherwise-correct neighbouring chunks, corrupted
bracketing anchors for _interpolate_stage2_words()'s affine warp — rather
than removing its cause, and measured against the same 17 known cases,
the result was *worse* than plain whisperx.align(), not better (median
timing offset against a reference SRT went from 0.27s at stage 1 to
6.93s at stage 2).

SECOND ATTEMPT, AND WHY IT WAS *ALSO* WRONG:
Kept the padded window but stopped handing it to align_one directly —
ran ffmpeg's own silencedetect filter over it first and sliced down to
just the detected non-silent span before aligning, on the reasoning that
real, purpose-built silence detection wouldn't have the first attempt's
problem, since MFA would never see the padding at all. That fixed the
trailing-smear mechanism specifically, but missed a second, different one
on a real (shorter) test file: silencedetect finds *any* non-silent
audio in the padded window, with no way to know whether it belongs to
THIS chunk or to whatever precedes/follows it. On a chunk near the start
of a file, with generous chunk_pad_before_sec reaching close to (or
clamped at) position 0, and without a clean, long-enough silence gap
between this chunk's real content and whatever came before it (dense,
fast-paced dialogue has little reason to leave one) — confirmed directly,
not hypothetical — the detected envelope reached back roughly 3 seconds
into unrelated preceding audio. MFA then had to explain the given text
across that whole (too-wide, wrong-content) span: one word's end matched
truth to within 2ms (the chunk's real trailing boundary, correctly
found), while the chunk's detected start was 3.1 seconds early, and two
of the chunk's own words absorbed the gap (0.24s of real duration
inflated to 2.35s; 0.70s inflated to 2.02s).

Two attempts, two different mechanisms, both traceable to the same root
decision: searching a generously padded window and trusting *anything*
found in it — MFA's own alignment search in the first case, real silence
detection in the second — to reliably isolate just this chunk's real
audio. Neither did, reliably. Left both here as a record of what was
tried and ruled out, same as this project's own
docs/timestamp-drift-investigation.md does for its six earlier MFA
integration problems — worth knowing before generous padding is tempting
to reach for a third time.

THE ACTUAL FIX — stop trying to search for it at all:
[slice_start, slice_end) is now the chunk's own claimed [start, end] from
WhisperX plus a small, FIXED edge margin (alignment.mfa.chunk_edge_margin_sec,
default well under a second) — not a generously padded search window, and
nothing tries to find real audio beyond it. A chunk affected by genuine
WhisperX timestamp drift will generally fail to align in a window this
tight (align_one can't find the given text where it isn't), and degrades
to stage 1 (WhisperX) timing for that chunk's own words via the same
non-fatal per-chunk path described below — which is not a regression:
it's exactly the timing those words would have had before MFA was
introduced at all. The trade being made explicitly: MFA no longer
attempts to rescue the ~1% of content genuinely affected by WhisperX
drift (17 of 1,538 lines in the original Independence Day count) — both
attempts at rescuing it corrupted a much larger fraction of otherwise-fine
content instead, in two different ways, so the safer default is to let
those specific words fall back to no-worse-than-baseline rather than risk
a third, cleverer way to search for them.

The cross-chunk plausibility check (align_with_mfa()'s monotonicity
comparison against the previous chunk's accepted anchors) is kept as a
cheap backstop regardless — a tight slice removes the room for either
padding failure mode to occur, but doesn't guarantee every chunk aligns
perfectly, and the check costs nothing to keep.

This tight-slicing version has NOT yet been re-run against a real MFA
install — same lack of a network path to conda-forge from where this was
written, see docs/timestamp-drift-investigation.md for what "validated
directly, not just reasoned about" has meant everywhere else in this file.
The re-validation path is the same one that caught both earlier attempts'
failures and should be trusted more than this docstring's own reasoning:
re-run against Independence Day's 17 hand-confirmed drift cases and the
reference SRT, AND against whatever shorter/denser-dialogue file exposed
the second attempt's failure, and check the resulting transcript's own
word-duration distribution directly (no word should come back multiple
seconds long) and specifically inspect chunks near the very start of each
job-segment and near tightly-packed dialogue with little pause between
lines — those are exactly the conditions that exposed problems in both
earlier attempts, and a design change motivated by one file's failure
mode needs checking against the file that surfaced it, not only the one
that surfaced the first. Checking only the aggregate offset number was
what let the first attempt's rejection-rate statistic look plausible
before the SRT comparison showed otherwise; don't repeat that shortcut
here either. Expect this version's aggregate offset to be closer to
stage 1's own (rather than better than stage 1 everywhere) precisely
because it deliberately stops trying to rescue the drift-affected
minority — the 17 known cases are the direct measure of what, if
anything, was given up by no longer padding.

A per-chunk MFA failure (timeout, bad exit, a discarded out-of-sequence
result) is NOT fatal to the whole segment the way it was before chunking —
it degrades just that chunk's words to stage 1 timing and moves on to the
next chunk, via the same _interpolate_stage2_words() path a chunk with no
alignable text already falls through today. alignment.mfa.fallback_to_whisperx
still controls the coarser, whole-segment case (MFA fundamentally unusable
right now — no conda, models not downloaded, G2P grapheme lookup failing —
see align_with_mfa()'s early checks): those are environment problems every
chunk would hit identically, so there's no reason to pay for N failed
attempts before giving up, and that setting's meaning is unchanged from
before chunking.

WHAT THIS FIXES vs. WHAT IT DOESN'T:
MFA can only place words that WhisperX's transcribe() pass already
recognized as text somewhere. If WhisperX's own recognition drops a stretch
of dialogue entirely (a *recall* failure — nothing to align because there's
no text for it), MFA has nothing to work with there either. That's a
separate, still-open problem (see docs/timestamp-drift-investigation.md,
which now specifically traces one such case — dense phrase repetition
confusing WhisperX's own decoder — with MFA's beam width already ruled out
as a contributing factor). What this fixes is specifically: for words
WhisperX *did* recognize, making sure their reported timestamp is actually
where they occur.

MFA is a TIMING correction only — never a text source. Word text
(casing, punctuation, numerals, everything) always comes from WhisperX's
own output, unconditionally; MFA's own dictionary/TextGrid vocabulary is
lowercase, punctuation-free, and drops anything outside its G2P model's
alphabet before it's even sent (see _sanitize_for_mfa()), so it was never
going to be a usable text source anyway. Concretely:

  1. Only the words that survive sanitization are sent to MFA at all
     (origin_index tracks which whisper_words position each one came
     from). Everything else — numerals, foreign-alphabet words, tokens
     over _MAX_MFA_TOKEN_LEN — is never in MFA's input to begin with.
  2. Of what comes back, only entries this pipeline can match to MFA's
     own input with high confidence (_confident_mfa_anchors() — direct
     1:1 index when the tier's count matches what was sent, which is
     the common case; a conservative difflib reconciliation that trusts
     only exact-match stretches when it doesn't) become "anchors" — a
     whisper_words position paired with a corrected (start, end).
  3. Every anchor's WORD TEXT still comes from whisper_words, never from
     MFA's own tier label — an anchor contributes timing only.
  4. Every word that ISN'T an anchor — sent to MFA but not confidently
     matched back, or never sent at all — gets an INTERPOLATED timestamp
     (_interpolate_stage2_words()): WhisperX's own relative timing
     between the nearest anchors on each side, affine-warped (scaled +
     shifted, preserving shape) to fit the corrected span exactly. A
     word past the last anchor or before the first keeps WhisperX's raw
     timing untouched — this corrects gaps BETWEEN confirmed points, it
     never extrapolates beyond them.

This replaced an earlier design (still visible in git history) where a
tier-count mismatch fell back to reconstructing the ENTIRE word list via
difflib fuzzy matching against MFA's own labels — including using MFA's
raw dictionary-form label as the "word" wherever that reconstruction
couldn't find a WhisperX counterpart to prefer instead. That turned out to
be a real, previously-live bug: on real movie-length, repetition-heavy
dialogue, a mismatched tier is common enough (see _confident_mfa_anchors()'s
own docstring) that this wasn't a rare edge case, and the failure mode
was bad in both directions at once — words WhisperX actually recognized
correctly could get silently swapped for the wrong neighbour, and MFA's
own recognized-but-never-validated label could appear as a "word" nobody
ever actually said. The design above closes that off structurally rather
than case-by-case: text is never at risk, because text is never sourced
from MFA in the first place, at any stage, under any fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
import subprocess
import tempfile
import difflib
import math
import time
from pathlib import Path
from typing import Optional

from utils import cfg_get, run_cmd


class MFAError(RuntimeError):
    """Raised when MFA alignment fails and no fallback is available/allowed."""


def _mfa_cmd(*args: str) -> list[str]:
    """
    Build a command that invokes `mfa` inside its own conda env, without
    ever putting that env's bin/ directory on this process's PATH.

    An earlier version of this change did exactly that -- a Dockerfile
    `ENV PATH="/opt/conda/envs/mfa/bin:${PATH}"` -- reasoning that it was
    a narrow addition just to make the `mfa` binary discoverable. It
    wasn't narrow: a conda env's bin/ contains a full, separate Python
    installation, and PATH is searched in order, so prepending it doesn't
    just add `mfa`, it shadows `python`/`pip` for every OTHER subprocess
    call anywhere in this codebase that invokes them by bare name.
    Confirmed directly, not hypothetically -- this broke Step 2's own
    demucs invocation, which resolved "python" to the MFA env's
    interpreter instead of the main pip-installed one and failed with
    "No module named demucs". Appending instead of prepending would have
    dodged that one specific collision while leaving the same category of
    risk in place for the next binary name that happens to exist in both
    environments. Not touching PATH for this at all, and going through
    `conda run` instead, removes the category rather than one instance of
    it -- `conda run -n <env> <cmd>` activates exactly what that one
    subprocess call needs (PATH, LD_LIBRARY_PATH, etc. -- kalpy is a
    compiled extension, so the library path matters too, not just PATH)
    and nothing else, for the lifetime of that one call only.

    MFA_CONDA_EXE / MFA_CONDA_ENV are set in the Dockerfile, right next to
    where the env is actually created, so this file doesn't hardcode an
    assumption about install layout that could silently drift out of
    sync with it -- they default here to match the Dockerfile's current
    values only as a fallback for running outside that exact image.
    """
    conda_exe = os.environ.get("MFA_CONDA_EXE", "/opt/conda/bin/conda")
    conda_env = os.environ.get("MFA_CONDA_ENV", "mfa")
    return [conda_exe, "run", "-n", conda_env, "mfa", *args]


def _mfa_python_cmd(*args: str) -> list[str]:
    """
    Same rationale as _mfa_cmd() immediately above -- conda run, never this
    process's own PATH -- but invoking the mfa env's own python3 directly
    rather than the `mfa` CLI entry point. Needed for exactly one thing:
    _get_g2p_graphemes() below has to import montreal_forced_aligner's own
    model classes to ask a model what its real grapheme set is, and no
    `mfa` subcommand exposes that as structured, parseable output.
    """
    conda_exe = os.environ.get("MFA_CONDA_EXE", "/opt/conda/bin/conda")
    conda_env = os.environ.get("MFA_CONDA_ENV", "mfa")
    return [conda_exe, "run", "-n", conda_env, "python3", *args]


# ── One-time-per-container setup ─────────────────────────────────────────────

def _ensure_mfa_ready(acoustic_model: str, dictionary: str, g2p_model: Optional[str],
                       log: logging.LoggerAdapter) -> None:
    """
    Download MFA's pretrained models and initialize its server/database if
    that hasn't happened yet in this container's lifetime.

    Deliberately NOT done at Docker image build time (an earlier version of
    this change tried exactly that, via `RUN mfa server init` in the
    Dockerfile) -- and deliberately not fixed by simply running that command
    as some other fixed build-time user either. Both would only be partial
    fixes; see the actual reasoning below, since it isn't obvious and it's
    the whole reason this function exists at all:

    MFA's server/database backend calls PostgreSQL's initdb under the hood,
    which unconditionally refuses to run as root (a hard-coded PostgreSQL
    safety check, not a flag) -- confirmed directly, this is the exact error
    a build-time `RUN mfa server init` produces, since Docker RUN steps
    execute as root by default. But creating a fixed non-root build-time
    user and running it as *that* user instead wouldn't actually be enough
    either: hush.sh's containers run with `--user "$(id -u):$(id -g)"` --
    an arbitrary, host-determined UID chosen at `docker run` time, not
    whatever UID a Dockerfile RUN step happened to use at build time.
    PostgreSQL data directories are tied to the UID that initializes them;
    a build-time UID essentially never matches the real runtime UID, so
    build-time initialization under any fixed user is set up to fail later
    for the same underlying reason, just with a different, more confusing
    error at a less obvious time. The only UID that's actually correct for
    this is whatever the container is really running as -- which isn't
    knowable until the container is already running. Hence: lazy, here, now,
    not in the Dockerfile.

    Idempotent via our own marker file under MFA_ROOT_DIR, rather than
    relying on MFA's own re-init call being graceful the second time --
    that's not something this change had a way to verify against a real
    install (see docs/timestamp-drift-investigation.md's testing-status
    notes), so this doesn't assume it either way.
    """
    root = Path(os.environ.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))
    marker = root / ".hush_mfa_initialized"
    if marker.exists():
        return

    root.mkdir(parents=True, exist_ok=True)
    log.info(
        "    First use of MFA in this container -- downloading models and "
        "initializing its database (one-time; if %s is your persistent "
        "/cache volume, later runs and later jobs skip this entirely).",
        root,
    )

    try:
        proc = subprocess.run(_mfa_cmd("server", "init"), capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise MFAError("mfa server init did not finish within 300s.")
    if proc.returncode != 0:
        raise MFAError(
            f"mfa server init failed (exit {proc.returncode}).\n"
            "If the error above says \"cannot be run as root\": something is "
            "invoking this as root rather than the container's real runtime "
            "UID -- it must run as whatever UID will later call "
            "`mfa align_one` (see docs/timestamp-drift-investigation.md and "
            "_ensure_mfa_ready's own docstring for why this can't be done at "
            "Docker build time at all).\n"
            f"stderr(tail): {proc.stderr[-2000:]}"
        )

    for model_type, name in (
        ("acoustic", acoustic_model),
        ("dictionary", dictionary),
        ("g2p", g2p_model),
    ):
        if not name:
            continue
        try:
            proc = subprocess.run(
                _mfa_cmd("model", "download", model_type, name),
                capture_output=True, text=True, timeout=1800,
            )
        except subprocess.TimeoutExpired:
            raise MFAError(f"mfa model download {model_type} {name} did not finish within 1800s.")
        if proc.returncode != 0:
            raise MFAError(
                f"mfa model download {model_type} {name} failed "
                f"(exit {proc.returncode}).\nstderr(tail): {proc.stderr[-2000:]}"
            )

    marker.write_text("initialized by profanity-hush's align_mfa.py\n")


# ── Public entry point ───────────────────────────────────────────────────────

def align_with_mfa(
    dialog_wav: Path,
    dialog_duration_sec: float,
    whisper_segments: list[dict],
    whisperx_words: list[dict],
    cfg: dict,
    log: logging.LoggerAdapter,
    work_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Align WhisperX's recognized text against dialog_wav using MFA, returning
    the same [{"word", "start", "end", "score"}, ...] schema that
    whisperx.align() produces in transcribe.py — a drop-in replacement at
    that call site, not a new data shape downstream needs to know about.

    dialog_duration_sec — this job-segment's own duration, i.e. what
        transcribe.py already computed as dur_sec for its own log line at
        this segment's call site. Needed here only to clamp padded chunk
        slices to a valid range (see CHUNKING below); not re-probed from
        dialog_wav itself since the caller already has it.

    whisper_segments — result["segments"] from wx_model.transcribe(), i.e.
        WhisperX's *pre-alignment* output: text (+ usually avg_logprob) per
        internally-decoded chunk, no word-level breakdown yet, PLUS that
        chunk's own (WhisperX-claimed, not-fully-trusted) start/end — used
        here only to decide how to group chunks and how wide a search
        window to hand MFA, never as the actual final timing for anything;
        see CHUNKING below and this module's own docstring for why that
        distinction matters.

    whisperx_words — transcribe.py's own stage_words[1]: whisperx.align()'s
        per-word output for this SAME segment (same words, same order —
        see _flatten_whisper_words() and _collect_whisperx_words(), which
        both walk this segment's text the same way), already computed by
        the time this is called since stage 1 always runs first. Used
        ONLY as the interpolation basis for whatever this function can't
        get a confident MFA timestamp for — never for word text, and
        never for anything at positions MFA *does* confidently cover.
        See _interpolate_stage2_words() for the mechanism, and this
        module's own docstring for why word text never comes from MFA at
        all, at any stage.

    CHUNKING: rather than one align_one call for this whole (typically
    ~30-minute) job-segment, this runs one call per short chunk of
    consecutive whisper_segments (_build_alignment_chunks(),
    alignment.mfa.chunk_target_sec) — see this module's own docstring for
    why, and for the padding + cross-chunk monotonicity check that keeps
    this from silently reintroducing a dependency on WhisperX's own
    (sometimes drift-affected) segment timestamps. A single chunk's MFA
    call failing — timeout, bad exit, or a result discarded by the
    monotonicity check — degrades only that chunk's words to stage 1
    (WhisperX) timing and continues with the rest; it does not raise and
    does not affect any other chunk. See _align_chunk()'s own docstring.

    Raises MFAError only for problems that would affect every chunk
    identically — no conda install, MFA's models/database not ready, the
    G2P model's grapheme set unreadable — checked once, up front, before
    any chunk is attempted, since there would be no point discovering the
    same environment problem N times over. If
    alignment.mfa.fallback_to_whisperx is true (the default), the caller
    (transcribe.py) is expected to catch MFAError and retry this segment
    through the whisperx.align() path entirely instead — this function
    itself does not know how to do that fallback, since it has no access
    to the whisperx align model/metadata transcribe.py is holding.
    """
    if not whisper_segments:
        return []

    acoustic_model = cfg_get(cfg, "alignment", "mfa", "acoustic_model")
    dictionary     = cfg_get(cfg, "alignment", "mfa", "dictionary")
    # allow_null=True (not default=): g2p_model IS a config.yaml schema key
    # (see config.yaml's alignment.mfa block) that may legitimately be set
    # to `null` to disable G2P fallback -- distinct from a key that's
    # absent from the schema entirely, which is what `default=` is for
    # (see utils.cfg_get()'s docstring). Using default= here would also
    # mask a genuine config.yaml typo/omission instead of raising.
    g2p_model      = cfg_get(cfg, "alignment", "mfa", "g2p_model", allow_null=True)
    beam           = int(cfg_get(cfg, "alignment", "mfa", "beam"))
    retry_beam     = int(cfg_get(cfg, "alignment", "mfa", "retry_beam"))
    fallback_allowed = bool(cfg_get(cfg, "alignment", "mfa", "fallback_to_whisperx"))
    chunk_target_sec = float(cfg_get(cfg, "alignment", "mfa", "chunk_target_sec"))
    chunk_edge_margin_sec = float(cfg_get(cfg, "alignment", "mfa", "chunk_edge_margin_sec"))
    chunk_timeout_sec    = float(cfg_get(cfg, "alignment", "mfa", "chunk_timeout_sec"))

    # Checked here, before _ensure_mfa_ready() (which is otherwise the first
    # thing to shell out) rather than only right before the first align_one
    # call -- a missing conda install should fail once, fast, with this
    # specific message, before any chunk is even built. Left where it was,
    # it wouldn't fire until after _ensure_mfa_ready() had already tried and
    # failed to run `conda run ...` itself, surfacing as a much less clear
    # raw FileNotFoundError instead.
    conda_exe = os.environ.get("MFA_CONDA_EXE", "/opt/conda/bin/conda")
    if not Path(conda_exe).exists():
        raise MFAError(
            f"alignment.backend is 'mfa' but {conda_exe} doesn't exist. "
            "MFA is a conda-forge package (needs the compiled `kalpy` Kaldi "
            "bindings, which aren't on PyPI) — see the Dockerfile's MFA "
            "install stage. `pip install montreal-forced-aligner` alone is "
            "not sufficient; it installs but fails at import time with "
            "`ModuleNotFoundError: No module named '_kalpy'`. If MFA is "
            "installed at a different location, set MFA_CONDA_EXE."
        )

    _ensure_mfa_ready(acoustic_model, dictionary, g2p_model, log)

    # None when no g2p_model is configured at all -- see _sanitize_for_mfa()'s
    # docstring for what that changes. When one is configured, this is the
    # model's own real alphabet (queried, not guessed -- see
    # _get_g2p_graphemes()'s docstring for why this replaced an earlier,
    # incorrect static guess).
    graphemes = _get_g2p_graphemes(g2p_model, log) if g2p_model else None

    whisper_words, whisper_scores = _flatten_whisper_words(whisper_segments)
    chunks = _build_alignment_chunks(whisper_segments, chunk_target_sec, log)

    tmp_ctx = None
    if work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="mfa_align_")
        work_dir = Path(tmp_ctx.name)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Same roles origin_index/anchors played before chunking existed --
    # origin_index[k] is still "which whisper_words position produced the
    # k'th sanitized token sent to MFA", anchors is still keyed the same
    # way -- just built up across many chunks' own local versions of both
    # instead of in one pass, so _interpolate_stage2_words() below needs no
    # changes at all: from where it sits, this looks exactly like it did
    # before chunking, just assembled differently.
    origin_index: list[int] = []
    anchors: dict[int, tuple[float, float]] = {}
    word_offset = 0        # whisper_words consumed by chunks processed so far
    mfa_word_offset = 0    # sanitized tokens sent to MFA by chunks so far
    last_accepted_end = 0.0
    n_ok = n_degraded = n_rejected = 0
    t_start = time.monotonic()

    try:
        for c_idx, chunk_segments in enumerate(chunks):
            c_start = chunk_segments[0].get("start")
            c_end = chunk_segments[-1].get("end")
            if c_start is None or c_end is None:
                # _build_alignment_chunks() only produces this when start/end
                # were missing for the WHOLE segment (see its own docstring)
                # -- reduces to the pre-chunking whole-file behaviour.
                slice_start, slice_end = 0.0, dialog_duration_sec
                label = f"chunk {c_idx + 1}/{len(chunks)}"
            else:
                slice_start = max(0.0, c_start - chunk_edge_margin_sec)
                slice_end = min(dialog_duration_sec, c_end + chunk_edge_margin_sec)
                label = f"chunk {c_idx + 1}/{len(chunks)} [{c_start:.1f}s-{c_end:.1f}s]"

            local_words, local_origin, local_anchors = _align_chunk(
                dialog_wav, chunk_segments, slice_start, slice_end,
                dictionary, acoustic_model, g2p_model, graphemes,
                beam, retry_beam, chunk_timeout_sec, fallback_allowed,
                work_dir / f"chunk_{c_idx:04d}", label, log,
            )

            if local_anchors:
                # See this module's docstring for why this check exists and
                # what it's specifically defending against: a chunk whose
                # audio slice, despite generous padding, still ended up
                # containing the wrong stretch of dialogue -- MFA will
                # confidently fit the given text somewhere in whatever
                # it's given, so a clean exit code alone doesn't rule that
                # out. Genuinely correct neighbouring chunks should show
                # negligible overlap regardless of how much padding either
                # one has, since MFA fits the given text to its best span
                # rather than spreading it across all the room on offer.
                chunk_min_start = min(s for s, _ in local_anchors.values())
                if chunk_min_start < last_accepted_end - _CHUNK_OVERLAP_TOLERANCE_SEC:
                    log.warning(
                        "  %s: MFA anchors land %.1fs before the previous "
                        "chunk's own anchors ended -- discarding as an "
                        "implausible result (likely a mis-slice, not a real "
                        "alignment); this chunk's words will use stage 1 "
                        "(WhisperX) timing instead.",
                        label, last_accepted_end - chunk_min_start,
                    )
                    local_anchors = {}
                    n_rejected += 1
                else:
                    last_accepted_end = max(last_accepted_end, max(e for _, e in local_anchors.values()))
                    n_ok += 1
            elif local_origin:
                n_degraded += 1
            # else: chunk had no alignable text at all -- not a failure,
            # nothing to count (see _align_chunk()'s docstring).

            origin_index.extend(word_offset + i for i in local_origin)
            for k, (s, e) in local_anchors.items():
                anchors[mfa_word_offset + k] = (s, e)
            word_offset += len(local_words)
            mfa_word_offset += len(local_origin)

            if (c_idx + 1) % 10 == 0 or c_idx == len(chunks) - 1:
                log.info(
                    "    MFA: %d/%d chunk(s) done (%d ok, %d degraded, %d "
                    "rejected), %.1fs elapsed.",
                    c_idx + 1, len(chunks), n_ok, n_degraded, n_rejected,
                    time.monotonic() - t_start,
                )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if chunks and n_ok == 0:
        log.warning(
            "    MFA anchored nothing at all for %s across %d chunk(s) -- "
            "this segment's timing will be entirely stage 1 (WhisperX). "
            "Not raised as a failure (fallback_to_whisperx's whole-segment "
            "meaning is unchanged, but a zero-anchor result already "
            "degrades safely through the same interpolation path a partial "
            "one does -- see this module's docstring).",
            dialog_wav.name, len(chunks),
        )

    words = _interpolate_stage2_words(
        whisper_words, whisper_scores, whisperx_words, origin_index, anchors, log,
    )
    _enforce_monotonic(words, log)
    return words


# ── Input prep ────────────────────────────────────────────────────────────────

# Typographic punctuation WhisperX routinely emits (curly quotes/apostrophes,
# en/em dashes, the single-character ellipsis glyph) mapped to plain ASCII
# *before* the real grapheme filter below runs -- so e.g. a curly apostrophe
# has a chance to survive as a straight one if the model's alphabet includes
# straight apostrophe, rather than being dropped outright as an unrecognized
# character either way.
_TYPOGRAPHIC_TO_ASCII = str.maketrans({
    "\u2018": "'", "\u2019": "'",   # ‘ ’  → '
    "\u201c": '"', "\u201d": '"',  # “ ”  → "
    "\u2013": "-", "\u2014": "-",  # – —  → -
    "\u2026": "...",               # …    → ...
})

# Catches the decoder-hallucination-loop artifact this pipeline already
# knows about (docs/timestamp-drift-investigation.md's "What's still open":
# a single garbled, repeated-token 'word' up to 190 characters long) --
# cheap to detect, and not a real word MFA could ever place correctly
# regardless of sanitization.
_MAX_MFA_TOKEN_LEN = 20

# How far a chunk's earliest MFA anchor is allowed to land before the
# previous chunk's own accepted anchors ended, on dialog_wav's shared
# absolute timeline, before align_with_mfa() discards the whole chunk as
# an implausible (likely mis-sliced) result rather than trusting it -- see
# that function's own comment at the check itself, and this module's
# docstring, for what this is specifically defending against. Deliberately
# generous relative to normal word-boundary jitter (sub-second) and even
# genuinely overlapping dialogue between two speakers (low single-digit
# seconds at most) -- a real mis-slice, per the drift magnitudes
# docs/timestamp-drift-investigation.md observed directly, should miss by
# much more than this, not hover right at the edge of it. An internal
# tuning constant rather than a config.yaml knob: unlike chunk_target_sec
# or the padding amounts, there's no real-world signal (audio length,
# observed drift range) a person configuring this pipeline would use to
# pick a different value for their own library.
_CHUNK_OVERLAP_TOLERANCE_SEC = 3.0

# Only used as a last resort when no g2p_model is configured at all (see
# _sanitize_for_mfa()) -- there's no model to ask for a real grapheme set
# in that case, and no G2P rewriter call for a bad character to crash
# either, so this only needs to be a reasonable generic guess, not a
# verified one.
_MFA_FALLBACK_SAFE_WORD = re.compile(r"^[a-z']+(-[a-z']+)*$")

# Decorative boundary punctuation to strip before the real grapheme check --
# deliberately everything in string.punctuation EXCEPT apostrophe and
# hyphen, which can be genuinely meaningful at a word's edge rather than
# just decoration (see _sanitize_for_mfa()'s docstring).
_MFA_BOUNDARY_PUNCT = "".join(c for c in string.punctuation if c not in ("'", "-"))

# MFA's own docs for its per-word text processing describe the exact
# algorithm: "Words will be lower cased and any graphemes that were not in
# the model's training data will be removed" -- i.e. clean_up_word() in
# MFA's own g2p/generator.py, run against g2p_model.meta["graphemes"]
# before the word ever reaches the rewriter:
#
#   def clean_up_word(word, graphemes):
#       new_word = [c for c in word if c in graphemes]
#       return "".join(new_word), missing_graphemes
#
# align_one's online single-utterance path (online/alignment.py's
# tokenize_utterance_text()) doesn't run this before calling
# g2p_model.rewriter(w) -- confirmed by the traceback, which shows the raw
# pynini.lib.rewrite.Error: Composition failure propagating straight up,
# uncaught. That's the actual bug: not a specific bad character, but the
# *absence* of this filtering step on this one code path, for every
# OOV word regardless of what's in it.
#
# An earlier version of this fix guessed the alphabet instead of querying
# it -- lowercase letters, apostrophe, hyphen -- reasoning that MFA's
# English dictionaries are lowercase ASCII with contractions/compounds.
# That guess was wrong: reprocessing the same episodes on 2026-07-30 still
# hit Composition failure on every one of them, past this guessed filter.
# Apostrophe and/or hyphen most likely aren't actually in english_us_mfa's
# trained grapheme set at all -- plausible if its G2P training data never
# needed contractions as separate forms, since those are already fully
# enumerated in the dictionary and never OOV in the first place -- but
# there's no need to keep guessing which characters are safe when the
# model itself can just be asked. _get_g2p_graphemes() does that.
_G2P_GRAPHEME_CACHE: dict[str, frozenset[str]] = {}

_G2P_GRAPHEME_SCRIPT = (
    "import json, sys\n"
    "from montreal_forced_aligner.models import G2PModel\n"
    "m = G2PModel(sys.argv[1])\n"
    "print(json.dumps(sorted(m.meta.get('graphemes') or [])))\n"
)


def _get_g2p_graphemes(g2p_model: str, log: logging.LoggerAdapter) -> frozenset[str]:
    """
    Query the real grapheme alphabet of the G2P model actually in use,
    directly from MFA's own G2PModel class running inside its own conda
    env -- not a static guess in this file. Cached on disk for the
    container's lifetime (same MFA_ROOT_DIR marker-file pattern as
    _ensure_mfa_ready()), since this needs a `conda run` round trip into
    the mfa env's own python3, not just a cheap CLI call, and doesn't
    change for a given downloaded model.
    """
    if g2p_model in _G2P_GRAPHEME_CACHE:
        return _G2P_GRAPHEME_CACHE[g2p_model]

    root = Path(os.environ.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))
    cache_path = root / f".hush_g2p_graphemes_{g2p_model}.json"
    if cache_path.exists():
        graphemes = frozenset(json.loads(cache_path.read_text()))
        _G2P_GRAPHEME_CACHE[g2p_model] = graphemes
        return graphemes

    proc = subprocess.run(
        _mfa_python_cmd("-c", _G2P_GRAPHEME_SCRIPT, g2p_model),
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise MFAError(
            f"could not read the grapheme set from G2P model {g2p_model!r} "
            f"(exit {proc.returncode}) -- can't sanitize text for it safely. "
            f"stderr(tail): {proc.stderr.strip()[-2000:]}"
        )
    graphemes = frozenset(json.loads(proc.stdout.strip()))
    if not graphemes:
        raise MFAError(f"G2P model {g2p_model!r} reported an empty grapheme set.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(sorted(graphemes)))
    log.info(
        "    Cached G2P grapheme set for %s (%d characters) -> %s",
        g2p_model, len(graphemes), cache_path,
    )
    _G2P_GRAPHEME_CACHE[g2p_model] = graphemes
    return graphemes


def _sanitize_for_mfa(word: str, graphemes: Optional[frozenset[str]]) -> str:
    """
    Best-effort cleanup of one WhisperX token before it goes into align_one's
    input text file. NOT applied to the copy kept in whisper_words / eventual
    transcript.json -- those must keep WhisperX's original casing and
    attached punctuation intact for matching.py's case-sensitive comparisons
    (see transcribe.py's "Casing policy" / "Punctuation policy" docstrings).

    `graphemes` is the target G2P model's own real alphabet from
    _get_g2p_graphemes() -- None only when no g2p_model is configured at
    all, in which case there's no G2P rewriter call this could crash and
    _MFA_FALLBACK_SAFE_WORD is a good-enough generic guess instead.

    Returns '' for a token that should be dropped from MFA's input entirely
    (equivalent to WhisperX never having recognized it there) rather than
    risk it reaching the G2P layer and crashing the whole segment. This is
    the same trade-off transcribe.py already documents for whisperx.align()
    itself: "Some tokens cannot be aligned (numerals, currency symbols,
    punctuation-only tokens)" -- dropping them here before MFA ever sees
    them is a more graceful version of the same fact, not a new one.

    A word with ANY character outside `graphemes` is dropped whole, not
    stripped down to its remaining valid characters. An earlier version of
    this function did the latter -- mirroring MFA's own clean_up_word(),
    which keeps the remainder for dictionary-generation coverage -- and it
    was wrong for this use case: confirmed directly against a real
    fallback transcript (2026-07-30), which turned out to contain a short
    run of genuine German dialogue ("Zurück in ein Minute. Spaß haben.").
    Stripping ü/ß out of "Zurück"/"Spaß" character-by-character produced
    "zurck"/"spa" -- fabricated strings that are *more* likely to crash
    G2P composition than either the original word or no word at all, since
    they're not real English orthography and not what was actually said.
    MFA's batch tooling can afford the partial-keep trade-off because a
    composition failure there just skips one dictionary entry inside a
    try/except; align_one has no such net, so the safer trade for this
    pipeline is: use a word as MFA sees it, or not at all.

    Decorative boundary punctuation (a trailing comma/period/exclamation
    mark, a leading quote) is stripped BEFORE that whole-word check, not
    left to trigger it -- transcribe.py deliberately keeps this attached
    for matching.py's purposes, but it was never part of the word's actual
    spelling, so treating it the same as an embedded foreign character
    would (and, in an earlier version of this function, did) wrongly drop
    the large majority of words in any real transcript, since most
    sentence-final words carry exactly this kind of attached punctuation.
    Apostrophe and hyphen are deliberately excluded from what counts as
    "boundary punctuation" here, since both can be genuinely meaningful at
    a word's edge (the dropped-g colloquialisms config.yaml's own comments
    call out -- "sayin'", "rockin'" -- end in a real apostrophe, not a
    decorative one) -- whether that survives into the word sent to MFA is
    left to the graphemes check right after, same as any other character.
    """
    w = word.translate(_TYPOGRAPHIC_TO_ASCII).lower().strip(_MFA_BOUNDARY_PUNCT)
    if not w:
        return ""
    if graphemes is not None:
        if any(c not in graphemes for c in w):
            return ""
    elif not _MFA_FALLBACK_SAFE_WORD.match(w):
        return ""
    if len(w) > _MAX_MFA_TOKEN_LEN:
        return ""
    return w


def _flatten_whisper_words(whisper_segments: list[dict]) -> tuple[list[str], list[float]]:
    """
    Pull (word_tokens, per_word_score) out of pre-alignment WhisperX
    segments. There's no per-word breakdown at this stage (that's what
    we're computing), so every word in a segment inherits that segment's
    own avg_logprob, converted from log-space to a 0-1-ish scale via
    exp() — the same shape of number as whisperx.align()'s per-word
    `score`, just one value per *segment* rather than per word.

    This is a real, deliberate reduction in granularity versus the
    whisperx.align() path's true per-word confidence — flagged here rather
    than silently passed off as equivalent. If your installed whisperx
    version's transcribe() call exposes word-level confidence pre-alignment
    (check `result["segments"][i].get("words")` before assuming it doesn't
    — this varies by version), prefer wiring that through instead of this
    per-segment fallback; nothing downstream (_confident_mfa_anchors(),
    _interpolate_stage2_words()) cares which granularity it's given, since
    a score is just carried through by index either way, never computed
    from anything MFA returns.
    """
    words: list[str] = []
    scores: list[float] = []
    for seg in whisper_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        seg_words = text.split()
        avg_logprob = seg.get("avg_logprob")
        seg_score = math.exp(avg_logprob) if avg_logprob is not None else 0.5
        words.extend(seg_words)
        scores.extend([seg_score] * len(seg_words))
    return words, scores


def _build_alignment_chunks(
    whisper_segments: list[dict], chunk_target_sec: float, log: logging.LoggerAdapter,
) -> list[list[dict]]:
    """
    Group whisper_segments into runs no longer than chunk_target_sec,
    splitting only BETWEEN segments, never within one -- so every chunk
    boundary this produces falls exactly where WhisperX's own segmentation
    already found a natural break, without this function needing any
    VAD/silence logic of its own. This decides how to group the TEXT only;
    see align_with_mfa() and this module's docstring for why the AUDIO
    each chunk is actually aligned against is deliberately not a tight
    slice at these same boundaries.

    Defensive fallback: whisper_segments are expected to carry "start" and
    "end" (standard Whisper/WhisperX segment fields; this codebase has
    just never had a reason to read them before now, since the whole-file
    design didn't need them) -- but that's never been directly confirmed
    against every whisperx version this pipeline might run against, the
    way e.g. avg_logprob's presence is discussed in
    _flatten_whisper_words()'s own docstring. If ANY segment is missing
    either one, this doesn't guess -- it logs once and returns everything
    as a single chunk, i.e. exactly the pre-chunking whole-segment
    behaviour, rather than silently mis-grouping some segments correctly
    and others not.
    """
    if any(seg.get("start") is None or seg.get("end") is None for seg in whisper_segments):
        log.warning(
            "    whisper_segments missing start/end on at least one entry -- "
            "can't chunk by natural boundaries safely, sending this whole "
            "segment to MFA as one chunk (the pre-chunking behaviour). If "
            "this shows up routinely, check what your installed whisperx "
            "version's transcribe() actually returns per segment."
        )
        return [list(whisper_segments)]

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_start = 0.0
    for seg in whisper_segments:
        if not current:
            current = [seg]
            current_start = seg["start"]
        elif seg["end"] - current_start <= chunk_target_sec:
            current.append(seg)
        else:
            chunks.append(current)
            current = [seg]
            current_start = seg["start"]
    if current:
        chunks.append(current)

    oversized = [c for c in chunks if len(c) == 1 and c[0]["end"] - c[0]["start"] > chunk_target_sec]
    if oversized:
        longest = max(c[0]["end"] - c[0]["start"] for c in oversized)
        log.warning(
            "    %d whisper segment(s) individually exceed the %.0fs MFA "
            "chunk target and couldn't be split further (no word-level "
            "timing exists yet to split on) -- sent to MFA as their own, "
            "larger-than-usual chunk. Longest: %.1fs.",
            len(oversized), chunk_target_sec, longest,
        )
    return chunks


def _slice_wav(src: Path, start_sec: float, end_sec: float, out: Path, log: logging.LoggerAdapter) -> None:
    """
    Extract [start_sec, end_sec) from src into out. Same approach and same
    reasoning as steps/segment.py's own _split(): -ss before -i for fast,
    exact input-side seeking on PCM WAV, -c copy since no re-encoding is
    needed or wanted -- MFA gets exactly dialog_wav's own sample
    rate/channel layout, just a short span of it (a chunk's own claimed
    boundaries plus a small fixed edge margin -- see align_with_mfa() and
    this module's docstring for why that margin is small and fixed rather
    than a generous, searched window).
    """
    run_cmd([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start_sec),
        "-i", str(src),
        "-t", str(max(end_sec - start_sec, 0.0)),
        "-c", "copy",
        str(out),
    ], log)


def _align_chunk(
    dialog_wav: Path,
    chunk_segments: list[dict],
    slice_start: float,
    slice_end: float,
    dictionary: str,
    acoustic_model: str,
    g2p_model: Optional[str],
    graphemes: Optional[frozenset[str]],
    beam: int,
    retry_beam: int,
    timeout_sec: float,
    fallback_allowed: bool,
    chunk_dir: Path,
    label: str,
    log: logging.LoggerAdapter,
) -> tuple[list[str], list[int], dict[int, tuple[float, float]]]:
    """
    Run one chunk's worth of alignment: sanitize its own text, slice
    [slice_start, slice_end) of dialog_wav, invoke align_one, parse and
    reconcile the result -- everything align_with_mfa() used to do once
    for a whole segment, scoped to one chunk instead.

    slice_start/slice_end are the chunk's own claimed boundaries plus a
    small, fixed edge margin (alignment.mfa.chunk_edge_margin_sec) -- NOT
    a generously padded search window. Two earlier versions of this tried
    generous padding, once handed to align_one directly and once trimmed
    first via real silence detection; both were validated against real
    audio and found to make timing worse than plain whisperx.align(), via
    two DIFFERENT mechanisms (a chunk's last word smeared across trailing
    padding; a chunk's search window reaching past its real content into
    unrelated preceding audio with no clean silence gap to stop it). See
    this module's own docstring for the full account of both. A chunk
    affected by genuine WhisperX timestamp drift will generally fail to
    align in this tight a window -- and correctly degrade to stage 1
    (WhisperX) timing for its own words via the same non-fatal path any
    other per-chunk failure uses, rather than risk a confidently-wrong
    result from a window generous enough to search but not sure what it
    might find.

    Returns (local_words, local_origin_index, local_anchors):
      local_words        -- this chunk's own flattened word tokens, in
                             order (_flatten_whisper_words(chunk_segments)).
      local_origin_index -- local_origin_index[k] is the position in
                             local_words that the k'th sanitized,
                             sent-to-MFA token came from -- same role
                             align_with_mfa()'s own origin_index used to
                             play for the whole segment, scoped to this
                             chunk; the caller shifts these into global
                             whisper_words-space.
      local_anchors       -- {k: (start, end)}, keyed the same way as
                              local_origin_index. start/end are already
                              converted to dialog_wav's own absolute
                              timeline (slice_start already added back in
                              -- the caller does NOT need to offset these
                              again, only re-key them into the global
                              mfa_input_words-position space). Empty ({})
                              whenever this chunk contributed nothing --
                              no alignable text, or align_one's own
                              attempts were all exhausted and
                              fallback_allowed is true -- which is not an
                              error: a position with no anchor already
                              means "use stage 1 timing here" regardless
                              of WHY it has none, so a degraded chunk needs
                              no separate signal beyond an empty dict.

    Raises MFAError only when fallback_allowed is False and this chunk's
    own align_one attempts were all exhausted -- i.e. only when config
    says a chunk-level failure should be fatal rather than degrade to
    stage 1 timing for just this chunk's words. Never raises for "chunk
    had no alignable text" -- MFA was never going to be asked to do
    anything for zero input, so there's nothing that failed.
    """
    local_words, _local_scores = _flatten_whisper_words(chunk_segments)

    local_mfa_input_words: list[str] = []
    local_origin_index: list[int] = []
    for i, w in enumerate(local_words):
        s = _sanitize_for_mfa(w, graphemes)
        if s:
            local_mfa_input_words.append(s)
            local_origin_index.append(i)
    if not local_mfa_input_words:
        log.debug("    %s: no alignable text after sanitizing -- skipped.", label)
        return local_words, local_origin_index, {}

    chunk_dir.mkdir(parents=True, exist_ok=True)
    wav_in = chunk_dir / "utt.wav"
    txt_in = chunk_dir / "utt.txt"
    out_dir = chunk_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    # See _find_output_textgrid()'s own docstring for why this exact path,
    # not just out_dir, is passed as align_one's OUTPUT_PATH -- unchanged
    # reasoning from before chunking, just per-chunk now.
    tg_target = out_dir / f"{wav_in.stem}.TextGrid"

    _slice_wav(dialog_wav, slice_start, slice_end, wav_in, log)
    txt_in.write_text(" ".join(local_mfa_input_words), encoding="utf-8")

    # Same two-attempt shape align_with_mfa() used before chunking (see
    # git history for that version) -- G2P first if configured, one
    # dictionary-only retry on a Composition failure. A timeout is now
    # ALSO treated as retry-worthy rather than an immediately-fatal,
    # uncaught subprocess.TimeoutExpired: a timeout doesn't tell us WHY a
    # chunk was slow, and disabling G2P is a real (if not guaranteed) way
    # a retry could come in faster, at the bounded cost of at most one
    # more timeout_sec before giving up on this one chunk.
    attempts = [("with G2P", g2p_model)] if g2p_model else [("dictionary-only", None)]
    if g2p_model:
        attempts.append(("dictionary-only retry (G2P disabled)", None))

    returncode: Optional[int] = None
    stdout = stderr = ""
    timed_out = False
    for attempt_label, attempt_g2p_model in attempts:
        if tg_target.exists():
            tg_target.unlink()
        cmd = _mfa_cmd(
            "align_one",
            str(wav_in), str(txt_in), dictionary, acoustic_model, str(tg_target),
            "--beam", str(beam),
            "--retry_beam", str(retry_beam),
            "--clean",
            "--single_speaker",
        )
        if attempt_g2p_model:
            cmd += ["--g2p_model_path", attempt_g2p_model]

        log.debug("    MFA (%s, %s): %s", label, attempt_label, " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            returncode, stdout, stderr, timed_out = proc.returncode, proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired:
            returncode, stdout, stderr, timed_out = None, "", "", True

        if returncode == 0:
            break
        if not timed_out and "Composition failure" not in stderr:
            break  # a different, non-G2P failure -- retrying without G2P won't help
        if timed_out:
            log.debug(
                "    %s: %s attempt didn't finish within %.0fs.",
                label, attempt_label, timeout_sec,
            )
        else:
            log.warning(
                "    %s: MFA G2P composition failure surviving sanitization "
                "(%s) -- retrying with G2P disabled.", label, attempt_label,
            )

    if returncode != 0:
        if timed_out:
            reason = f"align_one did not finish within {timeout_sec:.0f}s (alignment.mfa.chunk_timeout_sec)"
        else:
            hint = ""
            if "Composition failure" in stderr:
                hint = (
                    " (G2P composition failure that persisted even on the "
                    "dictionary-only retry with G2P disabled -- suggests "
                    "something other than a single bad OOV word.)"
                )
            reason = f"align_one exited {returncode}:{hint} {stderr.strip()[-1000:]}"
        if not fallback_allowed:
            raise MFAError(f"{label}: {reason}")
        log.warning(
            "  %s: MFA alignment failed -- this chunk's words will use "
            "stage 1 (WhisperX) timing instead. Reason: %s", label, reason,
        )
        return local_words, local_origin_index, {}

    try:
        textgrid_path = _find_output_textgrid(out_dir, tg_target, wav_in.stem, stdout, stderr, chunk_dir)
        mfa_words = _parse_textgrid_words(textgrid_path)
    except MFAError as exc:
        if not fallback_allowed:
            raise
        log.warning(
            "  %s: %s -- this chunk's words will use stage 1 (WhisperX) "
            "timing instead.", label, exc,
        )
        return local_words, local_origin_index, {}

    local_anchors_relative = _confident_mfa_anchors(local_mfa_input_words, mfa_words, log)
    # +slice_start here, once, is what lets the caller treat every chunk's
    # anchors as already being on dialog_wav's own shared absolute
    # timeline -- see this function's own docstring.
    local_anchors = {k: (s + slice_start, e + slice_start) for k, (s, e) in local_anchors_relative.items()}
    return local_words, local_origin_index, local_anchors


def _find_output_textgrid(
    out_dir: Path, tg_target: Path, stem: str, stdout: str, stderr: str, work_dir: Path,
) -> Path:
    """
    Locate align_one's output TextGrid, without full confidence about
    exactly where a given MFA version puts it for a single-utterance call
    (see the comment at this function's call site for why). stdout/stderr
    are passed as plain strings rather than a CompletedProcess so this can
    be called uniformly regardless of whether the caller has a real
    subprocess.CompletedProcess to hand it (a timed-out attempt never
    produces one at all). Checked in order, most-likely-correct first:

      1. tg_target -- the exact file path passed as align_one's own
         OUTPUT_PATH argument, on the theory that it's read as a literal
         target file for a single-utterance command (unlike corpus-mode
         `align`, which needs a directory).
      2. Anything matching *.TextGrid anywhere under out_dir, in case
         align_one still applies its own naming convention inside
         whatever it was given.
      3. Anything matching *.TextGrid anywhere under work_dir (one level
         up), in case OUTPUT_PATH's parent directory was created but MFA
         resolved the actual write location some other way entirely.

    If none of those hit, the exception below includes align_one's own
    captured stdout (which likely says where it thinks it wrote the
    result -- discarding that on a "successful" exit is what made the
    first version of this error message a dead end) and an actual
    directory listing, so a real failure here is diagnosable from the log
    alone rather than needing another round of blind guessing.
    """
    if tg_target.exists():
        return tg_target

    matches = list(out_dir.rglob("*.TextGrid")) if out_dir.exists() else []
    if matches:
        return matches[0]

    matches = list(work_dir.rglob("*.TextGrid"))
    if matches:
        return matches[0]

    def _listing(p: Path) -> str:
        if not p.exists():
            return f"{p} does not exist"
        return "\n".join(f"  {f.relative_to(p)}" for f in sorted(p.rglob("*"))) or f"{p} exists but is empty"

    raise MFAError(
        f"align_one exited 0 (reported success) but no .TextGrid was found at the "
        f"expected path ({tg_target}) or anywhere under {out_dir} or {work_dir}.\n"
        f"align_one's own stdout (tail):\n{stdout.strip()[-2000:]}\n"
        f"align_one's own stderr (tail):\n{stderr.strip()[-2000:]}\n"
        f"Contents of {work_dir}:\n{_listing(work_dir)}"
    )


# ── Output parsing ────────────────────────────────────────────────────────────

def _parse_textgrid_words(textgrid_path: Path) -> list[tuple[str, float, float]]:
    """
    Returns [(word, start, end), ...] in time order, silence/empty
    intervals dropped. MFA's default output ("long_textgrid" format) has a
    'words' tier and a 'phones' tier — only 'words' is used here; the
    phones tier is available in the same file if finer-grained boundary
    work is ever wanted later.
    """
    from praatio import textgrid  # deferred: only needed on this path

    tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=True)
    tier_name = "words" if "words" in tg.tierNames else tg.tierNames[0]
    word_tier = tg.getTier(tier_name)

    out: list[tuple[str, float, float]] = []
    for start, end, label in word_tier.entries:
        label = label.strip()
        if not label or label == "<eps>":
            # '' is MFA's silence marker; '<eps>' is Kaldi's internal
            # epsilon/no-output transition, which apparently can also show
            # up directly in the words tier for some inputs. Confirmed
            # against a real run, not theoretical: 1,804 of 12,895 "words"
            # in one transcript were the literal string "<eps>", not real
            # content -- silently inflating word counts and (more
            # importantly) standing in for genuinely-unaligned words in a
            # way that made them look like recognized-but-oddly-named
            # tokens rather than what they actually are, which is the same
            # thing an empty silence interval is: nothing to report here.
            continue
        if label in ("<unk>", "spn"):
            # MFA's marker for "detected speech here, but couldn't place it
            # in the dictionary/G2P" — keep it as a low-confidence token
            # rather than silently dropping the interval; a real word did
            # occur here even though MFA can't tell us which.
            label = "<unk>"
        out.append((label, float(start), float(end)))
    return out


# ── Anchoring MFA's output against WhisperX's original words ────────────────
#
# Two-step process, replacing the single-function reconstruction this used
# to be (still visible in git history as _restore_original_words() /
# _match_scores_and_words()):
#
#   1. _confident_mfa_anchors() decides WHICH mfa_input_words positions we
#      trust MFA's timing for at all -- direct 1:1 index in the common
#      case, a conservative equal-only difflib reconciliation otherwise.
#      Returns TIMES only, keyed by position -- never touches word text.
#   2. _interpolate_stage2_words() turns those anchors into a full,
#      same-length-as-whisper_words output list, filling every
#      non-anchored position (dropped by sanitization, or not confidently
#      matched back) by affine-warping WhisperX's own relative timing
#      between the nearest anchors on each side. Word text is
#      whisper_words[i] at every position, full stop -- anchored or
#      interpolated, this function has no code path that can produce
#      anything else.
#
# The old design's fallback reconstructed the ENTIRE word list from
# scratch via difflib whenever the tier-count assumption didn't hold,
# including reaching for MFA's own raw dictionary-form label as the
# "word" wherever that reconstruction had nothing of WhisperX's to prefer
# instead. On real movie-length, repetition-heavy dialogue that assumption
# failed often enough to matter (see _confident_mfa_anchors()'s own
# docstring for how often, on real input), and the failure mode compounded
# a timing problem with a text-correctness one at the same time: a word
# WhisperX transcribed correctly could get silently swapped for a
# neighbour, or replaced outright by a label MFA guessed but this pipeline
# never validated against anything. Splitting "where do I trust MFA's
# time" from "what do I show as the word" -- and making the second
# question always whisper_words[i], never conditionally MFA's label --
# closes that off structurally: the worst this design can now do to a
# word's TEXT is nothing at all, because there's no path left where MFA's
# own label ever reaches the output.

def _norm(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


def _confident_mfa_anchors(
    mfa_input_words: list[str],
    mfa_words: list[tuple[str, float, float]],
    log: logging.LoggerAdapter,
) -> dict[int, tuple[float, float]]:
    """
    Decide which positions in mfa_input_words (equivalently, origin_index)
    this pipeline trusts MFA's own (start, end) for. Returns {k: (start,
    end)} — a partial map in general, never word text.

    Primary path — direct index, not string matching: forced alignment
    (that's what "forced" means in the name) places every word it's
    *given* somewhere in the audio; it doesn't invent, merge, split, or
    drop word-tier entries relative to its own input list. So once
    _parse_textgrid_words() has already dropped the genuine silence/<eps>
    intervals (see that function's own docstring), what's left should
    correspond 1:1, in order, to mfa_input_words — i.e.
    len(mfa_words) == len(mfa_input_words) — and mfa_words[k] IS the
    aligned form of mfa_input_words[k], full stop. This is the fast,
    common-case path: every position becomes an anchor, no difflib
    involved at all.

    Fallback path — conservative difflib reconciliation, used when that
    count assumption doesn't hold: NOT a rare, hypothetical case on real
    input (see the warning below, and docs/timestamp-drift-investigation.md's
    "I've been sayin' it" passage for the kind of dense repetition that
    tends to trigger it) — routine enough that it needs to degrade safely
    rather than assume it won't happen. The two normalized token
    sequences (mfa_input_words vs. MFA's own labels — both already
    lowercase/punctuation-free, so this is a much better-conditioned
    comparison than diffing against WhisperX's original casing/punctuation
    would be) are reconciled with difflib, but ONLY 'equal' opcode
    stretches are trusted as anchors. 'replace'/'insert'/'delete' spans
    contribute NO anchors at all, on purpose — a wrong guess here would
    hand a word a plausible-looking but incorrect timestamp with no way
    to tell from the output alone, whereas a position this function
    simply doesn't cover just falls through to interpolation in
    _interpolate_stage2_words(), which is always safe by construction.
    Conservative in the same spirit as _parse_textgrid_words() keeping
    "<unk>"/"spn" as a visible low-confidence token rather than silently
    dropping it: prefer a visible, bounded gap over a silent, confident-
    looking mistake.
    """
    if len(mfa_words) == len(mfa_input_words):
        return {k: (start, end) for k, (_label, start, end) in enumerate(mfa_words)}

    input_norm = [_norm(w) for w in mfa_input_words]
    label_norm = [_norm(label) for label, _, _ in mfa_words]
    sm = difflib.SequenceMatcher(None, input_norm, label_norm, autojunk=False)
    anchors: dict[int, tuple[float, float]] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            _label, start, end = mfa_words[j1 + offset]
            anchors[i1 + offset] = (start, end)

    log.warning(
        "MFA's word tier has %d real entries but %d words were sent to it "
        "(after sanitization) -- expected these to match 1:1, since forced "
        "alignment shouldn't add/drop/merge word-tier entries relative to "
        "its own input. Reconciled the two normalized token sequences and "
        "recovered %d/%d position(s) as confident anchors (exact-match "
        "stretches only); the rest of this segment's words -- including "
        "every word never sent to MFA to begin with -- get an interpolated "
        "timestamp instead (see _interpolate_stage2_words()), never a "
        "guessed word or an MFA-derived label. If you see this warning "
        "routinely rather than as a one-off, please report it together "
        "with the segment's dialog audio -- it points at an MFA behavior "
        "this pipeline doesn't yet account for.",
        len(mfa_words), len(mfa_input_words), len(anchors), len(mfa_input_words),
    )
    return anchors


def _fill_gap(
    words: list[Optional[dict]],
    gap: list[int],
    whisper_words: list[str],
    whisper_scores: list[float],
    whisperx_words: list[dict],
    lo_i: Optional[int],
    hi_i: Optional[int],
    corrected_lo: Optional[float],
    corrected_hi: Optional[float],
) -> None:
    """
    Fill words[i] in place for every i in `gap` — a run of whisper_words
    positions with no confident MFA anchor of their own (see
    _confident_mfa_anchors()). Word text is always whisper_words[i];
    score is always whisper_scores[i] — only start/end vary by branch.

    UNBOUNDED (lo_i or hi_i is None — this gap touches the start/end of
    the segment, so there's no anchor on that side to interpolate
    toward): WhisperX's own raw per-word timing, untouched. Deliberately
    NOT extrapolated from the single nearest anchor on the other side —
    interpolating BETWEEN two confirmed points is safe by construction
    (it can't disagree with either one), extrapolating beyond the last
    confirmed point is a genuinely different, weaker claim this design
    doesn't make.

    BOUNDED (both sides anchored): affine-warp WhisperX's own
    [whisperx_words[lo_i].end, whisperx_words[hi_i].start] sub-timeline
    onto MFA's corrected [corrected_lo, corrected_hi] — same shape (each
    word's relative position and duration within the gap), rescaled to
    fit the corrected span exactly. This is deliberately a scale-and-shift
    of WhisperX's OWN relative timing, not a fresh estimate from word
    count or duration alone: preserves whatever real signal WhisperX's
    timing has about which words in the gap were short/long, quick/slow,
    while still being anchored to MFA's more trustworthy endpoints. Falls
    back to evenly-spaced slots across the whole gap if WhisperX's own
    timing for this stretch isn't usable for that (any word in the gap
    null, or the boundary span itself zero/negative width) — still
    correctly bounded either way, just without the finer within-gap
    shape when the input to preserve isn't there. See align_mfa.py's
    module docstring's account of why an earlier version of this module
    trusted MFA's own drift-prone WhisperX-internal timing far less than
    the corrected anchors either side of it — the same reasoning is why
    the correction is anchored on confirmed MFA points, not on
    WhisperX's own gap boundary as reported.
    """
    if lo_i is None or hi_i is None:
        for i in gap:
            w = whisperx_words[i]
            words[i] = {
                "word": whisper_words[i],
                "start": w.get("start"),
                "end": w.get("end"),
                "score": round(whisper_scores[i], 3),
            }
        return

    raw_lo = whisperx_words[lo_i].get("end")
    raw_hi = whisperx_words[hi_i].get("start")
    usable_shape = (
        raw_lo is not None and raw_hi is not None and raw_hi > raw_lo
        and all(
            whisperx_words[i].get("start") is not None and whisperx_words[i].get("end") is not None
            for i in gap
        )
    )
    corrected_span = max(corrected_hi - corrected_lo, 0.0)

    if usable_shape:
        scale = corrected_span / (raw_hi - raw_lo)
        for i in gap:
            w = whisperx_words[i]
            words[i] = {
                "word":  whisper_words[i],
                "start": round(corrected_lo + (w["start"] - raw_lo) * scale, 3),
                "end":   round(corrected_lo + (w["end"]   - raw_lo) * scale, 3),
                "score": round(whisper_scores[i], 3),
            }
        return

    # Evenly-spaced fallback -- still correctly bounded by MFA's two
    # anchors, just without WhisperX's own within-gap shape to warp.
    step = corrected_span / len(gap)
    t = corrected_lo
    for i in gap:
        words[i] = {
            "word":  whisper_words[i],
            "start": round(t, 3),
            "end":   round(t + step, 3),
            "score": round(whisper_scores[i], 3),
        }
        t += step


def _interpolate_stage2_words(
    whisper_words: list[str],
    whisper_scores: list[float],
    whisperx_words: list[dict],
    origin_index: list[int],
    anchors_by_k: dict[int, tuple[float, float]],
    log: logging.LoggerAdapter,
) -> list[dict]:
    """
    Build the final, same-length-as-whisper_words stage 2 output: MFA's
    corrected timing at every anchored position, an interpolated estimate
    everywhere else (see _fill_gap()). Word text and score always come
    from whisper_words/whisper_scores by direct index — this function has
    no code path that can substitute either one for anything else.

    anchors_by_k is keyed by position in mfa_input_words (equivalently
    origin_index) — the space _confident_mfa_anchors() works in, since
    that's what it compared against MFA's own tier. Re-keyed here to
    whisper_words-space (anchor_by_i) via origin_index, since that's the
    space every OTHER list in this function — whisper_words,
    whisper_scores, whisperx_words — is already in.

    A segment where zero anchors were recoverable degrades to exactly
    WhisperX's own stage 1 output, unmodified, word for word — not an
    error case, just the bottom of a continuous scale from "MFA anchored
    nothing" up to "MFA anchored everything," handled by the same code
    path throughout rather than a separate cutoff/threshold that would
    otherwise need its own justification for wherever it was drawn.
    """
    n = len(whisper_words)

    stage1_usable = len(whisperx_words) == n
    if not stage1_usable:
        log.warning(
            "  WhisperX's own per-word output (%d words) doesn't line up "
            "1:1 with the %d words recognized for this segment -- can't "
            "use it as an interpolation basis. Every word without its own "
            "confident MFA anchor is left with a null timestamp instead "
            "(same as an unalignable word has always had), rather than "
            "risk interpolating against a misaligned reference.",
            len(whisperx_words), n,
        )

    anchor_by_i = {origin_index[k]: t for k, t in anchors_by_k.items()}
    anchored = sorted(anchor_by_i)

    words: list[Optional[dict]] = [None] * n
    for i in anchored:
        start, end = anchor_by_i[i]
        words[i] = {
            "word":  whisper_words[i],
            "start": round(start, 3),
            "end":   round(end, 3),
            "score": round(whisper_scores[i], 3),
        }

    bounds = [None] + anchored + [None]
    for lo, hi in zip(bounds, bounds[1:]):
        start_i = 0 if lo is None else lo + 1
        end_i = n if hi is None else hi
        gap = list(range(start_i, end_i))
        if not gap:
            continue
        if not stage1_usable:
            for i in gap:
                words[i] = {
                    "word": whisper_words[i], "start": None, "end": None,
                    "score": round(whisper_scores[i], 3),
                }
            continue
        corrected_lo = anchor_by_i[lo][1] if lo is not None else None
        corrected_hi = anchor_by_i[hi][0] if hi is not None else None
        _fill_gap(
            words, gap, whisper_words, whisper_scores, whisperx_words,
            lo, hi, corrected_lo, corrected_hi,
        )

    return words  # type: ignore[return-value]


def _enforce_monotonic(words: list[dict], log: logging.LoggerAdapter) -> None:
    """
    Defensive final pass over the WHOLE word list: nudge any word whose
    start would precede the previous word's end forward by the minimum
    amount needed to keep the sequence non-decreasing, in place.

    Should be a no-op in the overwhelming majority of runs — MFA's own
    tier is monotonic by construction, and the affine warp in _fill_gap()
    preserves order exactly — but Step 5 (mute) depends on these
    intervals being sane, and this function is the one place in the
    pipeline that combines timing from two different sources (MFA anchors
    and warped/raw WhisperX timing) across gaps of varying width, so this
    stays cheap, unconditional insurance rather than an assumption argued
    for instead of checked.
    """
    prev_end: Optional[float] = None
    nudged = 0
    for w in words:
        if w["start"] is None or w["end"] is None:
            continue
        if prev_end is not None and w["start"] < prev_end:
            shift = prev_end - w["start"]
            w["start"] = round(w["start"] + shift, 3)
            w["end"] = round(w["end"] + shift, 3)
            nudged += 1
        if w["end"] < w["start"]:
            w["end"] = w["start"]
        prev_end = w["end"]
    if nudged:
        log.warning(
            "  %d word(s) had their timing nudged forward by "
            "_enforce_monotonic() to preserve chronological order after "
            "interpolation -- expected to be rare; if this fires often, "
            "the affine warp's inputs are worth a closer look.",
            nudged,
        )
