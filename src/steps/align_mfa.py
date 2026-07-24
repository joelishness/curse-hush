"""
Word-level alignment via Montreal Forced Aligner (MFA) — the default
alignment engine as of this version (config.yaml's alignment.backend: mfa).
Falls back to whisperx.align()'s wav2vec2/CTC alignment per-segment on
failure; see that file for the reasoning behind the switch. Short version:
whisperx.align() aligns *within* whatever segment boundary WhisperX's own
transcribe() pass already committed to, so a chunk-boundary timing error
upstream (see docs/timestamp-drift-investigation.md) propagates straight
through it uncorrected. MFA aligns the whole audio handed to it against the
whole recognized text in one pass, with no dependency on WhisperX's internal
~30s decode chunks at all, so it has nothing to inherit a chunk-boundary
error from in the first place. Validated directly, not just reasoned about
— see the same doc for the before/after numbers.

Imported unconditionally at the top of transcribe.py (not behind a
conditional "only if backend == mfa" guard) — this module's own top-level
imports are stdlib + utils only, so there's no cost to importing it even on
a job that ends up using whisperx for every segment.

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
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import difflib
import math
from pathlib import Path
from typing import Optional

from utils import cfg_get


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

    proc = subprocess.run(_mfa_cmd("server", "init"), capture_output=True, text=True, timeout=300)
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
        proc = subprocess.run(
            _mfa_cmd("model", "download", model_type, name),
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise MFAError(
                f"mfa model download {model_type} {name} failed "
                f"(exit {proc.returncode}).\nstderr(tail): {proc.stderr[-2000:]}"
            )

    marker.write_text("initialized by profanity-hush's align_mfa.py\n")


# ── Public entry point ───────────────────────────────────────────────────────

def align_with_mfa(
    dialog_wav: Path,
    whisper_segments: list[dict],
    cfg: dict,
    log: logging.LoggerAdapter,
    work_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Align WhisperX's recognized text against dialog_wav using MFA, returning
    the same [{"word", "start", "end", "score"}, ...] schema that
    whisperx.align() produces in transcribe.py — a drop-in replacement at
    that call site, not a new data shape downstream needs to know about.

    whisper_segments — result["segments"] from wx_model.transcribe(), i.e.
        WhisperX's *pre-alignment* output: text (+ usually avg_logprob) per
        internally-decoded chunk, no word-level breakdown yet. That
        breakdown is exactly what this function produces, via MFA instead
        of whisperx.align().

    Raises MFAError if alignment fails and
    alignment.mfa.fallback_to_whisperx is false. If true (the default),
    the caller (transcribe.py) is expected to catch MFAError and retry that
    segment through the whisperx.align() path instead — this function
    itself does not know how to do that fallback, since it has no access to
    the whisperx align model/metadata transcribe.py is holding.
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

    # Checked here, before _ensure_mfa_ready() (which is otherwise the first
    # thing to shell out) rather than only right before the align_one call
    # below -- a missing conda install should fail once, fast, with this
    # specific message. Left where it was, it wouldn't fire until after
    # _ensure_mfa_ready() had already tried and failed to run `conda run ...`
    # itself, surfacing as a much less clear raw FileNotFoundError instead.
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

    whisper_words, whisper_scores = _flatten_whisper_words(whisper_segments)
    transcript_text = " ".join(whisper_words)

    tmp_ctx = None
    if work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="mfa_align_")
        work_dir = Path(tmp_ctx.name)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        wav_in  = work_dir / "utt.wav"
        txt_in  = work_dir / "utt.txt"
        out_dir = work_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Passed as the exact target FILE path, not a bare directory --
        # align_one handles exactly one input, unlike corpus-mode `align`
        # (which needs a directory since it's writing one file per input
        # across many). A previous version of this passed just out_dir and
        # assumed align_one would place a conventionally-named file inside
        # it, matching `align`'s own convention -- that assumption produced
        # "MFA reported success but no .TextGrid found", confirmed against
        # a real run where align_one's exit code was 0 (a real success,
        # not one of the earlier root/PATH/passwd failures) but nothing
        # existed at the path this code was looking in. This is the more
        # likely correct reading of OUTPUT_PATH for a single-utterance
        # command, not a confirmed one -- see _find_output_textgrid's
        # fallback search and the error it raises if this guess is ALSO
        # wrong; that error is now rich enough to settle it either way
        # from one more run, rather than needing a third guess blind.
        tg_target = out_dir / f"{wav_in.stem}.TextGrid"
        _prepare_input_pair(dialog_wav, transcript_text, wav_in, txt_in)

        cmd = _mfa_cmd(
            "align_one",
            str(wav_in), str(txt_in), dictionary, acoustic_model, str(tg_target),
            "--beam", str(beam),
            "--retry_beam", str(retry_beam),
            "--clean",
            "--single_speaker",
        )
        if g2p_model:
            cmd += ["--g2p_model_path", g2p_model]

        log.debug("    MFA: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise MFAError(
                f"mfa align_one exited {proc.returncode} on {dialog_wav.name}: "
                f"{proc.stderr.strip()[-2000:]}"
            )

        textgrid_path = _find_output_textgrid(out_dir, tg_target, wav_in.stem, proc, work_dir)
        mfa_words = _parse_textgrid_words(textgrid_path)

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    scores = _match_scores(whisper_words, whisper_scores, [w for w, _, _ in mfa_words])
    words = [
        {"word": w, "start": round(start, 3), "end": round(end, 3), "score": round(score, 3)}
        for (w, start, end), score in zip(mfa_words, scores)
    ]
    return words


# ── Input prep ────────────────────────────────────────────────────────────────

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
    per-segment fallback; the matching logic in _match_scores() doesn't
    care which granularity it's given.
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


def _prepare_input_pair(dialog_wav: Path, transcript_text: str, wav_out: Path, txt_out: Path) -> None:
    """
    mfa align_one takes exactly one sound file + one text file — no corpus
    directory/speaker-folder structure needed (that's `mfa align`, for
    multi-file datasets; see docs/timestamp-drift-investigation.md for why
    align_one is the right subcommand here, not align).
    """
    shutil.copy(dialog_wav, wav_out)
    txt_out.write_text(transcript_text, encoding="utf-8")


def _find_output_textgrid(
    out_dir: Path, tg_target: Path, stem: str, proc: "subprocess.CompletedProcess", work_dir: Path,
) -> Path:
    """
    Locate align_one's output TextGrid, without full confidence about
    exactly where a given MFA version puts it for a single-utterance call
    (see the comment at this function's call site for why). Checked in
    order, most-likely-correct first:

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
        f"align_one's own stdout (tail):\n{proc.stdout.strip()[-2000:]}\n"
        f"align_one's own stderr (tail):\n{proc.stderr.strip()[-2000:]}\n"
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


# ── Score carry-over ──────────────────────────────────────────────────────────

def _norm(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


def _match_scores(
    whisper_words: list[str],
    whisper_scores: list[float],
    mfa_words: list[str],
) -> list[float]:
    """
    Map WhisperX's recognition-confidence scores onto MFA's word list, which
    won't always tokenize 1:1 with WhisperX's own output (punctuation
    handling, contractions, G2P-driven splits). Same difflib-based approach
    used earlier for the drift cross-reference — a straightforward
    positional zip silently mis-attributes scores the moment token counts
    diverge even slightly, which is common enough here not to risk it.
    """
    wn = [_norm(w) for w in whisper_words]
    mn = [_norm(w) for w in mfa_words]

    sm = difflib.SequenceMatcher(None, wn, mn, autojunk=False)
    out: list[Optional[float]] = [None] * len(mn)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[j1 + k] = whisper_scores[i1 + k]
        elif tag == "replace":
            span = whisper_scores[i1:i2] or whisper_scores
            avg = sum(span) / len(span) if span else 0.5
            for k in range(j1, j2):
                out[k] = avg
        # 'delete': whisper token with no MFA counterpart — contributes nothing.
        # 'insert': MFA token with no whisper counterpart — filled by the
        #           overall-average fallback below.

    overall_avg = sum(whisper_scores) / len(whisper_scores) if whisper_scores else 0.5
    return [s if s is not None else overall_avg for s in out]
