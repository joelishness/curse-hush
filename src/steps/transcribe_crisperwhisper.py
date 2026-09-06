"""
CrisperWhisper (nyrahealth/CrisperWhisper 2.0) — one of three independently-
toggleable transcription/alignment engines this pipeline supports (see
transcribe.py's own module docstring for the engine registry this plugs
into). Unlike whisperx/mfa, this engine transcribes each segment's audio
directly rather than re-timing WhisperX's own recognized text, so it needs
no WhisperX involvement of any kind — no shared model, no shared recognized
text, nothing.

WHY THIS IS DIFFERENT FROM WHISPERX/MFA:
whisperx and mfa (steps/align_mfa.py) both take WhisperX's own recognized
TEXT as fixed and only refine WHEN each word occurs. CrisperWhisper does
not fit that: it is an independently trained model that transcribes the
audio itself, so its own recognized words — casing, punctuation, verbatim
disfluencies ("um", stutters, false starts) it captures that WhisperX's
own intended-style transcription does not — can genuinely differ from
WhisperX's. Treating it as just another re-timer of the same text would be
wrong, not just imprecise.

That has one real, practical consequence worth knowing before setting
alignment.engines.crisperwhisper.final: true (this config's default —
see below): if this engine fails for a specific segment, that segment's
transcript.json contribution is simply empty for that span, UNLESS
another engine is also enabled (alignment.engines.whisperx.enabled or
alignment.engines.mfa.enabled) — there is no automatic fallback to
either of those, since they don't share this engine's recognized text.
See steps/transcribe.py's own module docstring for the full account of
how the final/authoritative transcript is resolved.

WHY TRY THIS AT ALL:
Every failure mode whisperx/mfa have hit in this project's own history
traces back to the same root shape: WhisperX's own segment-level timing
can drift when it decodes in fixed ~30s windows independently of each
other (an entirely unrecognized stretch leaves its real duration
unaccounted for in WhisperX's own running clock — see
docs/timestamp-drift-investigation.md), and every attempt at a separate
alignment stage to correct that after the fact needs to search for where
the real audio actually is, which has proven fragile in this codebase's
own history (see steps/align_mfa.py's module docstring for three
attempts, two of which made things worse on real validation runs).

CrisperWhisper's longform strategy is architecturally different in the
one respect that matters most here: chunk i's timing offset is always
`i * stride` — a fixed, externally-imposed value that does not depend on
how much of any PREVIOUS chunk's audio was successfully recognized. There
is no internal "how much time has the decoder consumed so far" state for
an unrecognized stretch to silently fail to update.

VALIDATION STATUS: tested directly against Independence Day (1996), the
same film and methodology used to validate MFA (see
docs/timestamp-drift-investigation.md) — found more accurate than both
whisperx and mfa on that comparison. That's why this config's default
promotes it to alignment.engines.crisperwhisper.final: true, with
whisperx/mfa both off by default (see config.yaml's own alignment.engines
comment). Re-validate the same way (Independence Day's known cases, the
reference SRT, checking both timing AND text — see "WHAT THIS DOES NOT
GIVE YOU" below for why text matters here specifically) after any change
to this module, this engine's own model/backend settings, or a
CrisperWhisper package upgrade.

WHAT THIS DOES NOT GIVE YOU:
- A per-word confidence score. CrisperWhisper's own WordTimestamp has no
  score field (unlike WhisperX's and MFA's own output) — every word from
  this engine carries score=None in transcript_crisperwhisper_NN.json.
  When this engine is final (the default), steps/matching.py DOES receive
  these words — confirmed by reading matching.py directly, it already
  treats a None score as 0.0 (`float(w["score"]) if w.get("score") is not
  None else 0.0`), the same graceful handling any other engine's own
  null-timestamp words already rely on, so this isn't a new code path
  this engine needed, just a new source of the same null.
- A word-for-word-identical transcript to whisperx/mfa, by design (see
  above) — comparing transcript_crisperwhisper_NN.json against
  transcript_whisperx_NN.json/transcript_mfa_NN.json means comparing two
  independently-recognized texts, not just two timings of the same text.
  A text-level diff is part of what enabling more than one engine's
  debug_subtitle is for.

MODEL LICENSING — READ BEFORE ENABLING:
The inference code (this package) is MIT. The model WEIGHTS are not:
standard models ("large"/"turbo"/"medium"/"small") are released under the
Nyra Health Non-Commercial Research License (free for research and other
non-commercial use; commercial use needs a separate license from Nyra) —
see https://huggingface.co/nyralabs/CrisperWhisper2.0_large/blob/main/LICENSE.md.
The "_pro" variants are commercial-license-only and not used by this
module's own default config. Confirm this fits your own use of this
pipeline before enabling alignment.engines.crisperwhisper.enabled.

BACKEND CHOICE — ct2, this package's own default preference, and why:
crisperwhisper's own backend="auto" resolution tries ct2 first, falling
back to transformers only if ct2 isn't installed at all — confirmed by
reading model.py's own _resolve_backend() directly, not assumed from the
README. ct2 (CTranslate2, the same engine faster-whisper is built on) is
typically faster on CPU than raw HuggingFace transformers, and — checked
directly, not assumed — DOES support CPU cleanly (engine.py resolves
device="auto" to "cpu" when no CUDA device is present, the same as this
image's own device).

The real, confirmed risk: crisperwhisper[ct2] depends on a forked package
(ctranslate2-crisperwhisper, confirmed against its real PyPI metadata)
that installs under the SAME `import ctranslate2` name the plain
ctranslate2 this image already has via faster-whisper uses (confirmed by
reading engine.py's own `import ctranslate2` line) — both write to
site-packages/ctranslate2/, so whichever installs LAST wins on disk. The
Dockerfile installs crisperwhisper[ct2] after faster-whisper specifically
so the fork's files win. This is defended against, not silent, either
way: CrisperWhisperModel's CT2Engine calls a fail-fast check
(_check_fork_apis(), confirmed by reading engine.py) immediately after
loading the ctranslate2 module, and raises a specifically-named error if
the fork's required methods aren't present — a wrong install order
surfaces as a clear, diagnosable exception at model-load time, not a
silent quality regression.

compute_type="float32" is passed explicitly on both backends, never left
at either one's own default — confirmed by reading crisperwhisper's
actual source (not just its README): the transformers backend's
compute_type maps directly to a torch dtype with no CPU-aware adjustment
(float16 on CPU is unreliable/slow on many kernels), and ct2's own
model-conversion step defaults its quantization to float16 too (a
smaller but real precision cost). float32 is a genuinely first-class,
fully-supported option on both backends, not a fallback.
"""

import logging
from pathlib import Path

from utils import cfg_get


def load_crisperwhisper_model(cfg: dict, log: logging.LoggerAdapter):
    """
    Load the CrisperWhisper model once for the whole job (same reasoning
    as transcribe.py's own wx_model, when that one loads at all: a
    multi-minute load time, not worth repeating per segment). Only
    called when alignment.engines.crisperwhisper.enabled is true — see
    transcribe.py's own per-segment loop.
    """
    try:
        from crisperwhisper import CrisperWhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "alignment.engines.crisperwhisper.enabled is true but the "
            "crisperwhisper package is not installed inside the "
            "container. Ensure the Dockerfile pip-installs "
            "crisperwhisper[ct2] (see its own comment there, and this "
            "module's own BACKEND CHOICE section, for why that extra)."
        ) from exc

    model_name   = cfg_get(cfg, "alignment", "engines", "crisperwhisper", "model")
    backend      = cfg_get(cfg, "alignment", "engines", "crisperwhisper", "backend")
    device       = cfg_get(cfg, "alignment", "engines", "crisperwhisper", "device")
    compute_type = cfg_get(cfg, "alignment", "engines", "crisperwhisper", "compute_type")

    log.info(
        "  Loading CrisperWhisper model '%s' (backend=%s, device=%s, compute_type=%s) ...",
        model_name, backend, device, compute_type,
    )
    import time
    t_load = time.monotonic()
    # backend is read from config (default "ct2" -- see config.yaml's own
    # comment) rather than left at crisperwhisper's own "auto", so a
    # future crisperwhisper release changing "auto"'s own preference
    # order can't silently change which backend this image uses without
    # that being a visible, deliberate config/code change.
    model = CrisperWhisperModel(
        model_name,
        backend=backend,
        device=device,
        compute_type=compute_type,
    )
    log.info("  ✓  CrisperWhisper model loaded in %.1f s.", time.monotonic() - t_load)
    return model


def transcribe_with_crisperwhisper(
    model,
    dialog_wav: Path,
    language: str,
    cfg: dict,
    log: logging.LoggerAdapter,
) -> list[dict]:
    """
    Run CrisperWhisper on one job-segment's own dialog stem, entirely
    independent of anything whisperx/mfa recognized (see this module's
    own docstring for why). Returns this pipeline's {"word","start","end",
    "score"} shape (score always None -- see docstring), timestamps
    segment-local (0-based within dialog_wav), matching every other
    engine's own convention (transcribe.py applies the global offset
    later, uniformly, for whichever engine's file is being merged).

    language -- the caller passes alignment.engines.crisperwhisper.
    language directly (this engine has its own, fully independent of
    alignment.engines.whisperx.language, since it shares no computation
    with that engine at all); falls back to "en" here if falsy (config
    null, or an empty string), mirroring the call site's own
    `language or "en"` -- NOTE this is not the same as whisperx's own
    null-means-auto-detect: this engine's transcribe() call has no
    auto-detect mode, so a null/empty language always becomes "en" here,
    never genuine detection. See config.yaml's own comment on
    alignment.engines.crisperwhisper.language.

    longform_strategy is left at the package's own default
    ("continuation") rather than exposed as a config knob here -- that
    default is the one this module's docstring reasons about; changing it
    would mean re-reasoning through a different mechanism entirely, not
    just a different tuning of the same one.
    """
    result = model.transcribe(
        str(dialog_wav),
        language=language or "en",
        word_timestamps=True,
    )

    words: list[dict] = []
    for w in (result.words or []):
        if not w.word:
            continue
        words.append({
            "word":  w.word,
            "start": w.start,
            "end":   w.end,
            "score": None,  # CrisperWhisper's WordTimestamp has no confidence field
        })
    return words
