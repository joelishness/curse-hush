"""
CrisperWhisper (nyrahealth/CrisperWhisper 2.0) as stage 3 -- see
transcribe.py's own module docstring for the alignment-stage registry
this plugs into, and for why stage 3 is a deliberate EXCEPTION to that
registry's normal assumption.

WHY THIS IS DIFFERENT FROM STAGES 1 AND 2:
Stages 1 (whisperx.align()) and 2 (MFA, steps/align_mfa.py) both take
WhisperX's own recognized TEXT as fixed and only refine WHEN each word
occurs -- transcribe.py's docstring states this as a structural
guarantee ("Neither stage changes what text was recognized"), and the
whole cascade-to-the-nearest-available-stage mechanism depends on every
stage sharing the same word list. CrisperWhisper does not fit that: it
is an independently trained model that transcribes the audio itself, so
its own recognized words -- casing, punctuation, verbatim disfluencies
("um", stutters, false starts) it captures that WhisperX's own
intended-style transcription does not -- can genuinely differ from
WhisperX's. Treating it as just another re-timer of the same text would
be wrong, not just imprecise.

Consequently: stage 3 is comparison-only. alignment.backend cannot be
set to "crisperwhisper" (see transcribe.py's _AUTHORITATIVE_TOOLS,
deliberately narrower than the full alignment-stage registry) -- it only
ever runs when alignment.crisperwhisper.enabled is true, purely to
produce transcript_3_NN.json as a third, independently-sourced
comparison track, the same role transcript_1.json/transcript_2.json have
played throughout this project's own investigation work. It is never
consulted by steps/matching.py, steps/mute.py, or anything else
censoring-relevant.

WHY TRY THIS AT ALL, GIVEN STAGES 1/2's HISTORY:
Every failure mode this pipeline has hit with WhisperX + a separate
alignment stage traces back to the same root shape: WhisperX's own
segment-level timing can drift when it decodes in fixed ~30s windows
independently of each other (an entirely unrecognized stretch leaves its
real duration unaccounted for in WhisperX's own running clock -- see
docs/timestamp-drift-investigation.md), and every attempt at a separate
alignment stage to correct that after the fact needs to search for
where the real audio actually is, which has proven fragile in this
codebase's own history (see steps/align_mfa.py's module docstring for
three attempts, two of which made things worse on real validation runs).

CrisperWhisper's longform strategy is architecturally different in the
one respect that matters most here: chunk i's timing offset is always
`i * stride` -- a fixed, externally-imposed value that does not depend
on how much of any PREVIOUS chunk's audio was successfully recognized.
There is no internal "how much time has the decoder consumed so far"
state for an unrecognized stretch to silently fail to update. Whether
this actually avoids the drift class of error in practice, on real
episodes, is exactly what re-running the validation this project has
done throughout (Independence Day's 17 known cases, the reference SRT)
needs to check -- this reasoning is why it's worth trying, not a
substitute for checking.

WHAT THIS DOES NOT GIVE YOU:
- A per-word confidence score. CrisperWhisper's own WordTimestamp has no
  score field (unlike WhisperX's and MFA's own output) -- every word
  from this stage carries score=None in transcript_3_NN.json. This is
  never actually consumed downstream: stage 3 is comparison-only (see
  above), so steps/matching.py never receives its words at all, the same
  way it never receives any non-authoritative stage's words. score=None
  here is about transcript_3_NN.json itself being an honest record of
  what this stage does and doesn't provide, not about protecting a
  downstream consumer that doesn't exist for this stage.
- A word-for-word-identical transcript to stages 1/2, by design (see
  above) -- comparing transcript_3_NN.json against transcript_1_NN.json/
  transcript_2_NN.json means comparing two independently-recognized
  texts, not just two timings of the same text. A text-level diff is
  part of what this comparison is for.

MODEL LICENSING -- READ BEFORE ENABLING:
The inference code (this package) is MIT. The model WEIGHTS are not:
standard models ("large"/"turbo"/"medium"/"small") are released under
the Nyra Health Non-Commercial Research License (free for research and
other non-commercial use; commercial use needs a separate license from
Nyra) -- see https://huggingface.co/nyralabs/CrisperWhisper2.0_large/blob/main/LICENSE.md.
The "_pro" variants are commercial-license-only and not used by this
module's own default config. Confirm this fits your own use of this
pipeline before enabling alignment.crisperwhisper.enabled.

BACKEND CHOICE -- transformers, not ct2, and why:
crisperwhisper[ct2] depends on a package named ctranslate2-crisperwhisper
(a fork, confirmed by inspecting its actual PyPI metadata -- not the
same distribution as the plain `ctranslate2` this image already installs
transitively via faster-whisper/whisperx). Different PyPI distribution
name, but NOT confirmed here to install under a different Python import
name -- if it also provides `import ctranslate2`, installing both in the
same environment risks one's files silently overwriting the other's in
site-packages, depending on install order, which could produce a
confusing runtime failure (or worse, a silent one) in whichever of
whisperx/faster-whisper or this module happens to import second. This
was not something there was a safe way to fully rule out from here (would
need the actual wheel's file listing compared against the installed
`ctranslate2` package's own, in the real image). The transformers backend
instead reuses this image's already-installed torch directly, with no new
native-binary package under a potentially-colliding import name -- lower
total risk for this specific image, at the cost of somewhat slower CPU
inference than a correctly-isolated ct2 setup would give. If throughput
becomes the priority, MFA's own conda isolation (see the Dockerfile's MFA
stage) is the precedent for how to safely add ct2 alongside the existing
stack -- its own separate environment, not merged into this one.

compute_type: this module ALWAYS passes compute_type="float32" explicitly
for a CPU device, rather than relying on the package's own "float16"
default -- confirmed by reading crisperwhisper's actual source
(transformers_engine.py's _resolve_torch_dtype()) that compute_type is
mapped directly to a torch dtype with NO automatic CPU/GPU-aware
adjustment: passing the default on a CPU device would silently attempt
torch.float16 inference on CPU, which is unreliable/slow on many CPU
kernels. This is exactly the kind of thing this project's own history
says to check by reading the real implementation rather than trusting a
quickstart example -- see steps/align_mfa.py's module docstring for why
that habit exists here at all.

NOT VALIDATED: everything about this module's real-world behavior --
transcription/timing QUALITY on real episodes, actual CPU throughput for
a whole film, and the ctranslate2 namespace question above -- is
reasoned from this package's own real source code (confirmed by
extracting and reading the actual installed wheel, not just its
README), not from a real run. Same standard this whole project holds
everything else to: re-validate against Independence Day's 17 known
drift cases and the reference SRT before drawing any conclusion about
whether this is actually better, worse, or just different from stages
1/2 on real content.
"""

import logging
from pathlib import Path

from utils import cfg_get


def load_crisperwhisper_model(cfg: dict, log: logging.LoggerAdapter):
    """
    Load the CrisperWhisper model once for the whole job (same reasoning
    as transcribe.py's own wx_model: multi-minute load time, not worth
    repeating per segment). Only called when
    alignment.crisperwhisper.enabled is true -- see transcribe.py's own
    per-segment loop.
    """
    try:
        from crisperwhisper import CrisperWhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "alignment.crisperwhisper.enabled is true but the crisperwhisper "
            "package is not installed inside the container. Ensure the "
            "Dockerfile pip-installs crisperwhisper[transformers] (see its "
            "own comment there for why that extra, not [ct2])."
        ) from exc

    model_name   = cfg_get(cfg, "alignment", "crisperwhisper", "model")
    device       = cfg_get(cfg, "alignment", "crisperwhisper", "device")
    compute_type = cfg_get(cfg, "alignment", "crisperwhisper", "compute_type")

    log.info(
        "  Loading CrisperWhisper model '%s' (device=%s, compute_type=%s) ...",
        model_name, device, compute_type,
    )
    import time
    t_load = time.monotonic()
    # backend="transformers" pinned explicitly -- see this module's own
    # docstring (BACKEND CHOICE) for why "auto" isn't trusted here even
    # though ct2 might resolve the same way today: a future crisperwhisper
    # release changing "auto"'s preference order shouldn't silently switch
    # this image onto the unverified ctranslate2 namespace path.
    model = CrisperWhisperModel(
        model_name,
        backend="transformers",
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
    Run CrisperWhisper on one job-segment's own dialog stem, independent
    of anything WhisperX (stages 1/2) recognized -- see this module's own
    docstring for why. Returns this pipeline's {"word","start","end",
    "score"} shape (score always None -- see docstring), timestamps
    segment-local (0-based within dialog_wav), matching every other
    stage's own convention (transcribe.py applies the global offset
    later, uniformly, for whichever stage's file is being merged).

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
