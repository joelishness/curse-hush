"""
CrisperWhisper (nyrahealth/CrisperWhisper 2.0) as stage 3 -- see
transcribe.py's own module docstring for the alignment-stage registry
this plugs into, and for why stage 3 is a deliberate EXCEPTION to that
registry's normal assumption.

WHY THIS IS DIFFERENT FROM STAGES 1 AND 2:
Stages 1 (whisperx.align()) and 2 (MFA, steps/align_mfa.py) both take
WhisperX's own recognized TEXT as fixed and only refine WHEN each word
occurs -- transcribe.py's docstring states this as a structural
guarantee ("Neither stage changes what text was recognized"). CrisperWhisper
does not fit that: it is an independently trained model that transcribes
the audio itself, so its own recognized words -- casing, punctuation,
verbatim disfluencies ("um", stutters, false starts) it captures that
WhisperX's own intended-style transcription does not -- can genuinely
differ from WhisperX's. Treating it as just another re-timer of the same
text would be wrong, not just imprecise.

That said: transcribe.py's own authoritative-resolution cascade turns out
not to actually depend on shared text at the mechanism level -- it just
picks the highest-numbered stage (up to alignment.backend's own number)
that produced ANY data for a given segment, it doesn't interpolate
between stages word-by-word (that word-level reconciliation is entirely
internal to stage 2's own chunking, see steps/align_mfa.py). So
alignment.backend CAN be set to "crisperwhisper" (with
alignment.crisperwhisper.enabled: true also required -- see
transcribe.py's own validation) -- the real, narrower consequence is a
style-consistency one, not a structural break: a segment where stage 3
fails falls back to stage 2/1's own recognized text for THAT segment,
which can read differently (casing, punctuation, disfluency handling)
than its CrisperWhisper-sourced neighbours. See transcribe.py's own
"Which stages actually run" for the full account of that tradeoff.
Regardless of alignment.backend's value, stage 3 only ever runs at all
when alignment.crisperwhisper.enabled is true -- its own switch, not
tied to alignment.dual_output (see that module for why).

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
  from this stage carries score=None in transcript_3_NN.json. If
  alignment.backend is "crisperwhisper", steps/matching.py DOES receive
  these words for whatever segments stage 3 succeeded on -- confirmed
  by reading matching.py directly, it already treats a None score as
  0.0 (`float(w["score"]) if w.get("score") is not None else 0.0`), the
  same graceful handling any stage's own null-timestamp words already
  rely on, so this isn't a new code path stage 3 needed, just a new
  source of the same null.
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

BACKEND CHOICE -- ct2, this package's own default preference, and why:
crisperwhisper's own backend="auto" resolution tries ct2 first, falling
back to transformers only if ct2 isn't installed at all -- confirmed by
reading model.py's _resolve_backend() directly, not assumed from the
README. ct2 (CTranslate2, the same engine faster-whisper is built on) is
typically faster on CPU than raw HuggingFace transformers, and -- checked
directly, not assumed -- DOES support CPU cleanly (engine.py resolves
device="auto" to "cpu" when no CUDA device is present, the same as this
image's own device). An earlier version of this module used the
transformers backend instead, reasoning (correctly, as far as it went)
that crisperwhisper[ct2] depends on a forked package
(ctranslate2-crisperwhisper, confirmed against its real PyPI metadata)
that installs under the SAME `import ctranslate2` name the plain
ctranslate2 this image already has via faster-whisper uses (confirmed by
reading engine.py's own `import ctranslate2` line) -- but concluded from
that alone that avoiding ct2 entirely was the safer choice, without
checking two things that change the picture:
  1. compute_type="float32" is available on ct2 too, as a genuinely
     first-class option (converter.py's own quantization parameter
     accepts "float32" directly, not as a fallback/workaround) -- so
     there is no actual quality cost to using ct2 over transformers when
     precision is set explicitly, only a potential speed gain.
  2. The import-collision risk is real but defended against, not silent:
     CrisperWhisperModel's CT2Engine calls a fail-fast check
     (_check_fork_apis(), confirmed by reading engine.py) immediately
     after loading the ctranslate2 module, and raises a specifically-
     named error if the fork's required methods aren't present. A wrong
     install order (see the Dockerfile's own comment on this) surfaces
     as a clear, diagnosable exception at model-load time, not a silent
     quality regression -- much lower actual risk than "might silently
     misbehave" implied.
Given both of those, ct2 -- the package's own default, typically faster,
no precision cost when compute_type is explicit -- is the better choice
for this pipeline; the earlier transformers-backend version was more
conservative than the evidence actually supported. Kept explicit here
(backend="ct2", not "auto") for the same reason as before: a future
crisperwhisper release changing "auto"'s own preference order shouldn't
silently change which backend this image uses without that being a
visible, deliberate config/code change.

compute_type: this module ALWAYS passes compute_type="float32" explicitly
on both backends, rather than relying on either one's own default --
confirmed by reading crisperwhisper's actual source that NEITHER backend
auto-adjusts for CPU: the transformers backend's compute_type maps
directly to a torch dtype with no CPU-aware adjustment
(transformers_engine.py's _resolve_torch_dtype()), and ct2's own
converter.py defaults its quantization parameter to "float16" too. Either
default, left unset on this image's CPU-only device, would silently use
lower-precision computation than intended (unreliable/slow torch.float16
on CPU for the transformers backend; a real, if smaller, quantization
precision loss for ct2's own default). This is exactly the kind of thing
this project's own history says to check by reading the real
implementation rather than trusting a quickstart example -- see
steps/align_mfa.py's module docstring for why that habit exists here at
all.

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
            "Dockerfile pip-installs crisperwhisper[ct2] (see its own "
            "comment there, and this module's own BACKEND CHOICE section, "
            "for why that extra)."
        ) from exc

    model_name   = cfg_get(cfg, "alignment", "crisperwhisper", "model")
    backend      = cfg_get(cfg, "alignment", "crisperwhisper", "backend")
    device       = cfg_get(cfg, "alignment", "crisperwhisper", "device")
    compute_type = cfg_get(cfg, "alignment", "crisperwhisper", "compute_type")

    log.info(
        "  Loading CrisperWhisper model '%s' (backend=%s, device=%s, compute_type=%s) ...",
        model_name, backend, device, compute_type,
    )
    import time
    t_load = time.monotonic()
    # backend is read from config (default "ct2" -- see config.yaml's own
    # comment) rather than left at crisperwhisper's own "auto", for the
    # same reason as before: a future crisperwhisper release changing
    # "auto"'s own preference order shouldn't silently change which
    # backend this image uses without that being a visible, deliberate
    # config change.
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
