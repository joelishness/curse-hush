# CrisperWhisper stage 3 — apply notes

Adds `transcript_3_NN.json` as a third, **independently-sourced** transcript
from a separately-trained model — not a re-timing of stages 1 (WhisperX) or
2 (MFA) the way they re-time each other. Updated from an earlier version of
this delivery: `alignment.backend` **can** be set to `"crisperwhisper"` (it
was wrongly hard-blocked before), and the recommended backend is now `ct2`,
not `transformers` — see both sections below for why.

## Applying

Same pattern as previous deliveries: `patches/*.patch` are unified diffs
against the original upload; `src/`, `config/`, and `Dockerfile` here mirror
your repo layout for direct copy. `tests/test_transcribe_crisperwhisper.py`
is new — copy to your `tests/`. **This changes `Dockerfile`** — a real
image rebuild is needed, not just a code drop-in.

If you applied the earlier version of this delivery (`[transformers]`,
comparison-only): this supersedes it. The Dockerfile now installs
`crisperwhisper[ct2]` instead, and `alignment.backend: crisperwhisper` is
now a valid, if optional, choice.

## Can `alignment.backend` be `crisperwhisper`? Yes — here's the real tradeoff

Stages 1 and 2 both re-time WhisperX's own recognized *text*, so falling
from stage 2 to stage 1 changes only a segment's timing, never its words.
Stage 3 transcribes independently, so its words can genuinely differ
(casing, punctuation, verbatim disfluencies WhisperX's style doesn't
capture). The pipeline's authoritative-resolution cascade turns out not to
actually depend on shared text at the mechanism level, though — it just
picks the highest-numbered stage that produced *any* data for a segment, it
doesn't interpolate between stages word-by-word. So there's no structural
reason to block the choice.

The real, narrower consequence: if stage 3 fails for one specific segment,
that segment falls back to stage 2/1's own text while its neighbors stay
CrisperWhisper-sourced — a possible style seam at that one boundary, not a
break. `alignment.crisperwhisper.enabled: true` is required alongside
`alignment.backend: crisperwhisper` (checked explicitly, raises a clear
error if you set one without the other) — otherwise stage 3 never runs at
all and every segment silently cascades to stage 2/1, which defeats the
point without telling you.

## Backend: `ct2`, not `transformers` — checked, not assumed, this time

An earlier version of this recommended `transformers`, reasoning that
`crisperwhisper[ct2]`'s dependency on a forked `ctranslate2-crisperwhisper`
package was too much unconfirmed risk. Checked more thoroughly this round
by reading the actual installed package source:

| | `transformers` | `ct2` |
|---|---|---|
| The package's own default preference | No — only used if `ct2` isn't installed | **Yes** — `backend="auto"` tries this first (confirmed in `model.py`) |
| CPU support | Yes | Yes — confirmed `device="auto"` resolves to `"cpu"` cleanly |
| Speed on CPU | Slower — raw PyTorch/HF | **Faster** — CTranslate2 is purpose-built for this; it's the same engine `faster-whisper` (already in this image) is built on |
| Quality at `compute_type: float32` | Full precision | **Full precision, confirmed** — `float32` is a first-class supported quantization option in `converter.py`, not a fallback |
| Dependency risk | None — reuses this image's existing `torch` | Real, confirmed: `ctranslate2-crisperwhisper` installs under the same `import ctranslate2` name as the plain package `faster-whisper` already needs. Mitigated by install order (this Dockerfile installs it *after* `faster-whisper`) **and** defended in code — `CrisperWhisperModel` calls a fail-fast check (`_check_fork_apis()`, confirmed in `engine.py`) right after loading, raising a specifically-named error if the wrong package won. A bad install order surfaces as a clear exception at model-load time, not silent bad output. |

Given `ct2` is the tool's own default, typically faster, and has no quality
cost when `compute_type` is set explicitly (which this config always does,
on both backends — see below), it's the better choice here. The earlier
`transformers` recommendation was more conservative than the evidence
actually supported.

**`compute_type: float32` is still set explicitly on `ct2` too** — its own
model-conversion step defaults to `float16` (a real, if smaller, precision
cost than the `transformers` backend's CPU-unsafe `float16` default), so
this config pins full precision on both backends rather than trusting
either one's own default.

## Facts worth knowing before enabling

- **License**: inference code is MIT; model **weights are not** — standard
  models (what this config defaults to: `large`) are under a non-commercial
  research license. `_pro` variants are commercial-license-only. Confirm
  this fits your use.
- **No per-word confidence score** — `WordTimestamp` doesn't have one, so
  `transcript_3_NN.json` (and, if `crisperwhisper` is authoritative,
  `transcript.json` for those segments) carries `score: None`.
  `matching.py` already handles that gracefully (confirmed by reading it:
  `float(w["score"]) if w.get("score") is not None else 0.0`).

## What I actually verified vs. what I couldn't

**Verified directly** (installed the real `crisperwhisper==2.0.2` wheel and
read the actual source, both backends): the public API shape, the longform
continuation mechanism's fixed-stride timing offset, both backends' CPU
dtype defaults and gotchas, `backend="auto"`'s real preference order, and
the `ctranslate2-crisperwhisper` fork's exact import-collision mechanics
and its own defensive fail-fast check. This pipeline's own conversion/
validation logic is tested in `tests/test_transcribe_crisperwhisper.py`.

**Not verified, and can't be from here** (no GPU, disk-constrained for a
full model download in this sandbox): actual transcription/timing quality
on real audio, actual CPU throughput, and whether the install-order
mitigation for the `ctranslate2` collision holds up in your actual image
build (rather than just in this Dockerfile's stated ordering). Same
standard as everything else in this project: re-run against Independence
Day's 17 known drift cases and the reference SRT before drawing any
conclusion about quality, and check your real build log for the
`_check_fork_apis` error to confirm the right `ctranslate2` package won.

## Enabling it

```yaml
alignment:
  backend: crisperwhisper   # or leave at mfa/whisperx and just enable below for comparison only
  crisperwhisper:
    enabled: true
```

Rebuild the image first — the package isn't there until you do.
