# MFA chunking, third pass — apply notes

Supersedes both previous `profanity-hush-mfa-chunking` deliveries. Applying
this replaces the padding-and-search approach entirely — don't layer this on
top of either previous version; it removes `_detect_speech_envelope()` and
the `chunk_pad_before_sec`/`chunk_pad_after_sec`/`chunk_silence_noise_db`/
`chunk_silence_min_dur_sec` config keys, replacing all of it with one small
fixed margin.

## What happened, plainly

Two different attempts at generous, searched padding were each validated
against real data and each found to make timing worse than plain
`whisperx.align()` — via two *different* mechanisms:

1. **Padding handed to `align_one` directly** — MFA smeared a chunk's last
   word across trailing silence rather than stopping cleanly (one word
   measured over 25 seconds). Median offset vs. a reference SRT: 0.27s
   (stage 1) → 6.93s (stage 2).
2. **Padding trimmed via real silence detection first** — fixed mechanism
   #1, but on a shorter test file with denser dialogue, the detected
   envelope reached ~3 seconds back into unrelated audio before the
   chunk's real content, because there was no clean silence gap for
   silence detection to stop at. One word's true 0.24s duration became
   2.35s; another's true 0.70s became 2.02s.

Both failures trace to the same decision: searching a generously padded
window and trusting whatever's found in it — MFA's own alignment search in
the first case, real silence detection in the second — to reliably isolate
just one chunk's real audio. Neither did, reliably, in two different ways.

## The fix this time: stop searching

`chunk_edge_margin_sec` (default `0.3`) replaces all four removed config
keys. It's a small, **fixed** margin added to a chunk's own claimed
`[start, end]` before slicing — not a search window, and nothing tries to
find real audio beyond it. A chunk affected by genuine WhisperX drift will
generally fail to align in a window this tight, and falls back to stage 1
(WhisperX) timing for its own words — the same non-fatal path any other
per-chunk failure already uses. That's not a regression: it's exactly the
timing those words would have had before MFA was introduced.

**The explicit trade:** MFA no longer attempts to rescue the ~1% of content
genuinely affected by WhisperX drift (17 of 1,538 lines in the original
Independence Day count). Both attempts at rescuing that 1% corrupted a much
larger fraction of otherwise-correct content instead, twice, in two
different ways — so the safer default now is to let that specific content
fall back to no-worse-than-baseline rather than keep searching for a third,
cleverer way to reach it.

The cross-chunk monotonicity check from the first version is kept as a
cheap backstop regardless (costs nothing, and a tight slice removes the
room for either padding failure mode but doesn't guarantee perfection).

## Tests

`tests/test_align_mfa_chunking.py` — the four envelope-detection tests are
gone (that function no longer exists). Two new ones replace them, including
a **direct regression test for the reported bug**: a chunk placed right
next to unrelated audio with no silence gap between them, confirming the
tight-margin slice never reaches back into it — proof by construction that
the second attempt's specific failure mode is now structurally impossible,
not just less likely.

All 9 tests pass: `PYTHONPATH=src python3 tests/test_align_mfa_chunking.py`
(needs `ffmpeg` on `PATH`).

## Validated vs. not — same standard, same gap

Still not run against a real MFA install (no path to conda-forge from
here). The re-validation path is the same one that caught both previous
failures: re-run against Independence Day's 17 known cases **and** whatever
shorter/denser file exposed the second attempt's problem, check the
resulting transcript's own word-duration distribution directly (not just
whether the aggregate number moved), and specifically look at chunks near
the start of each job-segment and near tightly-packed dialogue — those are
exactly the conditions that broke both earlier attempts.

One thing worth expecting going in, so it doesn't read as a new problem:
the aggregate SRT-offset number for this version should land close to
stage 1's own (not dramatically better everywhere), because it deliberately
stops trying to improve on the drift-affected minority. The 17 known cases
are the direct, specific measure of what — if anything — was given up by no
longer padding.
