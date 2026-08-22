"""
Regression tests for align_mfa.py's chunked-alignment design (see that
module's own docstring for the full reasoning). No existing tests/
convention existed in this repo before this change; added because this is
the one part of the MFA chunking work that reasoning alone can't fully
verify without a real MFA install (see align_mfa.py's module docstring's
"validated" section) -- these at least pin down the index/offset
arithmetic and the audio-slicing math precisely, so a future refactor that
breaks either one fails loudly here instead of producing a plausible-
looking but silently wrong timestamp downstream.

Run with:  PYTHONPATH=src python3 tests/test_align_mfa_chunking.py
Needs ffmpeg on PATH (already a hard requirement of this pipeline) for
TestSliceWav; no other external dependency, and nothing here touches a
real MFA/conda install.
"""
import logging
import math
import os
import struct
import sys
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("MFA_CONDA_EXE", "/bin/sh")  # anything that exists -- only the early Path.exists() check cares

from steps import align_mfa  # noqa: E402

log = logging.getLogger("test_align_mfa_chunking")
log.addHandler(logging.NullHandler())


def seg(start, end, text="word"):
    return {"start": start, "end": end, "text": text, "avg_logprob": -0.1}


def _run(fn):
    fn()
    print(f"  ok  {fn.__name__}")


# ── _build_alignment_chunks ──────────────────────────────────────────────

def test_chunks_pack_up_to_target():
    segs = [seg(i * 4, i * 4 + 4) for i in range(5)]  # 4s each, back to back
    chunks = align_mfa._build_alignment_chunks(segs, chunk_target_sec=10.0, log=log)
    assert [len(c) for c in chunks] == [2, 2, 1]
    spans = [(c[0]["start"], c[-1]["end"]) for c in chunks]
    assert spans == [(0, 8), (8, 16), (16, 20)]


def test_oversized_single_segment_stands_alone():
    segs = [seg(0, 5), seg(5, 65), seg(65, 70)]  # middle one is 60s alone
    chunks = align_mfa._build_alignment_chunks(segs, chunk_target_sec=25.0, log=log)
    assert [len(c) for c in chunks] == [1, 1, 1]
    assert chunks[1][0]["end"] - chunks[1][0]["start"] == 60.0


def test_missing_timestamps_falls_back_to_one_chunk():
    segs = [seg(0, 4), {"text": "no timestamps", "avg_logprob": -0.1}, seg(8, 12)]
    chunks = align_mfa._build_alignment_chunks(segs, chunk_target_sec=10.0, log=log)
    assert len(chunks) == 1 and len(chunks[0]) == 3


def test_empty_input():
    assert align_mfa._build_alignment_chunks([], chunk_target_sec=10.0, log=log) == []


# ── align_with_mfa's chunk-merging: offsets + monotonicity rejection ────

def test_merge_offsets_and_rejects_implausible_chunk():
    """
    Three chunks; the middle one 'succeeds' but its anchors land back
    inside the first chunk's own accepted territory -- exactly the
    mis-sliced-chunk failure mode the monotonicity check exists for. Its
    words must fall through to interpolation between their real
    neighbours, and its bad anchors must not appear anywhere in the
    output.
    """
    whisper_segments = [seg(0.0, 10.0, "one two three"), seg(10.0, 20.0, "four five"), seg(20.0, 30.0, "six seven")]
    fake_chunks = [[s] for s in whisper_segments]
    results = [
        (["one", "two", "three"], [0, 1, 2], {0: (1.0, 2.0), 1: (4.0, 5.0), 2: (7.0, 8.0)}),
        (["four", "five"], [0, 1], {0: (2.0, 3.0), 1: (5.0, 6.0)}),  # bad: overlaps chunk 0
        (["six", "seven"], [0, 1], {0: (21.0, 22.0), 1: (25.0, 26.0)}),
    ]
    whisperx_words = [{"word": w, "start": i * 3.0, "end": i * 3.0 + 2.0, "score": 0.5}
                       for i, w in enumerate(["one", "two", "three", "four", "five", "six", "seven"])]
    cfg = _cfg()

    with patch.object(align_mfa, "_ensure_mfa_ready", return_value=None), \
         patch.object(align_mfa, "_get_g2p_graphemes", return_value=None), \
         patch.object(align_mfa, "_build_alignment_chunks", return_value=fake_chunks), \
         patch.object(align_mfa, "_align_chunk", side_effect=results):
        words = align_mfa.align_with_mfa(
            dialog_wav=None, dialog_duration_sec=30.0,
            whisper_segments=whisper_segments, whisperx_words=whisperx_words,
            cfg=cfg, log=log,
        )

    assert [w["word"] for w in words] == ["one", "two", "three", "four", "five", "six", "seven"]
    assert (words[0]["start"], words[0]["end"]) == (1.0, 2.0)
    assert (words[2]["start"], words[2]["end"]) == (7.0, 8.0)
    assert (words[5]["start"], words[5]["end"]) == (21.0, 22.0)
    assert (words[6]["start"], words[6]["end"]) == (25.0, 26.0)
    # The part that matters: rejected chunk's words interpolate strictly
    # inside the [8.0, 21.0] gap, nowhere near its discarded 2.0-6.0 anchors.
    for i in (3, 4):
        assert 8.0 <= words[i]["start"] < words[i]["end"] <= 21.0, words[i]
    for a, b in zip(words, words[1:]):
        assert a["end"] <= b["start"] + 1e-9


def test_merge_handles_degraded_and_empty_chunks():
    """Outright chunk failure (attempted, no anchors) vs. a chunk with no
    alignable text at all are different code paths -- both must leave
    correct text/offsets behind them for the next chunk regardless."""
    whisper_segments = [seg(0.0, 5.0, "alpha beta"), seg(5.0, 10.0, "7"), seg(10.0, 15.0, "gamma delta")]
    fake_chunks = [[s] for s in whisper_segments]
    results = [
        (["alpha", "beta"], [0, 1], {}),   # attempted, MFA failed -> degrade
        (["7"], [], {}),                    # nothing survived sanitization
        (["gamma", "delta"], [0, 1], {0: (10.5, 11.5), 1: (13.5, 14.5)}),
    ]
    whisperx_words = [{"word": w, "start": i * 2.0, "end": i * 2.0 + 1.0, "score": 0.5}
                       for i, w in enumerate(["alpha", "beta", "7", "gamma", "delta"])]
    cfg = _cfg()

    with patch.object(align_mfa, "_ensure_mfa_ready", return_value=None), \
         patch.object(align_mfa, "_get_g2p_graphemes", return_value=None), \
         patch.object(align_mfa, "_build_alignment_chunks", return_value=fake_chunks), \
         patch.object(align_mfa, "_align_chunk", side_effect=results):
        words = align_mfa.align_with_mfa(
            dialog_wav=None, dialog_duration_sec=15.0,
            whisper_segments=whisper_segments, whisperx_words=whisperx_words,
            cfg=cfg, log=log,
        )

    assert [w["word"] for w in words] == ["alpha", "beta", "7", "gamma", "delta"]
    assert words[0]["start"] is not None and words[1]["start"] is not None
    assert (words[3]["start"], words[3]["end"]) == (10.5, 11.5)
    assert (words[4]["start"], words[4]["end"]) == (13.5, 14.5)


def _cfg():
    return {"alignment": {"mfa": {
        "acoustic_model": "english_mfa", "dictionary": "english_mfa", "g2p_model": None,
        "beam": 400, "retry_beam": 1000, "fallback_to_whisperx": True,
        "chunk_target_sec": 25, "chunk_edge_margin_sec": 0.3,
        "chunk_timeout_sec": 120,
    }}}


# ── _slice_wav: real ffmpeg, real audio, sub-ms precision check ─────────

def test_slice_wav_preserves_absolute_time(tmp_path=None):
    """
    Generates a real WAV with a precisely-timed beep, slices a padded
    window out of it the same way align_with_mfa/_align_chunk actually
    would, and confirms that slice_start + (position found within the
    slice) recovers the beep's true absolute time -- this is the exact
    arithmetic _align_chunk relies on to convert MFA's slice-relative
    output back onto dialog_wav's own timeline.
    """
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="slice_wav_test_"))
    sr = 44100
    duration = 30.0
    beep_start, beep_dur, beep_freq = 8.0, 0.5, 1000.0
    n = int(sr * duration)

    src = tmp / "synthetic_dialog.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            t = i / sr
            val = int(16000 * math.sin(2 * math.pi * beep_freq * t)) if beep_start <= t < beep_start + beep_dur else 0
            frames += struct.pack("<hh", val, val)
        w.writeframes(bytes(frames))

    # A chunk that (slightly wrongly, on purpose) claims [8.0, 8.6] with
    # this codebase's actual default padding (5 before, 25 after).
    claimed_start, claimed_end = 8.0, 8.6
    slice_start = max(0.0, claimed_start - 5.0)
    slice_end = min(duration, claimed_end + 25.0)

    out = tmp / "sliced.wav"
    align_mfa._slice_wav(src, slice_start, slice_end, out, log)

    with wave.open(str(out), "rb") as w:
        sr_out, n_out = w.getframerate(), w.getnframes()
        raw = w.readframes(n_out)
    left = struct.unpack(f"<{n_out * 2}h", raw)[0::2]
    onset_sample = next(i for i, v in enumerate(left) if abs(v) > 5000)
    onset_absolute = slice_start + onset_sample / sr_out

    assert abs(onset_absolute - 8.0) < 0.001, f"expected ~8.0s, got {onset_absolute:.4f}s"


def _make_wav(path, duration, active_spans, sr=44100, freq=440.0):
    """active_spans: list of (start,end) tuples that get a tone; everything
    else is silence."""
    n = int(sr * duration)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            t = i / sr
            active = any(s <= t < e for s, e in active_spans)
            val = int(16000 * math.sin(2 * math.pi * freq * t)) if active else 0
            frames += struct.pack("<hh", val, val)
        w.writeframes(bytes(frames))


# ── _align_chunk end-to-end: everything real except the align_one call ──

def test_align_chunk_slices_tight_margin_not_a_search_window():
    """
    Direct regression test for the second attempt's failure: a chunk very
    close to the start of a file, with OTHER (unrelated) audio activity
    before it and no silence gap wide enough to separate them cleanly --
    exactly the shape that made the previous (padded + silence-detected)
    design sweep ~3 real seconds of unrelated preceding audio into a
    chunk's alignment on a real run. Confirms the tight-margin design
    never reaches back past chunk_edge_margin_sec regardless of what's
    nearby, because there's no search step left that could reach further.
    """
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="align_chunk_test_"))
    src = tmp / "dialog.wav"
    # Unrelated activity from 0.0-3.0 (stands in for "whatever's earlier in
    # the file/scene"), essentially no gap, then this chunk's real content
    # from 3.1-5.6.
    _make_wav(src, 60.0, [(0.0, 3.0), (3.1, 5.6)])

    chunk_segments = [seg(3.1, 5.6, "hello world")]

    fake_proc = type("FakeProc", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    written_wav_durations = []

    real_slice_wav = align_mfa._slice_wav
    def spy_slice_wav(src_, start, end, out, log_):
        real_slice_wav(src_, start, end, out, log_)
        with wave.open(str(out), "rb") as w:
            written_wav_durations.append(w.getnframes() / w.getframerate())

    with patch.object(align_mfa, "_slice_wav", side_effect=spy_slice_wav), \
         patch("subprocess.run", return_value=fake_proc), \
         patch.object(align_mfa, "_find_output_textgrid", return_value=tmp / "fake.TextGrid"), \
         patch.object(align_mfa, "_parse_textgrid_words", return_value=[("hello", 0.3, 0.9), ("world", 1.0, 1.6)]):
        local_words, local_origin, local_anchors = align_mfa._align_chunk(
            dialog_wav=src, chunk_segments=chunk_segments,
            slice_start=max(0.0, 3.1 - 0.3), slice_end=5.6 + 0.3,  # margin, not a search window
            dictionary="english_mfa", acoustic_model="english_mfa", g2p_model=None, graphemes=None,
            beam=400, retry_beam=1000, timeout_sec=120, fallback_allowed=True,
            chunk_dir=tmp / "chunk_0000", label="test chunk", log=log,
        )

    assert len(written_wav_durations) == 1
    duration = written_wav_durations[0]
    # Should be essentially exactly the claim (2.5s) plus 2x margin (0.6s) --
    # ~3.1s -- and nowhere near reaching back to the unrelated 0.0-3.0
    # activity, regardless of there being no clean silence gap to stop at.
    assert 2.9 <= duration <= 3.3, (
        f"WAV handed to align_one was {duration:.2f}s -- expected ~3.1s "
        f"(claim + fixed margin). If this is closer to 5.9s, unrelated "
        f"preceding audio leaked in again."
    )


def test_align_chunk_uses_exact_margin_no_more_no_less():
    """
    Companion check: with nothing else nearby (clean silence on both
    sides), the slice should still be exactly [slice_start, slice_end] as
    given -- confirms there's no remaining trim/search step at all, not
    just that it's bounded in the adversarial case above.
    """
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="align_chunk_test_"))
    src = tmp / "dialog.wav"
    _make_wav(src, 30.0, [(12.0, 14.5)])

    chunk_segments = [seg(12.0, 14.5, "hello world")]
    fake_proc = type("FakeProc", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    written = []

    real_slice_wav = align_mfa._slice_wav
    def spy_slice_wav(src_, start, end, out, log_):
        real_slice_wav(src_, start, end, out, log_)
        with wave.open(str(out), "rb") as w:
            written.append(w.getnframes() / w.getframerate())

    slice_start, slice_end = 12.0 - 0.3, 14.5 + 0.3
    with patch.object(align_mfa, "_slice_wav", side_effect=spy_slice_wav), \
         patch("subprocess.run", return_value=fake_proc), \
         patch.object(align_mfa, "_find_output_textgrid", return_value=tmp / "fake.TextGrid"), \
         patch.object(align_mfa, "_parse_textgrid_words", return_value=[("hello", 0.3, 0.9), ("world", 1.0, 1.6)]):
        align_mfa._align_chunk(
            dialog_wav=src, chunk_segments=chunk_segments,
            slice_start=slice_start, slice_end=slice_end,
            dictionary="english_mfa", acoustic_model="english_mfa", g2p_model=None, graphemes=None,
            beam=400, retry_beam=1000, timeout_sec=120, fallback_allowed=True,
            chunk_dir=tmp / "chunk_0000", label="test chunk", log=log,
        )

    assert len(written) == 1
    assert abs(written[0] - (slice_end - slice_start)) < 0.05, (
        f"expected exactly the given [slice_start, slice_end] "
        f"({slice_end-slice_start:.2f}s), got {written[0]:.2f}s -- "
        f"something is still trimming/searching."
    )


if __name__ == "__main__":
    for fn in [
        test_chunks_pack_up_to_target,
        test_oversized_single_segment_stands_alone,
        test_missing_timestamps_falls_back_to_one_chunk,
        test_empty_input,
        test_merge_offsets_and_rejects_implausible_chunk,
        test_merge_handles_degraded_and_empty_chunks,
        test_slice_wav_preserves_absolute_time,
        test_align_chunk_slices_tight_margin_not_a_search_window,
        test_align_chunk_uses_exact_margin_no_more_no_less,
    ]:
        _run(fn)
    print("\nALL TESTS PASSED")
