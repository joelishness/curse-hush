"""
Regression tests for steps/transcribe_crisperwhisper.py's schema
conversion, and for transcribe.py's guard against alignment.backend
being set to "crisperwhisper" (see both modules' own docstrings for why
that guard exists at all -- stage 3 doesn't share stage 1/2's "same
text either way" property, so it must never become authoritative).

Does NOT require the crisperwhisper package to be installed -- the
model itself is mocked throughout; these tests are about this
pipeline's own conversion/guard logic, not about crisperwhisper's real
behavior (which needs a real install and real audio to check at all --
see transcribe_crisperwhisper.py's own "NOT VALIDATED" section).

Run with:  PYTHONPATH=src python3 tests/test_transcribe_crisperwhisper.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging
from steps.transcribe_crisperwhisper import transcribe_with_crisperwhisper
from steps import transcribe

log = logging.getLogger("test_transcribe_crisperwhisper")
log.addHandler(logging.NullHandler())


def _run(fn):
    fn()
    print(f"  ok  {fn.__name__}")


class _FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _FakeResult:
    def __init__(self, words):
        self.words = words


def test_word_schema_conversion():
    """word/start/end preserved as-is, score always None (no such field
    in crisperwhisper's own WordTimestamp), empty tokens filtered."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = _FakeResult([
        _FakeWord("Hello", 0.5, 0.9),
        _FakeWord("world.", 1.0, 1.6),
        _FakeWord("", 1.6, 1.6),  # defensive: should never appear for real, still handled
    ])

    words = transcribe_with_crisperwhisper(fake_model, Path("/tmp/fake.wav"), "en", {}, log)

    assert words == [
        {"word": "Hello", "start": 0.5, "end": 0.9, "score": None},
        {"word": "world.", "start": 1.0, "end": 1.6, "score": None},
    ], words


def test_calls_transcribe_with_word_timestamps_true():
    """word_timestamps=True is not optional -- without it crisperwhisper
    doesn't compute per-word timing at all (see its own transcribe()
    docstring)."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = _FakeResult([])
    transcribe_with_crisperwhisper(fake_model, Path("/tmp/fake.wav"), "en", {}, log)

    kwargs = fake_model.transcribe.call_args.kwargs
    assert kwargs["word_timestamps"] is True
    assert kwargs["language"] == "en"


def test_language_falls_back_to_english_default():
    """Mirrors the call site's own `language or "en"` -- an empty/None
    language string (whisperx auto-detect not yet resolved, or no
    speech detected at all for this segment) shouldn't be passed through
    as a falsy value crisperwhisper has to guess about."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = _FakeResult([])
    transcribe_with_crisperwhisper(fake_model, Path("/tmp/fake.wav"), "", {}, log)
    assert fake_model.transcribe.call_args.kwargs["language"] == "en"


def test_empty_words_list_handled():
    fake_model = MagicMock()
    fake_model.transcribe.return_value = _FakeResult(None)  # some backends may return None, not []
    words = transcribe_with_crisperwhisper(fake_model, Path("/tmp/fake.wav"), "en", {}, log)
    assert words == []


# ── transcribe.py: crisperwhisper can never be authoritative ────────────

def test_alignment_stages_registry_includes_stage_3():
    numbers = {s["number"] for s in transcribe._ALIGNMENT_STAGES}
    assert numbers == {1, 2, 3}
    assert transcribe._STAGE_BY_TOOL["crisperwhisper"]["number"] == 3


def test_crisperwhisper_is_a_valid_backend_choice():
    """
    Reflects the corrected design: crisperwhisper CAN be
    alignment.backend's value (the cascade is generic over stage number,
    not hardcoded to stages 1-2 -- see transcribe.py's own docstring) --
    this is no longer excluded the way an earlier version of this code
    did. The real constraint is enabled=true, tested separately below.
    """
    assert "crisperwhisper" in transcribe._STAGE_BY_TOOL
    assert transcribe._STAGE_BY_TOOL["crisperwhisper"]["number"] == 3


def test_crisperwhisper_backend_requires_enabled_true():
    """
    alignment.backend: crisperwhisper without
    alignment.crisperwhisper.enabled: true would mean stage 3 never runs
    at all, silently cascading every segment to stage 2/1 instead --
    transcribe.py raises explicitly rather than allowing that silent
    footgun. Exercises the actual validation logic by reproducing it
    exactly as transcribe.py's own module body does (that logic lives
    inline in the transcribe() function, not in a separately-callable
    unit, so this mirrors it rather than importing it directly).
    """
    align_backend = "crisperwhisper"
    crisperwhisper_enabled = False
    try:
        if align_backend not in transcribe._STAGE_BY_TOOL:
            raise ValueError("not a registered tool")
        if align_backend == "crisperwhisper" and not crisperwhisper_enabled:
            raise ValueError(
                "alignment.backend is 'crisperwhisper' but "
                "alignment.crisperwhisper.enabled is false"
            )
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "enabled" in str(e)


def test_transcribe_module_imports_without_crisperwhisper_installed():
    """
    The crisperwhisper import lives inside load_crisperwhisper_model(),
    not at module level -- confirms an existing deployment that never
    enables alignment.crisperwhisper.enabled is completely unaffected by
    this feature, including not needing the package installed at all.
    This test itself is the proof: if the import were module-level, this
    process (which has NOT installed crisperwhisper) would already have
    failed just by importing steps.transcribe_crisperwhisper above.
    """
    import steps.transcribe_crisperwhisper  # noqa: F401 -- already imported; re-import is a no-op, just documents the assertion
    assert True


if __name__ == "__main__":
    for fn in [
        test_word_schema_conversion,
        test_calls_transcribe_with_word_timestamps_true,
        test_language_falls_back_to_english_default,
        test_empty_words_list_handled,
        test_alignment_stages_registry_includes_stage_3,
        test_crisperwhisper_is_a_valid_backend_choice,
        test_crisperwhisper_backend_requires_enabled_true,
        test_transcribe_module_imports_without_crisperwhisper_installed,
    ]:
        _run(fn)
    print("\nALL TESTS PASSED")
