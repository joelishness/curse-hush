"""
Regression tests for steps/transcribe_crisperwhisper.py's schema
conversion, and for the alignment-engine registry it plugs into (see
steps/transcribe.py's own module docstring and utils.
validate_alignment_engines()) -- in particular, that CrisperWhisper is a
fully ordinary, independent engine that CAN be marked final like any
other, with no special-casing left over from an earlier version of this
codebase where it was comparison-only.

Does NOT require the crisperwhisper package to be installed -- the
model itself is mocked throughout; these tests are about this
pipeline's own conversion/registry/validation logic, not about
crisperwhisper's real behavior (which needs a real install and real
audio to check at all -- see transcribe_crisperwhisper.py's own
docstring for the validation status).

Run with:  PYTHONPATH=src python3 tests/test_transcribe_crisperwhisper.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging
from steps.transcribe_crisperwhisper import transcribe_with_crisperwhisper
from steps import transcribe
import utils

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


# ── steps/transcribe_crisperwhisper.py: schema conversion ───────────────

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
    language string (this engine's own alignment.engines.crisperwhisper.
    language left null, or not yet resolved) shouldn't be passed through
    as a falsy value crisperwhisper has to guess about. NOTE: unlike
    alignment.engines.whisperx.language, null here does NOT mean
    auto-detect -- this engine's transcribe() call has no such mode; see
    this function's own docstring."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = _FakeResult([])
    transcribe_with_crisperwhisper(fake_model, Path("/tmp/fake.wav"), "", {}, log)
    assert fake_model.transcribe.call_args.kwargs["language"] == "en"


def test_empty_words_list_handled():
    fake_model = MagicMock()
    fake_model.transcribe.return_value = _FakeResult(None)  # some backends may return None, not []
    words = transcribe_with_crisperwhisper(fake_model, Path("/tmp/fake.wav"), "en", {}, log)
    assert words == []


def test_transcribe_crisperwhisper_module_imports_without_crisperwhisper_installed():
    """
    The crisperwhisper import lives inside load_crisperwhisper_model(),
    not at module level -- confirms an existing deployment that never
    enables alignment.engines.crisperwhisper.enabled is completely
    unaffected by this engine, including not needing the package
    installed at all. This test itself is the proof: if the import were
    module-level, this process (which has NOT installed crisperwhisper)
    would already have failed just by importing
    steps.transcribe_crisperwhisper above.
    """
    import steps.transcribe_crisperwhisper  # noqa: F401 -- already imported; re-import is a no-op, just documents the assertion
    assert True


# ── steps/transcribe.py: the engine registry itself ──────────────────────

def test_engine_registry_has_all_three_engines():
    """
    utils.ALIGNMENT_ENGINE_NAMES is the single shared registry
    steps/transcribe.py's own dispatch and utils.
    validate_alignment_engines()/alignment_engines_summary() all key off
    -- confirms transcribe.py imports and re-exposes it as _ENGINES
    (rather than defining its own, separate list that could drift out of
    sync), and that every registered engine has a display label.
    """
    assert transcribe._ENGINES == utils.ALIGNMENT_ENGINE_NAMES
    assert set(transcribe._ENGINES) == {"whisperx", "mfa", "crisperwhisper"}
    for name in transcribe._ENGINES:
        assert transcribe._engine_label(name), f"{name} has no display label"


def test_crisperwhisper_can_be_marked_final_like_any_other_engine():
    """
    Reflects the corrected design: any registered engine, including
    crisperwhisper, CAN be alignment.engines.<name>.final: true -- there
    is no special-casing that excludes it the way an earlier version of
    this codebase did (when crisperwhisper was "comparison-only" and
    could never become authoritative). This is now this config's own
    DEFAULT (see config.yaml's alignment.engines.crisperwhisper block),
    following real-world validation against Independence Day (1996)
    finding it more accurate than both whisperx and mfa.
    """
    cfg = {"alignment": {"engines": {
        "whisperx":       {"enabled": False, "debug_subtitle": False, "final": False, "final_subtitle": False, "embed_subtitle": False},
        "mfa":            {"enabled": False, "debug_subtitle": False, "final": False, "final_subtitle": False, "embed_subtitle": False},
        "crisperwhisper": {"enabled": True,  "debug_subtitle": True,  "final": True,  "final_subtitle": True,  "embed_subtitle": False},
    }}}
    utils.validate_alignment_engines(cfg)  # must not raise

    toggles = {name: transcribe._engine_toggles(cfg, name) for name in transcribe._ENGINES}
    final_engines = [name for name in transcribe._ENGINES if toggles[name]["final"]]
    assert final_engines == ["crisperwhisper"]


def test_final_or_debug_subtitle_without_enabled_is_rejected():
    """
    alignment.engines.<name>.final: true (or debug_subtitle: true)
    without that same engine's own enabled: true is caught by
    utils.validate_alignment_engines() at startup -- an engine that
    never runs can't be authoritative, and there's nothing to render a
    debug subtitle from. Exercises the real validation function
    directly, not a re-derivation of its logic -- unlike an earlier
    version of this codebase's own tests for the (now-removed)
    alignment.backend cascade, which had to mirror inline pipeline logic
    because no standalone, importable check existed yet.
    """
    cfg = {"alignment": {"engines": {
        "whisperx":       {"enabled": False, "debug_subtitle": True, "final": False, "final_subtitle": False, "embed_subtitle": False},
        "mfa":            {"enabled": False, "debug_subtitle": False, "final": False, "final_subtitle": False, "embed_subtitle": False},
        "crisperwhisper": {"enabled": True,  "debug_subtitle": True,  "final": True,  "final_subtitle": True,  "embed_subtitle": False},
    }}}
    try:
        utils.validate_alignment_engines(cfg)
        raise AssertionError("should have raised ConfigError")
    except utils.ConfigError as e:
        assert "debug_subtitle" in str(e) and "enabled" in str(e)


def test_zero_or_multiple_final_engines_is_rejected():
    """Exactly one enabled engine must be final -- zero and two-plus are
    both rejected, each with a message naming the actual problem."""
    zero_final = {"alignment": {"engines": {
        "whisperx":       {"enabled": True, "debug_subtitle": False, "final": False, "final_subtitle": False, "embed_subtitle": False},
        "mfa":            {"enabled": False, "debug_subtitle": False, "final": False, "final_subtitle": False, "embed_subtitle": False},
        "crisperwhisper": {"enabled": False, "debug_subtitle": False, "final": False, "final_subtitle": False, "embed_subtitle": False},
    }}}
    try:
        utils.validate_alignment_engines(zero_final)
        raise AssertionError("should have raised ConfigError")
    except utils.ConfigError as e:
        assert "none" in str(e).lower()

    two_final = {"alignment": {"engines": {
        "whisperx":       {"enabled": True, "debug_subtitle": False, "final": True, "final_subtitle": False, "embed_subtitle": False},
        "mfa":            {"enabled": False, "debug_subtitle": False, "final": False, "final_subtitle": False, "embed_subtitle": False},
        "crisperwhisper": {"enabled": True,  "debug_subtitle": False, "final": True, "final_subtitle": False, "embed_subtitle": False},
    }}}
    try:
        utils.validate_alignment_engines(two_final)
        raise AssertionError("should have raised ConfigError")
    except utils.ConfigError as e:
        assert "more than one" in str(e).lower()


def test_mfa_enabled_without_whisperx_enabled_is_a_valid_configuration():
    """
    alignment.engines.mfa.enabled: true with alignment.engines.whisperx.
    enabled: false is NOT a validation error -- it's the documented,
    supported shape where WhisperX's own recognition+alignment runs
    internally as MFA's required input (see steps/transcribe.py's own
    module docstring) without being separately exposed. This test exists
    specifically so that invariant doesn't regress into an accidental
    validation error later.
    """
    cfg = {"alignment": {"engines": {
        "whisperx":       {"enabled": False, "debug_subtitle": False, "final": False, "final_subtitle": False, "embed_subtitle": False},
        "mfa":            {"enabled": True,  "debug_subtitle": True,  "final": True,  "final_subtitle": True,  "embed_subtitle": False},
        "crisperwhisper": {"enabled": False, "debug_subtitle": False, "final": False, "final_subtitle": False, "embed_subtitle": False},
    }}}
    utils.validate_alignment_engines(cfg)  # must not raise


if __name__ == "__main__":
    for fn in [
        test_word_schema_conversion,
        test_calls_transcribe_with_word_timestamps_true,
        test_language_falls_back_to_english_default,
        test_empty_words_list_handled,
        test_transcribe_crisperwhisper_module_imports_without_crisperwhisper_installed,
        test_engine_registry_has_all_three_engines,
        test_crisperwhisper_can_be_marked_final_like_any_other_engine,
        test_final_or_debug_subtitle_without_enabled_is_rejected,
        test_zero_or_multiple_final_engines_is_rejected,
        test_mfa_enabled_without_whisperx_enabled_is_a_valid_configuration,
    ]:
        _run(fn)
    print("\nALL TESTS PASSED")
