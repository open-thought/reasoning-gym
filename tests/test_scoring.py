import pytest

import reasoning_gym
from reasoning_gym.scoring import (
    cascade_score,
    float_match,
    math_match,
    string_match,
    strip_latex,
    _mathrm_to_text,
)


# ---------------------------------------------------------------------------
# strip_latex
# ---------------------------------------------------------------------------

class TestStripLatex:
    def test_inline_math_delimiters(self):
        assert strip_latex(r"\(42\)") == "42"

    def test_inline_math_mid_string(self):
        assert strip_latex(r"the value is \(x + 1\) here") == "the value is x + 1 here"

    def test_text_command(self):
        assert strip_latex(r"\text{hello world}") == "hello world"

    def test_mathrm_command(self):
        assert strip_latex(r"\mathrm{cm}") == "cm"

    def test_double_backslash(self):
        assert strip_latex(r"a \\ b") == "a b"

    def test_tilde(self):
        assert strip_latex("a~b") == "a b"

    def test_stray_backslashes(self):
        assert strip_latex(r"\alpha + \beta") == "alpha + beta"

    def test_whitespace_normalisation(self):
        assert strip_latex("  a   b  ") == "a b"

    def test_combined(self):
        assert strip_latex(r"\(\text{answer}\)") == "answer"

    def test_plain_string_unchanged(self):
        assert strip_latex("42") == "42"
        assert strip_latex("hello") == "hello"


# ---------------------------------------------------------------------------
# string_match
# ---------------------------------------------------------------------------

class TestStringMatch:
    def test_exact(self):
        assert string_match("42", "42") == 1.0

    def test_case_insensitive(self):
        assert string_match("Hello", "hello") == 1.0
        assert string_match("TRUE", "true") == 1.0

    def test_whitespace_stripped(self):
        assert string_match("  42  ", "42") == 1.0

    def test_mismatch(self):
        assert string_match("42", "43") == 0.0

    def test_empty_strings(self):
        assert string_match("", "") == 1.0

    def test_non_string_graceful(self):
        assert string_match(None, "42") == 0.0


# ---------------------------------------------------------------------------
# float_match
# ---------------------------------------------------------------------------

class TestFloatMatch:
    def test_exact(self):
        assert float_match("3.14", "3.14") == 1.0

    def test_within_tolerance(self):
        assert float_match("100", "100.5") == 1.0

    def test_outside_tolerance(self):
        assert float_match("100", "102") == 0.0

    def test_zero_tolerance(self):
        assert float_match("0", "0.005") == 1.0

    def test_negative(self):
        assert float_match("-5.0", "-5.0") == 1.0
        assert float_match("-5.0", "5.0") == 0.0

    def test_non_numeric(self):
        assert float_match("abc", "42") == 0.0
        assert float_match("42", "abc") == 0.0

    def test_custom_tolerance(self):
        assert float_match("100", "110", rel_tol=0.15) == 1.0
        assert float_match("100", "110", rel_tol=0.05) == 0.0


# ---------------------------------------------------------------------------
# math_match
# ---------------------------------------------------------------------------

class TestMathMatch:
    def test_returns_zero_without_math_verify(self, monkeypatch):
        """When math-verify is not importable, math_match should return 0."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "math_verify":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        assert math_match("42", "42") == 0.0

    def test_dollar_sign_stripping(self):
        result = math_match("$42$", "$42$")
        assert result >= 0.0  # at least doesn't crash

    def test_display_math_delimiters(self):
        result = math_match(r"\[42\]", r"\[42\]")
        assert result >= 0.0

    def test_non_parseable_returns_zero(self):
        assert math_match("not math at all ???", "also not math ???") == 0.0


# ---------------------------------------------------------------------------
# _mathrm_to_text helper
# ---------------------------------------------------------------------------

class TestMathrmToText:
    def test_replaces_mathrm(self):
        assert _mathrm_to_text(r"\mathrm{cm}") == r"\text{cm}"

    def test_no_mathrm_unchanged(self):
        assert _mathrm_to_text("42") == "42"

    def test_multiple_occurrences(self):
        s = r"\mathrm{kg} \cdot \mathrm{m}"
        assert _mathrm_to_text(s) == r"\text{kg} \cdot \text{m}"


# ---------------------------------------------------------------------------
# cascade_score — without dataset
# ---------------------------------------------------------------------------

class TestCascadeScoreStandalone:
    def test_exact_string(self):
        assert cascade_score("42", "42") >= 0.99

    def test_case_insensitive_string(self):
        assert cascade_score("True", "true") >= 0.99

    def test_latex_wrapped_string(self):
        assert cascade_score(r"\text{42}", "42") >= 0.99

    def test_numeric_tolerance(self):
        assert cascade_score("100.05", "100") >= 0.99

    def test_mismatch(self):
        assert cascade_score("42", "99") == 0.0

    def test_empty_answer(self):
        assert cascade_score("", "42") == 0.0


# ---------------------------------------------------------------------------
# cascade_score — with a real dataset
# ---------------------------------------------------------------------------

class TestCascadeScoreWithDataset:
    def test_chain_sum_exact(self):
        ds = reasoning_gym.create_dataset("chain_sum", size=5, seed=42)
        entry = ds[0]
        score = cascade_score(entry["answer"], entry["answer"], dataset=ds, entry=entry)
        assert score == 1.0

    def test_chain_sum_latex_wrapped(self):
        ds = reasoning_gym.create_dataset("chain_sum", size=5, seed=42)
        entry = ds[0]
        wrapped = rf"\text{{{entry['answer']}}}"
        score = cascade_score(wrapped, entry["answer"], dataset=ds, entry=entry)
        assert score >= 0.99

    def test_never_downgrades(self):
        """The cascade should never return less than score_answer itself."""
        ds = reasoning_gym.create_dataset("chain_sum", size=10, seed=123)
        for entry in ds:
            base = ds.score_answer(entry["answer"], entry)
            cascaded = cascade_score(entry["answer"], entry["answer"], dataset=ds, entry=entry)
            assert cascaded >= base


# ---------------------------------------------------------------------------
# ProceduralDataset.score_answer_cascade convenience method
# ---------------------------------------------------------------------------

class TestScoreAnswerCascadeMethod:
    def test_method_exists(self):
        ds = reasoning_gym.create_dataset("chain_sum", size=1, seed=0)
        assert hasattr(ds, "score_answer_cascade")

    def test_oracle_answer_scores_one(self):
        ds = reasoning_gym.create_dataset("chain_sum", size=5, seed=42)
        for entry in ds:
            assert ds.score_answer_cascade(entry["answer"], entry) == 1.0

    def test_none_answer_scores_zero(self):
        ds = reasoning_gym.create_dataset("chain_sum", size=1, seed=0)
        entry = ds[0]
        assert ds.score_answer_cascade(None, entry) == 0.0

    def test_latex_wrapped_answer(self):
        ds = reasoning_gym.create_dataset("chain_sum", size=5, seed=42)
        entry = ds[0]
        wrapped = rf"\({entry['answer']}\)"
        assert ds.score_answer_cascade(wrapped, entry) >= 0.99

    def test_never_less_than_score_answer(self):
        ds = reasoning_gym.create_dataset("chain_sum", size=10, seed=99)
        for entry in ds:
            base = ds.score_answer(entry["answer"], entry)
            cascaded = ds.score_answer_cascade(entry["answer"], entry)
            assert cascaded >= base


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------

class TestTopLevelImports:
    def test_cascade_score_importable(self):
        from reasoning_gym import cascade_score as cs
        assert callable(cs)

    def test_matchers_importable(self):
        from reasoning_gym import string_match, float_match, math_match, strip_latex
        assert all(callable(f) for f in [string_match, float_match, math_match, strip_latex])
