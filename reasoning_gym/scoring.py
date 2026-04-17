"""Scoring cascade utilities for reducing false negatives in answer verification.

Provides a multi-step fallback cascade that wraps any dataset's ``score_answer``
with progressively more lenient matchers:

    1. ``score_answer()``          -- environment's built-in verifier
    1b. ``score_answer()``         -- retry after stripping LaTeX wrappers
    2. ``string_match``            -- case-insensitive exact comparison
    3. ``float_match``             -- numeric comparison with tolerance
    4. ``math_match``              -- symbolic math via *math-verify*

The cascade can only *upgrade* a score, never downgrade it.

``math_match`` requires the optional ``math-verify`` package.  When it is not
installed the step is silently skipped (returns 0.0).  Install via::

    pip install reasoning-gym[scoring]
"""

import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .dataset import ProceduralDataset


# ---------------------------------------------------------------------------
# LaTeX normalisation
# ---------------------------------------------------------------------------

def strip_latex(s: str) -> str:
    """Remove common LaTeX wrappers and normalise whitespace.

    Handles ``\\(…\\)``, ``\\text{}``, ``\\mathrm{}``, double-backslash
    linebreaks, tildes, and stray backslashes.
    """
    s = re.sub(r"^\\\((.*)\\\)$", r"\1", s.strip())
    s = re.sub(r"\\\((.*?)\\\)", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\\\+", " ", s)
    s = re.sub(r"~", " ", s)
    s = re.sub(r"\\", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Individual matchers
# ---------------------------------------------------------------------------

def string_match(predicted: str, expected: str) -> float:
    """Case-insensitive exact string comparison after stripping whitespace."""
    try:
        return 1.0 if predicted.lower().strip() == expected.lower().strip() else 0.0
    except Exception:
        return 0.0


def float_match(
    predicted: str,
    expected: str,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01,
) -> float:
    """Numeric comparison with configurable tolerance.

    Accepts if ``|a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)``.
    Returns 0.0 for non-numeric strings.
    """
    try:
        a = float(predicted)
        b = float(expected)
        return 1.0 if abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol) else 0.0
    except Exception:
        return 0.0


def math_match(predicted: str, expected: str) -> float:
    """Symbolic math verification via *math-verify*, with numeric fallback.

    Strips dollar signs and common display-math delimiters before parsing.
    Falls back to :func:`float_match` on the parsed numeric values when
    symbolic ``verify`` returns ``False``.

    Returns 0.0 when ``math-verify`` is not installed.
    """
    try:
        from math_verify import parse, verify
    except ImportError:
        return 0.0

    try:
        a = expected.strip("$")
        b = predicted.strip("$")

        for delim_open, delim_close in [(r"\[", r"\]"), (r"\(", r"\)"), (r"\,", r"\,")]:
            if a.startswith(delim_open) and a.endswith(delim_close):
                a = a[2:-2].strip()
            if b.startswith(delim_open) and b.endswith(delim_close):
                b = b[2:-2].strip()

        pa = parse(f"${a}$")
        pb = parse(f"${b}$")

        if verify(pa, pb):
            return 1.0

        # Numeric fallback on the first parsed element
        try:
            va, vb = float(pa[0]), float(pb[0])
            return 1.0 if abs(va - vb) <= max(0.01 * max(abs(va), abs(vb)), 0.01) else 0.0
        except Exception:
            return 0.0
    except Exception:
        return 0.0


def _mathrm_to_text(s: str) -> str:
    r"""Replace ``\mathrm{…}`` with ``\text{…}`` for a second math_match attempt."""
    return re.sub(r"\\mathrm\{([^}]*)\}", r"\\text{\1}", s)


# ---------------------------------------------------------------------------
# Full cascade
# ---------------------------------------------------------------------------

def cascade_score(
    answer: str,
    expected: str,
    dataset: Optional["ProceduralDataset"] = None,
    entry: Optional[dict[str, Any]] = None,
) -> float:
    """Apply the multi-step scoring cascade.

    When *dataset* and *entry* are supplied the environment's own
    ``score_answer`` is tried first (steps 1 & 1b).  The remaining steps
    use only the raw answer strings and never require a dataset instance.

    The cascade can only upgrade — if an earlier step already returned
    a near-perfect score (>= 0.99) it is returned immediately.

    Args:
        answer:   The model's predicted answer string.
        expected: The gold / oracle answer string.
        dataset:  Optional :class:`ProceduralDataset` whose ``score_answer``
                  should be tried first.
        entry:    The dataset entry dict (must contain at least ``"answer"``).
                  Required when *dataset* is provided.

    Returns:
        A score in ``[0.0, 1.0]``.
    """
    best = 0.0

    # Step 1: environment's built-in verifier
    if dataset is not None and entry is not None:
        try:
            score = float(dataset.score_answer(answer, entry))
            if score >= 0.99:
                return score
            best = max(best, score)
        except Exception:
            pass

        # Step 1b: retry after stripping LaTeX
        cleaned = strip_latex(answer)
        if cleaned != answer:
            try:
                score = float(dataset.score_answer(cleaned, entry))
                if score >= 0.99:
                    return score
                best = max(best, score)
            except Exception:
                pass

    # Steps 2-5: string / float / math cascade
    for score in (
        string_match(answer, expected),
        string_match(strip_latex(answer), strip_latex(expected)),
        float_match(answer, expected),
        math_match(answer, expected),
        math_match(_mathrm_to_text(answer), _mathrm_to_text(expected)),
    ):
        if score >= 0.99:
            return score
        best = max(best, score)

    return best
