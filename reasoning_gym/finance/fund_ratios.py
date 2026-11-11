"""Fund performance ratio dataset (DPI, RVPI, TVPI, MOIC).

Generates private-equity / venture-capital fund performance multiple problems.
Each item presents a set of fund figures and asks the model to compute one
ratio, rounded to a fixed number of decimal places.
"""

import random
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, NamedTuple, Optional

from reasoning_gym import utils

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "fund_ratios"

# The complete set of fund performance metrics this dataset can generate.
SUPPORTED_METRICS = ("dpi", "rvpi", "tvpi", "moic")

# Per-metric prompt copy: long name, short name, and a plain-words definition of
# the formula. The definition text restates the exact formula in words so the
# prompt is self-contained.
METRIC_INFO = {
    "dpi": {
        "long": "Distributions to Paid-In",
        "short": "DPI",
        "definition": "DPI is cumulative distributions divided by called (paid-in) capital.",
    },
    "rvpi": {
        "long": "Residual Value to Paid-In",
        "short": "RVPI",
        "definition": "RVPI is the net asset value (NAV) divided by called (paid-in) capital.",
    },
    "tvpi": {
        "long": "Total Value to Paid-In",
        "short": "TVPI",
        "definition": ("TVPI is total value (NAV plus cumulative distributions) divided by called (paid-in) capital."),
    },
    "moic": {
        "long": "Multiple on Invested Capital",
        "short": "MOIC",
        "definition": (
            "MOIC is total value net of expenses (NAV plus distributions minus expenses) divided by "
            "invested capital net of expenses (called capital minus expenses)."
        ),
    },
}


@dataclass
class FundRatiosConfig:
    """Configuration for fund ratio task generation."""

    # Commitment range (total capital an LP has committed to the fund).
    min_commitment: float = 10_000_000
    max_commitment: float = 500_000_000

    # Called capital as a fraction of commitment, constrained to (0, 1].
    min_called_fraction: float = 0.3
    max_called_fraction: float = 1.0

    # Expenses as a fraction of called capital, constrained to [0, 1).
    min_expense_fraction: float = 0.05
    max_expense_fraction: float = 0.25

    # Distributions as a multiple of called capital. Capped at 1.5x (rather than
    # 2.0x) so that, combined with the NAV multiple below, total value (TVPI)
    # stays mostly within the realistic ~0.5-2.5x band observed for PE/VC funds
    # and implausible simultaneous very-high DPI and very-high RVPI scenarios
    # become rare. See the realism-iteration note in design.md.
    min_distribution_multiple: float = 0.0
    max_distribution_multiple: float = 1.5

    # NAV (residual value) as a multiple of called capital. Capped at 1.5x for
    # the same realism reason as the distribution multiple above.
    min_nav_multiple: float = 0.0
    max_nav_multiple: float = 1.5

    # Rounding applied to the computed answer.
    decimal_places: int = 2

    # Which metrics to sample from. A tuple (immutable) default avoids the
    # mutable default-argument pitfall and keeps configs hashable/deepcopy-safe.
    metrics: tuple[str, ...] = ("dpi", "rvpi", "tvpi", "moic")

    # Display currency code used in prompts.
    currency: str = "USD"

    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.size > 0, "size must be positive"

        # Range consistency (min <= max) for every configurable range.
        assert self.min_commitment <= self.max_commitment, "min_commitment must be <= max_commitment"
        assert (
            self.min_called_fraction <= self.max_called_fraction
        ), "min_called_fraction must be <= max_called_fraction"
        assert (
            self.min_expense_fraction <= self.max_expense_fraction
        ), "min_expense_fraction must be <= max_expense_fraction"
        assert (
            self.min_distribution_multiple <= self.max_distribution_multiple
        ), "min_distribution_multiple must be <= max_distribution_multiple"
        assert self.min_nav_multiple <= self.max_nav_multiple, "min_nav_multiple must be <= max_nav_multiple"

        # Commitment must be strictly positive.
        assert self.min_commitment > 0, "min_commitment must be positive"

        # Called-capital fraction must lie in (0, 1].
        assert self.min_called_fraction > 0, "min_called_fraction must be > 0"
        assert self.max_called_fraction <= 1, "max_called_fraction must be <= 1"

        # Expense fraction must lie in [0, 1) so that called - expenses > 0.
        assert self.min_expense_fraction >= 0, "min_expense_fraction must be >= 0"
        assert self.max_expense_fraction < 1, "max_expense_fraction must be < 1"

        # Distribution / NAV multiples must be non-negative.
        assert self.min_distribution_multiple >= 0, "min_distribution_multiple must be >= 0"
        assert self.min_nav_multiple >= 0, "min_nav_multiple must be >= 0"

        # Rounding precision must be non-negative.
        assert self.decimal_places >= 0, "decimal_places must be >= 0"

        # Metrics must be non-empty and a subset of the supported set.
        assert len(self.metrics) > 0, "metrics must be non-empty"
        assert set(self.metrics).issubset(set(SUPPORTED_METRICS)), f"metrics must be a subset of {SUPPORTED_METRICS}"


class FundFigures(NamedTuple):
    """The core fund figures a scenario is built from.

    All values are :class:`~decimal.Decimal` so downstream ratio arithmetic is
    exact and free of binary-float rounding surprises.
    """

    commitment: Decimal
    called: Decimal
    expenses: Decimal
    distributions: Decimal
    nav: Decimal


def _draw_uniform_decimal(rng: random.Random, low: float, high: float) -> Decimal:
    """Draw a uniform value in ``[low, high]`` as a ``Decimal``.

    The draw is performed in float space (matching the design's "uniform"
    wording) and then converted to ``Decimal`` via its string form, which keeps
    the decimal value exactly as displayed rather than its binary expansion.
    """
    if low == high:
        value = low
    else:
        value = rng.uniform(low, high)
    return Decimal(str(value))


def generate_figures(rng: random.Random, config: FundRatiosConfig) -> FundFigures:
    """Derive a consistent set of fund figures using the fraction-based algorithm.

    Figures are built from a commitment plus fractions/multiples so the
    invariants hold by construction without rejection sampling:

    - ``0 < called <= commitment`` (called_fraction in (0, 1])
    - ``0 <= expenses < called`` (expense_fraction in [0, 1)), hence the MOIC
      denominator ``called - expenses`` is strictly positive.

    All arithmetic uses :class:`~decimal.Decimal`.
    """
    # 1. Commitment, drawn uniformly and rounded to whole currency units.
    commitment = _draw_uniform_decimal(rng, config.min_commitment, config.max_commitment)
    commitment = commitment.quantize(Decimal("1"), rounding=ROUND_HALF_UP)

    # 2. Called capital as a fraction of commitment (fraction in (0, 1]).
    called_fraction = _draw_uniform_decimal(rng, config.min_called_fraction, config.max_called_fraction)
    called = commitment * called_fraction

    # 3. Expenses as a fraction of called capital (fraction in [0, 1)); because
    #    the fraction is strictly < 1, called - expenses > 0 by construction.
    expense_fraction = _draw_uniform_decimal(rng, config.min_expense_fraction, config.max_expense_fraction)
    expenses = called * expense_fraction

    # 4. Distributions and NAV as multiples of called capital.
    distribution_multiple = _draw_uniform_decimal(
        rng, config.min_distribution_multiple, config.max_distribution_multiple
    )
    nav_multiple = _draw_uniform_decimal(rng, config.min_nav_multiple, config.max_nav_multiple)
    distributions = called * distribution_multiple
    nav = called * nav_multiple

    return FundFigures(
        commitment=commitment,
        called=called,
        expenses=expenses,
        distributions=distributions,
        nav=nav,
    )


def _exact_ratio(metric: str, figures: FundFigures) -> Decimal:
    """Compute a metric's exact (unrounded) ratio from the figures.

    Formulas (see requirements / design):

    - ``total_value = nav + distributions``
    - ``DPI  = distributions / called``
    - ``RVPI = nav / called``
    - ``TVPI = total_value / called``
    - ``MOIC = (total_value - expenses) / (called - expenses)``
    """
    total_value = figures.nav + figures.distributions

    if metric == "dpi":
        return figures.distributions / figures.called
    if metric == "rvpi":
        return figures.nav / figures.called
    if metric == "tvpi":
        return total_value / figures.called
    if metric == "moic":
        # called - expenses > 0 by construction (expense_fraction < 1).
        return (total_value - figures.expenses) / (figures.called - figures.expenses)

    raise ValueError(f"Unknown metric: {metric!r}; expected one of {SUPPORTED_METRICS}")


def compute_ratio(metric: str, figures: FundFigures, decimal_places: int) -> tuple[Decimal, Decimal]:
    """Compute a metric's ratio from the figures.

    Returns a ``(exact, rounded)`` pair where ``exact`` is the full-precision
    ratio (useful for metadata's ``result``) and ``rounded`` is quantized to
    ``decimal_places`` using ``ROUND_HALF_UP`` (the stored answer).
    """
    exact = _exact_ratio(metric, figures)
    quantum = Decimal(1).scaleb(-decimal_places)  # e.g. decimal_places=2 -> Decimal("0.01")
    rounded = exact.quantize(quantum, rounding=ROUND_HALF_UP)
    return exact, rounded


def _format_currency(value: Decimal, currency: str) -> str:
    """Format a figure as a readable whole-currency amount, e.g. ``USD 80,000,000``."""
    whole = value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return f"{currency} {whole:,}"


class FundRatiosDataset(ProceduralDataset):
    """Generates private-equity / venture-capital fund performance ratio problems.

    Each item presents a set of fund figures and asks the model to compute one
    ratio (DPI, RVPI, TVPI, or MOIC), rounded to a configurable number of
    decimal places.
    """

    def __init__(self, config: FundRatiosConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single fund-ratio task.

        Args:
            idx: Index of the item to generate.

        Returns:
            dict with keys:
                - question: str, a self-contained natural-language prompt.
                - answer: str, the rounded ratio.
                - metadata: dict with figures, unrounded result, and difficulty.
        """
        # Deterministic RNG from base seed and idx (matches library convention).
        rng = random.Random(self.seed + idx)

        config: FundRatiosConfig = self.config

        figures = generate_figures(rng, config)

        # Sort the metrics so the choice is deterministic regardless of the
        # iteration order of the configured collection.
        metric = rng.choice(sorted(config.metrics))

        exact, rounded = compute_ratio(metric, figures, config.decimal_places)

        question = self._build_question(metric, figures, config)

        return {
            "question": question,
            "answer": str(rounded),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "metric": metric,
                "commitment": float(figures.commitment),
                "called": float(figures.called),
                "distributions": float(figures.distributions),
                "nav": float(figures.nav),
                "expenses": float(figures.expenses),
                "result": float(exact),
                "decimal_places": config.decimal_places,
                "difficulty": {
                    "commitment": (config.min_commitment, config.max_commitment),
                    "decimal_places": config.decimal_places,
                },
            },
        }

    def _build_question(self, metric: str, figures: FundFigures, config: FundRatiosConfig) -> str:
        """Build the metric-specific, self-contained question prompt.

        The expenses line is included only for MOIC (the only metric whose
        formula uses expenses); for DPI/RVPI/TVPI it is omitted so the prompt
        contains exactly the figures required.
        """
        info = METRIC_INFO[metric]
        currency = config.currency

        lines = [
            f"A private equity fund has the following figures (in {currency}):",
            f"- Committed capital: {_format_currency(figures.commitment, currency)}",
            f"- Called (paid-in) capital: {_format_currency(figures.called, currency)}",
            f"- Cumulative distributions: {_format_currency(figures.distributions, currency)}",
            f"- Net asset value (NAV) of remaining holdings: {_format_currency(figures.nav, currency)}",
        ]
        if metric == "moic":
            lines.append(
                f"- Fund expenses (management fees + carried interest): "
                f"{_format_currency(figures.expenses, currency)}"
            )

        lines.append("")
        lines.append(f"Calculate the fund's {info['long']} ({info['short']}). {info['definition']}")
        lines.append(
            f"Give your answer as a multiple rounded to {config.decimal_places} decimal places "
            f"(for example, 1.75). Provide only the number."
        )

        return "\n".join(lines)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score a model answer against the oracle with numeric tolerance.

        Fund ratios are decimal multiples, so string equality is too strict: a
        model may answer ``1.5``, ``1.50``, ``1.5x``, or ``+1.50``. We strip an
        optional trailing unit marker (``x``/``X`` or ``%``) and surrounding
        whitespace, then delegate to :func:`reasoning_gym.utils.compute_decimal_reward`,
        which tolerates sign, leading zeros, trailing decimals, and commas and
        returns 0.0 for ``None``/empty answers.
        """
        if answer is not None:
            cleaned = answer.strip()
            # Strip a single trailing unit marker ('x'/'X' or '%'), then any
            # whitespace that preceded it.
            if cleaned and cleaned[-1] in ("x", "X", "%"):
                cleaned = cleaned[:-1].strip()
            answer = cleaned
        # tolerate sign, leading zeros and trailing decimals, strip commas "+01,000.00" == "1000"
        return utils.compute_decimal_reward(answer, oracle_answer=entry["answer"], strip_commas=True)


class FundRatiosCurriculum(BaseCurriculum):
    """Difficulty curriculum for the fund-ratios dataset.

    Follows the ``chain_sum`` curriculum pattern. Two attributes scale
    difficulty:

    - ``commitment`` — a :class:`RangeAttributeDefinition` that scales the
      ``(min_commitment, max_commitment)`` range across increasing magnitudes
      (1e6 -> 1e9). Larger, less-round figures make the arithmetic harder. At
      the base level the lower/upper fields collapse onto the smallest
      magnitude, so ``min_commitment == max_commitment == 1e6`` (> 0), keeping
      the generated config valid.
    - ``decimal_places`` — a :class:`ScalarAttributeDefinition` over
      ``decimal_places`` with levels ``[2, 3, 4]``; more decimal places demand
      more precise division.
    """

    def __init__(self):
        super().__init__(FundRatiosCurriculum.__name__, FundRatiosConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="commitment",
                levels=[1_000_000, 10_000_000, 100_000_000, 1_000_000_000],
                default_level=0,  # Start with the smallest commitment magnitude
                description="Magnitude of the fund commitment (and derived figures)",
                lower_field_name="min_commitment",
                upper_field_name="max_commitment",
            ),
            ScalarAttributeDefinition(
                name="decimal_places",
                levels=[2, 3, 4],
                default_level=0,  # Start with 2 decimal places
                description="Number of decimal places the answer is rounded to",
                field_name="decimal_places",
            ),
        )


# Register the dataset
register_dataset(DATASET_NAME, FundRatiosDataset, FundRatiosConfig, FundRatiosCurriculum)
