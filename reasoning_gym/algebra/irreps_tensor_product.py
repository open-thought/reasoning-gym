"""O(3) irrep tensor-product dataset and symbolic helpers.

This first PR supports only binary tensor products of two single irreps.
It does not support direct sums, multi-factor products, or inputs where
either side is itself a sum of irreps.

The task uses e3nn-style notation as reference context only.  The runtime
implementation is intentionally pure symbolic Python and does not depend on
e3nn because the v1 oracle only needs token parsing, parity multiplication,
inclusive l-range decomposition, and canonical formatting/scoring.
"""

import random
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from ..dataset import ProceduralDataset
from ..factory import register_dataset

DATASET_NAME = "irreps_tensor_product"

Parity = Literal["e", "o"]

_IRREP_RE = re.compile(r"^(0|[1-9][0-9]*)([eEoO])$")

__all__ = [
    "DATASET_NAME",
    "Irrep",
    "Parity",
    "IrrepsTensorProductConfig",
    "IrrepsTensorProductDataset",
    "parse_irrep",
    "format_irrep",
    "multiply_parity",
    "decompose_tensor_product",
    "format_decomposition",
    "irrep_dimension",
]


@dataclass(frozen=True)
class Irrep:
    """An O(3) irrep token with degree and parity."""

    l: int
    parity: Parity

    def __post_init__(self) -> None:
        if type(self.l) is not int:
            raise TypeError("l must be an int")
        if self.l < 0:
            raise ValueError("l must be non-negative")
        if not isinstance(self.parity, str):
            raise TypeError("parity must be a string")
        if self.parity not in ("e", "o"):
            raise ValueError("parity must be 'e' or 'o'")


def parse_irrep(text: str) -> Irrep:
    """Parse a canonical single-irrep token into an Irrep."""

    if not isinstance(text, str):
        raise TypeError("irrep token must be a string")

    match = _IRREP_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError("invalid irrep token")

    l_text, parity_text = match.groups()
    return Irrep(l=int(l_text), parity=parity_text.lower())


def format_irrep(irrep: Irrep) -> str:
    """Format an Irrep as canonical lowercase ASCII text."""

    if not isinstance(irrep, Irrep):
        raise TypeError("irrep must be an Irrep")
    return f"{irrep.l}{irrep.parity}"


def multiply_parity(left: Parity, right: Parity) -> Parity:
    """Multiply normalized parity values."""

    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("parity values must be strings")
    if left not in ("e", "o") or right not in ("e", "o"):
        raise ValueError("parity values must be 'e' or 'o'")
    return "e" if left == right else "o"


def decompose_tensor_product(left: Irrep, right: Irrep) -> tuple[Irrep, ...]:
    """Return the canonical O(3) tensor-product decomposition."""

    if not isinstance(left, Irrep) or not isinstance(right, Irrep):
        raise TypeError("tensor-product operands must be Irrep values")

    product_parity = multiply_parity(left.parity, right.parity)
    min_l = abs(left.l - right.l)
    max_l = left.l + right.l
    return tuple(Irrep(l=l_value, parity=product_parity) for l_value in range(min_l, max_l + 1))


def format_decomposition(terms: tuple[Irrep, ...]) -> str:
    """Format a canonical decomposition tuple."""

    if not isinstance(terms, tuple):
        raise TypeError("decomposition must be a tuple")
    if not terms:
        raise ValueError("decomposition must be non-empty")
    if any(not isinstance(term, Irrep) for term in terms):
        raise TypeError("decomposition terms must be Irrep values")

    first_parity = terms[0].parity
    previous_l = terms[0].l - 1

    for term in terms:
        if term.parity != first_parity:
            raise ValueError("decomposition terms must share parity")
        if term.l != previous_l + 1:
            raise ValueError("decomposition terms must be consecutive and sorted")
        previous_l = term.l

    return " + ".join(format_irrep(term) for term in terms)


def irrep_dimension(irrep: Irrep) -> int:
    """Return the dimension of a single O(3) irrep."""

    if not isinstance(irrep, Irrep):
        raise TypeError("irrep must be an Irrep")
    return 2 * irrep.l + 1


def _normalize_answer_separators(answer: str) -> str:
    return answer.replace("⊕", "+")


def _parse_decomposition_answer(answer: str) -> tuple[Irrep, ...]:
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    normalized = _normalize_answer_separators(answer).strip()
    if not normalized:
        raise ValueError("answer must be non-empty")

    pieces = [piece.strip() for piece in normalized.split("+")]
    if any(not piece for piece in pieces):
        raise ValueError("answer contains an empty term")

    terms = tuple(parse_irrep(piece) for piece in pieces)

    # Reuse formatter validation so scorer rejects duplicates, wrong order,
    # mixed parity, and non-consecutive output ranges.
    format_decomposition(terms)
    return terms


@dataclass
class IrrepsTensorProductConfig:
    """Configuration for O(3) irrep tensor-product tasks."""

    min_l: int = 0
    max_l: int = 4
    include_even: bool = True
    include_odd: bool = True
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.min_l >= 0, "min_l must be >= 0"
        assert self.max_l >= self.min_l, "max_l must be >= min_l"
        assert self.include_even or self.include_odd, "at least one parity must be enabled"
        assert self.size > 0, "size must be positive"


class IrrepsTensorProductDataset(ProceduralDataset):
    """Generate rule-present O(3) binary tensor products of two single irreps."""

    def __init__(self, config: IrrepsTensorProductConfig):
        self._prompt_templates = [
            (
                "Decompose the O(3) tensor product {left} x {right}. "
                "Use this rule: output every irrep from |l1-l2| through l1+l2, inclusive, "
                "and multiply parity using e*e=e, e*o=o, o*e=o, o*o=e. "
                "Reply with the canonical ascending decomposition, for example: 1o + 2o + 3o."
            )
        ]
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(self.seed + idx)
        left = self._sample_irrep(rng)
        right = self._sample_irrep(rng)
        terms = decompose_tensor_product(left, right)

        left_text = format_irrep(left)
        right_text = format_irrep(right)
        answer = format_decomposition(terms)

        return {
            "question": rng.choice(self._prompt_templates).format(left=left_text, right=right_text),
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "seed": self.seed,
                "left_irrep": left_text,
                "right_irrep": right_text,
                "l1": left.l,
                "l2": right.l,
                "parity_1": left.parity,
                "parity_2": right.parity,
                "parity_case": f"{left.parity}{right.parity}",
                "parity_product": terms[0].parity,
                "output_l_min": terms[0].l,
                "output_l_max": terms[-1].l,
                "term_count": len(terms),
                "decomposition_terms": [format_irrep(term) for term in terms],
                "difficulty": {
                    "min_l": self.config.min_l,
                    "max_l": self.config.max_l,
                    "include_even": self.config.include_even,
                    "include_odd": self.config.include_odd,
                },
            },
        }

    def _sample_irrep(self, rng: random.Random) -> Irrep:
        parity_options: list[Parity] = []
        if self.config.include_even:
            parity_options.append("e")
        if self.config.include_odd:
            parity_options.append("o")

        return Irrep(
            l=rng.randint(self.config.min_l, self.config.max_l),
            parity=rng.choice(parity_options),
        )

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0

        try:
            predicted_terms = _parse_decomposition_answer(answer)
        except (TypeError, ValueError):
            return 0.0

        expected_terms = tuple(parse_irrep(term) for term in entry["metadata"]["decomposition_terms"])
        return 1.0 if predicted_terms == expected_terms else 0.0


register_dataset(DATASET_NAME, IrrepsTensorProductDataset, IrrepsTensorProductConfig)
