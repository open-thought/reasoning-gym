"""Syllogism reasoning task generator"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from random import Random
from typing import Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset
from ..utils import StrEnum

DATASET_NAME = "syllogism"
DATASET_VERSION = 2

SEMANTICS = (
    "Use only the statements given, ignoring real-world knowledge. Categories may be empty. "
    "All and No statements do not imply existence; Some statements do. "
    "The conclusion must hold whenever both statements are true."
)


class Quantifier(StrEnum):
    ALL = "All"
    NO = "No"
    SOME = "Some"
    SOME_NOT = "Some ... are not"


class Term:
    """Represents a categorical term used in syllogisms"""

    def __init__(self, name: str, plural: str):
        self.name = name
        self.plural = plural

    def __repr__(self) -> str:
        """Return string representation of the term"""
        return f"Term({self.name}, {self.plural})"


@dataclass
class SyllogismConfig:
    """Configuration for syllogism task generation"""

    # Control which quantifiers to use
    allow_all: bool = True
    allow_no: bool = True
    allow_some: bool = True
    allow_some_not: bool = True

    # Target invalid fraction; use the available class if the target is impossible.
    invalid_ratio: float = 0.3

    # Probability of generating inversion problems instead of syllogisms (0.0 to 1.0)
    inversion_probability: float = 0.3

    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert any(
            [self.allow_all, self.allow_no, self.allow_some, self.allow_some_not]
        ), "At least one quantifier type must be allowed"
        assert 0.0 <= self.invalid_ratio <= 1.0, "invalid_ratio must be between 0.0 and 1.0"
        assert 0.0 <= self.inversion_probability <= 1.0, "inversion_probability must be between 0.0 and 1.0"


class SyllogismDataset(ProceduralDataset):
    """Generates syllogism reasoning tasks"""

    # Default terms if none provided
    DEFAULT_TERMS = [
        # People
        Term("mortal", "mortals"),
        Term("human", "humans"),
        Term("child", "children"),
        Term("adult", "adults"),
        Term("parent", "parents"),
        Term("grandparent", "grandparents"),
        # Professions
        Term("philosopher", "philosophers"),
        Term("student", "students"),
        Term("teacher", "teachers"),
        Term("doctor", "doctors"),
        Term("scientist", "scientists"),
        Term("artist", "artists"),
        Term("musician", "musicians"),
        Term("writer", "writers"),
        Term("programmer", "programmers"),
        Term("engineer", "engineers"),
        Term("lawyer", "lawyers"),
        Term("chef", "chefs"),
        # Animals
        Term("animal", "animals"),
        Term("mammal", "mammals"),
        Term("dog", "dogs"),
        Term("cat", "cats"),
        Term("bird", "birds"),
        Term("fish", "fish"),
        Term("reptile", "reptiles"),
        Term("insect", "insects"),
        Term("butterfly", "butterflies"),
        Term("bee", "bees"),
        Term("ant", "ants"),
        Term("spider", "spiders"),
        Term("horse", "horses"),
        Term("elephant", "elephants"),
        Term("lion", "lions"),
        Term("tiger", "tigers"),
        Term("whale", "whales"),
        Term("dolphin", "dolphins"),
    ]

    def __init__(self, config: SyllogismConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.terms = self.DEFAULT_TERMS

    def _get_allowed_quantifiers(self) -> list[Quantifier]:
        """Get list of allowed quantifiers based on config"""
        quantifiers = []
        if self.config.allow_all:
            quantifiers.append(Quantifier.ALL)
        if self.config.allow_no:
            quantifiers.append(Quantifier.NO)
        if self.config.allow_some:
            quantifiers.append(Quantifier.SOME)
        if self.config.allow_some_not:
            quantifiers.append(Quantifier.SOME_NOT)
        return quantifiers

    @staticmethod
    @lru_cache(maxsize=None)
    def _entails_indexed(premises: tuple, conclusion: tuple, num_terms: int) -> bool:
        """Check all Venn-region occupancies (at most 256 models).

        One member per occupied region suffices for categorical statements.
        """
        if not 1 <= num_terms <= 3:
            raise ValueError("Expected one to three categories")

        def constraint(statement):
            quantifier, subject, predicate = statement
            # All/Some-not concern A minus B; No/Some concern A intersect B.
            predicate_present = quantifier in (Quantifier.NO, Quantifier.SOME)
            regions = sum(
                1 << region
                for region in range(1 << num_terms)
                if region & (1 << subject) and bool(region & (1 << predicate)) == predicate_present
            )
            if not isinstance(quantifier, Quantifier):
                raise ValueError(f"Unknown quantifier: {quantifier}")
            return regions, quantifier in (Quantifier.SOME, Quantifier.SOME_NOT)

        requirements = [constraint(premise) for premise in premises]
        conclusion_regions, conclusion_exists = constraint(conclusion)
        for occupied in range(1 << (1 << num_terms)):
            if all(bool(occupied & regions) == exists for regions, exists in requirements):
                if bool(occupied & conclusion_regions) != conclusion_exists:
                    return False
        return True

    @staticmethod
    def _entails(premises: tuple, conclusion: tuple) -> bool:
        """Check entailment with empty categories allowed."""
        terms = list(dict.fromkeys(term for statement in (*premises, conclusion) for term in statement[1:]))

        def indexed(statement):
            quantifier, subject, predicate = statement
            return quantifier, terms.index(subject), terms.index(predicate)

        return SyllogismDataset._entails_indexed(
            tuple(indexed(premise) for premise in premises), indexed(conclusion), len(terms)
        )

    @staticmethod
    def _is_valid_syllogism(
        premise1: tuple[Quantifier, "Term", "Term"],
        premise2: tuple[Quantifier, "Term", "Term"],
        conclusion: tuple[Quantifier, "Term", "Term"],
    ) -> bool:
        """Check a three-category syllogism."""
        common_terms = set(premise1[1:]) & set(premise2[1:])
        all_terms = set(premise1[1:]) | set(premise2[1:])
        if len(common_terms) != 1 or len(all_terms) != 3:
            return False
        if set(conclusion[1:]) != all_terms - common_terms:
            return False
        return SyllogismDataset._entails((premise1, premise2), conclusion)

    def _format_quantifier_statement(self, quantifier: Quantifier, subject: Term, predicate: Term) -> str:
        """Format a quantified statement in natural language"""
        if quantifier == Quantifier.SOME_NOT:
            return f"Some {subject.plural} are not {predicate.plural}"
        else:
            return f"{quantifier.value} {subject.plural} are {predicate.plural}"

    def _check_logical_equivalence(
        self, premise: tuple[Quantifier, Term, Term], conclusion: tuple[Quantifier, Term, Term]
    ) -> bool:
        """Check one-way entailment, not equivalence."""
        return self._entails((premise,), conclusion)

    @staticmethod
    @lru_cache(maxsize=None)
    def _candidate_forms(quantifiers: tuple[Quantifier, ...], inversion: bool) -> tuple:
        """Group allowed forms by validity."""
        candidates = ([], [])
        for q1, q2, qc in product(quantifiers, repeat=3):
            premise1, premise2 = (q1, 0, 1), (q2, 1, 2)
            for selected in (1, 2) if inversion else (0,):
                subject, predicate = {0: (0, 2), 1: (1, 0), 2: (2, 1)}[selected]
                conclusion = (qc, subject, predicate)
                valid = SyllogismDataset._entails_indexed((premise1, premise2), conclusion, 3)
                candidates[valid].append((premise1, premise2, conclusion, selected))
        return tuple(tuple(group) for group in candidates)

    def _generate_syllogism(self, rng: Random, idx: int) -> dict:
        """Generate a problem using both premises."""
        terms = rng.sample(self.terms, 3)
        inversion = rng.random() < self.config.inversion_probability
        target_valid = rng.random() >= self.config.invalid_ratio
        candidates = self._candidate_forms(tuple(self._get_allowed_quantifiers()), inversion)
        is_valid = target_valid if candidates[target_valid] else not target_valid
        premise1, premise2, conclusion, selected = rng.choice(candidates[is_valid])

        def format_statement(statement):
            quantifier, subject, predicate = statement
            return self._format_quantifier_statement(quantifier, terms[subject], terms[predicate])

        premise1_text, premise2_text, conclusion_text = map(format_statement, (premise1, premise2, conclusion))
        question = (
            f"{SEMANTICS}\n\n"
            f"Consider these statements:\n"
            f"1. {premise1_text}\n"
            f"2. {premise2_text}\n\n"
            f"Does it logically follow that:\n"
            f"{conclusion_text}?\n"
            f"(Answer Yes or No)"
        )
        metadata = {
            "source_dataset": DATASET_NAME,
            "source_index": idx,
            "dataset_version": DATASET_VERSION,
            "semantics": "modern_empty_categories_allowed",
            "premise1": premise1_text,
            "premise2": premise2_text,
            "conclusion": conclusion_text,
            "is_valid": is_valid,
            "type": "inversion" if inversion else "syllogism",
        }
        if inversion:
            metadata["selected_premise"] = selected
        return {"question": question, "answer": "Yes" if is_valid else "No", "metadata": metadata}

    def __getitem__(self, idx: int) -> dict:
        """Generate a single syllogism task"""
        rng = Random(self.seed + idx)
        return self._generate_syllogism(rng, idx)


class SyllogismCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(SyllogismCurriculum.__name__, SyllogismConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="allow_all",
                field_name="allow_all",
                levels=[True, True, True, True],
                description="Allow 'All' quantifier",
            ),
            ScalarAttributeDefinition(
                name="allow_no",
                field_name="allow_no",
                levels=[False, True, True, True],
                description="Allow 'No' quantifier",
            ),
            ScalarAttributeDefinition(
                name="allow_some",
                field_name="allow_some",
                levels=[False, False, True, True],
                description="Allow 'Some' quantifier",
            ),
            ScalarAttributeDefinition(
                name="allow_some_not",
                field_name="allow_some_not",
                levels=[False, False, False, True],
                description="Allow 'Some ... are not' quantifier",
            ),
        )


register_dataset(DATASET_NAME, SyllogismDataset, SyllogismConfig, SyllogismCurriculum)
