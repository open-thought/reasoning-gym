"""Syllogism reasoning task generator"""

from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import List, Optional, Tuple

from ..dataset import ProceduralDataset


class Quantifier(Enum):
    ALL = "All"
    NO = "No"
    SOME = "Some"
    SOME_NOT = "Some ... are not"


class Term:
    """Represents a categorical term used in syllogisms"""
    def __init__(self, name: str, plural: str):
        self.name = name
        self.plural = plural


@dataclass
class SyllogismConfig:
    """Configuration for syllogism task generation"""
    
    # Lists of terms to use in syllogisms
    terms: List[Term] = None  # Will be populated with defaults if None
    
    # Control which quantifiers to use
    allow_all: bool = True
    allow_no: bool = True
    allow_some: bool = True
    allow_some_not: bool = True
    
    # Whether to include invalid syllogisms as negative examples
    include_invalid: bool = True
    
    # Percentage of invalid examples if included (0.0 to 1.0)
    invalid_ratio: float = 0.3
    
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert any([self.allow_all, self.allow_no, self.allow_some, self.allow_some_not]), \
            "At least one quantifier type must be allowed"
        assert 0.0 <= self.invalid_ratio <= 1.0, \
            "invalid_ratio must be between 0.0 and 1.0"


class SyllogismDataset(ProceduralDataset):
    """Generates syllogism reasoning tasks"""

    # Default terms if none provided
    DEFAULT_TERMS = [
        Term("mortal", "mortals"),
        Term("human", "humans"),
        Term("animal", "animals"),
        Term("mammal", "mammals"),
        Term("dog", "dogs"),
        Term("cat", "cats"),
        Term("bird", "birds"),
        Term("fish", "fish"),
        Term("plant", "plants"),
        Term("tree", "trees"),
        Term("flower", "flowers"),
        Term("philosopher", "philosophers"),
        Term("student", "students"),
        Term("teacher", "teachers"),
    ]

    def __init__(self, config: SyllogismConfig):
        self.config = config
        if self.config.terms is None:
            self.config.terms = self.DEFAULT_TERMS
        self.config.validate()
        super().__init__(seed=config.seed, size=config.size)

    def _get_allowed_quantifiers(self) -> List[Quantifier]:
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

    def _is_valid_syllogism(self, premise1: Tuple[Quantifier, Term, Term],
                           premise2: Tuple[Quantifier, Term, Term],
                           conclusion: Tuple[Quantifier, Term, Term]) -> bool:
        """
        Check if a syllogism is logically valid.
        This is a simplified implementation - in practice you'd want more complete logic rules.
        """
        # Example rule: If both premises are universal affirmative (ALL),
        # conclusion must also be universal affirmative
        if (premise1[0] == Quantifier.ALL and premise2[0] == Quantifier.ALL):
            return conclusion[0] == Quantifier.ALL
        
        # Example rule: If one premise is negative (NO),
        # conclusion must be negative
        if (premise1[0] == Quantifier.NO or premise2[0] == Quantifier.NO):
            return conclusion[0] == Quantifier.NO
        
        # Add more validity rules here...
        return False

    def _generate_syllogism(self, rng: Random) -> dict:
        """Generate a single syllogism problem"""
        # Select three different terms
        terms = rng.sample(self.config.terms, 3)
        quantifiers = self._get_allowed_quantifiers()
        
        # Generate premises and conclusion
        premise1 = (rng.choice(quantifiers), terms[0], terms[1])
        premise2 = (rng.choice(quantifiers), terms[1], terms[2])
        conclusion = (rng.choice(quantifiers), terms[0], terms[2])

        # Decide if this should be a valid or invalid syllogism
        is_valid = True
        if self.config.include_invalid and rng.random() < self.config.invalid_ratio:
            is_valid = False
            # If should be invalid, regenerate conclusion until invalid
            while self._is_valid_syllogism(premise1, premise2, conclusion):
                conclusion = (rng.choice(quantifiers), terms[0], terms[2])

        # Format the syllogism as text
        premise1_text = f"{premise1[0].value} {premise1[1].plural} are {premise1[2].plural}"
        premise2_text = f"{premise2[0].value} {premise2[1].plural} are {premise2[2].plural}"
        conclusion_text = f"{conclusion[0].value} {conclusion[1].plural} are {conclusion[2].plural}"

        question = (
            f"Consider these statements:\n"
            f"1. {premise1_text}\n"
            f"2. {premise2_text}\n\n"
            f"Does it logically follow that:\n"
            f"{conclusion_text}?"
        )

        return {
            "question": question,
            "answer": "Yes" if is_valid else "No",
            "metadata": {
                "premise1": premise1_text,
                "premise2": premise2_text,
                "conclusion": conclusion_text,
                "is_valid": is_valid,
            }
        }

    def __getitem__(self, idx: int) -> dict:
        """Generate a single syllogism task"""
        rng = Random(self.seed + idx)
        return self._generate_syllogism(rng)


def syllogism_dataset(
    terms: List[Term] = None,
    allow_all: bool = True,
    allow_no: bool = True,
    allow_some: bool = True,
    allow_some_not: bool = True,
    include_invalid: bool = True,
    invalid_ratio: float = 0.3,
    seed: Optional[int] = None,
    size: int = 500,
) -> SyllogismDataset:
    """Create a SyllogismDataset with the given configuration."""
    config = SyllogismConfig(
        terms=terms,
        allow_all=allow_all,
        allow_no=allow_no,
        allow_some=allow_some,
        allow_some_not=allow_some_not,
        include_invalid=include_invalid,
        invalid_ratio=invalid_ratio,
        seed=seed,
        size=size,
    )
    return SyllogismDataset(config)
