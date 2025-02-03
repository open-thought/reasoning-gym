"""Syllogism reasoning task generator"""

from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import List, Optional, Tuple

from ..factory import ProceduralDataset, register_dataset


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

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert any(
            [self.allow_all, self.allow_no, self.allow_some, self.allow_some_not]
        ), "At least one quantifier type must be allowed"
        assert 0.0 <= self.invalid_ratio <= 1.0, "invalid_ratio must be between 0.0 and 1.0"


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
        self.terms = self.DEFAULT_TERMS if config.terms is None else config.terms

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

    def _is_valid_syllogism(
        self,
        premise1: Tuple[Quantifier, Term, Term],
        premise2: Tuple[Quantifier, Term, Term],
        conclusion: Tuple[Quantifier, Term, Term],
    ) -> bool:
        """
        Check if a syllogism is logically valid using classical logic rules.

        Rules implemented:
        1. Universal Affirmative (ALL):
           - If both premises are ALL, conclusion must be ALL
           - ALL A are B + ALL B are C → ALL A are C (Barbara)

        2. Universal Negative (NO):
           - If one premise is NO and other is ALL, conclusion must be NO
           - NO A are B + ALL C are B → NO A are C (Celarent)
           - ALL A are B + NO C are B → NO A are C (Cesare)

        3. Particular Affirmative (SOME):
           - If one premise is SOME and other is ALL, conclusion must be SOME
           - SOME A are B + ALL B are C → SOME A are C (Darii)
           - ALL A are B + SOME C are B → SOME A are C (Disamis)

        4. Particular Negative (SOME_NOT):
           - If one premise is SOME_NOT and other is ALL, conclusion can be SOME_NOT
           - SOME A are not B + ALL B are C → SOME A are not C (Ferio)
           - ALL A are B + SOME C are not B → SOME A are not C (Festino)

        5. Invalid combinations:
           - Two negative premises never yield a valid conclusion
           - Two particular premises never yield a valid conclusion
           - If both premises are particular, no valid conclusion
           - If conclusion is universal but either premise is particular, invalid
        """
        q1, t1_1, t1_2 = premise1
        q2, t2_1, t2_2 = premise2
        qc, tc_1, tc_2 = conclusion

        # Rule 5: Two negative premises -> invalid
        if q1 in (Quantifier.NO, Quantifier.SOME_NOT) and q2 in (Quantifier.NO, Quantifier.SOME_NOT):
            return False

        # Rule 5: Two particular premises -> invalid
        if q1 in (Quantifier.SOME, Quantifier.SOME_NOT) and q2 in (Quantifier.SOME, Quantifier.SOME_NOT):
            return False

        # Rule 5: Universal conclusion with particular premise -> invalid
        if qc in (Quantifier.ALL, Quantifier.NO) and (
            q1 in (Quantifier.SOME, Quantifier.SOME_NOT) or q2 in (Quantifier.SOME, Quantifier.SOME_NOT)
        ):
            return False

        # Rule 1: Barbara syllogism
        if q1 == Quantifier.ALL and q2 == Quantifier.ALL:
            if t1_2 == t2_1 and tc_1 == t1_1 and tc_2 == t2_2:
                return qc == Quantifier.ALL

        # Rule 2: Celarent syllogism
        if q1 == Quantifier.NO and q2 == Quantifier.ALL:
            if t1_2 == t2_1 and tc_1 == t1_1 and tc_2 == t2_2:
                return qc == Quantifier.NO

        # Rule 2: Cesare syllogism
        if q1 == Quantifier.ALL and q2 == Quantifier.NO:
            if t1_2 == t2_1 and tc_1 == t1_1 and tc_2 == t2_2:
                return qc == Quantifier.NO

        # Rule 3: Darii syllogism
        if q1 == Quantifier.SOME and q2 == Quantifier.ALL:
            if t1_2 == t2_1 and tc_1 == t1_1 and tc_2 == t2_2:
                return qc == Quantifier.SOME

        # Rule 3: Disamis syllogism
        if q1 == Quantifier.ALL and q2 == Quantifier.SOME:
            if t1_2 == t2_1 and tc_1 == t1_1 and tc_2 == t2_2:
                return qc == Quantifier.SOME

        # Rule 4: Ferio syllogism
        if q1 == Quantifier.SOME_NOT and q2 == Quantifier.ALL:
            if t1_2 == t2_1 and tc_1 == t1_1 and tc_2 == t2_2:
                return qc == Quantifier.SOME_NOT

        # Rule 4: Festino syllogism
        if q1 == Quantifier.ALL and q2 == Quantifier.SOME_NOT:
            if t1_2 == t2_1 and tc_1 == t1_1 and tc_2 == t2_2:
                return qc == Quantifier.SOME_NOT

        return False

    def _format_quantifier_statement(self, quantifier: Quantifier, subject: Term, predicate: Term) -> str:
        """Format a quantified statement in natural language"""
        if quantifier == Quantifier.SOME_NOT:
            return f"Some {subject.plural} are not {predicate.plural}"
        else:
            return f"{quantifier.value} {subject.plural} are {predicate.plural}"

    def _generate_syllogism(self, rng: Random) -> dict:
        """Generate a single syllogism problem"""
        # Select three different terms
        terms = rng.sample(self.terms, 3)
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
        premise1_text = self._format_quantifier_statement(premise1[0], premise1[1], premise1[2])
        premise2_text = self._format_quantifier_statement(premise2[0], premise2[1], premise2[2])
        conclusion_text = self._format_quantifier_statement(conclusion[0], conclusion[1], conclusion[2])

        question = (
            f"Consider these statements:\n"
            f"1. {premise1_text}\n"
            f"2. {premise2_text}\n\n"
            f"Does it logically follow that:\n"
            f"{conclusion_text}?\n"
            f"(Answer Yes or No)"
        )

        return {
            "question": question,
            "answer": "Yes" if is_valid else "No",
            "metadata": {
                "premise1": premise1_text,
                "premise2": premise2_text,
                "conclusion": conclusion_text,
                "is_valid": is_valid,
            },
        }

    def __getitem__(self, idx: int) -> dict:
        """Generate a single syllogism task"""
        rng = Random(self.seed + idx)
        return self._generate_syllogism(rng)


register_dataset("syllogism", SyllogismDataset, SyllogismConfig)
