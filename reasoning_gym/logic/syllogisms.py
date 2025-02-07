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
        self.terms = self.DEFAULT_TERMS

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

    @staticmethod
    def _compute_valid_patterns():
        """Compute all valid syllogistic patterns"""
        # The four figures of syllogism based on middle term position
        FIGURES = [
            # Figure 1: M-P, S-M
            ((1,1, 2,2, None), (2,1, None,1), (1,2, None,2)),
            # Figure 2: P-M, S-M
            ((1,2, 2,2, None), (2,1, None,1), (1,1, None,2)),
            # Figure 3: M-P, M-S
            ((1,1, 2,1, None), (None,2, None,1), (1,2, None,2)),
            # Figure 4: P-M, M-S
            ((1,2, 2,1, None), (2,2, None,1), (1,1, None,2))
        ]

        # All possible quantifier combinations
        QUANTIFIERS = ['ALL', 'NO', 'SOME', 'SOME_NOT']
        
        valid_patterns = []
        
        for fig_idx, (middle, subject, predicate) in enumerate(FIGURES, 1):
            for maj in QUANTIFIERS:
                for min in QUANTIFIERS:
                    for conc in QUANTIFIERS:
                        # Apply syllogistic rules
                        # Rule 1: Two negative premises -> invalid
                        if maj in ('NO', 'SOME_NOT') and min in ('NO', 'SOME_NOT'):
                            continue
                            
                        # Rule 2: Two particular premises -> invalid
                        if maj in ('SOME', 'SOME_NOT') and min in ('SOME', 'SOME_NOT'):
                            continue
                            
                        # Rule 3: Universal conclusion needs universal premises
                        if conc in ('ALL', 'NO') and (
                            maj in ('SOME', 'SOME_NOT') or 
                            min in ('SOME', 'SOME_NOT')):
                            continue

                        # Rule 4: Negative conclusion needs negative premise
                        if conc in ('NO', 'SOME_NOT') and not (
                            maj in ('NO', 'SOME_NOT') or 
                            min in ('NO', 'SOME_NOT')):
                            continue

                        # Rule 5: Particular conclusion from universal premises 
                        # is valid but weaker than possible
                        if conc in ('SOME', 'SOME_NOT') and (
                            maj in ('ALL', 'NO') and 
                            min in ('ALL', 'NO')):
                            continue

                        valid_patterns.append(
                            (maj, min, conc, middle, subject, predicate)
                        )
        
        return valid_patterns

    # Valid syllogism patterns computed from rules
    VALID_PATTERNS = _compute_valid_patterns()

    def _is_valid_syllogism(
        self,
        premise1: Tuple[Quantifier, Term, Term],
        premise2: Tuple[Quantifier, Term, Term],
        conclusion: Tuple[Quantifier, Term, Term],
    ) -> bool:
        """Check if a syllogism is logically valid using pattern matching."""
        q1, t1_1, t1_2 = premise1
        q2, t2_1, t2_2 = premise2
        qc, tc_1, tc_2 = conclusion

        # Invalid combinations
        if (q1 in (Quantifier.NO, Quantifier.SOME_NOT) and 
            q2 in (Quantifier.NO, Quantifier.SOME_NOT)):  # Two negative premises
            return False
        if (q1 in (Quantifier.SOME, Quantifier.SOME_NOT) and 
            q2 in (Quantifier.SOME, Quantifier.SOME_NOT)):  # Two particular premises
            return False
        if qc in (Quantifier.ALL, Quantifier.NO) and (
            q1 in (Quantifier.SOME, Quantifier.SOME_NOT) or 
            q2 in (Quantifier.SOME, Quantifier.SOME_NOT)):  # Universal conclusion with particular premise
            return False

        terms = ((t1_1, t1_2), (t2_1, t2_2), (tc_1, tc_2))
        quants = (q1.value, q2.value, qc.value)

        # Check against valid patterns
        for pattern_q1, pattern_q2, pattern_qc, middle, subject, predicate in self.VALID_PATTERNS:
            if (pattern_q1, pattern_q2, pattern_qc) != quants:
                continue

            # Get terms according to pattern positions
            def get_term(pos):
                if pos is None:
                    return None
                premise_idx, term_idx = pos
                return terms[premise_idx-1][term_idx-1]

            # Check term positions match pattern
            middle_terms = [get_term(p) for p in middle[:2] if p is not None]
            if len(middle_terms) >= 2 and len(set(middle_terms)) != 1:
                continue

            subject_terms = [get_term(p) for p in subject[:2] if p is not None]
            if len(subject_terms) >= 2 and len(set(subject_terms)) != 1:
                continue

            predicate_terms = [get_term(p) for p in predicate[:2] if p is not None]
            if len(predicate_terms) >= 2 and len(set(predicate_terms)) != 1:
                continue

            # Verify the terms are in correct positions
            if middle_terms and subject_terms and predicate_terms:
                return True

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

        target_valid = rng.random() > self.config.invalid_ratio  # Invert ratio to match meaning
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            # Generate premises and conclusion
            premise1 = (rng.choice(quantifiers), terms[0], terms[1])
            premise2 = (rng.choice(quantifiers), terms[1], terms[2])
            conclusion = (rng.choice(quantifiers), terms[0], terms[2])

            # Check if validity matches target
            is_valid = self._is_valid_syllogism(premise1, premise2, conclusion)
            print(attempts, is_valid, target_valid, terms, premise1, premise2)
            if is_valid == target_valid:
                break
                
            attempts += 1
        
        if attempts >= max_attempts:
            # If we couldn't find a matching syllogism, return a basic valid one
            premise1 = (Quantifier.ALL, terms[0], terms[1])
            premise2 = (Quantifier.ALL, terms[1], terms[2]) 
            conclusion = (Quantifier.ALL, terms[0], terms[2])
            is_valid = True

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
