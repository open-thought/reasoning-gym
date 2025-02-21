"""Syllogism reasoning task generator"""

from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import Optional

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
    def _is_valid_syllogism(
        premise1: tuple[Quantifier, "Term", "Term"],
        premise2: tuple[Quantifier, "Term", "Term"],
        conclusion: tuple[Quantifier, "Term", "Term"],
    ) -> bool:
        """
        Checks whether a given syllogism is valid under classical (Aristotelian) rules,
        including the distribution rule:
        - If a term is distributed in the conclusion, it must be distributed
          in the premise where it appears as subject/predicate.
        """

        # --- 1) Extract data ---
        q1, p1_subj, p1_pred = premise1
        q2, p2_subj, p2_pred = premise2
        q3, c_subj, c_pred = conclusion

        negative_set = {Quantifier.NO, Quantifier.SOME_NOT}
        particular_set = {Quantifier.SOME, Quantifier.SOME_NOT}
        universal_set = {Quantifier.ALL, Quantifier.NO}

        # --- 2) Identify a unique middle term ---
        premise1_terms = {p1_subj, p1_pred}
        premise2_terms = {p2_subj, p2_pred}
        common_terms = premise1_terms.intersection(premise2_terms)

        if len(common_terms) != 1:
            return False
        middle_term = next(iter(common_terms))

        # Gather all terms => must be exactly 3 distinct terms
        all_terms = premise1_terms.union(premise2_terms)
        if len(all_terms) != 3:
            return False

        # The conclusion must use the other two terms (not the middle)
        other_two = all_terms - {middle_term}
        conclusion_terms = {c_subj, c_pred}
        if conclusion_terms != other_two:
            return False

        # --- 3) Identify which premise is major vs. minor ---
        def premise_contains(premise, term):
            return (premise[1] == term) or (premise[2] == term)

        if premise_contains(premise1, c_pred):
            major = premise1
            minor = premise2
        elif premise_contains(premise2, c_pred):
            major = premise2
            minor = premise1
        else:
            return False

        # The minor premise must contain the conclusion's subject
        if not premise_contains(minor, c_subj):
            return False

        # --- 4) Quick checks (traditional “no two negative,” etc.) ---
        if (q1 in negative_set) and (q2 in negative_set):
            return False
        if (q1 in particular_set) and (q2 in particular_set):
            return False
        if q3 in universal_set:
            if (q1 in particular_set) or (q2 in particular_set):
                return False
        if q3 in negative_set:
            if not ((q1 in negative_set) or (q2 in negative_set)):
                return False

        # --- 5) Distribution checks ---
        def distribution(q: Quantifier):
            if q == Quantifier.ALL:  # A
                return (True, False)
            elif q == Quantifier.NO:  # E
                return (True, True)
            elif q == Quantifier.SOME:  # I
                return (False, False)
            elif q == Quantifier.SOME_NOT:  # O
                return (False, True)
            else:
                raise ValueError(f"Unknown quantifier: {q}")

        # Conclusion distribution
        dist_c_subj, dist_c_pred = distribution(q3)

        # Major premise distribution
        q_major, major_subj, major_pred = major
        dist_major_subj, dist_major_pred = distribution(q_major)

        # Minor premise distribution
        q_minor, minor_subj, minor_pred = minor
        dist_minor_subj, dist_minor_pred = distribution(q_minor)

        # If the conclusion's subject is distributed, check it in the minor premise
        if dist_c_subj:
            if c_subj == minor_subj:
                if not dist_minor_subj:
                    return False
            elif c_subj == minor_pred:
                if not dist_minor_pred:
                    return False

        # If the conclusion's predicate is distributed, check it in the major premise
        if dist_c_pred:
            if c_pred == major_subj:
                if not dist_major_subj:
                    return False
            elif c_pred == major_pred:
                if not dist_major_pred:
                    return False

        # If either premise is negative, the conclusion must be negative.
        if (q1 in negative_set) or (q2 in negative_set):
            if q3 not in negative_set:
                return False

        # If all checks pass, it's valid
        return True

    def _format_quantifier_statement(self, quantifier: Quantifier, subject: Term, predicate: Term) -> str:
        """Format a quantified statement in natural language"""
        if quantifier == Quantifier.SOME_NOT:
            return f"Some {subject.plural} are not {predicate.plural}"
        else:
            return f"{quantifier.value} {subject.plural} are {predicate.plural}"

    def _check_logical_equivalence(
        self, premise: tuple[Quantifier, Term, Term], conclusion: tuple[Quantifier, Term, Term]
    ) -> bool:
        """Check if a conclusion is logically equivalent to a premise"""
        p_quant, p_subj, p_pred = premise
        c_quant, c_subj, c_pred = conclusion

        # Direct inversion for universal negative
        if p_quant == Quantifier.NO:
            if c_quant == Quantifier.NO:
                return p_subj == c_pred and p_pred == c_subj
            return False

        # Particular inversion for universal affirmative
        if p_quant == Quantifier.ALL:
            if c_quant == Quantifier.SOME:
                return p_subj == c_pred and p_pred == c_subj
            return False

        # Rules for particular statements
        if p_quant == Quantifier.SOME:
            if c_quant == Quantifier.SOME:
                return p_subj == c_pred and p_pred == c_subj
            return False

        if p_quant == Quantifier.SOME_NOT:
            # Some A are not B does not imply Some B are not A
            return False

        return False

    def _generate_syllogism(self, rng: Random) -> dict:
        """Generate a single syllogism problem"""
        # Select three different terms
        terms = rng.sample(self.terms, 3)
        quantifiers = self._get_allowed_quantifiers()

        # Decide whether to generate a traditional syllogism or an inversion problem
        if rng.random() < self.config.inversion_probability:
            # Generate two premises, one will be used for inversion, the other as distractor
            quantifier1 = rng.choice(quantifiers)
            quantifier2 = rng.choice(quantifiers)
            term1, term2, term3 = terms  # Use all three terms

            # Create two different premises
            premise1 = (quantifier1, term1, term2)
            premise2 = (quantifier2, term2, term3)

            # Format both premises
            premise1_text = self._format_quantifier_statement(premise1[0], premise1[1], premise1[2])
            premise2_text = self._format_quantifier_statement(premise2[0], premise2[1], premise2[2])

            # Randomly select which premise to use for inversion
            if rng.random() < 0.5:
                premise = premise1
                selected_premise_num = 1
            else:
                premise = premise2
                selected_premise_num = 2

            # Decide whether to generate a valid or invalid inversion
            target_valid = rng.random() > self.config.invalid_ratio

            # Get the quantifier and terms from the selected premise
            premise_quantifier, premise_term1, premise_term2 = premise

            if target_valid:
                # Generate valid inversions
                if premise_quantifier == Quantifier.NO:
                    conclusion = (premise_quantifier, premise_term2, premise_term1)  # No B are A
                elif premise_quantifier == Quantifier.ALL:
                    conclusion = (Quantifier.SOME, premise_term2, premise_term1)  # Some B are A
                elif premise_quantifier == Quantifier.SOME:
                    conclusion = (premise_quantifier, premise_term2, premise_term1)  # Some B are A
                else:  # SOME_NOT - try a different quantifier
                    new_quantifier = rng.choice([q for q in quantifiers if q != Quantifier.SOME_NOT])
                    # Update the premise with the new quantifier
                    premise = (new_quantifier, premise_term1, premise_term2)
                    premise_quantifier = new_quantifier  # Update the quantifier for conclusion generation
                    if selected_premise_num == 1:
                        premise1 = premise
                        premise1_text = self._format_quantifier_statement(premise[0], premise[1], premise[2])
                    else:
                        premise2 = premise
                        premise2_text = self._format_quantifier_statement(premise[0], premise[1], premise[2])

                    # Handle the new quantifier
                    if new_quantifier == Quantifier.NO:
                        conclusion = (new_quantifier, premise_term2, premise_term1)
                    elif new_quantifier == Quantifier.ALL:
                        conclusion = (Quantifier.SOME, premise_term2, premise_term1)
                    else:  # SOME
                        conclusion = (new_quantifier, premise_term2, premise_term1)
            else:
                # Generate invalid inversions by sampling from inappropriate quantifiers
                if premise_quantifier == Quantifier.NO:
                    # For NO statements, use ALL or SOME
                    conclusion = (rng.choice([Quantifier.ALL, Quantifier.SOME]), premise_term2, premise_term1)
                elif premise_quantifier == Quantifier.ALL:
                    # For ALL statements, use ALL or NO
                    conclusion = (rng.choice([Quantifier.ALL, Quantifier.NO]), premise_term2, premise_term1)
                elif premise_quantifier == Quantifier.SOME:
                    # For SOME statements, use ALL or NO
                    conclusion = (rng.choice([Quantifier.ALL, Quantifier.NO]), premise_term2, premise_term1)
                else:  # SOME_NOT
                    # For SOME_NOT statements, use any other quantifier
                    conclusion = (
                        rng.choice([q for q in quantifiers if q != Quantifier.SOME_NOT]),
                        premise_term2,
                        premise_term1,
                    )

            conclusion_text = self._format_quantifier_statement(conclusion[0], conclusion[1], conclusion[2])
            is_valid = self._check_logical_equivalence(premise, conclusion)

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
                    "selected_premise": selected_premise_num,
                    "conclusion": conclusion_text,
                    "is_valid": is_valid,
                    "type": "inversion",
                },
            }

        # Traditional syllogism generation
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
                "type": "syllogism",
            },
        }

    def __getitem__(self, idx: int) -> dict:
        """Generate a single syllogism task"""
        rng = Random(self.seed + idx)
        return self._generate_syllogism(rng)


register_dataset("syllogism", SyllogismDataset, SyllogismConfig)
