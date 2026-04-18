import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "regex_puzzles"

TASK_TYPES = ("string_generation", "extraction", "dfa_state", "dfa_prefix")

REGEX_PATTERNS = [
    (r"[a-c]{2}[0-9]{3}", "two lowercase letters (a-c) followed by three digits"),
    (r"[A-Z]{3}[0-9]{2}", "three uppercase letters followed by two digits"),
    (r"[0-9]{2}-[0-9]{2}-[0-9]{4}", "a date in DD-MM-YYYY format (digits only)"),
    (r"[a-z]+@[a-z]+\.[a-z]{2,3}", "a simple email like name@domain.com"),
    (r"[01]{4}", "a 4-digit binary string"),
    (r"[A-Z][a-z]{2,5}", "a capitalized word (3-6 letters)"),
    (r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", "an IP-address-like pattern"),
    (r"#[0-9a-f]{6}", "a hex color code like #a1b2c3"),
]


@dataclass
class RegexPuzzlesConfig:
    min_dfa_states: int = 3
    max_dfa_states: int = 5
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.3, 0.25, 0.25, 0.2])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_dfa_states >= 2, "min_dfa_states must be >= 2"
        assert self.max_dfa_states >= self.min_dfa_states, "max_dfa_states must be >= min_dfa_states"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


def _gen_matching_string(pattern: str, rng: random.Random) -> str:
    """Generate a string matching a simple regex pattern via character-level generation."""
    import string

    result = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "[":
            end = pattern.index("]", i)
            char_class = pattern[i + 1 : end]
            i = end + 1
            reps = 1
            if i < len(pattern) and pattern[i] == "{":
                end_brace = pattern.index("}", i)
                quant = pattern[i + 1 : end_brace]
                if "," in quant:
                    lo, hi = quant.split(",")
                    reps = rng.randint(int(lo), int(hi))
                else:
                    reps = int(quant)
                i = end_brace + 1
            elif i < len(pattern) and pattern[i] == "+":
                reps = rng.randint(1, 5)
                i += 1

            chars = []
            j = 0
            while j < len(char_class):
                if j + 2 < len(char_class) and char_class[j + 1] == "-":
                    chars.extend(chr(c) for c in range(ord(char_class[j]), ord(char_class[j + 2]) + 1))
                    j += 3
                else:
                    chars.append(char_class[j])
                    j += 1
            for _ in range(reps):
                result.append(rng.choice(chars))
        elif pattern[i] == "\\":
            i += 1
            if pattern[i] == "d":
                result.append(str(rng.randint(0, 9)))
            elif pattern[i] == ".":
                result.append(".")
            elif pattern[i] == "$":
                result.append("$")
            i += 1
        else:
            result.append(pattern[i])
            i += 1
    return "".join(result)


class RegexPuzzlesDataset(ProceduralDataset):
    def __init__(self, config: RegexPuzzlesConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _make_string_generation(self, rng: random.Random) -> dict:
        pattern, desc = rng.choice(REGEX_PATTERNS)
        answer = _gen_matching_string(pattern, rng)
        question = (
            f"Generate a string that matches the regex pattern `{pattern}` "
            f"(i.e., {desc}). Give only the string, nothing else."
        )
        return {"question": question, "answer": answer, "task_type": "string_generation", "pattern": pattern}

    def _make_extraction(self, rng: random.Random) -> dict:
        pattern_str = r"\$\d+\.\d{2}"
        n = rng.randint(2, 4)
        prices = [f"${rng.randint(1, 999)}.{rng.randint(10, 99):02d}" for _ in range(n)]
        words = ["The price is", "costs", "for", "and", "total", "you pay", "item at"]
        text_parts = []
        for p in prices:
            text_parts.append(rng.choice(words))
            text_parts.append(p)
        text_parts.append(rng.choice(["today", "now", "in total"]))
        text = " ".join(text_parts)
        matches = re.findall(pattern_str, text)
        answer = ", ".join(matches)
        question = (
            f"Extract all dollar amounts (matching the pattern $X.XX) from the following text:\n"
            f"'{text}'\n"
            f"List them separated by commas in the order they appear."
        )
        return {"question": question, "answer": answer, "task_type": "extraction"}

    def _make_dfa(self, rng: random.Random) -> tuple[dict, list, str, list, str]:
        n = rng.randint(self.config.min_dfa_states, self.config.max_dfa_states)
        states = [f"q{i}" for i in range(n)]
        alphabet = ["a", "b"]
        transitions = {}
        for s in states:
            for c in alphabet:
                transitions[(s, c)] = rng.choice(states)
        accept = rng.sample(states, rng.randint(1, max(1, n // 2)))
        return transitions, states, states[0], accept, alphabet

    def _run_dfa(self, transitions: dict, start: str, input_str: str) -> str:
        state = start
        for c in input_str:
            state = transitions.get((state, c), state)
        return state

    def _make_dfa_state(self, rng: random.Random) -> dict:
        transitions, states, start, accept, alphabet = self._make_dfa(rng)
        input_len = rng.randint(3, 6)
        input_str = "".join(rng.choice(alphabet) for _ in range(input_len))
        final_state = self._run_dfa(transitions, start, input_str)

        trans_str = ", ".join(f"δ({s},{c})={transitions[(s,c)]}" for s in states for c in alphabet)
        question = (
            f"A DFA has states {{{', '.join(states)}}}, alphabet {{a, b}}, start state {start}.\n"
            f"Transitions: {trans_str}\n"
            f"After processing the input '{input_str}', what state is the DFA in? "
            f"Give only the state name."
        )
        return {"question": question, "answer": final_state, "task_type": "dfa_state"}

    def _make_dfa_prefix(self, rng: random.Random) -> dict:
        transitions, states, start, accept, alphabet = self._make_dfa(rng)
        input_len = rng.randint(4, 8)
        input_str = "".join(rng.choice(alphabet) for _ in range(input_len))

        longest_prefix = ""
        state = start
        for i, c in enumerate(input_str):
            state = transitions.get((state, c), state)
            if state in accept:
                longest_prefix = input_str[: i + 1]

        if not longest_prefix:
            if start in accept:
                longest_prefix = ""
            else:
                longest_prefix = "NONE"

        trans_str = ", ".join(f"δ({s},{c})={transitions[(s,c)]}" for s in states for c in alphabet)
        accept_str = ", ".join(accept)
        question = (
            f"A DFA has states {{{', '.join(states)}}}, alphabet {{a, b}}, "
            f"start state {start}, accept states {{{accept_str}}}.\n"
            f"Transitions: {trans_str}\n"
            f"What is the longest prefix of '{input_str}' that is accepted by this DFA? "
            f"If no prefix is accepted, answer 'NONE'."
        )
        return {"question": question, "answer": longest_prefix, "task_type": "dfa_prefix"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "string_generation": self._make_string_generation,
            "extraction": self._make_extraction,
            "dfa_state": self._make_dfa_state,
            "dfa_prefix": self._make_dfa_prefix,
        }
        result = generators[task_type](rng)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "task_type": result["task_type"],
                "difficulty": {
                    "min_dfa_states": self.config.min_dfa_states,
                    "max_dfa_states": self.config.max_dfa_states,
                },
                **({"pattern": result["pattern"]} if "pattern" in result else {}),
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        oracle = entry["answer"]
        if answer.strip() == oracle.strip():
            return 1.0
        task_type = entry["metadata"]["task_type"]

        if task_type == "string_generation":
            pattern = entry["metadata"]["pattern"]
            try:
                if re.fullmatch(pattern, answer.strip()):
                    return 1.0
            except re.error:
                pass
            return 0.0

        if task_type == "extraction":
            try:
                a_parts = [x.strip() for x in answer.split(",")]
                o_parts = [x.strip() for x in oracle.split(",")]
                if a_parts == o_parts:
                    return 1.0
                return 0.0
            except (ValueError, TypeError):
                return 0.0

        return 0.0


class RegexPuzzlesCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(RegexPuzzlesCurriculum.__name__, RegexPuzzlesConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="max_dfa_states",
                field_name="max_dfa_states",
                levels=[3, 5, 7, 10],
                description="Maximum DFA states",
            ),
        )


register_dataset(DATASET_NAME, RegexPuzzlesDataset, RegexPuzzlesConfig, RegexPuzzlesCurriculum)
