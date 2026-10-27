"""
Regex construction puzzles: find a string that fully matches several regular expressions
at once, while not matching a few others.

Each puzzle is built backwards from a hidden string. Every required pattern is a loosened
description of that string (characters widened into classes, runs turned into repeats, pieces
swapped for wildcards or alternations), so a solution always exists. Forbidden patterns are
built around near misses of the hidden string.
"""

import random
import re
from dataclasses import dataclass
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "regex_construction"

# Answers longer than this are rejected before any pattern is run on them.
MAX_ANSWER_LENGTH = 128

# At most this many * or + per pattern. Patterns never nest quantifiers, so together with the
# answer length cap this keeps matching cheap no matter what string we are asked to score.
MAX_UNBOUNDED = 2

# How often each kind of fragment is used when writing a pattern. Required patterns lean loose
# so no single one gives the answer away; forbidden ones lean literal so they rule out something
# specific.
LOOSE_WEIGHTS = {"literal": 2, "classes": 3, "repeat": 3, "wildcard": 2, "choice": 2}
TIGHT_WEIGHTS = {"literal": 6, "classes": 2, "repeat": 1, "wildcard": 1, "choice": 1}

RULES_TEMPLATE = """The string may only contain the characters {alphabet}.
"Fully matches" means the whole string has to match the pattern, not just part of it, as with Python's re.fullmatch.
If more than one string works, any of them is accepted.

Respond with only the string."""


@dataclass
class RegexConstructionConfig:
    """Configuration for regex construction puzzles"""

    alphabet: str = "abcdef012345"
    min_length: int = 5
    max_length: int = 10
    min_patterns: int = 2  # patterns the string has to match
    max_patterns: int = 3
    min_forbidden: int = 0  # patterns the string must not match
    max_forbidden: int = 1
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert len(self.alphabet) >= 3, "alphabet needs at least 3 characters"
        assert len(set(self.alphabet)) == len(self.alphabet), "alphabet must not contain duplicates"
        assert self.alphabet.isascii() and self.alphabet.isalnum(), "alphabet may only contain letters and digits"
        assert 2 <= self.min_length <= self.max_length, "need 2 <= min_length <= max_length"
        assert self.max_length <= MAX_ANSWER_LENGTH // 2, f"max_length must be <= {MAX_ANSWER_LENGTH // 2}"
        assert 1 <= self.min_patterns <= self.max_patterns, "need 1 <= min_patterns <= max_patterns"
        assert 0 <= self.min_forbidden <= self.max_forbidden, "need 0 <= min_forbidden <= max_forbidden"


class RegexConstructionDataset(ProceduralDataset):
    """Generates puzzles where a string has to satisfy a set of regex constraints"""

    def __init__(self, config: RegexConstructionConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(self.seed + idx)
        cfg = self.config

        length = rng.randint(cfg.min_length, cfg.max_length)
        target = self._make_target(rng, length)

        patterns = []
        for _ in range(rng.randint(cfg.min_patterns, cfg.max_patterns)):
            patterns.append(self._required_pattern(rng, target, patterns))

        forbidden = []
        for _ in range(rng.randint(cfg.min_forbidden, cfg.max_forbidden)):
            forbidden.append(self._forbidden_pattern(rng, target, patterns, forbidden))

        sections = ["Find a string that fully matches every one of these regular expressions:\n" + _bullets(patterns)]
        if forbidden:
            sections.append("It must not fully match any of these:\n" + _bullets(forbidden))
        sections.append(RULES_TEMPLATE.format(alphabet=", ".join(cfg.alphabet)))

        return {
            "question": "\n\n".join(sections),
            "answer": target,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "alphabet": cfg.alphabet,
                "patterns": patterns,
                "forbidden": forbidden,
                "length": length,
                "difficulty": {
                    "length": (cfg.min_length, cfg.max_length),
                    "num_patterns": (cfg.min_patterns, cfg.max_patterns),
                    "num_forbidden": (cfg.min_forbidden, cfg.max_forbidden),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Any string that satisfies all the constraints gets full credit, anything else gets 0."""
        if not isinstance(answer, str):
            return 0.0
        answer = answer.strip()
        # models like to wrap the string in quotes or backticks
        if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in "\"'`":
            answer = answer[1:-1]
        if not answer or len(answer) > MAX_ANSWER_LENGTH:
            return 0.0

        metadata = entry["metadata"]
        if any(c not in metadata["alphabet"] for c in answer):
            return 0.0
        if not all(re.fullmatch(p, answer) for p in metadata["patterns"]):
            return 0.0
        if any(re.fullmatch(p, answer) for p in metadata["forbidden"]):
            return 0.0
        return 1.0

    def _make_target(self, rng: random.Random, length: int) -> str:
        # built from short runs so that repeat counts like a{3} come up naturally
        s = ""
        while len(s) < length:
            s += rng.choice(self.config.alphabet) * rng.choice([1, 1, 1, 2, 3])
        return s[:length]

    def _required_pattern(self, rng: random.Random, target: str, existing: list[str]) -> str:
        for _ in range(10):
            p = self._pattern_for(rng, target, LOOSE_WEIGHTS)
            if p != target and p not in existing:
                break
        return p

    def _forbidden_pattern(self, rng: random.Random, target: str, patterns: list[str], existing: list[str]) -> str:
        for _ in range(10):
            # Near misses that still satisfy the required patterns are preferred, since those
            # are the strings a solver could actually land on.
            near = [self._mutate(rng, target) for _ in range(20)]
            tempting = [s for s in near if all(re.fullmatch(p, s) for p in patterns)]
            base = rng.choice(tempting or near)
            p = self._pattern_for(rng, base, TIGHT_WEIGHTS)
            if p not in existing and not re.fullmatch(p, target):
                return p
        # a near miss is never equal to the target, so the bare string is always a valid fallback
        return base

    def _mutate(self, rng: random.Random, s: str) -> str:
        """Return a string that differs from s by one replaced, inserted or deleted character"""
        chars = list(s)
        i = rng.randrange(len(chars))
        op = rng.choice(["replace", "replace", "insert", "delete"] if len(chars) > 1 else ["replace", "insert"])
        if op == "replace":
            chars[i] = rng.choice([c for c in self.config.alphabet if c != chars[i]])
        elif op == "insert":
            chars.insert(i, rng.choice(self.config.alphabet))
        else:
            del chars[i]
        return "".join(chars)

    def _pattern_for(self, rng: random.Random, s: str, weights: dict[str, int]) -> str:
        """Split s into chunks and write each chunk as a regex fragment that matches it"""
        kinds = list(weights)
        parts = []
        unbounded = 0
        prev_kind = None
        i = 0
        while i < len(s):
            chunk = s[i : i + rng.randint(1, 4)]
            i += len(chunk)
            kind = rng.choices(kinds, weights=[weights[k] for k in kinds])[0]
            if kind == prev_kind == "wildcard":
                kind = "classes"  # avoid things like .{2}.* next to each other
            prev_kind = kind

            if kind == "literal":
                parts.append(chunk)
            elif kind == "classes":
                parts.append("".join(self._char_class(self._widen(rng, c)) for c in chunk))
            elif kind == "repeat":
                q = self._quantifier(rng, len(chunk), allow_open=unbounded < MAX_UNBOUNDED)
                parts.append(self._char_class(self._widen(rng, chunk)) + q)
            elif kind == "wildcard":
                q = self._quantifier(rng, len(chunk), allow_open=unbounded < MAX_UNBOUNDED)
                parts.append("." + q)
            else:
                options = [chunk, self._mutate(rng, chunk)]
                rng.shuffle(options)
                parts.append(f"({options[0]}|{options[1]})")

            if kind in ("repeat", "wildcard") and parts[-1][-1] in "*+":
                unbounded += 1

        return "".join(parts)

    def _widen(self, rng: random.Random, chars: str) -> list[str]:
        """The distinct characters of chars plus one or two others from the alphabet, sorted"""
        picked = set(chars)
        others = [c for c in self.config.alphabet if c not in picked]
        picked.update(rng.sample(others, min(len(others), rng.randint(1, 2))))
        return sorted(picked)

    def _char_class(self, chars: list[str]) -> str:
        if len(chars) == len(self.config.alphabet):
            return "."
        if len(chars) == 1:
            return chars[0]
        # collapse runs of consecutive characters, e.g. a b c d -> a-d
        runs = []
        for c in chars:
            if runs and ord(c) == ord(runs[-1][-1]) + 1:
                runs[-1] += c
            else:
                runs.append(c)
        return "[" + "".join(r[0] + "-" + r[-1] if len(r) >= 3 else r for r in runs) + "]"

    def _quantifier(self, rng: random.Random, n: int, allow_open: bool) -> str:
        """A quantifier that accepts n repetitions: exactly, within a small range, or open ended"""
        style = rng.choice(["exact", "range", "open"] if allow_open else ["exact", "range"])
        if style == "open":
            return rng.choice(["+", "*"])
        lo, hi = n, n
        if style == "range":
            lo = rng.randint(max(0, n - 2), n)
            hi = rng.randint(n, n + 2)
        if lo == hi:
            return "" if n == 1 else f"{{{n}}}"
        if (lo, hi) == (0, 1):
            return "?"
        return f"{{{lo},{hi}}}"


def _bullets(patterns: list[str]) -> str:
    return "\n".join(f"- {p}" for p in patterns)


class RegexConstructionCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(RegexConstructionCurriculum.__name__, RegexConstructionConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="length",
                levels=[5, 10, 15, 20, 30],
                description="Length of the hidden string",
                lower_field_name="min_length",
                upper_field_name="max_length",
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="num_patterns",
                levels=[1, 2, 3, 4],
                description="Number of patterns the string has to match",
                lower_field_name="min_patterns",
                upper_field_name="max_patterns",
            ),
            RangeAttributeDefinition(
                name="num_forbidden",
                levels=[0, 1, 2, 3],
                description="Number of patterns the string must not match",
                lower_field_name="min_forbidden",
                upper_field_name="max_forbidden",
            ),
        )


register_dataset(DATASET_NAME, RegexConstructionDataset, RegexConstructionConfig, RegexConstructionCurriculum)
