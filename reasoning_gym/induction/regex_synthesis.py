"""Regex synthesis: infer a regular expression from positive and negative example strings.

A hidden target regex is generated from a small grammar over a short alphabet. The model sees strings
that the target matches and strings it does not match, and must write a regex that separates them.
Answers are checked on the shown examples and on additional hidden strings, so listing the shown
positives (overfitting) does not earn reward. Many different regexes are correct.

Model answers are run with ``safe_regex`` (a linear-time NFA matcher) rather than Python's ``re``,
so a pathological answer such as ``(a+)+b`` cannot hang the scorer.
"""

import json
import re
import string
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset
from .safe_regex import MAX_PATTERN_LENGTH, MAX_REPEAT, UnsupportedRegexError, compile_safe_regex

DATASET_NAME = "regex_synthesis"

QUESTION_TEMPLATE = """Find a regular expression that matches every string in the "Match" list and none of the strings in the "Do not match" list. All strings use only the characters {alphabet}. "" denotes the empty string.

Match:
{positives}

Do not match:
{negatives}

Rules:
- The regex must match the whole string (like Python's re.fullmatch), so ^ and $ are optional.
- Your regex will also be checked on other strings of the same kind that are not shown here, so describe the general pattern instead of listing the examples.
- Allowed syntax: literal characters, ., character classes such as [ab], [a-c] or [^a], groups (...) and (?:...), alternation |, the quantifiers *, +, ?, {{m}}, {{m,}} and {{m,n}} (with m, n <= {max_repeat}), and the escapes \\d, \\w and \\s. Back-references, lookarounds and other advanced features are not allowed.
- The regex must be at most {max_length} characters long.

Answer with the regular expression only, for example: ab*(c|d)
"""

# Generator AST nodes (tuples):
#   ("lit", ch) | ("cls", chars, negated) | ("words", [word, ...]) | ("group", word)
#   ("quant", node, lo, hi_or_None)


def _render_class(chars: tuple, negated: bool) -> str:
    """Render a character class, compressing runs of 3+ consecutive letters into ranges."""
    chars = sorted(chars)
    parts = []
    i = 0
    while i < len(chars):
        j = i
        while j + 1 < len(chars) and ord(chars[j + 1]) == ord(chars[j]) + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{chars[i]}-{chars[j]}")
        else:
            parts.extend(chars[i : j + 1])
        i = j + 1
    return "[" + ("^" if negated else "") + "".join(parts) + "]"


def _render(node) -> str:
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "cls":
        return _render_class(node[1], node[2])
    if kind == "words":
        return "(" + "|".join(node[1]) + ")"
    if kind == "group":
        return "(" + node[1] + ")"
    if kind == "quant":
        _, inner, lo, hi = node
        if (lo, hi) == (0, None):
            suffix = "*"
        elif (lo, hi) == (1, None):
            suffix = "+"
        elif (lo, hi) == (0, 1):
            suffix = "?"
        elif hi is None:
            suffix = f"{{{lo},}}"
        elif lo == hi:
            suffix = f"{{{lo}}}"
        else:
            suffix = f"{{{lo},{hi}}}"
        return _render(inner) + suffix
    raise AssertionError(f"unknown node {kind}")


def _sample(rng: Random, node, alphabet: str, path: tuple = (), features: Optional[set] = None) -> str:
    """Sample one string matched by ``node``.

    If ``features`` is given, the choices made while sampling are recorded in it (which character of a class,
    which alternative, and how many repetitions), keyed by the position of the node in the AST. This lets the
    generator pick shown examples that together demonstrate every part of the target regex.
    """
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "cls":
        _, chars, negated = node
        pool = [c for c in alphabet if c not in chars] if negated else list(chars)
        ch = rng.choice(pool)
        if features is not None:
            features.add((path, "char", ch))
        return ch
    if kind == "words":
        word = rng.choice(node[1])
        if features is not None:
            features.add((path, "word", word))
        return word
    if kind == "group":
        return node[1]
    if kind == "quant":
        _, inner, lo, hi = node
        count = rng.randint(lo, lo + 3 if hi is None else hi)
        if features is not None:
            # for unbounded quantifiers, 0 / 1 / "2 or more" are the interesting cases
            features.add((path, "count", min(count, 2) if hi is None else count))
        return "".join(_sample(rng, inner, alphabet, path + (0,), features) for _ in range(count))
    raise AssertionError(f"unknown node {kind}")


def _normalize_answer(answer: str) -> Optional[str]:
    """Strip whitespace and common wrappers (code fences, backticks, /.../) from a model answer."""
    s = answer.strip()
    if len(s) > 4 * MAX_PATTERN_LENGTH:
        return None
    if s.startswith("```") and s.endswith("```") and len(s) >= 6:
        s = s[3:-3]
        first_line, _, rest = s.partition("\n")
        if rest and re.fullmatch(r"[A-Za-z]*\s*", first_line):  # drop a language tag such as ```regex
            s = rest
        s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] == "`":
        s = s[1:-1].strip()
    if len(s) >= 2 and s[0] == s[-1] == "/":
        s = s[1:-1]
    return s


@dataclass
class RegexSynthesisConfig:
    """Configuration for regex synthesis task generation"""

    min_components: int = 2  # Minimum number of concatenated parts in the target regex
    max_components: int = 4  # Maximum number of concatenated parts in the target regex
    alphabet_size: int = 3  # Strings use the first `alphabet_size` lowercase letters
    operator_level: int = 2  # 1: literals, classes, ? * +   2: adds alternation   3: adds groups, {m,n}, [^...]
    num_examples: int = 8  # Number of shown positive and of shown negative examples (each)
    num_hidden: int = 10  # Number of hidden positive and of hidden negative test strings (each)
    max_string_length: int = 16  # Maximum length of any example string

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_components <= self.max_components <= 12, "need 1 <= min_components <= max_components <= 12"
        assert 2 <= self.alphabet_size <= 26, "alphabet_size must be between 2 and 26"
        assert self.operator_level in (1, 2, 3), "operator_level must be 1, 2 or 3"
        assert self.num_examples >= 1, "num_examples must be at least 1"
        assert self.num_hidden >= 1, "num_hidden must be at least 1"
        assert self.max_string_length >= 4, "max_string_length must be at least 4"


class RegexSynthesisDataset(ProceduralDataset):
    """Generates regex synthesis (inductive programming) tasks"""

    MAX_ATTEMPTS = 500

    def __init__(self, config: RegexSynthesisConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    # ---------------------------------------------------------------- generation

    def _random_word(self, rng: Random, alphabet: str, min_len: int, max_len: int) -> str:
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))

    def _random_class(self, rng: Random, alphabet: str, allow_negated: bool):
        if allow_negated and rng.random() < 0.25:
            k = rng.randint(1, max(1, min(2, len(alphabet) - 2)))
            return ("cls", tuple(sorted(rng.sample(alphabet, k))), True)
        k = rng.randint(2, max(2, min(3, len(alphabet) - 1)))
        return ("cls", tuple(sorted(rng.sample(alphabet, k))), False)

    def _component(self, rng: Random, alphabet: str, level: int):
        kinds = ["lit", "lit", "cls", "cls"]
        if level >= 2:
            kinds += ["words", "words"]
        if level >= 3:
            kinds += ["group", "bounded"]
        kind = rng.choice(kinds)

        if kind in ("lit", "cls"):
            atom = ("lit", rng.choice(alphabet)) if kind == "lit" else self._random_class(rng, alphabet, level >= 3)
            if rng.random() < 0.5:
                lo, hi = rng.choice([(0, 1), (0, None), (1, None)])
                return ("quant", atom, lo, hi)
            return atom
        if kind == "words":
            words = set()
            target = rng.randint(2, 3)
            for _ in range(100):  # bounded: a small alphabet may not allow `target` prefix-free words
                if len(words) == target:
                    break
                word = self._random_word(rng, alphabet, 1, 3)
                # keep alternatives prefix-free so they are easier to tell apart from examples
                if not any(w.startswith(word) or word.startswith(w) for w in words):
                    words.add(word)
            if len(words) < 2:
                words.add(rng.choice(alphabet) + rng.choice(alphabet) + rng.choice(alphabet))
            node = ("words", sorted(words))
            if level >= 3 and rng.random() < 0.4:
                lo, hi = rng.choice([(0, 1), (0, None), (1, None)])
                return ("quant", node, lo, hi)
            if rng.random() < 0.25:
                return ("quant", node, 0, 1)
            return node
        if kind == "group":
            lo, hi = rng.choice([(0, None), (1, None), (0, 1)])
            return ("quant", ("group", self._random_word(rng, alphabet, 2, 3)), lo, hi)
        # bounded repetition of a literal or class
        atom = ("lit", rng.choice(alphabet)) if rng.random() < 0.5 else self._random_class(rng, alphabet, False)
        lo = rng.randint(1, 3)
        hi = rng.choice([lo, lo + 1, lo + 2]) if lo > 1 else rng.choice([2, 3])  # never render a plain {1}
        return ("quant", atom, lo, hi)

    def _mutate(self, rng: Random, s: str, alphabet: str) -> str:
        chars = list(s)
        for _ in range(rng.randint(1, 2)):
            op = rng.choice(["substitute", "insert", "delete", "swap"])
            if op == "insert" or not chars:
                chars.insert(rng.randint(0, len(chars)), rng.choice(alphabet))
            elif op == "substitute":
                chars[rng.randrange(len(chars))] = rng.choice(alphabet)
            elif op == "delete":
                del chars[rng.randrange(len(chars))]
            elif len(chars) >= 2:
                i = rng.randrange(len(chars) - 1)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    def _try_generate(self, rng: Random, alphabet: str, num_components: int) -> Optional[dict]:
        cfg = self.config
        needed = cfg.num_examples + cfg.num_hidden
        components = [self._component(rng, alphabet, cfg.operator_level) for _ in range(num_components)]
        target = "".join(_render(c) for c in components)
        matcher = compile_safe_regex(target)

        # Positives: distinct samples from the target language, with the choices made while sampling them.
        pool: dict[str, set] = {}
        for _ in range(needed * 40):
            features: set = set()
            s = "".join(_sample(rng, c, alphabet, (i,), features) for i, c in enumerate(components))
            if len(s) <= cfg.max_string_length:
                pool.setdefault(s, set()).update(features)
                if len(pool) == 2 * needed:
                    break
        if len(pool) < needed:
            return None  # language too small (or strings too long) to hide test strings

        # Shown positives: greedily cover as many sampling choices as possible, so every alternative,
        # class character and repetition case that appears in hidden strings is demonstrated.
        candidates = list(pool)
        rng.shuffle(candidates)
        covered: set = set()
        shown: list[str] = []
        while len(shown) < cfg.num_examples:
            best = max(candidates, key=lambda c: len(pool[c] - covered))
            shown.append(best)
            covered |= pool[best]
            candidates.remove(best)
        hidden = candidates[: cfg.num_hidden]
        positives = shown + hidden
        # fraction of hidden positives built only from choices that some shown positive also demonstrates
        coverage = sum(pool[h] <= covered for h in hidden) / len(hidden)
        seen = set(pool)

        # Negatives: mostly near-misses (small edits of positives), some random strings.
        max_len = max(len(s) for s in positives)
        negatives: list[str] = []
        for _ in range(needed * 50):
            if rng.random() < 0.7:
                s = self._mutate(rng, rng.choice(positives), alphabet)
            else:
                s = self._random_word(rng, alphabet, 0, max_len + 1)
            if len(s) <= cfg.max_string_length and s not in seen and not matcher.fullmatch(s):
                seen.add(s)
                negatives.append(s)
                if len(negatives) == needed:
                    break
        if len(negatives) < needed:
            return None  # target matches (almost) everything

        return {
            "target": target,
            "positives": shown,
            "negatives": negatives[: cfg.num_examples],
            "hidden_positives": hidden,
            "hidden_negatives": negatives[cfg.num_examples :],
            "feature_coverage": coverage,
        }

    def __getitem__(self, idx: int) -> dict:
        """Generate a single regex synthesis task"""
        rng = Random(self.seed + idx)
        cfg = self.config
        alphabet = string.ascii_lowercase[: cfg.alphabet_size]

        for _ in range(self.MAX_ATTEMPTS):
            num_components = rng.randint(cfg.min_components, cfg.max_components)
            task = self._try_generate(rng, alphabet, num_components)
            if task is not None:
                break
        else:
            raise RuntimeError(f"Could not generate a regex synthesis task for config {cfg}")

        question = QUESTION_TEMPLATE.format(
            alphabet=", ".join(alphabet),
            positives="\n".join(json.dumps(s) for s in task["positives"]),
            negatives="\n".join(json.dumps(s) for s in task["negatives"]),
            max_repeat=MAX_REPEAT,
            max_length=MAX_PATTERN_LENGTH,
        )

        return {
            "question": question,
            "answer": task["target"],
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "alphabet": alphabet,
                "target_regex": task["target"],
                "positive_examples": task["positives"],
                "negative_examples": task["negatives"],
                "hidden_positives": task["hidden_positives"],
                "hidden_negatives": task["hidden_negatives"],
                "num_components": num_components,
                "feature_coverage": round(task["feature_coverage"], 4),
                "difficulty": {
                    "num_components": (cfg.min_components, cfg.max_components),
                    "alphabet_size": cfg.alphabet_size,
                    "operator_level": cfg.operator_level,
                },
            },
        }

    # ---------------------------------------------------------------- scoring

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score a regex answer.

        - 0.0 if the answer is missing, invalid, uses unsupported syntax, or misclassifies any shown example.
        - Otherwise (fraction of hidden positives matched) * (fraction of hidden negatives rejected).
          Using the product means that overfitting to the shown positives, or matching everything that is
          not a shown negative, scores 0. A fully correct regex scores 1.0.
        """
        if not isinstance(answer, str):
            return 0.0
        pattern = _normalize_answer(answer)
        if not pattern:
            return 0.0
        try:
            regex = compile_safe_regex(pattern)
        except UnsupportedRegexError:
            return 0.0

        metadata = entry["metadata"]
        if not all(regex.fullmatch(s) for s in metadata["positive_examples"]):
            return 0.0
        if any(regex.fullmatch(s) for s in metadata["negative_examples"]):
            return 0.0

        hidden_pos = metadata["hidden_positives"]
        hidden_neg = metadata["hidden_negatives"]
        pos_acc = sum(regex.fullmatch(s) for s in hidden_pos) / len(hidden_pos) if hidden_pos else 1.0
        neg_acc = sum(not regex.fullmatch(s) for s in hidden_neg) / len(hidden_neg) if hidden_neg else 1.0
        return pos_acc * neg_acc


class RegexSynthesisCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(RegexSynthesisCurriculum.__name__, RegexSynthesisConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="num_components",
                levels=[2, 3, 4, 6],
                description="Number of concatenated parts in the target regex",
                lower_field_name="min_components",
                upper_field_name="max_components",
            ),
            ScalarAttributeDefinition(
                name="operator_level",
                levels=[1, 2, 3],
                description="Regex operators used by the target (1: ?*+ and classes, 2: +alternation, 3: +groups, {m,n}, [^...])",
                field_name="operator_level",
            ),
            ScalarAttributeDefinition(
                name="alphabet_size",
                levels=[2, 3, 4, 5],
                description="Number of distinct characters in the strings",
                field_name="alphabet_size",
            ),
        )


register_dataset(DATASET_NAME, RegexSynthesisDataset, RegexSynthesisConfig, RegexSynthesisCurriculum)
