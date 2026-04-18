import cmath
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "complex_advanced"

TASK_TYPES = ("polar", "euler", "inverse", "sqrt", "quadratic")


@dataclass
class ComplexAdvancedConfig:
    min_real: int = 1
    max_real: int = 10
    min_imag: int = 1
    max_imag: int = 10
    decimal_places: int = 4
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.max_real >= self.min_real, "max_real must be >= min_real"
        assert self.max_imag >= self.min_imag, "max_imag must be >= min_imag"
        assert self.min_real >= 1, "min_real must be >= 1"
        assert self.min_imag >= 1, "min_imag must be >= 1"
        assert self.decimal_places >= 1, "decimal_places must be >= 1"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type, must be in {TASK_TYPES}"
        assert len(self.task_weights) == len(self.task_types), "task_weights must match task_types length"
        assert self.size > 0, "size must be positive"


def _fmt(val: float, dp: int) -> str:
    return f"{val:.{dp}f}"


def _fmt_complex(z: complex, dp: int) -> str:
    r, i = round(z.real, dp), round(z.imag, dp)
    if abs(i) < 10 ** (-(dp + 1)):
        return _fmt(r, dp)
    if abs(r) < 10 ** (-(dp + 1)):
        return f"{_fmt(i, dp)}i"
    sign = "+" if i >= 0 else "-"
    return f"{_fmt(r, dp)} {sign} {_fmt(abs(i), dp)}i"


class ComplexAdvancedDataset(ProceduralDataset):
    def __init__(self, config: ComplexAdvancedConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _rand_complex(self, rng: random.Random, allow_neg: bool = True) -> complex:
        r = rng.randint(self.config.min_real, self.config.max_real)
        i = rng.randint(self.config.min_imag, self.config.max_imag)
        if allow_neg:
            r *= rng.choice([1, -1])
            i *= rng.choice([1, -1])
        return complex(r, i)

    def _make_polar(self, rng: random.Random) -> dict:
        z = self._rand_complex(rng)
        dp = self.config.decimal_places
        r, theta = cmath.polar(z)
        answer = f"modulus={_fmt(r, dp)}, argument={_fmt(theta, dp)}"
        a, b = int(z.real), int(z.imag)
        sign = "+" if b >= 0 else "-"
        question = (
            f"Convert the complex number {a} {sign} {abs(b)}i to polar form. "
            f"Give the modulus and argument (in radians), both rounded to {dp} decimal places. "
            f"Format: modulus=<value>, argument=<value>"
        )
        return {"question": question, "answer": answer, "task_type": "polar", "z": (a, b)}

    def _make_euler(self, rng: random.Random) -> dict:
        z = self._rand_complex(rng)
        dp = self.config.decimal_places
        r, theta = cmath.polar(z)
        rect = cmath.rect(r, theta)
        answer = _fmt_complex(rect, dp)
        question = (
            f"Express {_fmt(r, dp)}(cos({_fmt(theta, dp)}) + i*sin({_fmt(theta, dp)})) "
            f"in rectangular form a + bi, rounded to {dp} decimal places."
        )
        return {"question": question, "answer": answer, "task_type": "euler", "r": r, "theta": theta}

    def _make_inverse(self, rng: random.Random) -> dict:
        z = self._rand_complex(rng)
        dp = self.config.decimal_places
        inv = 1.0 / z
        answer = _fmt_complex(inv, dp)
        a, b = int(z.real), int(z.imag)
        sign = "+" if b >= 0 else "-"
        question = (
            f"Find the multiplicative inverse of {a} {sign} {abs(b)}i. "
            f"Express your answer in the form a + bi, rounded to {dp} decimal places."
        )
        return {"question": question, "answer": answer, "task_type": "inverse", "z": (a, b)}

    def _make_sqrt(self, rng: random.Random) -> dict:
        w = self._rand_complex(rng)
        z = w * w
        dp = self.config.decimal_places
        root1 = _fmt_complex(w, dp)
        root2 = _fmt_complex(-w, dp)
        answer = f"{root1}, {root2}"
        zr, zi = round(z.real, dp), round(z.imag, dp)
        sign = "+" if zi >= 0 else "-"
        question = (
            f"Find the two square roots of {_fmt(zr, dp)} {sign} {_fmt(abs(zi), dp)}i. "
            f"Give both roots rounded to {dp} decimal places, separated by a comma."
        )
        return {"question": question, "answer": answer, "task_type": "sqrt", "w": (int(w.real), int(w.imag))}

    def _make_quadratic(self, rng: random.Random) -> dict:
        dp = self.config.decimal_places
        use_complex = rng.choice([True, False])
        if use_complex:
            p = rng.randint(self.config.min_real, self.config.max_real) * rng.choice([1, -1])
            q = rng.randint(self.config.min_imag, self.config.max_imag)
            r1 = complex(p, q)
            r2 = complex(p, -q)
        else:
            r1 = complex(rng.randint(-self.config.max_real, self.config.max_real), 0)
            r2 = complex(rng.randint(-self.config.max_real, self.config.max_real), 0)

        a_coeff = 1
        b_coeff = -a_coeff * (r1 + r2)
        c_coeff = a_coeff * r1 * r2
        b_int, c_int = round(b_coeff.real), round(c_coeff.real)

        terms = [f"x^2"]
        if b_int > 0:
            terms.append(f"+ {b_int}x")
        elif b_int < 0:
            terms.append(f"- {abs(b_int)}x")
        if c_int > 0:
            terms.append(f"+ {c_int}")
        elif c_int < 0:
            terms.append(f"- {abs(c_int)}")
        eq_str = " ".join(terms)

        ans1 = _fmt_complex(r1, dp)
        ans2 = _fmt_complex(r2, dp)
        answer = f"{ans1}, {ans2}"

        question = (
            f"Solve the quadratic equation {eq_str} = 0. "
            f"Give both solutions rounded to {dp} decimal places, separated by a comma. "
            f"For complex solutions, use the form a + bi."
        )
        return {
            "question": question,
            "answer": answer,
            "task_type": "quadratic",
            "roots": [(r1.real, r1.imag), (r2.real, r2.imag)],
        }

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "polar": self._make_polar,
            "euler": self._make_euler,
            "inverse": self._make_inverse,
            "sqrt": self._make_sqrt,
            "quadratic": self._make_quadratic,
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
                    "min_real": self.config.min_real,
                    "max_real": self.config.max_real,
                    "min_imag": self.config.min_imag,
                    "max_imag": self.config.max_imag,
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        oracle = entry["answer"]
        if answer.strip() == oracle.strip():
            return 1.0

        task_type = entry["metadata"]["task_type"]
        try:
            if task_type == "polar":
                return self._score_polar(answer, oracle)
            elif task_type in ("sqrt", "quadratic"):
                return self._score_pair(answer, oracle)
            else:
                return self._score_single_complex(answer, oracle)
        except Exception:
            return 0.0

    def _score_polar(self, answer: str, oracle: str) -> float:
        def parse_polar(s: str) -> tuple[float, float]:
            parts = {}
            for part in s.split(","):
                k, v = part.split("=")
                parts[k.strip()] = float(v.strip())
            return parts["modulus"], parts["argument"]

        am, aa = parse_polar(answer)
        om, oa = parse_polar(oracle)
        mod_err = abs(am - om)
        arg_err = abs(aa - oa)
        return min(1.0, math.exp(-(mod_err + arg_err)))

    def _score_single_complex(self, answer: str, oracle: str) -> float:
        az = self._parse_complex(answer)
        oz = self._parse_complex(oracle)
        if az is None or oz is None:
            return 0.0
        return min(1.0, math.exp(-abs(az - oz)))

    def _score_pair(self, answer: str, oracle: str) -> float:
        a_parts = [s.strip() for s in answer.split(",")]
        o_parts = [s.strip() for s in oracle.split(",")]
        if len(a_parts) < 2 or len(o_parts) < 2:
            return 0.0
        a_vals = [
            self._parse_complex(a_parts[0] + ("" if "i" in a_parts[0] else "")),
            self._parse_complex(a_parts[1] + ("" if "i" in a_parts[1] else "")),
        ]
        o_vals = [self._parse_complex(o_parts[0]), self._parse_complex(o_parts[1])]
        if any(v is None for v in a_vals + o_vals):
            return 0.0
        d1 = min(
            abs(a_vals[0] - o_vals[0]) + abs(a_vals[1] - o_vals[1]),
            abs(a_vals[0] - o_vals[1]) + abs(a_vals[1] - o_vals[0]),
        )
        return min(1.0, math.exp(-d1))

    @staticmethod
    def _parse_complex(s: str) -> Optional[complex]:
        try:
            s = s.strip().replace(" ", "").replace("i", "j")
            if "j" not in s:
                return complex(float(s), 0)
            if s == "j":
                return 1j
            if s == "-j":
                return -1j
            if s.endswith("j") and "+" not in s[1:] and "-" not in s[1:]:
                coef = s[:-1] or "1"
                if coef == "-":
                    coef = "-1"
                return complex(0, float(coef))
            if "+j" in s:
                s = s.replace("+j", "+1j")
            if "-j" in s:
                s = s.replace("-j", "-1j")
            return complex(s)
        except (ValueError, TypeError):
            return None


class ComplexAdvancedCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(ComplexAdvancedCurriculum.__name__, ComplexAdvancedConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="max_real",
                field_name="max_real",
                levels=[5, 10, 50, 100],
                description="Maximum real part magnitude",
            ),
            ScalarAttributeDefinition(
                name="max_imag",
                field_name="max_imag",
                levels=[5, 10, 50, 100],
                description="Maximum imaginary part magnitude",
            ),
        )


register_dataset(DATASET_NAME, ComplexAdvancedDataset, ComplexAdvancedConfig, ComplexAdvancedCurriculum)
