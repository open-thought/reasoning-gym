import json
from pathlib import Path
from random import Random

# more variability
def generate_math_task(
    rng: Random, num_terms: int, num_digits: int, op: list[str] = ["+", "-", "*"]
) -> tuple[str, int]:
    parts = []

    def add_terms(remaining: int):
        num_left = rng.randint(1, remaining)
        num_right = remaining - num_left

        if num_left > 1 and rng.random() > 0.5:
            if rng.random() > 0.5:
                parts.append("-(")
            else:
                parts.append("(")
            add_terms(num_left)
            parts.append(")")
        else:
            for i in range(num_left):
                c = rng.randint(-(10**num_digits) + 1, 10**num_digits - 1)
                parts.append(str(c))
                if i + 1 < num_left:
                    parts.append(rng.choice(op))

        if num_right > 0:
            parts.append(rng.choice(op))
            add_terms(num_right)

    add_terms(num_terms)

    space_parts = []
    for p in parts:
        while rng.random() < 0.15:
            space_parts.append(" ")
        space_parts.append(p)

    term = " ".join(space_parts)
    ground_truth = eval(term)

    return term, ground_truth


def generate_task_file():
    rng = Random(42)

    num_tasks = 100_000
    i = 0

    output_filename = "math_tasks.jsonl"
    file_path = Path(output_filename)
    with file_path.open("w", encoding="utf-8") as f:
        while i < num_tasks:
            num_terms = rng.randint(2, 6)
            num_digits = rng.randint(1, 6)
            term, ground_truth = generate_math_task(
                rng, num_terms=num_terms, num_digits=num_digits
            )
            if abs(ground_truth) > 10**8 or abs(ground_truth) < 10:
                continue

            question_templates = [
                "{0}",
                "{0} =",
                "{0} = ?",
                "What is {0}?",
                "Solve {0}",
            ]

            template = rng.choice(question_templates)
            formatted_task = template.format(term)

            entry = {
                "id": str(i),
                "question": formatted_task,
                "answer": str(ground_truth),
                "num_terms": num_terms,
                "num_digits": num_digits,
            }

            json.dump(entry, f)
            f.write("\n")
            i += 1




class BasicIntArithmeticTaskConfig:
    def __init__(
        self,
        min_digits: int = 1,
        max_digits: int = 5,
        min_terms: int = 2,
        max_terms: int = 8,
    ):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.min_terms = min_terms
        self.max_terms = max_terms
        self.operators = ["+", "-"]

    def validate(self):
        assert self.min_digits > 0
        assert self.max_digits >= self.min_digits
        assert self.min_terms > 1
        assert self.max_terms >= self.min_terms
        assert len(self.operators) > 0


def generate_task(rng: Random, cfg: BasicIntArithmeticTaskConfig) -> str:
    num_terms = rng.randint(cfg.min_terms, cfg.max_terms)
    num_digits = rng.randint(cfg.min_digits, cfg.max_digits)
    constants = [rng.randint(0, 10**num_digits) for _ in range(num_terms)]
    operators = [rng.choice(cfg.operators) for _ in range(num_terms - 1)]

    buffer = []

    ground_truth = constants[0]

    buffer.append(f"{constants[0]}")
    for i, op in enumerate(operators):
        c = constants[i + 1]
        buffer.append(op)
        buffer.append(f"{c}")

        if op == "+":
            ground_truth += c
        elif op == "-":
            ground_truth -= c
        else:
            RuntimeError("Unsupported operator")

    buffer.append(f"")

    question_templates = [
        "{0}",
        "{0} =",
        "{0} = ?",
        "What is {0}?",
        "Solve {0}",
        "Calculate {0}",
        # 'evaluate {0}',
        # 'do me a favor and calculate {0}',
        # 'Give me the result of {0}',
        # 'Help me solve this: {0}',
        # 'calculator: {0}',
        # 'Tell me the result of the following expression {0}',
    ]

    template = rng.choice(question_templates)
    task = " ".join(buffer)
    formatted_task = template.format(task)

    return formatted_task, str(ground_truth), num_terms, num_digits
