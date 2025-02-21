from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

VERT = "│"
HORIZ = "─"
RBRANCH = "├"
LUP = "┘"
LDOWN = "┐"
RUP = "└"
RDOWN = "┌"


def _repeat(s: str, n: int) -> str:
    return s * n


def _matrix_put(matrix: list[list[str]], h: int, w: int, x: int, y: int, s: str, direction: str):
    """Place a string `s` into the 2D `matrix` starting at (x,y),
    advancing in `direction` ('RIGHT' or 'DOWN')."""
    if x >= w or y >= h:
        raise IndexError(f"_matrix_put: point ({x}, {y}) out of bounds!")
    for c in s:
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        matrix[y][x] = c
        if direction == "RIGHT":
            x += 1
        elif direction == "DOWN":
            y += 1


def _get_excel_name(index: int) -> str:
    """
    Convert a zero-based integer `index` into an Excel-like column name:
    0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, etc.
    """
    result = ""
    index += 1
    while index > 0:
        rem = (index - 1) % 26
        result = chr(ord("A") + rem) + result
        index = (index - 1) // 26
    return result


@dataclass
class CircuitLogicConfig:
    """
    Configuration for circuit logic task generation.

    :param num_terms: Number of terms (sub-expressions) to generate
    :param min_inputs: Minimum inputs per term
    :param max_inputs: Maximum inputs per term
    :param neg_prob: Probability (0.0-1.0) that an input is negated
    :param allow_reuse: Whether inputs can be reused
    :param size: Number of items in the dataset
    :param seed: Random seed
    """

    num_terms: int = 5
    min_inputs: int = 2
    max_inputs: int = 4
    neg_prob: float = 0.3
    allow_reuse: bool = True
    size: int = 100
    seed: Optional[int] = None

    def validate(self):
        assert 1 <= self.min_inputs <= self.max_inputs, "Invalid input range"
        assert 1 <= self.num_terms, "Invalid number of terms"
        assert 0.0 <= self.neg_prob <= 1.0, "neg_prob must be between 0 and 1"


class CircuitLogicDataset(ProceduralDataset):
    """
    Generates random digital logic circuits (in ASCII) together with:
      - a random Boolean expression,
      - random input assignments,
      - the final evaluated output.

    Each item in the dataset is a dict with:
       {
           "question": <str>,
           "answer": <str>,
           "metadata": {
               "diagram": <ASCII circuit diagram>,
               "expression": <str>,
               "term_strings": <list of term_strings>,
               "assignments": <dict of input->0/1>,
               "final_gate": <str>,
               "inputs": <list of input names>,
           }
       }
    """

    def __init__(self, config: CircuitLogicConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.config.validate()

        self.internal_ops = [
            ("AND", "&", "&"),
            ("NAND", "↑", "↑"),
            ("XOR", "⊕", "⊕"),
        ]
        self.final_gate_options = [
            ("OR", "+"),
            ("NOR", "↓"),
            ("XOR", "⊕"),
            ("AND", "&"),
        ]

    def __len__(self) -> int:
        return self.config.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self) -> dict[str, Any]:
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Generate one random circuit logic item using ASCII drawing.
        """
        rng = Random(self.seed + idx if self.seed is not None else None)
        return self._generate_circuit(
            rng=rng,
            num_terms=self.config.num_terms,
            min_inputs=self.config.min_inputs,
            max_inputs=self.config.max_inputs,
            neg_prob=self.config.neg_prob,
            allow_reuse=self.config.allow_reuse,
        )

    def _generate_circuit(
        self, rng: Random, num_terms: int, min_inputs: int, max_inputs: int, neg_prob: float, allow_reuse: bool
    ) -> dict[str, Any]:
        """
        Generate circuit logic (ASCII drawing + expression + evaluation)
        """
        final_gate_name, final_gate_sym = rng.choice(self.final_gate_options)
        final_gate_width = 2 + len(final_gate_sym)

        distinct_inputs: list[str] = []

        def get_random_input() -> str:
            if allow_reuse and distinct_inputs and rng.random() < 0.5:
                return rng.choice(distinct_inputs)
            else:
                name = _get_excel_name(len(distinct_inputs))
                distinct_inputs.append(name)
                return name

        term_ops: list[tuple[str, str, str]] = []
        term_strings: list[str] = []
        for _ in range(num_terms):
            op = rng.choice(self.internal_ops)
            term_ops.append(op)

            term_length = rng.randint(min_inputs, max_inputs)
            parts = []
            for __ in range(term_length):
                inp = get_random_input()
                neg = rng.random() < neg_prob
                parts.append(inp + ("'" if neg else ""))
            # Join the parts with the operator’s join symbol.
            term_str = op[1].join(parts)
            term_strings.append(term_str)

        expression_for_display = final_gate_sym.join(f"({t})" for t in term_strings)
        # use || separator internally that doesn't clash with other symbols...
        separator = "||"
        expression_for_internal = separator.join(term_strings)

        expr = []  # will hold a list of tuples (op_used, term_input_list)
        inputs_set = set()
        term_inputs_map = {}
        input_ypos = 0

        outer_terms = expression_for_internal.split(separator)
        for op_chosen, term in zip(term_ops, outer_terms):
            op_used = op_chosen
            # If the join symbol appears in the term, split by it; otherwise (single literal) use it as-is.
            if op_used[1] in term:
                input_strs = term.split(op_used[1])
            else:
                input_strs = [term]

            curr_term = []
            for part in input_strs:
                if not part:
                    continue
                neg = part.endswith("'")
                name = part[:-1] if neg else part
                inputs_set.add(name)
                curr_term.append({"name": name, "ypos": input_ypos, "neg": neg})
                term_inputs_map.setdefault(name, []).append({"ypos": input_ypos, "neg": neg})
                input_ypos += 1

            expr.append((op_used, curr_term))
            # Add a gap after each term.
            input_ypos += 1

        inputs_list = sorted(list(inputs_set))
        total_term_inputs = sum(len(t) for (_, t) in expr)
        height = len(inputs_list) + total_term_inputs + len(expr) - 1

        max_input_len = max((len(s) for s in inputs_list), default=0)
        input_width = max_input_len + 2
        width = 0

        width += input_width
        width += len(inputs_list) * 2
        not_and_start = width
        width += 7  # space for gates ...
        width += len(expr) + 1
        gate_start = width
        width += final_gate_width
        width += 4  # additional wiring space
        width += 4
        width += len(expression_for_display)

        # Create an empty drawing matrix.
        matrix = [[" " for _ in range(width)] for _ in range(height)]
        base_y = len(inputs_list)

        x = width - 8 - len(expression_for_display)
        y = base_y + ((height - base_y) // 2)
        _matrix_put(matrix, height, width, x, y, _repeat(HORIZ, 4) + " OUT", "RIGHT")

        x = gate_start
        out_gate_center = base_y + ((height - base_y) // 2) - (len(expr) // 2)
        if len(expr) == 1:
            _matrix_put(matrix, height, width, x, out_gate_center, _repeat(HORIZ, final_gate_width), "RIGHT")
        else:
            _matrix_put(matrix, height, width, x, out_gate_center, _repeat(HORIZ, len(expr)), "DOWN")
            _matrix_put(matrix, height, width, x + 1, out_gate_center, _repeat(VERT, len(expr)), "DOWN")
            for i, ch in enumerate(final_gate_sym):
                _matrix_put(matrix, height, width, x + 2 + i, out_gate_center, _repeat(ch, len(expr)), "DOWN")
                _matrix_put(matrix, height, width, x + 3 + i, out_gate_center, _repeat(ch, len(expr)), "DOWN")

        # Draw internal wiring (for the internal gate section).
        x = not_and_start
        y = base_y
        for op, term_inputs in expr:
            layers = [""] * 7
            for ti in term_inputs:
                layers[0] += HORIZ
                layers[1] += ">" if ti["neg"] else HORIZ
                layers[2] += "o" if ti["neg"] else HORIZ
                layers[3] += HORIZ
                # If multiple inputs, we connect them vertically
                layers[4] += VERT if len(term_inputs) > 1 else HORIZ
                layers[5] += op[2] if (len(term_inputs) > 1) else HORIZ
                layers[6] += op[2] if (len(term_inputs) > 1) else HORIZ

            for i in range(7):
                _matrix_put(matrix, height, width, x + i, y, layers[i], "DOWN")
            y += len(term_inputs) + 1

        x = 0
        y = 0
        for inp in inputs_list:
            label = f"{inp}: " + _repeat(HORIZ, input_width - (len(inp) + 2))
            _matrix_put(matrix, height, width, x, y, label, "RIGHT")
            y += 1

        x = input_width
        for idx, inp in enumerate(inputs_list):
            y = idx
            length = len(inputs_list) * 2 - 1 - (idx * 2)
            _matrix_put(matrix, height, width, x, y, _repeat(HORIZ, length) + LDOWN, "RIGHT")

        num = 0
        offset = len(inputs_list) * 2 - 1
        for inp in inputs_list:
            y_breaks = [base_y + ti["ypos"] for ti in term_inputs_map.get(inp, [])]
            y_breaks.sort()
            for yb in y_breaks:
                _matrix_put(
                    matrix, height, width, x + offset, yb, _repeat(HORIZ, len(inputs_list) * 2 - offset), "RIGHT"
                )
            y_start = num + 1
            max_break = max(y_breaks) if y_breaks else y_start
            branch = list(_repeat(VERT, max_break - y_start + 1))
            for yb in y_breaks:
                pos = yb - y_start
                if 0 <= pos < len(branch):
                    branch[pos] = RBRANCH
            branch[-1] = RUP
            _matrix_put(matrix, height, width, x + offset, y_start, "".join(branch), "DOWN")
            offset -= 2
            num += 1

        x = not_and_start + 7
        out_y = out_gate_center
        breakx = len(expr) // 2
        for op, term_inputs in expr:
            in_y = base_y + (term_inputs[0]["ypos"] + term_inputs[-1]["ypos"]) // 2
            # horizontal to branch
            _matrix_put(matrix, height, width, x, in_y, _repeat(HORIZ, abs(breakx) + 1), "RIGHT")
            # horizontal from branch up/down to final gate column
            _matrix_put(
                matrix, height, width, x + abs(breakx) + 1, out_y, _repeat(HORIZ, len(expr) - abs(breakx)), "RIGHT"
            )

            if in_y < out_y:
                branch = LDOWN + _repeat(VERT, out_y - in_y - 1) + RUP
                _matrix_put(matrix, height, width, x + abs(breakx) + 1, in_y, branch, "DOWN")
            elif in_y > out_y:
                branch = RDOWN + _repeat(VERT, in_y - out_y - 1) + LUP
                _matrix_put(matrix, height, width, x + abs(breakx) + 1, out_y, branch, "DOWN")

            out_y += 1
            breakx -= 1

        ascii_diagram = "\n".join("".join(row).rstrip() for row in matrix)

        assignments = {}
        for inp in inputs_list:
            assignments[inp] = rng.choice([0, 1])

        term_values = []
        for op_used, term_inputs in expr:
            op_name = op_used[0]
            values = []
            for literal in term_inputs:
                val = assignments[literal["name"]]
                if literal["neg"]:
                    val = 1 - val
                values.append(val)

            if op_name == "AND":
                term_val = 1 if all(v == 1 for v in values) else 0
            elif op_name == "NAND":
                term_val = 0 if all(v == 1 for v in values) else 1
            elif op_name == "XOR":
                tmp = 0
                for v in values:
                    tmp ^= v
                term_val = tmp
            else:
                term_val = 0
            term_values.append(term_val)

        # Evaluate final gate based on term values
        if final_gate_name == "OR":
            final_result = 1 if any(v == 1 for v in term_values) else 0
        elif final_gate_name == "NOR":
            final_result = 0 if any(v == 1 for v in term_values) else 1
        elif final_gate_name == "XOR":
            final_result = sum(term_values) % 2
        elif final_gate_name == "AND":
            final_result = 1 if all(v == 1 for v in term_values) else 0
        else:
            raise ValueError(f"Unknown gate type: {final_gate_name}")

        lines = []
        lines.append("Below is a randomly generated logic circuit.\n")
        lines.append(ascii_diagram)
        lines.append("\n")
        legend_lines = []
        legend_lines.append("Legend for gates:")
        for op_name, _, draw_sym in self.internal_ops:
            legend_lines.append(f"{draw_sym*2}: {op_name}")
        if neg_prob > 0:
            legend_lines.append(f">o: Negate")
        if final_gate_sym not in self.internal_ops:
            legend_lines.append(f"{final_gate_sym*2}: {final_gate_name}")
        legend_str = "\n".join(legend_lines)

        lines.append(legend_str)
        lines.append("")
        lines.append("Given the following input assignments:")
        for inp in inputs_list:
            lines.append(f"  {inp} = {assignments[inp]}")
        lines.append("")
        lines.append("What is the final output?")

        answer_str = str(final_result)
        question_str = "\n".join(lines)

        return {
            "question": question_str,
            "answer": answer_str,
            "metadata": {
                "expression": expression_for_display,
                "assignments": assignments,
                "term_strings": term_strings,
                "final_gate": final_gate_name,
                "inputs": inputs_list,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None or len(answer) == 0:
            return 0.0

        oracle_answer = entry["answer"]
        if oracle_answer == answer:
            return 1.0
        elif oracle_answer == answer.strip():
            return len(oracle_answer) / len(answer)

        return 0.01


register_dataset("circuit_logic", CircuitLogicDataset, CircuitLogicConfig)
