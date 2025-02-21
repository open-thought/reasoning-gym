from dataclasses import dataclass, field

ARC_PROMPT_TEMPLATE = """Find the common rule that maps an input grid to an output grid, given the examples below.

{examples}
Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
{input_grid}
"""


@dataclass
class BoardFormattingOptions:
    alphabet: list[str] = field(default_factory=lambda: [str(i) for i in range(10)])
    col_delimiter: str = " "
    row_delimiter: str = "\n"
    array_brackets: bool = False


def format_board(
    board: list[list[int]], formatting_options: BoardFormattingOptions, with_board_shape: bool = False
) -> str:
    """
    Format a board as a string
    """
    alphabet = formatting_options.alphabet
    col_delimiter = formatting_options.col_delimiter
    row_delimiter = formatting_options.row_delimiter
    array_brackets = formatting_options.array_brackets

    h, w = len(board), len(board[0])
    buffer = []

    if with_board_shape:
        buffer.append(f"Shape: {h}x{w}\n")

    if array_brackets:
        buffer.append(f"[")
        for row in range(h):
            if row > 0 and row_delimiter:
                buffer.append(row_delimiter)
            buffer.append("[")
            for col in range(w):
                if col > 0 and col_delimiter:
                    buffer.append(col_delimiter)
                value = board[row][col]
                buffer.append(alphabet[value])
            buffer.append("]")
        buffer.append("]")
    else:
        for row in range(h):
            if row > 0 and row_delimiter:
                buffer.append(row_delimiter)
            for col in range(w):
                if col > 0 and col_delimiter:
                    buffer.append(col_delimiter)
                value = board[row][col]
                buffer.append(alphabet[value])

    return "".join(buffer)


def format_board_pair(
    index: int,
    pair: dict[str, list[list[int]]],
    formatting_options: BoardFormattingOptions,
) -> str:
    """
    Format a board pair as a string
    """
    input_element = format_board(
        pair["input"],
        formatting_options=formatting_options,
    )
    output_element = format_board(
        pair["output"],
        formatting_options=formatting_options,
    )
    return f"Example {index}:\n\nInput:\n{input_element}\nOutput:\n{output_element}\n\n"


def parse_board(formatted_str: str, formatting_options: BoardFormattingOptions) -> tuple[tuple[int, ...], ...]:
    """
    Convert a formatted board string back to a tuple grid using formatting options
    """
    lines = [line.strip() for line in formatted_str.split("\n") if not line.strip().startswith("Shape: ")]
    grid_str = "\n".join(lines).strip()

    if formatting_options.array_brackets:
        if grid_str.startswith("[") and grid_str.endswith("]"):
            grid_str = grid_str[1:-1].strip()

    rows = grid_str.split(formatting_options.row_delimiter)
    if not rows:
        return tuple()

    grid = []
    for row in rows:
        row = row.strip()
        if formatting_options.array_brackets:
            if row.startswith("[") and row.endswith("]"):
                row = row[1:-1].strip()
        cells = row.split(formatting_options.col_delimiter)
        try:
            grid.append(
                tuple(
                    formatting_options.alphabet.index(cell.strip())
                    for cell in cells
                    if cell.strip()  # Handle empty strings from trailing delimiters
                )
            )
        except ValueError as e:
            valid_chars = ", ".join(f"'{c}'" for c in formatting_options.alphabet)
            raise ValueError(f"Invalid character in board string. Valid options: {valid_chars}") from e

    return tuple(grid)
