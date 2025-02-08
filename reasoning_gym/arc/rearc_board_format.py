from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BoardFormattingOptions:
    alphabet: list[str]
    col_delimiter: str
    row_delimiter: str
    array_brackets: bool


def format_arc_task(
    input_grid: Tuple[Tuple[int, ...], ...], output_grid: Tuple[Tuple[int, ...], ...], options: BoardFormattingOptions
) -> str:

    buffer = []
    if options.task_identifier:
        buffer.append(f"ARC Task: {options.task_identifier}")

    buffer.append("\nInput Grid:")
    buffer.append(format_board(input_grid, options))

    buffer.append("\n\nOutput Grid:")
    buffer.append(format_board(output_grid, options))

    return "\n".join(buffer)


def format_board(
    board: List[List[int]], formatting_options: BoardFormattingOptions, with_board_shape: bool = False
) -> str:
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
    pair: dict[str, List[List[int]]],
    formatting_options: BoardFormattingOptions,
) -> str:
    input_element = format_board(
        pair["input"],
        formatting_options=formatting_options,
    )
    output_element = format_board(
        pair["output"],
        formatting_options=formatting_options,
    )
    return f"Example {index}:\n\nInput:\n{input_element}\nOutput:\n{output_element}\n\n"
