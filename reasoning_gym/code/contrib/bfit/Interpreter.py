#!/usr/bin/env python3

import argparse
import sys


def create_jumps_dictionary(program):
    lbraces = list()
    res = dict()

    for index, command in enumerate(program):
        if command == "[":
            lbraces.append(index)
        elif command == "]":
            if len(lbraces) == 0:
                raise SyntaxError("Brainfuck: mismatched parentheses (at index: %s)" % index)

            lbrace_index = lbraces.pop()
            res[lbrace_index] = index
            res[index] = lbrace_index

    if len(lbraces) > 0:
        raise SyntaxError("Brainfuck: mismatched parentheses (at indexes: %s)" % str(lbraces))
    return res


def brainfuck(program, bits=8):

    jumps = create_jumps_dictionary(program)
    data = dict()
    data_pointer = 0

    instruction_pointer = 0

    while instruction_pointer < len(program):
        command = program[instruction_pointer]

        if command == ">":
            data_pointer += 1
        elif command == "<":
            data_pointer -= 1
        elif command == "+":
            data[data_pointer] = data.get(data_pointer, 0) + 1
            if data[data_pointer] == 2**bits:
                data[data_pointer] = 0
        elif command == "-":
            data[data_pointer] = data.get(data_pointer, 0) - 1
            if data[data_pointer] == -1:
                data[data_pointer] = 2**bits - 1
        elif command == ",":
            data[data_pointer] = ord(sys.stdin.read(1)) % 256
        elif command == ".":
            print(chr(data.get(data_pointer, 0)), end="", flush=True)
        elif command == "[":
            if data.get(data_pointer, 0) == 0:
                instruction_pointer = jumps[instruction_pointer]
        elif command == "]":
            if data.get(data_pointer, 0) != 0:
                instruction_pointer = jumps[instruction_pointer]
        else:  # everything else is comment
            pass

        instruction_pointer += 1

    if data_pointer != 0:
        print(
            "WARNING (interpreter) - at the end of the execution the data pointer is %s instead of 0 (possibly a compiler issue)"
            % str(data_pointer)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("--bits", "-b", "--interpreter-bits", type=int, default=8, help="Amount of bits each cell uses")

    args = parser.parse_args()
    with open(args.filepath, "r") as f:
        code = f.read()

    brainfuck(code, args.bits)
