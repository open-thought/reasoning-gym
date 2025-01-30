#!/usr/bin/env python3

import argparse
import os

import Interpreter
from Compiler import Compiler, Minify


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", metavar="input_file", nargs=1, help="Path to the input code file")
    parser.add_argument("--output", "-o", metavar="output_file", nargs=1, help="Path to output Brainfuck file")
    parser.add_argument("--run", "-r", action="store_true", help="Run the Brainfuck file after compilation")
    parser.add_argument("--minify", "-m", action="store_true", help="Minifies the compiled code")
    parser.add_argument("--optimize", "-opt", action="store_true", help="syntax optimization")

    args = parser.parse_args()

    input_file = args.filepath[0]
    if args.output:
        output_file = args.output[0]
    else:
        output_file_basename = os.path.splitext(os.path.basename(input_file))[0] + ".bf"
        output_file = os.path.join(os.path.dirname(input_file), output_file_basename)

    run_file = args.run
    minify_file = args.minify
    optimize = args.optimize

    return input_file, output_file, run_file, minify_file, optimize


def compile_file():
    input_file, output_file, run_file, minify_bf_code, optimize_code = process_args()
    print("Compiling file '%s'..." % input_file)

    with open(input_file, "rb") as f:
        code = f.read().decode("utf8")

    brainfuck_code = Compiler.compile(code, optimize_code)
    brainfuck_code += "\n"

    if minify_bf_code:
        brainfuck_code = Minify.minify(brainfuck_code)

    with open(output_file, "wt") as f:
        f.write(brainfuck_code)

    print("Compiled successfully to '%s'" % output_file)

    if run_file:
        print("Running compiled code...")
        Interpreter.brainfuck(brainfuck_code)


if __name__ == "__main__":
    compile_file()
