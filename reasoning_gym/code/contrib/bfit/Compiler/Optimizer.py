from .General import get_NUM_token_value
from .Token import Token

"""
This file holds functions that optimize code on syntax-level. For example:
The tokens corresponding to the code "3*5" will be replaced in-place by a token that represents "15"
"""


def optimize_once(tokens):
    # performs one pass on the tokens and optimizes them in-place if possible
    # optimization based on a list of rules

    def optimize_binop(tokens, start_index):
        # optimize arithmetic operations. E.g replace 1+2 with 3

        # need to be careful not to optimize (1+2*3) to (3*3)
        if (
            tokens[start_index + 1].data in ["*", "/", "%"]
            or (start_index + 3 >= len(tokens))
            or (tokens[start_index + 3].data not in ["*", "/", "%"])
        ):
            num1, num2 = get_NUM_token_value(tokens[start_index]), get_NUM_token_value(tokens[start_index + 2])
            op = tokens[start_index + 1].data
            if op == "+":
                val = num1 + num2
            elif op == "-":
                val = num1 - num2
                if val < 0:  # cannot optimize negative values
                    return False
            elif op == "*":
                val = num1 * num2
            elif op in ["/", "%"]:
                if num2 == 0:
                    print("WARNING (optimizer) - division by zero at %s" % str(tokens[start_index]))
                    return False
                if op == "/":
                    val = num1 // num2
                else:
                    val = num1 % num2
            else:
                raise NotImplementedError(op)

            # remove the 3 old tokens and replace them with new one
            new_token = Token(
                Token.NUM,
                tokens[start_index].line,
                tokens[start_index].column,
                data=str(val),
                original_tokens=tokens[start_index : start_index + 3],
            )

            for _ in range(3):
                tokens.pop(start_index)
            tokens.insert(start_index, new_token)
            return True

        return False

    def optimize_printint(tokens, start_index):
        # replace printint(50) with print("50")
        # since printing strings compiles into less Brainfuck code than printing ints
        if tokens[start_index].data == "printint":
            tokens[start_index] = Token(
                Token.PRINT, tokens[start_index].line, tokens[start_index].column, original_tokens=[tokens[start_index]]
            )
            tokens[start_index + 2] = Token(
                Token.STRING,
                tokens[start_index].line,
                tokens[start_index].column,
                data=str(tokens[start_index + 2].data),
                original_tokens=[tokens[start_index + 2]],
            )
            return True

        return False

    rules = [
        ([Token.NUM, Token.BINOP, Token.NUM], optimize_binop),  # arithmetic operations
        ([Token.ID, Token.LPAREN, Token.NUM, Token.RPAREN], optimize_printint),  # printint(50) to print("50")
    ]

    # try to match one of the rules to the tokens in a "sliding window" style
    i = 0
    while i < len(tokens):
        optimized = False
        for tokens_sequence, optimization_function in rules:
            if i + len(tokens_sequence) <= len(tokens):
                if all(tokens_sequence[n] == tokens[i + n].type for n in range(len(tokens_sequence))):
                    if optimization_function(tokens, i):
                        optimized = True
        if optimized:
            continue  # don't increment i, try to optimize the same location again
        i += 1


def optimize(tokens):
    # optimize tokens again and again until there is nothing left to optimize
    prev_tokens = [token.type for token in tokens]
    while True:
        optimize_once(tokens)
        print(".", end="")
        current_tokens = [token.type for token in tokens]
        if current_tokens == prev_tokens:
            break
        prev_tokens = current_tokens
