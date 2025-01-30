import re

from .Optimizer import optimize
from .Token import Token


class LexicalErrorException(Exception):
    pass


def analyze(text):
    """
    :returns list of tokens in the text
    raises exception in case of lexical error
    """

    rules = [
        (r"\s+", Token.WHITESPACE),
        ("void", Token.VOID),
        ("int", Token.INT),
        ("bool", Token.INT),  # treat bool as int
        ("char", Token.INT),  # treat char as int
        ("true", Token.TRUE),
        ("false", Token.FALSE),
        ("&&", Token.AND),
        (r"\|\|", Token.OR),
        (r"\!", Token.NOT),
        ("return", Token.RETURN),
        ("if", Token.IF),
        ("else", Token.ELSE),
        ("while", Token.WHILE),
        ("for", Token.FOR),
        ("do", Token.DO),
        ("print", Token.PRINT),
        ("switch", Token.SWITCH),
        ("case", Token.CASE),
        ("default", Token.DEFAULT),
        ("break", Token.BREAK),
        ("continue", Token.CONTINUE),  # todo
        (":", Token.COLON),
        (";", Token.SEMICOLON),
        (",", Token.COMMA),
        (r"\(", Token.LPAREN),
        (r"\)", Token.RPAREN),
        (r"\{", Token.LBRACE),
        (r"\}", Token.RBRACE),
        (r"\[", Token.LBRACK),
        (r"\]", Token.RBRACK),
        (r"=|\+=|-=|\*=|/=|%=|<<=|>>=|&=|\|=|\^=", Token.ASSIGN),
        (r"\?", Token.TERNARY),
        (r"<=|>=|==|!=|<|>", Token.RELOP),
        (r"\+\+", Token.INCREMENT),
        ("--", Token.DECREMENT),
        (r"\+|-|\*|/|%", Token.BINOP),
        (r"\*\*|//|%%", Token.UNARY_MULTIPLICATIVE),
        ("<<|>>", Token.BITWISE_SHIFT),
        ("~", Token.BITWISE_NOT),
        ("&", Token.BITWISE_AND),
        (r"\|", Token.BITWISE_OR),
        (r"\^", Token.BITWISE_XOR),
        ("([a-zA-Z_][a-zA-Z0-9_]*)", Token.ID),
        (r"(\d+)", Token.NUM),
        (r"(0x[A-Fa-f\d]+)", Token.NUM),  # hexadecimal number
        ("(0o[0-7]+)", Token.NUM),  # octal number
        ("(0b[01]+)", Token.NUM),  # binary number
        (r'\"(\\\"|[^"])*"', Token.STRING),
        (r"\'(\\\'|(\\)?[^\'])\'", Token.CHAR),
        ("//.*(\\n|$)", Token.COMMENT),
        (r"/\*[\s\S]*?\*/", Token.COMMENT),  # multiline comments
        (".", Token.UNIDENTIFIED),
    ]

    rules = [(re.compile(r), t) for r, t in rules]

    tokens = []

    # create a mapping of [line number] to [offset of that line from the beginning of the text]
    newline = re.compile("\n")
    lines = [0] + [m.end() for m in re.finditer(newline, text)]

    i = 0
    while i < len(text):
        current_matches = []
        for regex, token_type in rules:
            m = regex.match(text, i)
            if m:
                current_matches.append((m, token_type))

        # pick the token that fits the longest match
        # if tie - pick the one defined first in the rules list
        longest_match, max_i, matched_token = None, i, None
        for match, token_type in current_matches:
            if match.end() > max_i:
                longest_match, max_i, matched_token = match, match.end(), token_type

        # calculate line and column
        line, column = None, None
        for line_idx in range(len(lines) - 1):
            if lines[line_idx] <= longest_match.start() < lines[line_idx + 1]:
                line, column = line_idx + 1, (longest_match.start() - lines[line_idx]) + 1  # humans count from 1 :)
                break
        if not line:
            line, column = len(lines), (longest_match.start() - lines[-1]) + 1

        if matched_token in [Token.COMMENT, Token.WHITESPACE]:
            pass  # do nothing
        elif matched_token == Token.UNIDENTIFIED:
            raise LexicalErrorException("Unidentified Character '%s' (line %s column %s)" % (text[i], line, column))
        elif matched_token in [Token.STRING, Token.CHAR]:
            # remove quotes at beginning and end, un-escape characters
            tokens.append(
                Token(matched_token, line, column, longest_match.group()[1:-1].encode("utf8").decode("unicode_escape"))
            )
        elif matched_token in [
            Token.NUM,
            Token.ID,
            Token.BINOP,
            Token.RELOP,
            Token.ASSIGN,
            Token.UNARY_MULTIPLICATIVE,
            Token.BITWISE_SHIFT,
        ]:
            tokens.append(Token(matched_token, line, column, longest_match.group()))
        else:
            tokens.append(Token(matched_token, line, column))
        i = longest_match.end()

    return tokens


def tests():
    def test1():
        # test token priorities: INT should not be confused with ID even if ID contains "int"
        text = "my international int ; int; pints; international;"
        res = analyze(text)

        expected = [
            Token.ID,
            Token.ID,
            Token.INT,
            Token.SEMICOLON,
            Token.INT,
            Token.SEMICOLON,
            Token.ID,
            Token.SEMICOLON,
            Token.ID,
            Token.SEMICOLON,
        ]
        assert len(res) == len(expected) and all(res[i].type == expected[i] for i in range(len(res)))

    def test2():
        text = "true !||!false falsek  k||y+-a&&x"
        res = analyze(text)

        expected = [
            Token.TRUE,
            Token.NOT,
            Token.OR,
            Token.NOT,
            Token.FALSE,
            Token.ID,
            Token.ID,
            Token.OR,
            Token.ID,
            Token.BINOP,
            Token.BINOP,
            Token.ID,
            Token.AND,
            Token.ID,
        ]
        assert len(res) == len(expected) and all(res[i].type == expected[i] for i in range(len(res)))

    def test3():
        text = "1+2"
        tokens = analyze(text)
        expected = [Token.NUM, Token.BINOP, Token.NUM]
        assert len(tokens) == len(expected) and all(tokens[i].type == expected[i] for i in range(len(tokens)))
        optimize(tokens)
        assert len(tokens) == 1 and tokens[0].type == Token.NUM and tokens[0].data == "3"

        text = "1+2+3"
        tokens = analyze(text)
        expected = [Token.NUM, Token.BINOP, Token.NUM, Token.BINOP, Token.NUM]
        assert len(tokens) == len(expected) and all(tokens[i].type == expected[i] for i in range(len(tokens)))
        optimize(tokens)
        assert len(tokens) == 1 and tokens[0].type == Token.NUM and tokens[0].data == "6"

        # make sure it is not optimized to 9 (3*3)
        text = "1+2*3"
        tokens = analyze(text)
        expected = [Token.NUM, Token.BINOP, Token.NUM, Token.BINOP, Token.NUM]
        assert len(tokens) == len(expected) and all(tokens[i].type == expected[i] for i in range(len(tokens)))
        optimize(tokens)
        assert len(tokens) == 1 and tokens[0].type == Token.NUM and tokens[0].data == "7"

        # test all arithmetic operations
        text = "(1+2*3/6)+(1%3)*(6-1)"
        tokens = analyze(text)
        expected = [
            Token.LPAREN,
            Token.NUM,
            Token.BINOP,
            Token.NUM,
            Token.BINOP,
            Token.NUM,
            Token.BINOP,
            Token.NUM,
            Token.RPAREN,
            Token.BINOP,
            Token.LPAREN,
            Token.NUM,
            Token.BINOP,
            Token.NUM,
            Token.RPAREN,
            Token.BINOP,
            Token.LPAREN,
            Token.NUM,
            Token.BINOP,
            Token.NUM,
            Token.RPAREN,
        ]
        assert len(tokens) == len(expected) and all(tokens[i].type == expected[i] for i in range(len(tokens)))
        optimize(tokens)
        assert tokens[1].data == "2" and tokens[5].data == "1" and tokens[9].data == "5"

    # todo find a better way to test?
    test1()
    test2()
    test3()


if __name__ == "__main__":
    tests()
