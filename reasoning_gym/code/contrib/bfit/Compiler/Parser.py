from .Exceptions import BFSemanticError, BFSyntaxError
from .General import is_token_literal
from .Token import Token


class Parser:
    """
    Used to easily iterate tokens
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token_index = 0

    # parsing tokens
    def current_token(self):
        if self.current_token_index >= len(self.tokens):
            return None
        else:
            return self.token_at_index(self.current_token_index)

    def advance_token(self, amount=1):
        self.current_token_index += amount

    def advance_to_token_at_index(self, token_index):
        self.current_token_index = token_index

    def token_at_index(self, index):
        assert index < len(self.tokens)
        return self.tokens[index]

    def next_token(self, next_amount=1):
        return self.token_at_index(self.current_token_index + next_amount)

    def find_matching(self, starting_index=None):
        """
        :return: the index of the token that matches the current token
        :param starting_index (optional) - the index of the token we want to match

        for example, if current token is {
        it returns the index of the matching }
        """
        if starting_index is None:
            starting_index = self.current_token_index

        tokens = self.tokens
        token_to_match = tokens[starting_index]
        if token_to_match.type == Token.LBRACE:
            inc = Token.LBRACE
            dec = Token.RBRACE
        elif token_to_match.type == Token.LBRACK:
            inc = Token.LBRACK
            dec = Token.RBRACK
        elif token_to_match.type == Token.LPAREN:
            inc = Token.LPAREN
            dec = Token.RPAREN
        else:
            raise BFSemanticError("No support for matching %s" % str(token_to_match))

        i = starting_index
        cnt = 0
        while i < len(tokens):
            if tokens[i].type == inc:
                cnt += 1
            elif tokens[i].type == dec:
                cnt -= 1

            if cnt == 0:
                return i

            i += 1

        raise BFSyntaxError("Did not find matching %s for %s" % (dec, str(token_to_match)))

    def check_next_tokens_are(self, tokens_list, starting_index=None):
        if starting_index is None:
            starting_index = self.current_token_index

        # used for "assertion" and print a nice message to the user
        if starting_index + len(tokens_list) >= len(self.tokens):
            raise BFSyntaxError("Expected %s after %s" % (str(tokens_list), str(self.tokens[starting_index])))
        for i in range(0, len(tokens_list)):
            if self.tokens[starting_index + 1 + i].type != tokens_list[i]:
                raise BFSyntaxError(
                    "Expected %s after %s"
                    % (str(tokens_list[i]), [str(t) for t in self.tokens[starting_index : starting_index + 1 + i]])
                )

    def check_next_token_is(self, token, starting_index=None):
        self.check_next_tokens_are([token], starting_index=starting_index)

    def check_current_tokens_are(self, tokens_list):
        self.check_next_tokens_are(tokens_list, starting_index=self.current_token_index - 1)

    def check_current_token_is(self, token):
        self.check_current_tokens_are([token])

    def compile_array_initialization_list(self):
        # {1, 2, 3, ...} or {array_initialization_list, array_initialization_list, array_initialization_list, ...} or string
        # parses the definition and returns a list (of list of list ....) of literal tokens (NUM, CHAR, TRUE, FALSE)

        list_tokens = []

        if self.current_token().type == Token.STRING:
            string_token = self.current_token()
            line, column = string_token.line, string_token.column
            for char in string_token.data:
                list_tokens.append(Token(Token.NUM, line, column, str(ord(char))))

            self.advance_token()  # point to after STRING
            return list_tokens

        assert self.current_token().type == Token.LBRACE
        self.advance_token()  # skip to after LBRACE

        while is_token_literal(self.current_token()) or self.current_token().type == Token.LBRACE:
            if self.current_token().type == Token.LBRACE:  # list of (literals | list)
                list_tokens.append(self.compile_array_initialization_list())
            else:  # literal
                list_tokens.append(self.current_token())
                self.advance_token()  # skip literal

            if self.current_token().type not in [Token.COMMA, Token.RBRACE]:
                raise BFSyntaxError("Unexpected %s (expected comma (,) or RBRACE (}))" % self.current_token())

            if self.current_token().type == Token.COMMA:
                self.advance_token()  # skip comma
            if self.current_token().type == Token.RBRACE:
                break

        self.check_current_token_is(Token.RBRACE)
        self.advance_token()  # skip RBRACE
        return list_tokens
