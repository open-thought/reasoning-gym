#!/usr/bin/env python3
from .Exceptions import BFSemanticError, BFSyntaxError
from .FunctionCompiler import FunctionCompiler
from .Functions import check_function_exists, get_function_object, insert_function_object
from .General import get_literal_token_code, is_token_literal, unpack_literal_tokens_to_array_dimensions
from .Globals import (
    create_variable_from_definition,
    get_global_variables_size,
    get_variable_dimensions,
    get_variable_size,
    insert_global_variable,
)
from .Lexical_analyzer import analyze
from .LibraryFunctionCompiler import insert_library_functions
from .Optimizer import optimize
from .Parser import Parser
from .Token import Token

"""
This file is responsible for creating FunctionCompiler objects and global variables objects
And finally, return the code of the main function
"""


class Compiler:
    def __init__(self, code, optimize_code=False):
        tokens = analyze(code)
        if optimize_code:
            optimize(tokens)
        self.parser = Parser(tokens)

    # global variables and functions
    def create_function_object(self):
        # function: (INT | VOID) ID LPAREN expression_list RPAREN LBRACE statements RBRACE
        # returns function named tuple

        if self.parser.current_token().type not in [Token.VOID, Token.INT]:
            raise BFSemanticError(
                "Function return type can be either void or int, not '%s'" % str(self.parser.current_token())
            )

        self.parser.check_next_tokens_are([Token.ID, Token.LPAREN])

        # save all tokens of this function
        function_name = self.parser.next_token(next_amount=1).data
        RPAREN_index = self.parser.find_matching(
            starting_index=self.parser.current_token_index + 2
        )  # first find RPAREN
        self.parser.check_next_token_is(Token.LBRACE, starting_index=RPAREN_index)
        RBRACE_index = self.parser.find_matching(starting_index=RPAREN_index + 1)  # then find RBRACE

        # take all tokens between INT and RBRACE and pass them to function object
        function_tokens = self.parser.tokens[self.parser.current_token_index : RBRACE_index + 1]
        # skip function definition
        self.parser.advance_to_token_at_index(RBRACE_index + 1)

        function = FunctionCompiler(function_name, function_tokens)
        return function

    def compile_global_variable_definition(self):
        # INT ID (ASSIGN NUM | (LBRACK NUM RBRACK)+ (ASSIGN LBRACE ... RBRACE)?)? SEMICOLON
        # returns code that initializes this variable, and advances pointer according to variable size

        self.parser.check_current_tokens_are([Token.INT, Token.ID])
        ID_token = self.parser.next_token()
        variable = create_variable_from_definition(self.parser, advance_tokens=True)
        insert_global_variable(variable)

        # if this is set to False, then the compiler assumes that initially all cells are set to zero
        # if this is set to True, then the compiler zeros each cell before using it (may generate a lot of unnecessary BF code)
        ZERO_CELLS_BEFORE_USE = False

        code = "[-]" if ZERO_CELLS_BEFORE_USE else ""
        if get_variable_size(variable) > 1:  # its an array
            if self.parser.current_token().type == Token.SEMICOLON:
                # array definition - INT ID (LBRACK NUM RBRACK)+ SEMICOLON
                self.parser.advance_token()  # skip SEMICOLON
                code = (code + ">") * get_variable_size(variable)  # advance to after this variable
                return code
            elif self.parser.current_token().type == Token.ASSIGN and self.parser.current_token().data == "=":
                # array definition and initialization - INT ID (LBRACK NUM RBRACK)+ ASSIGN ((LBRACE ... RBRACE)+|STRING) SEMICOLON
                self.parser.advance_token()  # skip ASSIGN

                if self.parser.current_token().type not in [Token.LBRACE, Token.STRING]:
                    raise BFSyntaxError("Expected LBRACE or STRING at '%s'" % self.parser.current_token())

                literal_tokens_list = self.parser.compile_array_initialization_list()
                self.parser.check_current_token_is(Token.SEMICOLON)
                self.parser.advance_token()  # skip SEMICOLON

                array_dimensions = get_variable_dimensions(variable)
                unpacked_literals_list = unpack_literal_tokens_to_array_dimensions(
                    ID_token, array_dimensions, literal_tokens_list
                )

                for literal in unpacked_literals_list:
                    code += get_literal_token_code(literal)  # evaluate this literal and point to next array element
                return code
            else:
                raise BFSyntaxError(
                    "Unexpected %s in array definition. Expected SEMICOLON (;) or ASSIGN (=)"
                    % self.parser.current_token()
                )

        elif self.parser.current_token().type == Token.SEMICOLON:  # no need to initialize
            self.parser.advance_token()  # skip SEMICOLON
            code += ">"  # advance to after this variable
        else:
            self.parser.check_current_token_is(Token.ASSIGN)
            if self.parser.current_token().data != "=":
                raise BFSyntaxError(
                    "Unexpected %s when initializing global variable. Expected ASSIGN (=)" % self.parser.current_token()
                )
            self.parser.advance_token()  # skip ASSIGN

            if not is_token_literal(self.parser.current_token()):
                raise BFSemanticError(
                    "Unexpected '%s'. expected literal (NUM | CHAR | TRUE | FALSE )" % str(self.parser.current_token())
                )

            code += get_literal_token_code(self.parser.current_token())

            self.parser.check_next_token_is(Token.SEMICOLON)
            self.parser.advance_token(amount=2)  # skip (NUM|CHAR|TRUE|FALSE) SEMICOLON

        return code

    def process_global_definitions(self):
        """
        Iterate through all tokens
        When encountering function definition - create Function object and pass it the function's tokens
        When encountering global variable definition - create Variable object
        Returns code that initializes global variables and advances the pointer to after them
        """
        code = ""
        token = self.parser.current_token()
        while token is not None and token.type in [Token.VOID, Token.INT, Token.SEMICOLON]:
            if token.type == Token.SEMICOLON:  # can have random semicolons ;)
                self.parser.advance_token()
                token = self.parser.current_token()
                continue
            self.parser.check_next_token_is(Token.ID)

            if self.parser.next_token(next_amount=2).type == Token.LPAREN:
                function = self.create_function_object()
                insert_function_object(function)
            elif token.type is Token.INT and self.parser.next_token(next_amount=2).type in [
                Token.SEMICOLON,
                Token.ASSIGN,
                Token.LBRACK,
            ]:
                code += self.compile_global_variable_definition()
            else:
                raise BFSyntaxError(
                    "Unexpected '%s' after '%s'. Expected '(' (function definition) or one of: '=', ';', '[' (global variable definition)"
                    % (str(self.parser.next_token(next_amount=2)), str(self.parser.next_token()))
                )

            token = self.parser.current_token()

        if self.parser.current_token() is not None:  # we have not reached the last token
            untouched_tokens = [str(t) for t in self.parser.tokens[self.parser.current_token_index :]]
            raise BFSyntaxError("Did not reach the end of the code. Untouched tokens:\n%s" % untouched_tokens)

        return code

    def compile(self):
        insert_library_functions()
        code = (
            self.process_global_definitions()
        )  # code that initializes global variables and advances pointer to after them

        check_function_exists(Token(Token.ID, 0, 0, "main"), 0)
        code += get_function_object("main").get_code(get_global_variables_size())
        code += "<" * get_global_variables_size()  # point to the first cell to end the program nicely :)
        return code


def compile(code, optimize_code=False):
    """
    :param code:  C-like code (string)
    :param optimize_code:  syntax optimization (bool)
    :return code:  Brainfuck code (string)
    """
    compiler = Compiler(code, optimize_code)
    brainfuck_code = compiler.compile()
    return brainfuck_code


if __name__ == "__main__":
    print("This file cannot be directly run")
    print("Please import it and use the 'compile' function")
    print("Which receives a C-like code (string) and returns Brainfuck code (string)")
