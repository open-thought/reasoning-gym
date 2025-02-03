from collections import namedtuple

from .General import dimensions_to_size, get_NUM_token_value
from .Token import Token

"""
This file holds the program's functions and global variables
(as global variables, hehe)
And related functions
"""

global_variables = list()  # Global list of global variables


# variables
def get_global_variables():
    return global_variables


def insert_global_variable(variable):
    get_global_variables().append(variable)


def get_global_variables_size():
    return sum(get_variable_size(variable) for variable in get_global_variables())


def create_variable(name, type, dimensions):
    # return variable named tuple
    variable = namedtuple("variable", ["name", "type", "size", "cell_index"])

    variable.name = name
    variable.type = type
    variable.dimensions = dimensions  # list of array dimensions sizes (for non-arrays it will be [1])
    variable.cell_index = None  # will be updated when we insert this variable into an ids map

    return variable


def get_variable_size(variable):
    # return total variable size
    return dimensions_to_size(variable.dimensions)


def get_variable_dimensions(variable):
    return variable.dimensions


def is_variable_array(variable):
    return variable.dimensions != [1]


def create_variable_from_definition(parser, index=None, advance_tokens=False):
    """
    processes the variable definition at index, and returns the variable named tuple
    if index is None, then assumes we start at the current_token_index
    if advance_tokens is True, then modifies current_token_index accordingly using parser.advance_token()
    """

    if index is None:
        index = parser.current_token_index

    assert parser.tokens[index].type == Token.INT

    parser.check_next_token_is(Token.ID, starting_index=index)
    ID = parser.tokens[index + 1].data

    if advance_tokens:
        parser.advance_token(amount=2)  # skip INT ID

    if parser.tokens[index + 2].type == Token.LBRACK:  # array (support multi-dimensional arrays)
        dimensions = []  # element[i] holds the size of dimension[i]
        while parser.tokens[index + 2].type == Token.LBRACK:
            parser.check_next_tokens_are([Token.LBRACK, Token.NUM, Token.RBRACK], starting_index=index + 1)
            dimensions.append(get_NUM_token_value(parser.tokens[index + 3]))

            if advance_tokens:
                parser.advance_token(amount=3)  # skip LBRACK NUM RBRACK
            index += 3
    else:
        dimensions = [1]

    return create_variable(ID, Token.INT, dimensions)
