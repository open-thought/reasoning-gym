from copy import deepcopy

from .Exceptions import BFSemanticError

functions = dict()  # Global dictionary of function_name --> FunctionCompiler objects


def insert_function_object(function):
    functions[function.name] = function


def get_function_object(name):
    """
    must return a copy of the function
    because we might need to compile function recursively
    and if we don't work on different copies then we will interfere with the current token pointer etc

    for example:
        int increase(int n) { return n+1;}
        int main() {int x = increase(increase(1));}

    while compiling the first call, we start a compilation of the same function object in the second call
    """
    return deepcopy(functions[name])


def check_function_exists(function_token, parameters_amount):
    function_name = function_token.data
    if function_name not in functions:
        raise BFSemanticError("Function '%s' is undefined" % str(function_token))

    function = functions[function_name]
    if len(function.parameters) != parameters_amount:
        raise BFSemanticError(
            "Function '%s' has %s parameters (called it with %s parameters)"
            % (str(function_token), len(function.parameters), parameters_amount)
        )
