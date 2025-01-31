from .Functions import insert_function_object
from .Token import Token


class LibraryFunctionCompiler:
    def __init__(self, name, type, parameters, code):
        self.name = name
        self.type = type
        self.parameters = parameters
        self.code = code

    def get_code(self, current_stack_pointer):
        return self.code


def get_readint_code():
    # res, tmp, input, loop
    # tmp is used for multiplication
    """
    res = 0
    loop = 1

    while loop
        loop = 0
        input = input()
        if input != newline # todo add a eof check as well. run it in several interpreters to look for common ways for "end of number" input
            loop = 1
            res *= 10 + char_to_digit(input)
    """

    code = "[-]"  # clear res = 0
    code += ">[-]"  # tmp = 0
    code += ">>[-]+"  # loop = 1

    code += "["  # while loop == 1
    code += "[-]"  # loop = 0
    code += "<"  # point to input
    code += ","  # input character
    code += "----------"  # sub 10 (check for newline)

    code += "["  # if input is not newline
    code += ">"  # point to loop
    code += "+"  # loop = 1

    # multiply res by 10 and add the input digit
    code += "<<<"  # point to res
    code += "[>+<-]"  # move res to tmp
    code += ">"  # point to tmp
    code += "[<++++++++++>-]"  # res = tmp * 10, tmp = 0
    code += ">"  # point to input
    code += "-" * (
        0x30 - 10
    )  # convert character to a digit by subtracting 0x30 from it (we already subtracted 10 before)
    code += "[<<+>>-]"  # res += input
    code += "]"  # end if

    code += ">"  # point to loop
    code += "]"  # end while

    code += "<<<"  # point to res

    return code


def get_printint_code():
    # return_cell value_to_print_cell

    code = ">"  # point to value_to_print cell
    code += ">[-]" * 8 + "<" * 8  # zero some cells

    code += ">++++++++++<"  # div amount
    code += "[->-[>+>>]>[+[<+>-]>+>>]<<<<<]"  # value_to_print/10
    code += ">[-]"  # zero d-n%d
    code += ">>"  # point to div result

    code += ">++++++++++<"  # div amount
    code += "[->-[>+>>]>[+[<+>-]>+>>]<<<<<]"  # res/10
    code += ">[-]"  # zero d-n%d
    code += ">>"  # point to div result

    code += "["  # if the first digit is not 0
    code += ">++++++[<++++++++>-]<."  # add 48 to the first digit and print it
    code += "<<"
    code += "+>"  # set is_over_100 to true
    code += "+>"  # add 1 to the second digit so it prints even when it's 0
    code += "[-]"  # zero the first digit
    code += "]"  # end if

    code += "<"  # point to the second digit

    code += "["  # if the second digit is not 0
    code += "<[>-<-]"  # if is_over_100 is true then subtract 1 from the second digit
    code += "++++++[>++++++++<-]>."  # add 48 to the second digit and print it
    code += "[-]"  # zero the second digit
    code += "]"  # end if

    code += "<<"  # point to the cell after the third digit
    code += "++++++[<++++++++>-]<."  # add 48 to the third digit and print it
    code += "[-]"  # zero the third digit
    code += "<<"  # point to value_to_print_cell which is 0

    code += "<"  # point to return_cell
    return code


def get_readchar_code():
    # read input into "return value cell". no need to move the pointer
    code = ","
    return code


def get_printchar_code():
    # point to parameter, output it, and then point back to "return value cell"
    code = ">.<"
    return code


def insert_library_functions():
    readint = LibraryFunctionCompiler("readint", Token.INT, list(), get_readint_code())
    insert_function_object(readint)

    printint = LibraryFunctionCompiler("printint", Token.VOID, [Token.INT], get_printint_code())
    insert_function_object(printint)

    readchar = LibraryFunctionCompiler("readchar", Token.INT, list(), get_readchar_code())
    insert_function_object(readchar)

    printchar = LibraryFunctionCompiler("printchar", Token.VOID, [Token.INT], get_printchar_code())
    insert_function_object(printchar)
