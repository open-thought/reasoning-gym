from z3 import *


def column(matrix, i):
    return [matrix[j][i] for j in range(len(matrix))]

def instanciate_int_constrained(name, s, card):
    x = Int(name)
    # Each int represent an index in p[name]
    s.add(x >= 0, x <= card - 1)
    return x

def count_solutions(s, max=1e9):
    count = 0
    while s.check() == sat:
        count += 1
        if count >= max:
            return count
        m = s.model()

        # Create a new constraint the blocks the current model
        block = []
        for d in m:
            # d is a declaration
            if d.arity() > 0:
                raise Z3Exception("uninterpreted functions are not supported")
            # create a constant from declaration
            c = d()
            if is_array(c) or c.sort().kind() == Z3_UNINTERPRETED_SORT:
                raise Z3Exception("arrays and uninterpreted sorts are not supported")
            block.append(c != m[d])

        s.add(Or(block))
    return count
