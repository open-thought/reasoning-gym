from itertools import count
import random
import sympy
from sympy import Expr, Symbol, lambdify


def evaluate_expr(rng: random.Random, expr: Expr):
    vars = list(expr.free_symbols)
    values = [rng.randint(0, 9) for _ in range(len(vars))]
    f = lambdify(vars, expr)
    result = f(*values)

    expr_str = str(expr)
    for var, val in zip(vars, values):
        expr_str = expr_str.replace(str(var), str(val))

    return {
        "expr": expr_str,
        "result": result,
    }


def generate_expr_tree(rng: random.Random, min_depth: int = 2, max_depth: int = 5, var_prefix: str = "x") -> Expr:
    counter = count(1)
    primitives = [sympy.Add, sympy.Mul]

    def _generate(depth):
        if depth < min_depth:
            expr = rng.choice(primitives)
            return expr(_generate(depth + 1), _generate(depth + 1))
        if depth >= max_depth:
            return Symbol(f"{var_prefix}{next(counter)}")
        if rng.random() < 0.5:
            expr = rng.choice(primitives)
            return expr(_generate(depth + 1), _generate(depth + 1))
        else:
            return Symbol(f"{var_prefix}{next(counter)}")

    return _generate(0)


if __name__ == '__main__':
    rng = random.Random()
    x = generate_expr_tree(rng)
    print(x)
    print(evaluate_expr(rng, x))
