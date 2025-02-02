from random import Random
from typing import Dict, Any

def generate_from_variables(n: int, ball_type: str, color: str, frac_1: float, frac_2: float) -> Dict[str, Any]:
    first_calc = int(n * frac_1)
    final_calc = int(first_calc * frac_2)
    
    question = f"A juggler can juggle {n} balls. {frac_1:.0%} of the balls are {ball_type} balls, and {frac_2:.0%} of the {ball_type} balls are {color}. How many {color} {ball_type} balls are there?"
    
    answer_cot = f"{ball_type} balls:{n} * {frac_1}={first_calc}\n{color} {ball_type} balls:{first_calc}*{frac_2}={final_calc} balls\n#### {final_calc}"

    return {
        'question': question,
        'answer': str(final_calc),
        'answer_cot': answer_cot,
        'answer_value': final_calc,
        'variables': {
            'total_balls': n,
            'ball_type': ball_type,
            'color': color,
            'fraction_first': frac_1,
            'fraction_second': frac_2,
            'first_calculation': first_calc,
            'final_calculation': final_calc
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    ball_types = ["golf", "tennis"]
    colors = ["blue", "red", "green", "yellow", "white"]
    fractions = [0.5, 0.25, 0.75]

    ball_type = rng.choice(ball_types)
    color = rng.choice(colors)
    frac_1 = rng.choice(fractions) 
    frac_2 = rng.choice(fractions)
    
    # Generate n that ensures integer results
    n = int(rng.randint(10, int(100 * difficulty)))
    while not (n * frac_1).is_integer() or not (n * frac_1 * frac_2).is_integer():
        n = int(rng.randint(10, int(100 * difficulty)))

    result = generate_from_variables(n, ball_type, color, frac_1, frac_2)
    
    return {
        'question': result['question'],
        'answer': result['answer'],
        'metadata': {
            'difficulty': difficulty,
            'answer_value': result['answer_value'],
            'answer_cot': result['answer_cot'],
            'variables': result['variables']
        }
    }

def original_example() -> Dict[str, Any]:
    return generate_from_variables(16, "golf", "blue", 0.5, 0.5)
