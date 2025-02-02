from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, color1: str, color2: str, 
                          n1: int, n2: int, frac1: float, mult1: float) -> Dict[str, Any]:
    
    n1_result = int(n1 * frac1)
    n2_result = int(n2 * mult1)
    total = n1_result + n2_result
    
    question = f"{name1} has {n1} tubes of {color1} paint and {n2} tubes of {color2} paint. {name2} has half as many tubes of {color1} paint as {name1}, and three times as many tubes of {color2} paint as {name1}. How many tubes of paint does {name2} have?"
    
    answer_cot = f"{name2} has {n1}*{frac1}={n1_result} tubes of {color1} paint\n" \
                 f"{name2} has {n2}*{mult1}={n2_result} tubes of {color2} paint\n" \
                 f"{name2} has a total of {n1_result}+{n2_result}={total} tubes of paint\n" \
                 f"#### {total}"
    
    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'name1': name1,
            'name2': name2,
            'color1': color1,
            'color2': color2,
            'n1': n1,
            'n2': n2,
            'frac1': frac1,
            'mult1': mult1,
            'n1_result': n1_result,
            'n2_result': n2_result
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Ben", "James", "John", "Michael", "William", "David", "Richard", "Joseph"]
    colors = ["blue", "red", "green", "yellow", "purple", "orange"]
    
    name1, name2 = rng.sample(names, 2)
    color1, color2 = rng.sample(colors, 2)
    
    # Generate numbers that ensure integer results
    n1 = int(rng.randint(2, int(20 * difficulty)))
    n2 = int(rng.randint(2, int(20 * difficulty)))
    frac1 = 0.5  # half
    mult1 = 3.0  # three times
    
    result = generate_from_variables(name1, name2, color1, color2, n1, n2, frac1, mult1)
    
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
    return generate_from_variables("Ben", "Jasper", "blue", "yellow", 4, 3, 0.5, 3.0)
