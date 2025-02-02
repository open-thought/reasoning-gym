from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, fruit: str, n1: int, n2: int, d1: str, d2: str, 
                          d3: str, mult: int) -> Dict[str, Any]:
    first_two_days = n1 + n2
    friday_amount = mult * n1
    total = first_two_days + friday_amount
    
    question = f"{name} picks {n1} {fruit}s on {d1}. Then he picks {n2} {fruit}s on {d2}. On {d3}, he picks {mult} times the number of {fruit}s he did on {d1}. How many {fruit}s does {name} have?"
    
    answer_cot = f"Combining {d1} and {d2}, {name} has {n1} {fruit}s + {n2} {fruit}s = {first_two_days} {fruit}s.\nOn {d3}, he picks {mult} * {n1} {fruit}s = {friday_amount} {fruit}s.\nAltogether, {name} has {first_two_days} {fruit}s + {friday_amount} {fruit}s = {total} {fruit}s.\n#### {total}"

    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'name': name,
            'fruit': fruit,
            'day1_amount': n1,
            'day2_amount': n2, 
            'day1': d1,
            'day2': d2,
            'day3': d3,
            'multiplier': mult,
            'day3_amount': friday_amount,
            'total': total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["John", "James", "William", "Michael", "David", "Robert", "Thomas"]
    fruits = ["banana", "apple", "orange", "pear", "peach", "plum"]
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    multipliers = ["double", "triple", "quadruple"]
    mult_values = {"double": 2, "triple": 3, "quadruple": 4}
    
    name = rng.choice(names)
    fruit = rng.choice(fruits)
    d1, d2, d3 = rng.sample(weekdays, 3)
    mult_word = rng.choice(multipliers)
    mult = mult_values[mult_word]
    
    n1 = int(rng.randint(30, int(400 * difficulty)))
    n2 = int(rng.randint(50, int(400 * difficulty)))

    result = generate_from_variables(name, fruit, n1, n2, d1, d2, d3, mult)
    
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
    return generate_from_variables("John", "banana", 4, 6, "Wednesday", "Thursday", 
                                 "Friday", 3)
