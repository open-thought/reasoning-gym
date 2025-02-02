from random import Random
from typing import Dict, Any

def generate_from_variables(family: str, item: str, total: int, n1: int, n2: int,
                          flavor1: str, flavor2: str, flavor3: str) -> Dict[str, Any]:
    n3 = total - (n1 + n2)
    
    question = f"The {family} family is busy making {item}s. So far, they've made {total} {item}s. They have {n1} {flavor1} {item}s, {n2} {flavor2} {item}s, and some {flavor3} {item}s. How many {flavor3} {item}s have they made?"
    
    answer_cot = f"The total number of pieces of {flavor1} and {flavor2} {item}s is {n1} + {n2} = {n1+n2}.\nTherefore, they made {total} - {n1+n2} = {n3} {flavor3} {item}s.\n#### {n3}"

    return {
        'question': question,
        'answer': str(n3),
        'answer_cot': answer_cot,
        'answer_value': n3,
        'variables': {
            'family': family,
            'item': item,
            'total': total,
            'n1': n1,
            'n2': n2,
            'n3': n3,
            'flavor1': flavor1,
            'flavor2': flavor2,
            'flavor3': flavor3
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    families = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
    items = ["cupcake", "muffin", "brownie", "biscuit"] 
    flavors = ["vanilla", "strawberry", "blueberry", "lemon", "peanut butter"]

    family = rng.choice(families)
    item = rng.choice(items)
    flavor1, flavor2, flavor3 = rng.sample(flavors, 3)

    total = int(rng.randrange(5000, int(10000 * difficulty), 25))
    n1 = int(rng.randint(1000, int(3000 * difficulty)))
    n2 = int(rng.randint(1000, int(3000 * difficulty)))

    while n1 + n2 >= total:
        n1 = int(rng.randint(1000, int(3000 * difficulty)))
        n2 = int(rng.randint(1000, int(3000 * difficulty)))

    result = generate_from_variables(family, item, total, n1, n2, flavor1, flavor2, flavor3)

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
    return generate_from_variables("Adams", "cookie", 7995, 2595, 3075,
                                 "rainbow", "oatmeal", "chocolate chip")
