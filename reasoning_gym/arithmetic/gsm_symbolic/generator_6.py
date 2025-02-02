from random import Random
from typing import Dict, Any

def generate_from_variables(n_girls: int, place: str, multiplier: int) -> Dict[str, Any]:
    n_boys = n_girls * multiplier
    total_kids = n_girls + n_boys
    
    question = f"There are {n_girls} girls in the {place}. If there are {multiplier} times the number of boys in the {place}, how many kids are in the {place}?"
    
    answer_cot = f"There are {n_girls} girls x {multiplier} boys/girl = {n_boys} boys in the {place}.\nIn total there are {n_girls} girls + {n_boys} boys = {total_kids} kids in the {place}\n#### {total_kids}"

    return {
        'question': question,
        'answer': str(total_kids),
        'answer_cot': answer_cot,
        'answer_value': total_kids,
        'variables': {
            'n_girls': n_girls,
            'place': place,
            'multiplier': multiplier,
            'n_boys': n_boys,
            'total_kids': total_kids
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    places = ['park', 'yard', 'field', 'playground', 'garden']
    multipliers = [2, 3, 4] # twice, triple, quadruple

    place = rng.choice(places)
    multiplier = rng.choice(multipliers)
    
    # Scale n_girls with difficulty but ensure result is valid
    n_girls = int(rng.randint(5, int(50 * difficulty)))
    while n_girls * (multiplier + 1) > 200:
        n_girls = int(rng.randint(5, int(50 * difficulty)))

    result = generate_from_variables(n_girls, place, multiplier)
    
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
    return generate_from_variables(6, 'park', 2)
