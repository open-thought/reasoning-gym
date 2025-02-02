from random import Random
from typing import Dict, Any

def generate_from_variables(structure: str, n1: int, color1: str, color2: str, 
                          color3: str, obj: str, mult: int, total: int) -> Dict[str, Any]:
    n2 = n1 * mult
    n3 = total - n1 - n2
    
    question = f"A {structure} is made out of {n1} {color1} {obj}s, {mult} times as many {color2} {obj}s, and an unknown number of {color3} {obj}s. If there are {total} {obj}s in the {structure} in total, how many {color3} {obj}s are there?"
    
    answer_cot = f"There are {n1}*{mult} = {n2} {color2} {obj}s in the {structure}.\nThere are {total}-{n1}-{n2} = {n3} {color3} {obj}s in the {structure}.\n#### {n3}"

    return {
        'question': question,
        'answer': str(n3),
        'answer_cot': answer_cot, 
        'answer_value': n3,
        'variables': {
            'structure': structure,
            'n1': n1,
            'n2': n2,
            'n3': n3,
            'color1': color1,
            'color2': color2,
            'color3': color3,
            'obj': obj,
            'mult': mult,
            'total': total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    structures = ["building", "pyramid", "stack", "tower"]
    objects = ["brick", "cube", "tile", "block"]
    colors = ["green", "purple", "orange", "pink", "white", "black"]
    
    structure = rng.choice(structures)
    obj = rng.choice(objects)
    color1, color2, color3 = rng.sample(colors, 3)
    
    n1 = int(rng.randint(2, int(10 * difficulty)))
    mult = 2  # "twice" as specified in original
    n2 = n1 * mult
    
    # Ensure total is greater than n1 + n2
    min_total = n1 + n2 + 1
    total = int(rng.randint(min_total, min_total + int(20 * difficulty)))
    
    result = generate_from_variables(structure, n1, color1, color2, color3, obj, mult, total)
    
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
    return generate_from_variables("tower", 4, "blue", "yellow", "red", "block", 2, 32)
