from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, fruit: str, total: int, n1: int, n2: int, n3: int,
                          sibling1: str, sibling2: str) -> Dict[str, Any]:
    slice2 = n1 + n2
    slice3 = slice2 + n3
    total_eaten = n1 + slice2 + slice3
    
    question = f"{name} sliced an {fruit} into {total} pieces. She ate {n1} slice, her {sibling1} ate {n2} more than her, and her {sibling2} ate {n3} more than her {sibling1}. How many slices of {fruit} did they all eat?"
    
    answer_cot = f"Her {sibling1} ate {n1} + {n2} = {slice2} slices.\nHer {sibling2} ate {slice2} + {n3} = {slice3} slices.\nThey ate a total of {n1} + {slice2} + {slice3} = {total_eaten} slices.\n#### {total_eaten}"

    return {
        'question': question, 
        'answer': str(total_eaten),
        'answer_cot': answer_cot,
        'answer_value': total_eaten,
        'variables': {
            'name': name,
            'fruit': fruit,
            'total_slices': total,
            'first_person_slices': n1,
            'second_person_extra': n2,
            'third_person_extra': n3,
            'sibling1': sibling1,
            'sibling2': sibling2,
            'total_eaten': total_eaten
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Doxa"]
    fruits = ["orange", "pear", "peach", "mango", "kiwi", "apple"]
    siblings = ["brother", "sister", "cousin", "friend"]
    
    name = rng.choice(names_female)
    fruit = rng.choice(fruits)
    sibling1, sibling2 = rng.sample(siblings, 2)
    
    total = int(rng.randint(6, int(33 * difficulty)))
    n1 = int(rng.randint(3, int(15 * difficulty)))
    n2 = int(rng.randint(5, int(13 * difficulty)))
    n3 = int(rng.randint(3, int(14 * difficulty)))
    
    # Ensure conditions are met
    while n1 + (n1 + n2) + (n1 + n2 + n3) > total:
        n1 = int(rng.randint(3, int(15 * difficulty)))
        n2 = int(rng.randint(5, int(13 * difficulty)))
        n3 = int(rng.randint(3, int(14 * difficulty)))
    
    result = generate_from_variables(name, fruit, total, n1, n2, n3, sibling1, sibling2)
    
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
    return generate_from_variables("Doxa", "apple", 8, 1, 1, 1, "sister", "brother")
