from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, age1: int, years: int, 
                          relation_type: str, mult: int) -> Dict[str, Any]:
    future_age = age1 * mult
    current_age = future_age - years
    
    question = f"{name1} is {age1} years old. In {years} years his {relation_type} {name2} will be {mult} times as old as {name1} is now. How old is {name2} right now?"
    
    answer_cot = f"{years} years from now {name2} will be {age1}*{mult}={future_age}.\nRight now {name2} is {future_age}-{years}={current_age} years old.\n#### {current_age}"
    
    return {
        'question': question,
        'answer': str(current_age),
        'answer_cot': answer_cot,
        'answer_value': current_age,
        'variables': {
            'name1': name1,
            'name2': name2,
            'age1': age1,
            'years': years,
            'relation_type': relation_type,
            'mult': mult,
            'future_age': future_age,
            'current_age': current_age
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_male = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard']
    names_female = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan']
    relation_types = ['sister', 'cousin']
    
    name1 = rng.choice(names_male)
    name2 = rng.choice(names_female)
    relation_type = rng.choice(relation_types)
    
    age1 = int(rng.randint(8, int(25 * difficulty)))
    years = int(rng.randint(2, int(10 * difficulty)))
    mult = int(rng.randint(2, int(5 * difficulty)))
    
    # Ensure conditions are met
    while age1 * mult - years <= 0:
        age1 = int(rng.randint(8, int(25 * difficulty)))
        years = int(rng.randint(2, int(10 * difficulty)))
        mult = int(rng.randint(2, int(5 * difficulty)))
    
    result = generate_from_variables(name1, name2, age1, years, relation_type, mult)
    
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
    return generate_from_variables('Brett', 'Angela', 14, 4, 'sister', 3)
