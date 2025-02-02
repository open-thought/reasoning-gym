from random import Random
from typing import Dict, Any

def generate_from_variables(comet_name: str, name: str, relative: str, orbit_period: int, 
                          relative_age: int, multiple: int) -> Dict[str, Any]:
    second_viewing_age = relative_age * multiple
    first_viewing_age = second_viewing_age - orbit_period
    
    question = f"Comet {comet_name} orbits the sun every {orbit_period} years. {name}'s {relative} saw the Comet when he was {relative_age} years old. {name} saw the comet a second time when he was {multiple} times the age his {relative} was when he saw the Comet. How old was {name} when he saw the Comet for the first time?"
    
    answer_cot = f"{name} saw the Comet for the second time when he was {relative_age} years * {multiple}= {second_viewing_age} years old.\nComet {comet_name} can be seen every {orbit_period} years, so {name} saw the comet for the first time when he was {second_viewing_age} years - {orbit_period} years = {first_viewing_age} years old.\n#### {first_viewing_age}"
    
    return {
        'question': question,
        'answer': str(first_viewing_age),
        'answer_cot': answer_cot,
        'answer_value': first_viewing_age,
        'variables': {
            'comet_name': comet_name,
            'name': name,
            'relative': relative,
            'orbit_period': orbit_period,
            'relative_age': relative_age,
            'multiple': multiple,
            'second_viewing_age': second_viewing_age,
            'first_viewing_age': first_viewing_age
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    comets = ['Halley', 'Hale-Bopp', 'Hyakutake', 'Encke']
    names = ['William', 'James', 'John', 'Robert', 'Michael', 'David']
    relatives = ['dad', 'father', 'uncle', 'grandfather']
    multiples = ['two', 'three', 'four']
    
    comet_name = rng.choice(comets)
    name = rng.choice(names)
    relative = rng.choice(relatives)
    multiple = rng.choice(multiples)
    multiple_num = {'two': 2, 'three': 3, 'four': 4}[multiple]
    
    orbit_period = int(rng.randrange(50, int(101 * difficulty), 5))
    relative_age = int(rng.randint(20, int(51 * difficulty)))
    
    # Ensure conditions are met
    while (multiple_num * relative_age >= 100 or
           multiple_num * relative_age <= orbit_period or
           (multiple_num * relative_age - orbit_period) % 1 != 0):
        relative_age = int(rng.randint(20, int(51 * difficulty)))
    
    result = generate_from_variables(comet_name, name, relative, orbit_period,
                                   relative_age, multiple_num)
    
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
    return generate_from_variables('Halley', 'Bill', 'dad', 75, 30, 3)
