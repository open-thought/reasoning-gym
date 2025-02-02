from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, obj: str, surface: str, 
                          capacity: int, total: int, num_trays: int) -> Dict[str, Any]:
    max_capacity = capacity * num_trays
    leftover = total - max_capacity
    
    question = f"{name} places {obj}s on the {surface}. Each {surface} can hold {capacity} {obj}s. If he has {total} {obj}s and {num_trays} {surface}s, how many {obj}s won't he be able to place on the {surface}?"
    
    answer_cot = f"{name} will be able to place a total of {capacity} x {num_trays} = {max_capacity} {obj}s.\nTherefore, there are {total} - {max_capacity} = {leftover} {obj}s that he won't be able to place on the {surface}.\n#### {leftover}"
    
    return {
        'question': question,
        'answer': str(leftover),
        'answer_cot': answer_cot,
        'answer_value': leftover,
        'variables': {
            'name': name,
            'obj': obj,
            'surface': surface,
            'capacity_per_tray': capacity,
            'total_items': total,
            'num_trays': num_trays,
            'max_capacity': max_capacity,
            'leftover': leftover
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard']
    objects = ['olive', 'almond', 'cookie', 'cracker', 'banana'] 
    surfaces = ['plate', 'table', 'bowl', 'tray', 'basket']
    
    name = rng.choice(names)
    obj = rng.choice(objects)
    surface = rng.choice(surfaces)
    
    capacity = int(rng.randint(20, int(51 * difficulty)))
    num_trays = int(rng.randint(2, int(7 * difficulty)))
    
    # Ensure total is greater than max capacity
    max_capacity = capacity * num_trays
    total = max_capacity + int(rng.randint(1, int(20 * difficulty)))
    
    result = generate_from_variables(name, obj, surface, capacity, total, num_trays)
    
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
    return generate_from_variables('Jaime', 'egg', 'tray', 24, 64, 2)
