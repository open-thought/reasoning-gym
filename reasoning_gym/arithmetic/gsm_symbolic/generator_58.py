from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, container: str, liquid: str, volume: int, unit: str,
                          num_containers: int, calories: int) -> Dict[str, Any]:
    total_volume = volume * num_containers
    total_calories = total_volume * calories
    
    question = f"A {container} of {liquid} is {volume} {unit}s of {liquid}. {name} drinks {num_containers} {container}s of {liquid}. If {liquid} has {calories} calories per {unit} how many calories did he consume?"
    
    answer_cot = f"He drank {volume}*{num_containers}={total_volume} {unit}s of {liquid}.\nSo he drank {total_volume}*{calories}={total_calories} calories of {liquid}\n#### {total_calories}"
    
    return {
        'question': question,
        'answer': str(total_calories),
        'answer_cot': answer_cot,
        'answer_value': total_calories,
        'variables': {
            'name': name,
            'container': container,
            'liquid': liquid,
            'volume': volume,
            'unit': unit,
            'num_containers': num_containers,
            'calories': calories,
            'total_volume': total_volume,
            'total_calories': total_calories
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["John", "Mike", "James", "David", "Robert", "William"]
    containers = ["cup", "bottle", "carton"] 
    liquids = ["juice", "soda", "sparkling water", "tea", "lemonade"]
    units = ["ounce", "mL", "cc", "oz"]

    name = rng.choice(names)
    container = rng.choice(containers)
    liquid = rng.choice(liquids)
    unit = rng.choice(units)
    
    volume = int(rng.randint(6, int(16 * difficulty)))
    num_containers = int(rng.randint(2, int(6 * difficulty)))
    calories = int(rng.randint(2, int(10 * difficulty)))

    result = generate_from_variables(name, container, liquid, volume, unit,
                                   num_containers, calories)
    
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
    return generate_from_variables("John", "glass", "milk", 8, "ounce", 2, 3)
