from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, items: str, food: str, location: str, container: str,
                          num_jars: int, per_jar: int, per_pan: int) -> Dict[str, Any]:
    total_items = num_jars * per_jar
    num_pans = total_items // per_pan
    
    question = f"{name} has {num_jars} jars of {items} in her {location}. Each jar of {items} can decorate {per_jar} {food}s. {name} wants to bake enough {food}s to use up all of her {items}. If each {container} holds {per_pan} {food}s, how many {container}s worth of {food}s should she bake?"
    
    answer_cot = f"She has enough {items} for {num_jars} * {per_jar} = {total_items} {food}s.\nShe needs {total_items} / {per_pan} = {num_pans} {container}s to bake all of the {food}s.\n#### {num_pans}"

    return {
        'question': question,
        'answer': str(num_pans),
        'answer_cot': answer_cot,
        'answer_value': num_pans,
        'variables': {
            'name': name,
            'items': items,
            'food': food,
            'location': location,
            'container': container,
            'num_jars': num_jars,
            'per_jar': per_jar,
            'per_pan': per_pan,
            'total_items': total_items
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ['Mary', 'Sarah', 'Emma', 'Elizabeth', 'Catherine']
    items = ['sprinkles', 'frosting', 'icing', 'chocolate chips'] 
    foods = ['cupcake', 'cookie', 'brownie', 'muffin']
    locations = ['pantry', 'cupboard', 'kitchen cabinet', 'storage room']
    containers = ['pan', 'tray', 'baking sheet', 'rack']
    
    name = rng.choice(names_female)
    item = rng.choice(items)
    food = rng.choice(foods)
    location = rng.choice(locations)
    container = rng.choice(containers)
    
    # Generate numbers ensuring divisibility
    per_pan = int(rng.randint(6, int(24 * difficulty)))
    per_jar = int(rng.randint(6, int(20 * difficulty)))
    num_jars = int(rng.randint(3, int(15 * difficulty)))
    
    # Ensure total is divisible by per_pan
    total = num_jars * per_jar
    while total % per_pan != 0:
        num_jars = int(rng.randint(3, int(15 * difficulty)))
        total = num_jars * per_jar

    result = generate_from_variables(name, item, food, location, container,
                                   num_jars, per_jar, per_pan)
    
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
    return generate_from_variables('Mary', 'sprinkles', 'cupcake', 'pantry', 'pan',
                                 6, 8, 12)
