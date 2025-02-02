from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, event: str, food: str, obj: str,
                          package_husband: int, used_items: int, 
                          total_remaining: int) -> Dict[str, Any]:
    
    total_items = total_remaining + used_items
    package_size = total_items - package_husband
    
    question = f"{name} was preparing for a {event} at her house, where she intended to serve {food}. She noticed that she was out of plastic {obj}, so she bought a new package of {obj}. Later, her husband also bought a package of {package_husband} new {obj} and gave them to {name}. While {name} was making the {food}, she used {used_items} of the {obj} to sample her {food}. Later, when she went to set the table, she had a total of {total_remaining} {obj}. How many {obj} were in the package that {name} bought?"
    
    answer_cot = f"The total number of {obj} from {name} and her husband was {total_remaining}+{used_items}={total_items} {obj}.\nSince the husband bought a package of {package_husband} {obj}, then {name}'s package contained {total_items}-{package_husband}={package_size} {obj}.\n#### {package_size}"

    return {
        'question': question,
        'answer': str(package_size),
        'answer_cot': answer_cot,
        'answer_value': package_size,
        'variables': {
            'name': name,
            'event': event,
            'food': food, 
            'obj': obj,
            'husband_package': package_husband,
            'used_items': used_items,
            'remaining': total_remaining,
            'total': total_items,
            'package_size': package_size
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['Emma', 'Olivia', 'Sophia', 'Isabella', 'Ava', 'Mia', 'Charlotte']
    events = ['lunch party', 'birthday party', 'potluck party', 'baby shower', 'game night']
    foods = ['roast chicken', 'grilled salmon', 'beef stew', 'vegetable lasagna',
            'stuffed peppers', 'shrimp scampi', 'creme brulee'] 
    objects = ['spoons', 'forks', 'plates']

    name = rng.choice(names)
    event = rng.choice(events)
    food = rng.choice(foods)
    obj = rng.choice(objects)

    package_husband = int(rng.randint(5, int(20 * difficulty)))
    used_items = int(rng.randint(3, int(10 * difficulty)))
    
    # Calculate total_remaining to satisfy conditions
    package_size = int(rng.randint(10, int(30 * difficulty)))
    total_items = package_size + package_husband
    total_remaining = total_items - used_items

    result = generate_from_variables(name, event, food, obj,
                                   package_husband, used_items, total_remaining)
    
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
    return generate_from_variables("Julia", "dinner party", "stew", "spoons",
                                 5, 3, 12)
