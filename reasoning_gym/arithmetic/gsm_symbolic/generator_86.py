from random import Random
from typing import Dict, Any
import math

def generate_from_variables(animals: str, unit: str, o1: str, o2: str, o3: str, o4: str,
                          n1: int, n2: int, n3: int, n4: int,
                          w1: int, w2: int, w3: int, w4: int,
                          total: int) -> Dict[str, Any]:
    # Calculate weights
    sugar_weight = n4 * w4
    carrot_weight = n3 * w3  
    hay_weight = n1 * w1
    oat_weight = n2 * w2
    total_weight = sugar_weight + carrot_weight + hay_weight + oat_weight
    trips = total_weight // total

    question = f"A farmer is buying feed for his {animals}. He buys a variety of {o1}, {o2}, {o3} and {o4}. Since {o4} are a rare treat, he only buys {n4} {w4}-{unit} boxes of them for the whole stable. He only wants enough {o3} to feed the {animals} while the vegetables are fresh, so he buys {n3} {w3}-{unit} bags. {o1} is the main diet of his {animals}, so he buys {n1} {w1}-{unit} bales. {o2} are a staple to supplement the {o1}, so he buys {n2} {w2}-{unit} sacks. If his farm truck can carry {total} {unit}s at a time, how many trips does the farmer need to transport all the feed?"

    answer_cot = f"The farmer is buying {n4} * {w4} = {sugar_weight} {unit}s of {o4}.\nHe is buying {n3} * {w3} = {carrot_weight} {unit}s of {o3}.\nHe is buying {n1} * {w1} = {hay_weight} {unit}s of {o1}.\nHe is buying {n2} * {w2} = {oat_weight} {unit}s of {o2}.\nThe weight of all the feed is {sugar_weight} + {carrot_weight} + {hay_weight} + {oat_weight} = {total_weight} {unit}s.\nThus, the farmer needs {total_weight} / {total} = {trips} trips to transport all the feed in his farm truck.\n#### {trips}"

    return {
        'question': question,
        'answer': str(trips),
        'answer_cot': answer_cot,
        'answer_value': trips,
        'variables': {
            'animals': animals,
            'unit': unit,
            'feed_types': [o1, o2, o3, o4],
            'quantities': [n1, n2, n3, n4],
            'weights': [w1, w2, w3, w4],
            'truck_capacity': total,
            'total_weight': total_weight
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    animals = rng.choice(['horses', 'cows', 'sheep', 'pigs', 'alpacas'])
    unit = rng.choice(['pound', 'kilogram'])
    feed_options = ["hay", "corn", "oats", "apples", "wheat"]
    o1, o2, o4 = rng.sample(feed_options, 3)
    o3 = rng.choice(["carrots", "beets", "cucumbers"])
    
    # Scale ranges by difficulty
    n4 = int(rng.randint(4, int(8 * difficulty)))
    n3 = int(rng.randint(11, int(15 * difficulty)))
    n2 = int(rng.randint(15, int(20 * difficulty)))
    n1 = int(rng.randint(31, int(35 * difficulty)))
    
    w4 = int(rng.randint(3, int(8 * difficulty)))
    w3 = int(rng.randint(5, int(10 * difficulty)))
    w2 = int(rng.randint(15, int(20 * difficulty)))
    w1 = int(rng.randint(35, int(45 * difficulty)))
    
    # Ensure weight conditions are met
    while not (w4*n4 < w3*n3 < w2*n2 < w1*n1):
        w4 = int(rng.randint(3, int(8 * difficulty)))
        w3 = int(rng.randint(5, int(10 * difficulty)))
        w2 = int(rng.randint(15, int(20 * difficulty)))
        w1 = int(rng.randint(35, int(45 * difficulty)))

    total_weight = n1*w1 + n2*w2 + n3*w3 + n4*w4
    # Find truck capacity that divides total weight
    total = total_weight // rng.randint(2, 4)
    
    result = generate_from_variables(animals, unit, o1, o2, o3, o4,
                                   n1, n2, n3, n4, w1, w2, w3, w4, total)
    
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
    return generate_from_variables('horses', 'pound', 'hay', 'oats', 'carrots', 'sugar cubes',
                                 42, 20, 4, 2,
                                 75, 65, 12, 1,
                                 2250)
