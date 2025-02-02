from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, plants_received: int, plants_per_ledge: int, 
                          num_ledges: int, plants_to_give: int) -> Dict[str, Any]:
    
    initial_plants = plants_per_ledge * num_ledges
    total_plants = initial_plants + plants_received 
    plants_given = num_ledges * plants_to_give
    remaining_plants = total_plants - plants_given

    question = f"{name} is an avid gardener. Yesterday, she received {plants_received} new potted plants from her favorite plant nursery. She already has {plants_per_ledge} potted plants on each of the {num_ledges} window ledges of her large country home. Feeling generous, she has decided that she will give {plants_to_give} potted plant from each ledge to friends and family tomorrow. How many potted plants will {name} remain with?"

    answer_cot = f"Yesterday, before receiving the plants, {name} had {num_ledges}*{plants_per_ledge} = {initial_plants} potted plants\nAfter receiving an additional {plants_received} plants, she therefore had a total of {initial_plants} + {plants_received} = {total_plants} potted plants\nTomorrow, {name}'s plant giveaway will be {num_ledges}*{plants_to_give} = {plants_given} potted plants.\nShe will therefore remain with {total_plants} - {plants_given} = {remaining_plants} potted plants.\n#### {remaining_plants}"

    return {
        'question': question,
        'answer': str(remaining_plants),
        'answer_cot': answer_cot,
        'answer_value': remaining_plants,
        'variables': {
            'name': name,
            'plants_received': plants_received,
            'plants_per_ledge': plants_per_ledge,
            'num_ledges': num_ledges,
            'plants_to_give': plants_to_give,
            'initial_plants': initial_plants,
            'total_plants': total_plants,
            'plants_given': plants_given,
            'remaining_plants': remaining_plants
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Mary", "Emma", "Sophia", "Isabella", "Olivia", "Ava", "Mia"]
    
    name = rng.choice(names)
    plants_received = int(rng.randint(20, int(50 * difficulty)))
    plants_per_ledge = int(rng.randint(7, int(13 * difficulty)))
    num_ledges = int(rng.randint(50, int(70 * difficulty)))
    plants_to_give = int(rng.randint(3, int(8 * difficulty)))
    
    # Ensure condition: w * r + x - w*n > 0
    while (num_ledges * plants_per_ledge + plants_received - num_ledges * plants_to_give) <= 0:
        plants_per_ledge = int(rng.randint(7, int(13 * difficulty)))
        plants_to_give = int(rng.randint(3, int(8 * difficulty)))

    result = generate_from_variables(name, plants_received, plants_per_ledge, 
                                   num_ledges, plants_to_give)
    
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
    return generate_from_variables("Mary", 18, 2, 40, 1)
