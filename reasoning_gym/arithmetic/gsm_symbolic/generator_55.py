from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, place: str, fruit: str, location: str,
                          insect1: str, insect2: str, n: int, frac: str) -> Dict[str, Any]:
    num_insect1 = int(n * 0.5)  # half as many bugs as ants
    total_insects = n + num_insect1
    
    question = f"{name} went to their {place} to pick some {fruit} and found {frac} as many {insect1} as {insect2} in the {location}. If there were {n} {insect2}, calculate the total number of insects in the {location}."
    
    answer_cot = f"If there were {n} {insect2}, the total number of {insect1} in the {location} is {frac} * {n} {insect2} = {num_insect1} {insect1}\nThe total number of insects in the {location} is {num_insect1} {insect1} + {n} {insect2} = {total_insects} insects\n#### {total_insects}"
    
    return {
        'question': question,
        'answer': str(total_insects),
        'answer_cot': answer_cot,
        'answer_value': total_insects,
        'variables': {
            'name': name,
            'place': place,
            'fruit': fruit,
            'location': location,
            'insect1': insect1,
            'insect2': insect2,
            'n': n,
            'frac': frac,
            'num_insect1': num_insect1
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Dax", "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Riley"]
    places = ["orchard", "backyard", "greenhouse", "allotment"] 
    fruits = ["strawberries", "cherries", "blueberries", "raspberries"]
    locations = ["garden", "field", "plot", "patch"]
    insects = ["beetles", "ladybugs", "grasshoppers", "caterpillars", "bees", "wasps"]
    
    name = rng.choice(names)
    place = rng.choice(places)
    fruit = rng.choice(fruits)
    location = rng.choice(locations)
    insect1, insect2 = rng.sample(insects, 2)
    
    n = int(rng.randint(20, int(200 * difficulty)))
    # Ensure n is even for "half as many"
    if n % 2 == 1:
        n += 1
        
    result = generate_from_variables(name, place, fruit, location, insect1, insect2, n, "half")
    
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
    return generate_from_variables("Dax", "farm", "apples", "garden",
                                 "bugs", "ants", 50, "half")
