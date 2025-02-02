from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, food: str, animal1: str, animal2: str, 
                          n1: int, n2: int, k1: int, k2: int, unit: str) -> Dict[str, Any]:
    animal2_amount = 2 * n1 - n2  # Amount per sheep
    animal2_total = k2 * animal2_amount  # Total for sheep
    animal1_total = k1 * n1  # Total for goats
    total = animal1_total + animal2_total

    question = f"{name} is feeding his livestock {food}. Each {animal1} needs {n1} {unit}, and each {animal2} needs {n2} {unit} less than twice the amount each {animal1} needs. If there are {k1} {animal1}s and {k2} {animal2}s, how many {unit} of {food} does {name} need?"

    answer_cot = f"First figure out how much {food} each {animal2} needs: {n1} {unit} * 2 - {n2} = {animal2_amount} {unit}/{animal2}\n" \
                 f"Now figure out how much {food} total the {animal2}s need: {animal2_amount} {unit}/{animal2} * {k2} {animal2} = {animal2_total} {unit}\n" \
                 f"Now figure out how much {food} total the {animal1}s need: {n1} {unit}/{animal1} * {k1} {animal1}s = {animal1_total} {unit}\n" \
                 f"Now add the two amounts of {food} to find the total: {animal2_total} {unit} + {animal1_total} {unit} = {total} {unit}\n#### {total}"

    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'name': name,
            'food': food,
            'animal1': animal1,
            'animal2': animal2,
            'n1': n1,
            'n2': n2,
            'k1': k1,
            'k2': k2,
            'unit': unit,
            'animal2_amount': animal2_amount,
            'animal2_total': animal2_total,
            'animal1_total': animal1_total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["John", "Michael", "David", "James", "Robert", "William", "Richard"]
    foods = ["hay", "grain", "feed", "silage"]
    animals = ["goat", "cow", "horse", "donkey", "llama", "alpaca", "pig", "turkey", "duck"]
    units = ["pounds", "kilograms", "kg"]

    name = rng.choice(names)
    food = rng.choice(foods)
    animal1, animal2 = rng.sample(animals, 2)
    unit = rng.choice(units)

    n1 = int(rng.randint(3, int(15 * difficulty)))
    n2 = int(rng.randint(1, int(10 * difficulty)))
    
    # Ensure 2*n1 - n2 > 0
    while 2*n1 - n2 <= 0:
        n1 = int(rng.randint(3, int(15 * difficulty)))
        n2 = int(rng.randint(1, int(10 * difficulty)))

    k1 = int(rng.randint(10, int(50 * difficulty)))
    k2 = int(rng.randint(10, int(50 * difficulty)))

    result = generate_from_variables(name, food, animal1, animal2, n1, n2, k1, k2, unit)

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
    return generate_from_variables("Nate", "hay", "goat", "sheep", 5, 3, 15, 12, "pounds")
