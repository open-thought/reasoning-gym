from random import Random
from typing import Dict, Any

def generate_from_variables(fruit1: str, fruit2: str, n1: int, n2: int, 
                          frac1: float, frac2: float, spill: int, total: int) -> Dict[str, Any]:
    n1_after_spill = n1 - spill
    water_fruit1 = (n1_after_spill * frac1) 
    water_fruit2 = (n2 * frac2)
    total_water = water_fruit1 + water_fruit2
    
    question = f"I have {n1} liters of {fruit1} drink that are {frac1} water and I wish to add it to {n2} liters of {fruit2} drink that is {frac2} water. But as I pour it, I spill {spill} liter of the {fruit1} drink. How much water is in the remaining {total} liters?"

    answer_cot = f"There are {n2} x {frac2} = {water_fruit2} liters of water from the {n2} liters {fruit2} drink.\nAfter {spill} liter of {fruit1} drink was spilled, there were {n1} - {spill} = {n1_after_spill} liters of {fruit1} drink left.\nOut of the {n1_after_spill} liters, {n1_after_spill} x {frac1} = {water_fruit1} liters are water.\nThus, there are a total of {water_fruit2} + {water_fruit1} = {total_water} liters of water out of the {total} liters.\n#### {int(total_water)}"

    return {
        'question': question,
        'answer': str(int(total_water)),
        'answer_cot': answer_cot,
        'answer_value': int(total_water),
        'variables': {
            'fruit1': fruit1,
            'fruit2': fruit2,
            'initial_amount1': n1,
            'initial_amount2': n2,
            'water_fraction1': frac1,
            'water_fraction2': frac2,
            'spilled_amount': spill,
            'total_volume': total,
            'water_content1': water_fruit1,
            'water_content2': water_fruit2
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    fruits = ["apple", "grape", "mango", "peach", "lemon"]
    fractions = {"two-thirds": 2/3, "three-fifths": 3/5, "three-quarters": 3/4, 
                "one-half": 1/2, "four-fifths": 4/5}
    
    fruit1, fruit2 = rng.sample(fruits, 2)
    frac1 = rng.choice(list(fractions.values()))
    frac2 = rng.choice(list(fractions.values()))
    
    n1 = int(rng.randint(9, int(21 * difficulty)))
    n2 = int(rng.randint(12, int(31 * difficulty)))
    spill = int(rng.randint(3, min(7, n1)))
    
    # Ensure conditions are met
    while not (n1 + n2 - spill > 0 and 
              (n2 * frac2).is_integer() and
              ((n1 - spill) * frac1).is_integer()):
        n1 = int(rng.randint(9, int(21 * difficulty)))
        n2 = int(rng.randint(12, int(31 * difficulty)))
        spill = int(rng.randint(3, min(7, n1)))
    
    total = n1 + n2 - spill
    
    result = generate_from_variables(fruit1, fruit2, n1, n2, frac1, frac2, spill, total)
    
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
    return generate_from_variables("orange", "pineapple", 10, 15, 2/3, 3/5, 1, 24)
