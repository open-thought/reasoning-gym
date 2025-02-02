from random import Random
from typing import Dict, Any
from fractions import Fraction

def generate_from_variables(name: str, unit: str, weight_large: int, weight_medium: int, 
                          weight_small: Fraction, num_large: int, num_medium: int, 
                          num_small: int, total_amount: int) -> Dict[str, Any]:
    
    large_used = num_large * weight_large
    medium_used = num_medium * weight_medium
    small_used = float(num_small * weight_small)
    total_used = large_used + medium_used + small_used
    remaining = total_amount - total_used
    
    question = f"{name} wants to make different sized ice cubes with {total_amount} {unit}s of water. He can make giant cubes that use {weight_large} {unit}s per cube, medium cubes that use {weight_medium} {unit}s, and small cubes that use {weight_small} an {unit}. If he makes {num_large} giant cubes, {num_medium} medium cubes, and {num_small} small cubes, how many {unit}s of water does he have left?"
    
    answer_cot = f"The giant cubes used up {large_used} {unit}s of water because {num_large} times {weight_large} equals {large_used}.\n" \
                 f"The medium cubes used up {medium_used} {unit}s of water because {num_medium} times {weight_medium} equals {medium_used}.\n" \
                 f"The small cubes used up {int(small_used)} {unit}s of water because {num_small} times {weight_small} equals {int(small_used)}.\n" \
                 f"This means that {name} has used up {int(total_used)} {unit}s of water because {large_used} plus {medium_used} plus {int(small_used)} equals {int(total_used)}.\n" \
                 f"{name} has {int(remaining)} {unit}s of water left because {total_amount} minus {int(total_used)} equals {int(remaining)}.\n" \
                 f"#### {int(remaining)}"

    return {
        'question': question,
        'answer': str(int(remaining)),
        'answer_cot': answer_cot,
        'answer_value': int(remaining),
        'variables': {
            'name': name,
            'unit': unit,
            'weight_large': weight_large,
            'weight_medium': weight_medium,
            'weight_small': weight_small,
            'num_large': num_large,
            'num_medium': num_medium,
            'num_small': num_small,
            'total_amount': total_amount,
            'remaining': remaining
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Peter", "John", "Michael", "David", "James", "Robert", "William"]
    units = ["ounce", "gram", "milliliter"]
    
    name = rng.choice(names)
    unit = rng.choice(units)
    
    weight_large = int(rng.randint(7, int(14 * difficulty)))
    weight_medium = int(rng.randint(3, weight_large - 1))
    weight_small = Fraction(1, 2)
    
    num_large = int(rng.randint(2, int(8 * difficulty)))
    num_medium = int(rng.randint(4, int(12 * difficulty)))
    num_small = rng.choice([14, 24, 15])
    
    # Calculate needed total to ensure positive remainder
    used = num_large * weight_large + num_medium * weight_medium + float(num_small * weight_small)
    total_amount = int(used + rng.randint(1, int(10 * difficulty)))

    result = generate_from_variables(name, unit, weight_large, weight_medium, weight_small,
                                   num_large, num_medium, num_small, total_amount)
    
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
    return generate_from_variables("Peter", "ounce", 4, 2, Fraction(1,2), 3, 7, 8, 32)
