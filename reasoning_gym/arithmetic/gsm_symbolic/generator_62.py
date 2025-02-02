from random import Random
from typing import Dict, Any

def generate_from_variables(item: str, num_slices: int, name1: str, name2: str, 
                          slices_per_day: int, multiplier: int, unit: str) -> Dict[str, Any]:
    
    second_person_slices = slices_per_day * multiplier
    total_daily_slices = slices_per_day + second_person_slices
    days_lasting = num_slices // total_daily_slices
    
    question = f"A {item} has {num_slices} {unit}. If {name1} can eat {slices_per_day} {unit} a day while {name2} can eat {multiplier} times as much, how many days will the {item} last?"
    
    answer_cot = f"{name2} can eat {slices_per_day} x {multiplier} = {second_person_slices} {unit} a day.\nTogether, {name1} and {name2} can eat {slices_per_day} + {second_person_slices} = {total_daily_slices} {unit} a day.\nSo, a {item} will last for {num_slices}/{total_daily_slices} = {days_lasting} days.\n#### {days_lasting}"

    return {
        'question': question,
        'answer': str(days_lasting),
        'answer_cot': answer_cot,
        'answer_value': days_lasting,
        'variables': {
            'item': item,
            'num_slices': num_slices,
            'name1': name1,
            'name2': name2,
            'slices_per_day': slices_per_day,
            'multiplier': multiplier,
            'second_person_slices': second_person_slices,
            'total_daily_slices': total_daily_slices,
            'unit': unit
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    items = ["pizza", "cake", "pie", "lasagna"]
    units = ["pieces", "portions", "servings"]
    names = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Elijah", "Charlotte", "James"]
    
    item = rng.choice(items)
    unit = rng.choice(units)
    name1, name2 = rng.sample(names, 2)
    
    slices_per_day = int(rng.randint(2, int(6 * difficulty)))
    multiplier = 2  # Using 'twice' as specified in original
    
    # Ensure total is divisible by daily consumption
    daily_total = slices_per_day + (slices_per_day * multiplier)
    num_days = rng.randint(2, int(8 * difficulty))
    num_slices = daily_total * num_days
    
    result = generate_from_variables(item, num_slices, name1, name2, slices_per_day, 
                                   multiplier, unit)
    
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
    return generate_from_variables("loaf of bread", 24, "Abby", "Josh", 2, 2, "slices")
