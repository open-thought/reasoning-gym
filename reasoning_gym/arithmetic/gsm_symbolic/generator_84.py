from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, miles: str, time_cold: int, extra_time: int, 
                          multiplier: float, distance: int) -> Dict[str, Any]:
    time_warm = extra_time + multiplier * time_cold
    time_cold_total = distance * time_cold
    time_warm_total = distance * time_warm
    time_difference = time_warm_total - time_cold_total

    question = f"When the water is cold {name} swims a {miles} in {time_cold} minutes. When the water is warm {name} swims a {miles} in {extra_time} minutes more than {multiplier:.0f} times as long. How much longer does {name} take to swim {distance} {miles}s on a hot day than a cold day?"
    
    answer_cot = f"Cold water {miles} = {time_cold} minutes\n" \
                 f"Warm water {miles} = {extra_time}+{multiplier:.0f}({time_cold})={time_warm} minutes\n" \
                 f"{distance} {miles}s in cold water: {distance}({time_cold})={time_cold_total} minutes\n" \
                 f"{distance} {miles}s in warm water: {distance}({time_warm})={time_warm_total} minutes\n" \
                 f"{name} takes {time_warm_total}-{time_cold_total}={time_difference} minutes longer\n" \
                 f"#### {time_difference}"

    return {
        'question': question,
        'answer': str(time_difference),
        'answer_cot': answer_cot,
        'answer_value': time_difference,
        'variables': {
            'name': name,
            'unit': miles,
            'time_cold': time_cold,
            'extra_time': extra_time,
            'multiplier': multiplier,
            'distance': distance,
            'time_warm': time_warm,
            'time_cold_total': time_cold_total,
            'time_warm_total': time_warm_total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Ray", "Jim", "Bob", "Tom", "Mike", "John", "Steve"]
    units = ["mile", "kilometer"]
    
    name = rng.choice(names)
    unit = rng.choice(units)
    
    time_cold = int(rng.randint(10, int(50 * difficulty)))
    extra_time = int(rng.randint(1, int(10 * difficulty)))
    multiplier = 2.0  # "twice" specified in original
    distance = int(rng.randint(2, int(10 * difficulty)))
    
    # Check conditions
    while time_cold >= 60 or \
          extra_time + multiplier * time_cold >= 60 or \
          distance * (extra_time + multiplier * time_cold) - distance * time_cold <= 0:
        time_cold = int(rng.randint(10, int(50 * difficulty)))
        extra_time = int(rng.randint(1, int(10 * difficulty)))
        distance = int(rng.randint(2, int(10 * difficulty)))
    
    result = generate_from_variables(name, unit, time_cold, extra_time, multiplier, distance)
    
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
    return generate_from_variables("Ray", "mile", 16, 2, 2.0, 3)
