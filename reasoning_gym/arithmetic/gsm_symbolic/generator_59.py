from random import Random
from typing import Dict, Any

def generate_from_variables(time_per_interval: int, distance_per_interval: int, total_distance: int) -> Dict[str, Any]:
    intervals = total_distance // distance_per_interval
    total_time = intervals * time_per_interval
    
    question = f"A fog bank rolls in from the ocean to cover a city. It takes {time_per_interval} minutes to cover every {distance_per_interval} miles of the city. If the city is {total_distance} miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the whole city?"
    
    answer_cot = f"The city will be covered in {total_distance} / {distance_per_interval} = {intervals} intervals of {time_per_interval} minutes.\nThus, it will take {intervals} * {time_per_interval} = {total_time} minutes for the fog to cover the whole city.\n#### {total_time}"
    
    return {
        'question': question,
        'answer': str(total_time),
        'answer_cot': answer_cot,
        'answer_value': total_time,
        'variables': {
            'time_per_interval': time_per_interval,
            'distance_per_interval': distance_per_interval,
            'total_distance': total_distance,
            'intervals': intervals
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    # Generate random values scaled by difficulty
    distance_per_interval = int(rng.randint(2, int(100 * difficulty)))
    time_per_interval = int(rng.randint(2, int(500 * difficulty)))
    
    # Ensure total distance is divisible by distance_per_interval
    num_intervals = rng.randint(2, int(20 * difficulty))
    total_distance = distance_per_interval * num_intervals
    
    # Ensure total_distance is in valid range
    while total_distance > 100:
        num_intervals = rng.randint(2, int(20 * difficulty))
        total_distance = distance_per_interval * num_intervals

    result = generate_from_variables(time_per_interval, distance_per_interval, total_distance)
    
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
    return generate_from_variables(10, 3, 42)
