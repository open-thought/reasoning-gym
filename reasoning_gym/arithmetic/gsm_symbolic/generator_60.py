from random import Random
from fractions import Fraction
from typing import Dict, Any

def generate_from_variables(name: str, unit: str, total_dist: int, beach_dist: int, 
                          sidewalk_dist: int, speed_mult: int, beach_time: int) -> Dict[str, Any]:
    
    beach_rate = Fraction(beach_dist, beach_time)
    sidewalk_rate = beach_rate * speed_mult
    sidewalk_time = int(sidewalk_dist / sidewalk_rate)
    total_time = beach_time + sidewalk_time

    question = f"{name} walks {total_dist} {unit}s every day on her favorite walking trail, which includes {beach_dist} {unit}s of walking on the beach and {sidewalk_dist} {unit}s of walking on the sidewalk. On the sidewalk, {name} walks at twice the rate of speed that she does on the beach. If {beach_time} minutes of her walk is spent on the beach, how long does it take for her to complete the entire {total_dist}-{unit} walk, in minutes?"

    answer_cot = f"On the beach, {name} walks at a rate of {beach_dist} {unit}s per {beach_time} minutes, or {beach_dist}/{beach_time} = {beach_rate} {unit}s per minute.\nOn the sidewalk, she walks at {speed_mult} times the rate of speed as when she is on the sand, or {speed_mult} * {beach_rate} = {sidewalk_rate} {unit}s per minute.\nTo walk {sidewalk_dist} {unit}s on the sidewalk, it takes her {sidewalk_dist}÷{sidewalk_rate}={sidewalk_time} minutes.\nThus, in total, it takes {name} {beach_time}+{sidewalk_time}={total_time} minutes to walk her favorite route.\n#### {total_time}"

    return {
        'question': question,
        'answer': str(total_time),
        'answer_cot': answer_cot,
        'answer_value': total_time,
        'variables': {
            'name': name,
            'unit': unit,
            'total_distance': total_dist,
            'beach_distance': beach_dist,
            'sidewalk_distance': sidewalk_dist,
            'speed_multiplier': speed_mult,
            'beach_time': beach_time,
            'sidewalk_time': sidewalk_time
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['Emma', 'Sophia', 'Isabella', 'Olivia', 'Ava', 'Mia', 'Emily']
    units = ['mile', 'kilometer', 'block']
    
    name = rng.choice(names)
    unit = rng.choice(units)
    speed_mult = 2  # Fixed as "twice" in question
    
    beach_time = int(rng.randint(40, int(70 * difficulty)))
    beach_dist = int(rng.randint(10, int(20 * difficulty)))
    sidewalk_dist = int(rng.randint(10, int(20 * difficulty)))
    total_dist = beach_dist + sidewalk_dist

    # Ensure mathematical consistency
    while not (beach_dist < beach_time and 
              speed_mult * beach_dist < beach_time and
              beach_time % beach_dist == 0):
        beach_time = int(rng.randint(40, int(70 * difficulty)))
        beach_dist = int(rng.randint(10, int(20 * difficulty)))
        sidewalk_dist = int(rng.randint(10, int(20 * difficulty)))
        total_dist = beach_dist + sidewalk_dist

    result = generate_from_variables(name, unit, total_dist, beach_dist, 
                                   sidewalk_dist, speed_mult, beach_time)
    
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
    return generate_from_variables("Grandma", "mile", 3, 2, 1, 2, 40)
