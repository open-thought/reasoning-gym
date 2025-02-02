from random import Random
from typing import Dict, Any

def generate_from_variables(park_name: str, unit: str, length1: int, length2: int, 
                          speed1: int, speed2: int) -> Dict[str, Any]:
    time1 = length1 // speed1
    time2 = length2 // speed2
    time_diff = time1 - time2
    
    question = f"The biggest waterslide at {park_name} is {length1} {unit} long, and people slide down at {speed1} {unit}/minute. The second biggest waterslide is {length2} {unit} long, but steeper, so people slide down at {speed2} {unit}/minute. How much longer does it take to ride the biggest slide compared to the second biggest slide?"
    
    answer_cot = f"First find the ride length of the biggest slide: {length1} {unit} / {speed1} {unit}/minute = {time1} minutes\nThen find the ride length of the second biggest slide: {length2} {unit} / {speed2} {unit}/minute = {time2} minutes\nThen subtract the ride length of the second longest slide from the longest slide to find the difference: {time1} minutes - {time2} minutes = {time_diff} minutes\n#### {time_diff}"

    return {
        'question': question,
        'answer': str(time_diff),
        'answer_cot': answer_cot, 
        'answer_value': time_diff,
        'variables': {
            'park_name': park_name,
            'unit': unit,
            'length1': length1,
            'length2': length2,
            'speed1': speed1,
            'speed2': speed2,
            'time1': time1,
            'time2': time2
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    parks = ["Splash World", "Aqua Adventure", "Water Wonderland", "Neptunes Kingdom"]
    units = ['yards', 'meters']
    
    park_name = rng.choice(parks)
    unit = rng.choice(units)
    
    length1 = int(rng.randrange(250, int(401 * difficulty), 10))
    length2 = int(rng.randrange(200, int(301 * difficulty), 10))
    speed1 = int(rng.randrange(40, int(81 * difficulty), 5))
    speed2 = int(rng.randrange(60, int(101 * difficulty), 5))
    
    # Ensure conditions are met
    while (length1 <= length2 or speed2 <= speed1 or 
           length1 % speed1 != 0 or length2 % speed2 != 0 or 
           (length1 // speed1) <= (length2 // speed2)):
        length1 = int(rng.randrange(250, int(401 * difficulty), 10))
        length2 = int(rng.randrange(200, int(301 * difficulty), 10))
        speed1 = int(rng.randrange(40, int(81 * difficulty), 5))
        speed2 = int(rng.randrange(60, int(101 * difficulty), 5))

    result = generate_from_variables(park_name, unit, length1, length2, speed1, speed2)
    
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
    return generate_from_variables("Five Flags", "feet", 300, 240, 60, 80)
