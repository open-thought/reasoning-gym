from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, day1: str, day2: str, day3: str, 
                          time1: int, time2: int, mult: int) -> Dict[str, Any]:
    combined_time = time1 + time2
    target_time = combined_time * mult
    
    question = f"On {day3}, {name} wants to exercise for {mult} the amount of time he did on {day2} and {day1} combined. On {day1} he exercised for {time1} minutes. On {day2} he exercised for {time2} minutes. How many minutes does he have to exercise on {day3} to reach his goal?"
    
    answer_cot = f"On {day1} and {day2} he exercised a total of {combined_time} minutes because {time1} + {time2} = {combined_time}\nOn {day3} he has to exercise for {target_time} minutes because {combined_time} x {mult} = {target_time}\n#### {target_time}"

    return {
        'question': question,
        'answer': str(target_time),
        'answer_cot': answer_cot,
        'answer_value': target_time,
        'variables': {
            'name': name,
            'day1': day1,
            'day2': day2,
            'day3': day3,
            'time1': time1,
            'time2': time2,
            'multiplier': mult,
            'combined_time': combined_time,
            'target_time': target_time
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Peter", "John", "Michael", "David", "James", "Robert", "William"]
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    multipliers = [2, 3, 4]
    
    name = rng.choice(names)
    day1, day2, day3 = rng.sample(weekdays, 3)
    mult = rng.choice(multipliers)
    
    time1 = int(rng.randint(10, int(60 * difficulty)))
    time2 = int(rng.randint(10, int(60 * difficulty)))
    
    # Check conditions
    while (time1 + time2) <= 0 or ((time1 + time2) * mult / 60) >= 14:
        time1 = int(rng.randint(10, int(60 * difficulty)))
        time2 = int(rng.randint(10, int(60 * difficulty)))
        
    result = generate_from_variables(name, day1, day2, day3, time1, time2, mult)
    
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
    return generate_from_variables("Peter", "Sunday", "Monday", "Tuesday", 23, 16, 2)
