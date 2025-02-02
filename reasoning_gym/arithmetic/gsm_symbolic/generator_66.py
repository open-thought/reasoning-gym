from random import Random
from typing import Dict, Any
from fractions import Fraction

def generate_from_variables(name: str, weekdays: list, hour1: int, hour2: int, hour3: int,
                          min1: int, min2: int, total_hours: int, num_wed_episodes: int) -> Dict[str, Any]:
    
    mon, tue, wed, thu, fri = weekdays
    
    question = f"{name} watches TV after he finishes his homework every night. On {mon} and {tue}, he watched a {hour1}-hour episode of his favorite show each night. On {wed}, he watched a few episodes of a {min1}-minute show. On {thu}, he finished homework early and watched a {hour2}-hour episode and a {min2}-minute show. On {fri}, he got to stay up late for the weekend, so he watched two {hour3}-hour episodes. If he watched {total_hours} hours of TV in all, how many {min1}-minute episodes did he watch on {wed}?"

    answer_cot = f"Let {wed[0]} be the number of episodes he watched on {wed}.\n" \
                 f"After {mon}, he had {total_hours} - {hour1} = {total_hours-hour1} hours of TV left.\n" \
                 f"After {tue}, he had {total_hours-hour1} - {hour1} = {total_hours-2*hour1} hours of TV left.\n" \
                 f"After {thu}, he had {total_hours-2*hour1} - {hour2} - {Fraction(min2,60)} = {total_hours-2*hour1-hour2-Fraction(min2,60)} hours of TV left.\n" \
                 f"After {fri}, he had {total_hours-2*hour1-hour2-Fraction(min2,60)} - {2*hour3} = {total_hours-2*hour1-hour2-Fraction(min2,60)-2*hour3} hours of TV left.\n" \
                 f"Each {min1}-minute episode is {Fraction(min1,60)} hour.\n" \
                 f"Thus, {wed[0]} = {num_wed_episodes} episodes.\n#### {num_wed_episodes}"

    return {
        'question': question,
        'answer': str(num_wed_episodes),
        'answer_cot': answer_cot,
        'answer_value': num_wed_episodes,
        'variables': {
            'name': name,
            'weekdays': weekdays,
            'hour1': hour1,
            'hour2': hour2,
            'hour3': hour3,
            'min1': min1,
            'min2': min2,
            'total_hours': total_hours,
            'num_wed_episodes': num_wed_episodes
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    name = rng.choice(names)
    weekdays_sample = weekdays.copy()  # Keep original order for this problem
    
    hour1 = int(rng.randint(3, int(7 * difficulty)))
    hour2 = int(rng.randint(2, int(7 * difficulty)))
    hour3 = int(rng.randint(2, int(6 * difficulty)))
    
    min1 = int(rng.randint(1, int(12 * difficulty))) * 5  # Ensure divisible by 5
    min2 = int(rng.randint(1, int(11 * difficulty))) * 5  # Ensure divisible by 5
    
    # Calculate num_wed_episodes to ensure total_hours works out
    num_wed_episodes = int(rng.randint(1, int(8 * difficulty)))
    
    # Calculate total hours from all components
    total_hours = 2*hour1 + hour2 + min2/60 + 2*hour3 + (num_wed_episodes * min1/60)
    
    result = generate_from_variables(name, weekdays_sample, hour1, hour2, hour3,
                                   min1, min2, total_hours, num_wed_episodes)
    
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
    return generate_from_variables("Frankie", 
                                 ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                                 1, 1, 1, 30, 30, 7, 3)
