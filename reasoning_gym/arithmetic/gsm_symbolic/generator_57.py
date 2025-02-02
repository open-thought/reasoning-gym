from random import Random
from typing import Dict, Any

def generate_from_variables(n1: int, sport1: str, sport2: str, sport3: str, 
                          n2: int, n3: int, multiplier: int) -> Dict[str, Any]:
    n_volleyball = n1 * multiplier
    n_soccer = n2 + n3
    total = n1 + n_volleyball + n_soccer
    
    question = f"There are {n1} students playing {sport1} and twice that number playing {sport2}. There are {n2} boys and {n3} girls playing {sport3}. If each student only participates in one group, how many students are there in total?"
    
    answer_cot = f"There are {n1} x {multiplier} = {n_volleyball} students playing {sport2}.\nThere are {n2} + {n3} = {n_soccer} students playing {sport3}.\nIn total there are {n1} + {n_volleyball} + {n_soccer} = {total} students.\n#### {total}"

    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'tennis_players': n1,
            'volleyball_players': n_volleyball,
            'soccer_boys': n2,
            'soccer_girls': n3,
            'total_soccer': n_soccer,
            'total_students': total,
            'sports': [sport1, sport2, sport3]
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    sports = ['basketball', 'badminton', 'table tennis', 'football', 'volleyball']
    sport1, sport2, sport3 = rng.sample(sports, 3)
    
    # Generate numbers based on difficulty
    n1 = int(rng.randint(4, int(21 * difficulty)))
    n2 = int(rng.randint(10, int(31 * difficulty)))
    n3 = int(rng.randint(10, int(31 * difficulty)))
    multiplier = 2  # "twice" that number
    
    # Check condition
    while n1 * multiplier + n2 + n3 > 250:
        n1 = int(rng.randint(4, int(21 * difficulty)))
        n2 = int(rng.randint(10, int(31 * difficulty)))
        n3 = int(rng.randint(10, int(31 * difficulty)))
    
    result = generate_from_variables(n1, sport1, sport2, sport3, n2, n3, multiplier)
    
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
    return generate_from_variables(6, 'tennis', 'volleyball', 'soccer', 16, 22, 2)
