from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, game1: str, game2: str, period: str,
                          time1: int, time2: int, num1: int, num2: int) -> Dict[str, Any]:
    total_time1 = time1 * num1
    total_time2 = time2 * num2
    total_time = total_time1 + total_time2

    question = f"It takes {name} {time1} minutes to finish a {game1} and {time2} minutes to finish a {game2}. Over the {period} she solved {num1} {game1}s and {num2} {game2}s. How much time did she spend playing these games?"

    answer_cot = f"It takes {time1} minutes to complete a {game1} and she completed {num1} for a total of {time1}*{num1} = {total_time1} minutes\n" \
                 f"It takes {time2} minutes to complete a {game2} and she completed {num2} for a total of {time2}*{num2} = {total_time2} minutes\n" \
                 f"She spent {total_time1} minutes on {game1}s and {total_time2} minutes on {game2}s for a total of {total_time1}+{total_time2} = {total_time} minutes\n" \
                 f"#### {total_time}"

    return {
        'question': question,
        'answer': str(total_time),
        'answer_cot': answer_cot,
        'answer_value': total_time,
        'variables': {
            'name': name,
            'game1': game1,
            'game2': game2,
            'period': period,
            'time1': time1,
            'time2': time2,
            'num1': num1,
            'num2': num2,
            'total_time1': total_time1,
            'total_time2': total_time2
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte"]
    games = ["word puzzle", "jigsaw puzzle", "chess puzzle", "riddle", "brain teaser"]
    periods = ["weekend", "vacation", "holiday", "day off", "free time"]
    
    name = rng.choice(names)
    game1, game2 = rng.sample(games, 2)
    period = rng.choice(periods)
    
    time1 = int(rng.randint(5, int(30 * difficulty)))
    time2 = int(rng.randint(3, int(20 * difficulty)))
    while time2 >= time1:  # ensure time1 > time2
        time1 = int(rng.randint(5, int(30 * difficulty)))
        time2 = int(rng.randint(3, int(20 * difficulty)))
        
    num1 = int(rng.randint(2, int(10 * difficulty)))
    num2 = int(rng.randint(4, int(15 * difficulty)))
    
    result = generate_from_variables(name, game1, game2, period, time1, time2, num1, num2)
    
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
    return generate_from_variables("Carmen", "crossword puzzle", "sudoku puzzle", 
                                 "weekend", 10, 5, 3, 8)
