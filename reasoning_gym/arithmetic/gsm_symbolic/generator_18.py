from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, game: str, n1: int, n2: int, frac: float) -> Dict[str, Any]:
    score2 = int(frac * n1 + n2)
    total = n1 + score2

    question = f"{name1} scored {n1} points in one game of {game}. {name2} scored {n2} more than {frac:.0%} as many as {name1}. How many points did {name1} and {name2} have in total?"

    answer_cot = f"{name1} = {n1} points\n{name2} = {frac} * {n1} + {n2} = {score2} points\n{n1} + {score2} = {total} points\nTogether, {name1} and {name2} scored {total} points.\n#### {total}"
    
    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'name1': name1,
            'name2': name2,
            'game': game,
            'score1': n1,
            'bonus': n2,
            'fraction': frac,
            'score2': score2,
            'total_score': total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles", 
             "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"]
    games = ["bowling", "darts", "archery", "basketball", "tennis"]
    fractions = [0.5] # Could add more fractions if needed
    
    name1, name2 = rng.sample(names, 2)
    game = rng.choice(games)
    frac = rng.choice(fractions)
    
    n1 = int(rng.randint(200, int(500 * difficulty)))
    n2 = int(rng.randint(5, int(50 * difficulty)))
    
    # Ensure fraction calculation results in integer
    while not float(frac * n1).is_integer():
        n1 = int(rng.randint(200, int(500 * difficulty)))

    result = generate_from_variables(name1, name2, game, n1, n2, frac)
    
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
    return generate_from_variables("Pierson", "Nikita", "bowling", 278, 11, 0.5)
