from random import Random
from typing import Dict, Any

def generate_from_variables(item: str, n1: int, c1: str, c2: str, c3: str, p: int) -> Dict[str, Any]:
    more_cards = int(p/100 * n1)
    n2 = n1 + more_cards
    n3 = n1 + n2
    total = n3 + n3
    
    question = f"In a set of {item}'s cards, there are {n1} {c1} cards, and {p}% more {c2} cards. {c3} cards are as many as the sum of {c1} and {c2} cards. How many cards of all mentioned colors are there?"

    answer_cot = f"There are {p}/100 * {n1} = {more_cards} more {c2} cards than {c1} cards.\n" \
                 f"Which means there are {n1} + {more_cards} = {n2} {c2} cards.\n" \
                 f"{c3} cards make up to {n1} + {n2} = {n3} cards.\n" \
                 f"So in total, there are {n3} + {n3} = {total} cards of different colors.\n" \
                 f"#### {total}"

    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'item': item,
            'n1': n1,
            'c1': c1,
            'c2': c2, 
            'c3': c3,
            'p': p,
            'more_cards': more_cards,
            'total': total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    items = ["magician", "artist", "chef", "scientist", "athlete"]
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    
    item = rng.choice(items)
    c1, c2, c3 = rng.sample(colors, 3)
    
    n1 = int(rng.randint(20, int(81 * difficulty)))
    
    # Generate p ensuring division results in integer
    while True:
        p = int(rng.randint(20, min(90, int(100 * difficulty))))
        if (p/100 * n1).is_integer():
            break
            
    result = generate_from_variables(item, n1, c1, c2, c3, p)
    
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
    return generate_from_variables("magician", 15, "red", "green", "yellow", 60)
