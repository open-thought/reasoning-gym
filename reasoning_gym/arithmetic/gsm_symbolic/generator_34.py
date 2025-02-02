from random import Random
from typing import Dict, Any

def generate_from_variables(event: str, item: str, family: str, n: int, m: int, total: int) -> Dict[str, Any]:
    twins_total = 2 * n
    remaining = total - twins_total
    friends_found = remaining - m
    
    question = f"The {event} team hid {total} {item}. The {family} twins each found {n} {item}. All the other {item} except {m} were found by their friends. How many {item} did the friends find?"
    
    answer_cot = f"The {family} twins found, {n} * 2 = {twins_total} {item}.\nThe number that remained hidden was {total} - {twins_total} = {remaining} {item}\nSince {m} {item} were not found, this means the friends found {remaining} - {m} = {friends_found} {item}\n#### {friends_found}"
    
    return {
        'question': question,
        'answer': str(friends_found),
        'answer_cot': answer_cot,
        'answer_value': friends_found,
        'variables': {
            'event': event,
            'item': item,
            'family': family,
            'items_per_twin': n,
            'unfound_items': m,
            'total_items': total,
            'twins_total': twins_total,
            'friends_found': friends_found
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    events = ["Halloween candy hunt", "Treasure hunt", "Scavenger hunt", "Charity fundraiser"]
    items = ["eggs", "treats", "toys", "coins", "tokens", "balls", "candies", "goodies"]
    families = ["Johnson", "Williams", "Mirzakhani", "Lopez", "Garcia", "Lee"]
    
    event = rng.choice(events)
    item = rng.choice(items)
    family = rng.choice(families)
    
    total = int(rng.randint(50, int(201 * difficulty)) // 10 * 10)
    n = int(rng.randint(10, int(51 * difficulty)))
    m = int(rng.randint(5, int(21 * difficulty)))
    
    # Ensure conditions are met
    while 2 * n + m >= total:
        n = int(rng.randint(10, int(51 * difficulty)))
        m = int(rng.randint(5, int(21 * difficulty)))
    
    result = generate_from_variables(event, item, family, n, m, total)
    
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
    return generate_from_variables("Easter egg hunt", "eggs", "Smith", 30, 10, 100)
