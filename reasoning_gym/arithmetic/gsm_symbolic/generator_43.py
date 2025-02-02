from random import Random
from typing import Dict, Any

def generate_from_variables(person1: str, item: str, n: int, relation: str, k: int) -> Dict[str, Any]:
    other_amount = n - k
    total = n + other_amount
    
    question = f"A {person1} has {n} {item}s. His {relation} has {k} fewer {item}s than he has. How many {item}s do they have together?"
    
    answer_cot = f"His {relation} has {n} - {k} = {other_amount} {item}s.\nTogether, they have {n} + {other_amount} = {total} {item}s.\n#### {total}"
    
    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'person1': person1,
            'item': item,
            'first_amount': n,
            'relation': relation,
            'difference': k,
            'second_amount': other_amount,
            'total': total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    people = ["student", "boy", "child", "kid"]
    items = ["marble", "sticker", "toy", "book", "pencil"] 
    relations = ["sister", "brother", "friend", "cousin"]
    
    person1 = rng.choice(people)
    item = rng.choice(items)
    relation = rng.choice(relations)
    
    n = int(rng.randint(5, int(21 * difficulty)))
    k = int(rng.randint(2, min(n-1, int(10 * difficulty))))
    
    result = generate_from_variables(person1, item, n, relation, k)
    
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
    return generate_from_variables("boy", "card", 5, "brother", 3)
