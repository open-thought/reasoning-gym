from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, objects: str, n: int, obstacle: str, 
                          frac: float, k: int, fake_num: int, fake_object: str) -> Dict[str, Any]:
    
    dropped = int(n * frac)
    remaining = n - dropped
    found = k
    after_finding = remaining + found
    final = after_finding - fake_num
    
    question = f"{name} has a bag of {objects} with {n} inside. He tripped over {obstacle} while carrying it and dropped {dropped} of them. He scrambled to search for them but only came up with {k}. When he went back home, he inspected the {objects} further. {fake_num} of them he picked up wasn't a {objects}, but actually {fake_object} so he got rid of it. How many {objects} did {name} end up with?"
    
    answer_cot = f"{name} dropped his {objects} and was left with {n}*{1-frac}={remaining} {objects}.\n" \
                 f"He searched and found some of his lost {objects}, getting him back to {remaining}+{k}={after_finding} {objects}.\n" \
                 f"He went home and removed {fake_object}, leaving him with {after_finding}-{fake_num}={final} {objects}.\n" \
                 f"#### {final}"
    
    return {
        'question': question,
        'answer': str(final),
        'answer_cot': answer_cot,
        'answer_value': final,
        'variables': {
            'name': name,
            'objects': objects,
            'initial_count': n,
            'obstacle': obstacle,
            'fraction_dropped': frac,
            'found_count': k,
            'fake_count': fake_num,
            'fake_object': fake_object,
            'remaining': remaining,
            'after_finding': after_finding,
            'final_count': final
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
    objects = ["marbles", "coins", "buttons", "beads", "pebbles"]
    obstacles = ["rock", "stick", "toy", "root"]
    fake_objects = ["buttons", "coins", "pebbles", "beads"]
    fractions = [0.5, 0.25, 0.75]
    
    name = rng.choice(names)
    obj = rng.choice(objects)
    obstacle = rng.choice(obstacles)
    fake_object = rng.choice([x for x in fake_objects if x != obj])
    frac = rng.choice(fractions)
    
    n = int(rng.randrange(10, int(101 * difficulty), 2))
    fake_num = int(rng.randint(2, min(10, int(n * frac))))
    k = int(rng.randint(fake_num + 1, min(int(n * frac), int(20 * difficulty))))
    
    # Ensure conditions are met
    while not (isinstance(n * frac, int) and k < n * frac and k > fake_num):
        n = int(rng.randrange(10, int(101 * difficulty), 2))
        k = int(rng.randint(fake_num + 1, min(int(n * frac), int(20 * difficulty))))
    
    result = generate_from_variables(name, obj, n, obstacle, frac, k, fake_num, fake_object)
    
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
    return generate_from_variables("Brendan", "marbles", 10, "pebble", 0.5, 3, 1, "bead")
