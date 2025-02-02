from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, n1: int, n2: int, 
                          k1: int, k2: int) -> Dict[str, Any]:
    total_puppies = n1 + n2
    spotted_puppies = k1 + k2
    percentage = int(100 * spotted_puppies / total_puppies)
    
    question = f"{name1}'s dog has {n1} puppies, {k1} of which have spots. {name2}'s dog has {n2} puppies, {k2} of which have spots. What percentage of all the puppies have spots?"
    
    answer_cot = f"First find the total number of puppies: {n1} puppies + {n2} puppies = {total_puppies} puppies\n" \
                 f"Then find the total number of puppies with spots: {k1} puppies + {k2} puppies = {spotted_puppies} puppies\n" \
                 f"Then divide the number of spotted puppies by the total number of puppies and multiply by 100% to find the percentage of puppies with spots: {spotted_puppies} puppies / {total_puppies} puppies * 100% = {percentage}%\n" \
                 f"#### {percentage}"

    return {
        'question': question,
        'answer': str(percentage),
        'answer_cot': answer_cot,
        'answer_value': percentage,
        'variables': {
            'name1': name1,
            'name2': name2,
            'puppies1': n1,
            'puppies2': n2,
            'spotted1': k1,
            'spotted2': k2,
            'total_puppies': total_puppies,
            'total_spotted': spotted_puppies
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Jennifer", "Michael", "Christopher", "Jessica", "Matthew", "Ashley", 
             "Joshua", "Amanda", "Daniel", "David", "James", "Robert", "John", "Joseph"]
    
    name1, name2 = rng.sample(names, 2)
    
    # Scale ranges by difficulty but ensure values remain integers
    n1 = int(rng.randrange(950, int(1050 * difficulty), 5))
    n2 = int(rng.randrange(400, int(650 * difficulty), 5))
    k1 = int(rng.randrange(170, int(290 * difficulty), 10))
    k2 = int(rng.randrange(120, int(170 * difficulty), 10))
    
    # Ensure conditions are met
    while (k1 + k2) >= (n1 + n2) or (n1 + n2) % (k1 + k2) != 0:
        n1 = int(rng.randrange(950, int(1050 * difficulty), 5))
        n2 = int(rng.randrange(400, int(650 * difficulty), 5))
        k1 = int(rng.randrange(170, int(290 * difficulty), 10))
        k2 = int(rng.randrange(120, int(170 * difficulty), 10))
    
    result = generate_from_variables(name1, name2, n1, n2, k1, k2)
    
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
    return generate_from_variables("Jennifer", "Brandon", 8, 12, 3, 4)
