from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, city: str, celebrity_type: str,
                          vacation_type: str, n1: int, n2: int, s1: int, s2: int, 
                          goal: int) -> Dict[str, Any]:
    signatures_collected = s1 + s2
    signatures_needed = goal - signatures_collected
    
    question = f"{name1} and {name2} are sisters from {city} who love collecting signatures from {celebrity_type}. During their {vacation_type} from school, the sisters spend every afternoon collecting signatures. After {n1} weeks, {name1} and {name2} compare their autograph books, counting up the number of signatures each sister has collected. {name1} has {s1} signatures in her book, and {name2} has {s2}. The sisters have {n2} more weeks of {vacation_type}, and they decide they want to reach {goal} signatures between them by the end of the summer. How many signatures do the sisters need to collect to reach their goal?"
    
    answer_cot = f"{name1} and {name2} have already collected {s1} + {s2} signatures = {signatures_collected} signatures.\nSince their goal is {goal}, they need to collect {goal} - {signatures_collected} signatures. {goal} - {signatures_collected} = {signatures_needed} signatures\n#### {signatures_needed}"
    
    return {
        'question': question, 
        'answer': str(signatures_needed),
        'answer_cot': answer_cot,
        'answer_value': signatures_needed,
        'variables': {
            'name1': name1,
            'name2': name2,
            'city': city,
            'celebrity_type': celebrity_type,
            'vacation_type': vacation_type,
            'weeks_passed': n1,
            'weeks_remaining': n2,
            'signatures1': s1,
            'signatures2': s2,
            'goal': goal,
            'signatures_collected': signatures_collected,
            'signatures_needed': signatures_needed
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ['Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte', 'Carol', 'Jennifer']
    cities = ['Los Angeles', 'New York', 'Chicago', 'Houston', 'Phoenix']
    celebrity_types = ['movie stars', 'athletes', 'musicians', 'politicians', 'authors']
    vacation_types = ['winter break', 'spring break', 'summer break', 'fall break']
    
    name1, name2 = rng.sample(names_female, 2)
    city = rng.choice(cities)
    celebrity_type = rng.choice(celebrity_types)
    vacation_type = rng.choice(vacation_types)
    
    n1 = int(rng.randint(3, int(8 * difficulty)))
    n2 = int(rng.randint(2, int(5 * difficulty))) 
    s1 = int(rng.randint(15, int(40 * difficulty)))
    s2 = int(rng.randint(30, int(60 * difficulty)))
    goal = int(rng.randrange(90, int(150 * difficulty), 5))
    
    # Ensure conditions are met
    while s1 + s2 >= goal:
        s1 = int(rng.randint(15, int(40 * difficulty)))
        s2 = int(rng.randint(30, int(60 * difficulty)))
        
    result = generate_from_variables(name1, name2, city, celebrity_type, vacation_type,
                                   n1, n2, s1, s2, goal)
    
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
    return generate_from_variables('Carol', 'Jennifer', 'Los Angeles', 'celebrities',
                                 'summer break', 5, 3, 20, 44, 100)
