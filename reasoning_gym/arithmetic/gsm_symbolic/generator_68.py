from random import Random
from typing import Dict, Any

def generate_from_variables(n0: int, r: int, d: int, disease: str) -> Dict[str, Any]:
    # Calculate infected people after each day
    day1_new = n0 * r
    day1_total = n0 + day1_new
    
    day2_new = day1_total * r 
    day2_total = day1_total + day2_new
    
    day3_new = day2_total * r
    day3_total = day2_total + day3_new
    
    question = f"A {disease} infects {n0} people. Every day, each infected person infects {r} others. How many people are infected after {d} days?"
    
    answer_cot = f"On the first day, the original {n0} people infect {r} people each, so {n0} * {r} = {day1_new} more people are infected.\n" \
                 f"There are {n0} + {day1_new} = {day1_total} infected people after the first day.\n" \
                 f"On the second day, {day1_total} * {r} = {day2_new} more people are infected.\n" \
                 f"There are {day1_total} + {day2_new} = {day2_total} infected people after the second day.\n" \
                 f"On the third day, {day2_total} * {r} = {day3_new} more people are infected. Therefore, there are {day2_total} + {day3_new} = {day3_total} infected people after three days.\n" \
                 f"#### {day3_total}"

    return {
        'question': question,
        'answer': str(day3_total),
        'answer_cot': answer_cot,
        'answer_value': day3_total,
        'variables': {
            'initial_infected': n0,
            'infection_rate': r,
            'days': d,
            'disease_type': disease
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    diseases = ['virus', 'bacteria', 'parasite', 'infection']
    
    disease = rng.choice(diseases)
    n0 = int(rng.randint(5, int(21 * difficulty)))
    r = int(rng.randint(2, int(8 * difficulty)))
    d = 3  # Fixed at 3 days per problem description
    
    # Check condition: n0 * (r + 1)**d < 20000
    while n0 * (r + 1)**d >= 20000:
        n0 = int(rng.randint(5, int(21 * difficulty)))
        r = int(rng.randint(2, int(8 * difficulty)))
    
    result = generate_from_variables(n0, r, d, disease)
    
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
    return generate_from_variables(10, 6, 3, 'plague')
