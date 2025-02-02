from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, mult: int, n: int) -> Dict[str, Any]:
    n_mult = n * mult
    daily_total = n + n_mult
    weekly_total = daily_total * 7
    
    question = f"{name1} operates the cash register exactly {mult} times as fast as her less-experienced colleague {name2}. Daily, {name2} processes {n} customers. What is the total weekly production for the two if they work all days of the week?"
    
    answer_cot = f"While {name2} is processing {n} orders in a day, {name1} processes {n} orders/day * {mult} = {n_mult} orders/day.\n" \
                 f"In a day, they process {n_mult} orders/day + {n} orders/day = {daily_total} orders together.\n" \
                 f"The total number of orders the two processes in a week is {daily_total} orders/day * 7 days/week = {weekly_total} orders\n" \
                 f"#### {weekly_total}"

    return {
        'question': question,
        'answer': str(weekly_total),
        'answer_cot': answer_cot,
        'answer_value': weekly_total,
        'variables': {
            'name1': name1,
            'name2': name2,
            'multiplier': mult,
            'base_rate': n,
            'fast_rate': n_mult,
            'daily_total': daily_total,
            'weekly_total': weekly_total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ["Julie", "Sarah", "Emma", "Sophia", "Olivia", "Isabella", "Mia", "Charlotte"]
    multi_times = [2, 3, 4]
    
    name1, name2 = rng.sample(names_female, 2)
    mult = rng.choice(multi_times)
    n = int(rng.randint(30, int(100 * difficulty)))
    
    # Ensure conditions are met
    while not (n * mult).is_integer() or not ((n + n * mult) * 7).is_integer():
        n = int(rng.randint(30, int(100 * difficulty)))
    
    result = generate_from_variables(name1, name2, mult, n)
    
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
    return generate_from_variables("Julie", "Jewel", 2, 50)
