from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, food1: str, food2: str, mult: int, n: int, 
                          m: int, k: int) -> Dict[str, Any]:
    initial_food1 = n * mult  # initial corn
    initial_total = initial_food1 + n  # initial total
    bought_food1 = m - k  # bought corn
    bought_total = bought_food1 + m  # total bought
    final_total = initial_total + bought_total  # final total

    question = f"At {name}'s house, there is {mult} times as much {food1} as {food2}. He has a total of {n} {food2} in his house. {name} bought {m} more {food2} at the store and {k} fewer {food1} than the number of {food2}. Find the combined total of the number of {food1} and {food2} {name} has in the house?"
    
    answer_cot = f"Before buying any {food1} and {food2}, {name} had {mult} times as many {food1} as {food2}, which is {n} {food2} * {mult} {food1}/{food2} = {initial_food1} {food1}\nThe total number of {food1} and {food2} that {name} had before is {initial_food1} {food1} + {n} {food2} = {initial_total} items\nWhen he bought {k} fewer {food1} than {food2}, he bought {m} {food1} - {k} {food1} = {bought_food1} {food1}\nIn total, he bought {bought_food1} {food1} + {m} {food2} = {bought_total} items\nAfter the purchases, {name} has {initial_total} items + {bought_total} items = {final_total} total {food1} and {food2} combined.\n#### {final_total}"

    return {
        'question': question,
        'answer': str(final_total),
        'answer_cot': answer_cot,
        'answer_value': final_total,
        'variables': {
            'name': name,
            'food1': food1,
            'food2': food2,
            'multiplier': mult,
            'initial_amount': n,
            'bought_amount': m,
            'difference': k,
            'initial_total': initial_total,
            'bought_total': bought_total,
            'final_total': final_total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Allan", "John", "Michael", "David", "James", "Robert", "William"]
    foods = ["corn", "apple", "banana", "orange", "pear", "grape", "fig", 
            "persimmon", "plum", "kiwi"]
    multipliers = ["twice", "three times", "four times"]

    name = rng.choice(names)
    food1, food2 = rng.sample(foods, 2)
    mult = rng.randint(2, int(4 * difficulty))
    
    n = int(rng.randint(20, int(100 * difficulty)))
    m = int(rng.randint(30, int(100 * difficulty)))
    k = int(rng.randint(10, min(m, int(50 * difficulty))))

    result = generate_from_variables(name, food1, food2, mult, n, m, k)
    
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
    return generate_from_variables("Allan", "corn", "cannolis", 2, 40, 60, 40)
