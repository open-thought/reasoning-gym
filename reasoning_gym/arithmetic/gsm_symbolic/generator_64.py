from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, n1: int, d1: int, n2: int, d2: int) -> Dict[str, Any]:
    first_period = n1 * d1
    second_period = n2 * d2
    total_eggs = first_period + second_period
    dozens = total_eggs // 12
    
    question = f"If {name} eats {n1} eggs a day for {d1} days and then increases it to {n2} eggs a day for {d2} days, how many dozens of eggs will {name} need for {d1+d2} days?"
    
    answer_cot = f"He starts off eating {n1} eggs a day for {d1} days for a total of {n1}*{d1} = {first_period} eggs\n" \
                 f"Then he increases it to {n2} eggs a day for {d2} days for a total of {n2}*{d2} = {second_period} eggs\n" \
                 f"All total he will eat {first_period}+{second_period} = {total_eggs} eggs\n" \
                 f"There are 12 eggs in 1 dozen and he will eat {total_eggs} eggs which is {total_eggs}/12 = {dozens} dozen eggs\n" \
                 f"#### {dozens}"

    return {
        'question': question,
        'answer': str(dozens),
        'answer_cot': answer_cot,
        'answer_value': dozens,
        'variables': {
            'name': name,
            'eggs_per_day_first': n1,
            'days_first': d1,
            'eggs_per_day_second': n2, 
            'days_second': d2,
            'total_eggs': total_eggs,
            'dozens': dozens
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Chester", "James", "John", "Robert", "Michael", "William", "David"]
    name = rng.choice(names)
    
    # Generate random values scaled by difficulty
    n1 = int(rng.randint(2, int(6 * difficulty)))
    n2 = int(rng.randint(4, int(8 * difficulty)))
    while n2 <= n1:
        n2 = int(rng.randint(4, int(8 * difficulty)))
        
    d1 = int(rng.randint(20, int(110 * difficulty)))
    d2 = int(rng.randint(20, int(110 * difficulty)))
    
    # Ensure results are divisible by 12
    while (n1 * d1 + n2 * d2) % 12 != 0:
        d1 = int(rng.randint(20, int(110 * difficulty)))
        d2 = int(rng.randint(20, int(110 * difficulty)))

    result = generate_from_variables(name, n1, d1, n2, d2)
    
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
    return generate_from_variables("Chester", 3, 30, 5, 30)
