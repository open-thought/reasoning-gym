from random import Random
from typing import Dict, Any
import math

def generate_from_variables(name: str, food: str, peel_rate: int, batch_size: int, 
                          time_per_batch: int, total_amount: int) -> Dict[str, Any]:
    
    peel_time = total_amount // peel_rate
    num_batches = total_amount // batch_size  
    cook_time = num_batches * time_per_batch
    total_time = peel_time + cook_time

    question = f"{name} can peel {peel_rate} {food}s a minute and saute {batch_size} {food}s in {time_per_batch} minutes. How long will it take her to peel and saute {total_amount} {food}s?"
    
    answer_cot = f"First find how long it takes {name} to peel the {food}: {total_amount} {food} / {peel_rate} {food}/minute = {peel_time} minutes\n" \
                 f"Then find how many batches of {food} she needs to cook: {total_amount} {food} / {batch_size} {food}/batch = {num_batches} batches\n" \
                 f"Then multiply the number of batches by the time per batch to find the total cook time: {num_batches} batches * {time_per_batch} minutes/batch = {cook_time} minutes\n" \
                 f"Then add the peeling time to find the total time {name} spends: {cook_time} minutes + {peel_time} minutes = {total_time} minutes\n" \
                 f"#### {total_time}"

    return {
        'question': question,
        'answer': str(total_time),
        'answer_cot': answer_cot,
        'answer_value': total_time,
        'variables': {
            'name': name,
            'food': food,
            'peel_rate': peel_rate,
            'batch_size': batch_size,
            'time_per_batch': time_per_batch, 
            'total_amount': total_amount,
            'peel_time': peel_time,
            'cook_time': cook_time
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ["Emily", "Sarah", "Emma", "Sophia", "Olivia", "Ava", "Isabella", "Mia"]
    foods = ["shrimp", "onion", "carrot", "mushroom", "clam"]
    
    name = rng.choice(names_female)
    food = rng.choice(foods)
    
    peel_rate = int(rng.randint(4, int(15 * difficulty)))
    batch_size = int(rng.randrange(20, int(50 * difficulty), 5))
    time_per_batch = int(rng.randint(5, int(20 * difficulty)))
    
    # Ensure total is divisible by both peel_rate and batch_size
    lcm = peel_rate * batch_size // math.gcd(peel_rate, batch_size)
    num_lcm = rng.randint(1, int(4 * difficulty))
    total_amount = lcm * num_lcm
    
    result = generate_from_variables(name, food, peel_rate, batch_size, time_per_batch, total_amount)
    
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
    return generate_from_variables("Emily", "shrimp", 6, 30, 10, 90)
