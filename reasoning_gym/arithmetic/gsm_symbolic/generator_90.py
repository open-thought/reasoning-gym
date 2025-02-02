from random import Random
from typing import Dict, Any

def generate_from_variables(device: str, currency: str, rate1: float, rate2: float, 
                          threshold: int, total_mins: int) -> Dict[str, Any]:
    first_period = threshold
    second_period = total_mins - threshold
    
    cost1 = first_period * rate1
    cost2 = second_period * rate2
    total_cost = int(cost1 + cost2)
    
    question = f"To make a call from a {device}, you must pay {currency}{rate1} for each minute of your call. After {threshold} minutes, that price drops to {currency}{rate2} per minute. How much would a {total_mins}-minute call cost?"
    
    answer_cot = f"First {threshold} minutes would be a cost of {threshold} * {rate1} = {currency}{int(cost1)}.\nAfter that, there are {total_mins} - {threshold} = {second_period} minutes of the call left.\nAnd these {second_period} minutes cost {second_period} * {rate2} = {currency}{int(cost2)}.\nSo in total, the {total_mins}-minute call would cost {int(cost1)} + {int(cost2)} = {currency}{total_cost}.\n#### {total_cost}"
    
    return {
        'question': question,
        'answer': str(total_cost),
        'answer_cot': answer_cot,
        'answer_value': total_cost,
        'variables': {
            'device': device,
            'currency': currency,
            'rate1': rate1,
            'rate2': rate2,
            'threshold': threshold,
            'total_mins': total_mins,
            'first_period_cost': int(cost1),
            'second_period_cost': int(cost2)
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    devices = ['payphone', 'phone booth', 'hotel room phone']
    currencies = ['$', '£', '€']
    
    device = rng.choice(devices)
    currency = rng.choice(currencies)
    
    # Generate rates ensuring rate2 < rate1
    rate1 = round(rng.uniform(0.2, 0.5 * difficulty), 2)
    rate2 = round(rng.uniform(0.1, rate1), 2)
    
    threshold = int(rng.randint(10, int(50 * difficulty)))
    total_mins = int(rng.randint(threshold + 10, int(100 * difficulty)))
    
    # Ensure calculations result in integers
    while not (threshold * rate1).is_integer() or not ((total_mins - threshold) * rate2).is_integer():
        rate1 = round(rng.uniform(0.2, 0.5 * difficulty), 2)
        rate2 = round(rng.uniform(0.1, rate1), 2)
    
    result = generate_from_variables(device, currency, rate1, rate2, threshold, total_mins)
    
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
    return generate_from_variables('payphone', '$', 0.25, 0.2, 16, 36)
