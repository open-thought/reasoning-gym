from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, currency: str, initial_amount: float, 
                          quantity: int, item: str, store_type: str, 
                          unit_price: float) -> Dict[str, Any]:
    total_cost = quantity * unit_price
    remaining = initial_amount - total_cost
    
    question = f"{name} has {currency}{initial_amount:.2f} and wants to buy {quantity} {item}s from a bin at the {store_type} store. Each {item} costs {currency}{unit_price:.2f}. How much money does {name} have left after paying for the {item}s?"
    
    answer_cot = f"{name} paid {quantity} * {currency}{unit_price:.2f} = {currency}{total_cost:.2f} for the {item}s.\n{name} has {currency}{initial_amount:.2f} - {currency}{total_cost:.2f} = {currency}{int(remaining)} left.\n#### {int(remaining)}"

    return {
        'question': question,
        'answer': str(int(remaining)),
        'answer_cot': answer_cot,
        'answer_value': int(remaining),
        'variables': {
            'name': name,
            'currency': currency,
            'initial_amount': initial_amount,
            'quantity': quantity,
            'item': item,
            'store_type': store_type,
            'unit_price': unit_price,
            'total_cost': total_cost,
            'remaining': remaining
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["David", "John", "Michael", "James", "William", "Robert"]
    currencies = ["$", "€", "£"]
    items = ["screw", "nail", "washer", "nut", "anchor"]
    store_types = ["hardware", "home improvement", "construction supply"]
    
    name = rng.choice(names)
    currency = rng.choice(currencies)
    item = rng.choice(items)
    store_type = rng.choice(store_types)
    
    # Generate values ensuring conditions are met
    quantity = int(rng.randint(15, int(60 * difficulty)))
    unit_price = round(rng.uniform(0.01, min(1.0, 1.0 * difficulty)), 2)
    
    # Ensure initial amount is sufficient and result is integer
    total_cost = quantity * unit_price
    remaining = rng.randint(1, int(100 * difficulty))
    initial_amount = total_cost + remaining
    
    result = generate_from_variables(name, currency, initial_amount, quantity, 
                                   item, store_type, unit_price)
    
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
    return generate_from_variables("David", "$", 12.48, 16, "bolt", "hardware", 0.03)
