from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, num_bills: int, bill_value: int, num_items1: int, 
                          price1: int, num_items2: int, price2: int, item1: str, 
                          item2: str, currency: str) -> Dict[str, Any]:
    
    initial_amount = num_bills * bill_value
    spent_items1 = num_items1 * price1
    spent_items2 = num_items2 * price2
    total_spent = spent_items1 + spent_items2
    remaining = initial_amount - total_spent

    question = f"{name} has {num_bills} {currency}{bill_value} bills. He buys {num_items1} {item1}s for {currency}{price1} each. He also buys {num_items2} packs of {item2}s for {currency}{price2} each. How much money does he have left?"
    
    answer_cot = f"{name} starts off with {num_bills} * {currency}{bill_value} = {currency}{initial_amount}.\n" \
                 f"{name} spends {num_items1} {item1}s * {currency}{price1} = {currency}{spent_items1} on {item1}s.\n" \
                 f"{name} spends {num_items2} packs of {item2}s * {currency}{price2} = {currency}{spent_items2} on {item2}s.\n" \
                 f"Total {name} has spent {currency}{spent_items1} + {currency}{spent_items2} = {currency}{total_spent}.\n" \
                 f"{name} has {currency}{initial_amount} - {currency}{total_spent} = {currency}{remaining} remaining.\n#### {remaining}"

    return {
        'question': question,
        'answer': str(remaining),
        'answer_cot': answer_cot,
        'answer_value': remaining,
        'variables': {
            'name': name,
            'num_bills': num_bills,
            'bill_value': bill_value,
            'num_items1': num_items1,
            'price1': price1,
            'num_items2': num_items2,
            'price2': price2,
            'item1': item1,
            'item2': item2,
            'currency': currency,
            'initial_amount': initial_amount,
            'total_spent': total_spent
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Craig", "John", "Michael", "David", "James", "Robert", "William"]
    items1 = ["toy car", "action figure", "coloring book", "puzzle", "board game"]
    items2 = ["sticker", "candy bar", "trading card", "pencil", "eraser"]
    currencies = ["$", "€", "£"]
    bills = [(5, 5), (10, 10), (20, 20), (50, 50), (100, 100)]
    
    name = rng.choice(names)
    item1 = rng.choice(items1)
    item2 = rng.choice(items2)
    currency = rng.choice(currencies)
    bill_value = rng.choice(bills)[1]
    
    num_bills = int(rng.randint(1, int(10 * difficulty)))
    num_items1 = int(rng.randint(2, int(15 * difficulty)))
    num_items2 = int(rng.randint(2, int(10 * difficulty)))
    price1 = int(rng.randint(1, int(10 * difficulty)))
    price2 = int(rng.randint(1, int(10 * difficulty)))
    
    # Ensure total cost doesn't exceed available money
    while (num_items1 * price1 + num_items2 * price2) > (num_bills * bill_value):
        num_items1 = int(rng.randint(2, int(15 * difficulty)))
        num_items2 = int(rng.randint(2, int(10 * difficulty)))
        
    result = generate_from_variables(name, num_bills, bill_value, num_items1, price1,
                                   num_items2, price2, item1, item2, currency)
    
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
    return generate_from_variables("Craig", 2, 20, 6, 2, 3, 3, 
                                 "squirt gun", "water balloon", "$")
