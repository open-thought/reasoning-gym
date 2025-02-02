from random import Random
from typing import Dict, Any

def generate_from_variables(item1: str, item2: str, shop: str, currency: str, 
                          price1: int, price2: int, n1: int, n2: int) -> Dict[str, Any]:
    total1 = n1 * price1
    total2 = n2 * price2
    diff = total1 - total2
    
    question = f"A loaf of {item1} at the {shop} costs {currency}{price1}. {item2}s cost {currency}{price2} each. How much more do {n1} loaves of {item1} cost than {n2} {item2}s?"
    
    answer_cot = f"{n1} loaves of {item1} cost {n1} * {currency}{price1} = {currency}{total1}.\n" \
                 f"{n2} {item2}s cost {n2} * {currency}{price2} = {currency}{total2}.\n" \
                 f"The loaves of {item1} cost {currency}{total1} - {currency}{total2} = {currency}{diff} more than the {item2}s.\n" \
                 f"#### {diff}"

    return {
        'question': question,
        'answer': str(diff),
        'answer_cot': answer_cot,
        'answer_value': diff,
        'variables': {
            'item1': item1,
            'item2': item2,
            'shop': shop,
            'currency': currency,
            'price1': price1,
            'price2': price2,
            'n1': n1,
            'n2': n2,
            'total1': total1,
            'total2': total2
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    items1 = ["bread", "sourdough"]
    items2 = ["bagel", "muffin", "croissant", "biscuit"]
    shops = ["bakery", "cafe", "store", "market"]
    currencies = ["$", "£", "€"]
    
    item1 = rng.choice(items1)
    item2 = rng.choice(items2)
    shop = rng.choice(shops)
    currency = rng.choice(currencies)
    
    price1 = int(rng.randint(2, int(10 * difficulty)))
    price2 = int(rng.randint(1, int(5 * difficulty)))
    n1 = int(rng.randint(2, int(10 * difficulty)))
    n2 = int(rng.randint(2, int(10 * difficulty)))
    
    # Ensure condition: n1 * price1 > n2 * price2
    while n1 * price1 <= n2 * price2:
        n1 = int(rng.randint(2, int(10 * difficulty)))
        n2 = int(rng.randint(2, int(10 * difficulty)))

    result = generate_from_variables(item1, item2, shop, currency, price1, price2, n1, n2)
    
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
    return generate_from_variables("bread", "bagel", "bakery", "$", 2, 1, 3, 2)
