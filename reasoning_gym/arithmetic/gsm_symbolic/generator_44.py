from random import Random
from typing import Dict, Any

def generate_from_variables(store: str, color1: str, color2: str, color3: str, 
                          n1: int, n2: int, n3: int, p1: int, p2: int, p3: int,
                          currency: str) -> Dict[str, Any]:
    total1 = n1 * p1
    total2 = n2 * p2 
    total3 = n3 * p3
    grand_total = total1 + total2 + total3

    question = f"There are currently {n1} {color1} balls, {n2} {color2} balls, and {n3} {color3} balls in the {store}. {color1} balls cost {currency}{p1}, {color2} balls cost {currency}{p2} and {color3} balls cost {currency}{p3}. How much will the {store} have received after all the balls are sold?"

    answer_cot = f"For the {color1} balls, {n1} balls * {currency}{p1}/ball = {currency}{total1}.\nFor the {color2} balls, {n2} balls * {currency}{p2}/ball = {currency}{total2}.\nFor the {color3} balls, {n3} balls * {currency}{p3}/ball = {currency}{total3}.\nFor all balls, {currency}{total1} + {currency}{total2} + {currency}{total3} = {currency}{grand_total}.\n#### {grand_total}"

    return {
        'question': question,
        'answer': str(grand_total),
        'answer_cot': answer_cot,
        'answer_value': grand_total,
        'variables': {
            'store': store,
            'colors': [color1, color2, color3],
            'quantities': [n1, n2, n3],
            'prices': [p1, p2, p3],
            'currency': currency,
            'subtotals': [total1, total2, total3],
            'total': grand_total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    stores = ["store", "shop", "market", "warehouse"]
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink"]
    currencies = ["$", "€", "£"]
    
    store = rng.choice(stores)
    color1, color2, color3 = rng.sample(colors, 3)
    currency = rng.choice(currencies)
    
    n1 = int(rng.randint(3, int(20 * difficulty)))
    n2 = int(rng.randint(5, int(30 * difficulty)))
    n3 = int(rng.randint(15, int(50 * difficulty)))
    
    p1 = int(rng.randint(5, int(15 * difficulty)))
    p2 = int(rng.randint(3, int(10 * difficulty)))
    p3 = int(rng.randint(2, int(8 * difficulty)))

    result = generate_from_variables(store, color1, color2, color3, n1, n2, n3, 
                                   p1, p2, p3, currency)
    
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
    return generate_from_variables("store", "red", "blue", "green", 3, 11, 25,
                                 9, 5, 3, "$")
