from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, shop: str, item: str, item1: str, item2: str, 
                          item3: str, n1: int, n2: int, n3: int, p1: int, p2: int, 
                          p3: int) -> Dict[str, Any]:
    cost1 = n1 * p1
    cost2 = n2 * p2 
    cost3 = n3 * p3
    total_cost = cost1 + cost2 + cost3

    question = f"{name} went to the {shop} and bought various types of {item}. She bought {n1} dozen {item1} which cost ${p1} per dozen, {n2} dozen {item2} which cost ${p2} per dozen, and {n3} dozen {item3} for ${p3} per dozen. How much was the total cost?"

    answer_cot = f"The total charge for the {item1} was {n1} x ${p1} = ${cost1}.\nThe total charge for the {item2} was {n2} x ${p2} = ${cost2}.\nThe total charge for the {item3} was {n3} x ${p3} = ${p3*n3}.\nTherefore the total amount {name} paid for the {item} was ${cost1} + ${cost2} + ${cost3} = ${total_cost}.\n#### {total_cost}"

    return {
        'question': question,
        'answer': str(total_cost),
        'answer_cot': answer_cot,
        'answer_value': total_cost,
        'variables': {
            'name': name,
            'shop': shop,
            'item': item,
            'item1': item1,
            'item2': item2,
            'item3': item3,
            'n1': n1,
            'n2': n2,
            'n3': n3,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'cost1': cost1,
            'cost2': cost2,
            'cost3': cost3
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Charlotte", "Mia", "Amelia"]
    shops = ["bakery", "patisserie", "confectionery", "cafe"]
    items = ["pastries", "baked goods", "desserts", "treats"]
    item1_options = ["donuts", "croissants", "eclairs", "danishes"]
    item2_options = ["mini cupcakes", "macarons", "cookies", "tarts"]
    item3_options = ["mini cheesecakes", "brownies", "muffins", "scones"]

    name = rng.choice(names_female)
    shop = rng.choice(shops)
    item = rng.choice(items)
    item1 = rng.choice(item1_options)
    item2 = rng.choice(item2_options)
    item3 = rng.choice(item3_options)

    n1 = int(rng.randint(1, int(10 * difficulty)))
    n2 = int(rng.randint(4, int(10 * difficulty)))
    n3 = int(rng.randint(2, int(10 * difficulty)))
    
    p1 = int(rng.randint(11, int(21 * difficulty)))
    p2 = int(rng.randint(73, int(90 * difficulty)))
    p3 = int(rng.randint(112, int(120 * difficulty)))

    result = generate_from_variables(name, shop, item, item1, item2, item3,
                                   n1, n2, n3, p1, p2, p3)
    
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
    return generate_from_variables("Toula", "bakery", "pastries", "donuts", 
                                 "mini cupcakes", "mini cheesecakes",
                                 3, 2, 6, 68, 80, 55)
