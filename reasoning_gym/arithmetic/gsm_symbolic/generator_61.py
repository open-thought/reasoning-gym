from random import Random
from typing import Dict, Any
import numpy as np

def generate_from_variables(name: str, location: str, shop: str, item1: str, item2: str, 
                          item3: str, unit: str, cur: str, total: float, n1: int, n2: int,
                          n12: int, k: int, n3: int, p1: float, p2: float, p3: float,
                          discount: float) -> Dict[str, Any]:
    
    # Calculate costs
    item1_cost = n1*p1 + n2*(1-discount)*p1 + k*p1  # Cost of item1 with discount applied
    item2_cost = p2  # Cost of item2
    item3_cost = n3*p3  # Cost of item3
    total_spent = int(item1_cost + item2_cost + item3_cost)
    money_left = total - total_spent

    question = f"{name} went to the {location} for vacation. Her parents gave her {cur}{total} to buy whatever she wanted. At the {shop}, {item1} was on sale for \"Buy {n1} {unit}s at {cur}{p1} per {unit}, get {n2} {unit}s {discount} off.\" She scooped up {n12} {unit}s. She also bought a mixed bag of {item2} for {cur}{p2} and {n3} {item3} that were {cur}{p3} each. How much money does {name} have left?"

    answer_cot = f"{item1} is {n1} {unit}s for {cur}{p1} and gets {n2} {unit}s {discount} off. So {discount} off of {n2} {unit}s is {cur}{n2*discount}*{p1} = {cur}{n2*discount*p1}. The rest of {k} {unit}s does not have discount and come at {k*p1} so total is {n1}*{p1} + {n2}*{1-discount}*{p1} + {k}*{p1} = {item1_cost}\n{n3} {item3} at {cur}{p3} each is {n3}*{p3}={cur}{n3*p3}\nWhen you add all her purchases, {cur}{item1_cost}+{cur}{p2}+{cur}{n3*p3} = {cur}{total_spent}\nShe had {cur}{total} and spent {cur}{total_spent} so she had {cur}{total}-{cur}{total_spent} = {cur}{money_left} left over\n#### {money_left}"

    return {
        'question': question,
        'answer': str(money_left),
        'answer_cot': answer_cot,
        'answer_value': money_left,
        'variables': {
            'name': name,
            'location': location,
            'shop': shop,
            'item1': item1,
            'item2': item2,
            'item3': item3,
            'unit': unit,
            'currency': cur,
            'total_money': total,
            'n1': n1,
            'n2': n2,
            'n12': n12,
            'k': k,
            'n3': n3,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'discount': discount,
            'total_spent': total_spent
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ['Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte']
    locations = ['beach', 'boardwalk', 'pier', 'coast']
    shops = ['souvenir store', 'gift shop', 'beach shop', 'seaside store']
    items1 = ['fudge', 'saltwater taffy', 'rock candy', 'cotton candy']
    items2 = ['sand dollars', 'starfish', 'sea glass', 'coral pieces'] 
    items3 = ['postcards', 'keychains', 'stickers', 'pins']
    units = ['pound', 'kilogram', 'kg']
    currencies = ['$', '£', '€']
    fraction_nums = [0.25, 0.33, 0.5, 0.67, 0.75]

    name = rng.choice(names_female)
    location = rng.choice(locations)
    shop = rng.choice(shops)
    item1 = rng.choice(items1)
    item2 = rng.choice(items2)
    item3 = rng.choice(items3)
    unit = rng.choice(units)
    cur = rng.choice(currencies)
    
    total = int(rng.randint(1200, int(1500 * difficulty)))
    n1 = int(rng.randint(15, int(18 * difficulty)))
    n2 = int(rng.randint(4, int(10 * difficulty)))
    k = int(rng.randint(2, int(5 * difficulty)))
    n12 = n1 + n2 + k
    n3 = int(rng.randint(11, int(19 * difficulty)))
    p1 = int(rng.randint(20, int(24 * difficulty)))
    p2 = round(rng.uniform(11.25, 12.00), 2)
    p3 = round(rng.uniform(20.25, 21.25), 2)
    discount = rng.choice(fraction_nums[:4])

    # Ensure conditions are met
    while not (n2 < n1 and 
              n12 == n1 + n2 + k and 
              0 <= k < n1 and
              int(n1*p1 + n2*(1-discount)*p1 + k*p1 + p2 + n3*p3) == n1*p1 + n2*(1-discount)*p1 + k*p1 + p2 + n3*p3 and
              n1*p1 + n2*(1-discount)*p1 + k*p1 + p2 + n3*p3 < total):
        n1 = int(rng.randint(15, int(18 * difficulty)))
        n2 = int(rng.randint(4, int(10 * difficulty)))
        k = int(rng.randint(2, int(5 * difficulty)))
        n12 = n1 + n2 + k
        p1 = int(rng.randint(20, int(24 * difficulty)))

    result = generate_from_variables(name, location, shop, item1, item2, item3, unit, cur,
                                   total, n1, n2, n12, k, n3, p1, p2, p3, discount)
    
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
    return generate_from_variables('Sally', 'seashore', 'trinket shop', 'taffy', 'seashells',
                                 'magnets', 'pound', '$', 10, 1, 1, 2, 0, 4, 3, 1.50, 0.25, 0.5)
