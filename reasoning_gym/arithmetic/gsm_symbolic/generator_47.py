from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, n1: int, c1: float, n2: int, c2: float, 
                          c3: int, obj1: str, obj2: str, currency: str) -> Dict[str, Any]:
    cost1 = n1 * c1
    cost2 = n2 * c2
    total_cost = cost1 + cost2 + c3
    
    question = f"{name} went to buy some school supplies. He bought {n1} {obj1} which cost {currency}{c1} each, {n2} {obj2} which cost {currency}{c2} each, and a rim of bond paper which cost {currency}{c3}. How much did {name} spend on everything?"
    
    answer_cot = f"{name} spent {n1} x {currency}{c1} = {currency}{int(cost1)} for the {obj1}.\n" \
                 f"He also spent {n2} x {currency}{c2} = {currency}{int(cost2)} for the {obj2}.\n" \
                 f"Therefore, {name} spent a total of {currency}{int(cost1)} + {currency}{int(cost2)} + {currency}{c3} = {currency}{int(total_cost)} for the school supplies.\n" \
                 f"#### {int(total_cost)}"

    return {
        'question': question,
        'answer': str(int(total_cost)),
        'answer_cot': answer_cot,
        'answer_value': int(total_cost),
        'variables': {
            'name': name,
            'items1_count': n1,
            'item1_cost': c1,
            'items2_count': n2,
            'item2_cost': c2,
            'paper_cost': c3,
            'item1': obj1,
            'item2': obj2,
            'currency': currency
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["John", "Michael", "David", "James", "William", "Robert", "Thomas"]
    items = ["notebooks", "pencils", "erasers", "crayons", "colored pencils", "markers", "rulers", "folders"]
    currencies = ["$", "€", "£"]
    
    name = rng.choice(names)
    obj1, obj2 = rng.sample(items, 2)
    currency = rng.choice(currencies)
    
    n1 = int(rng.randrange(6, int(25 * difficulty), 2))
    c1 = round(rng.uniform(2.25, 11.5 * difficulty) * 4) / 4  # Round to nearest 0.25
    n2 = int(rng.randrange(4, int(15 * difficulty), 2))
    c2 = round(rng.uniform(8.25, 19.5 * difficulty) * 4) / 4
    c3 = int(rng.randint(10, int(25 * difficulty)))
    
    # Ensure conditions are met
    while not (n1 * c1).is_integer() or not (n2 * c2).is_integer():
        c1 = round(rng.uniform(2.25, 11.5 * difficulty) * 4) / 4
        c2 = round(rng.uniform(8.25, 19.5 * difficulty) * 4) / 4
    
    result = generate_from_variables(name, n1, c1, n2, c2, c3, obj1, obj2, currency)
    
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
    return generate_from_variables("Raphael", 4, 1.5, 2, 4, 20, "pens", "notebooks", "$")
