from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, fish: str, day: str, w1: int, w2: int, w3: int, 
                          n: int, unit: str, cur: str, price: float) -> Dict[str, Any]:
    total = int((w1 + w2) * price + (n-2)*w3*price)
    
    question = f"{name} caught {n} {fish}s last {day}, the first {fish} he caught weighs {w1} {unit}s, the second {fish} he caught weighs {w2} {unit}s, and the last {fish} he caught weighs {w3} {unit}s. If a {unit} of {fish} costs {cur}{price:.2f}, how much will he earn after selling all the {fish}s to the market?"
    
    answer_cot = f"{name} will earn {w1} x {cur}{price:.2f} = {cur}{w1*price:.2f} from the first {fish}.\n" \
                 f"He will earn {w2} x {cur}{price:.2f} = {cur}{w2*price:.2f} for the second {fish}.\n" \
                 f"The rest of the {fish}s are {n}-2 = {n-2}. He will earn {w3} x {cur}{price:.2f} = {cur}{w3*price:.2f} per each of them. So he will earn {n-2} * {w3*price:.2f} = {(n-2)*w3*price:.2f}\n" \
                 f"Therefore, the total amount he will earn for all the {fish}s is {cur}{w1*price:.2f} + {cur}{w2*price:.2f} + {cur}{(n-2)*w3*price:.2f}= {cur}{total}.\n#### {total}"

    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'name': name,
            'fish': fish,
            'day': day,
            'weight1': w1,
            'weight2': w2, 
            'weight3': w3,
            'num_fish': n,
            'unit': unit,
            'currency': cur,
            'price': price
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["John", "Michael", "David", "James", "Robert", "William", "Richard"]
    fish = ["salmon", "cod", "trout", "steelhead"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    units = ["kilogram", "pound", "kg"]
    currencies = ["$", "€", "£"]
    
    name = rng.choice(names)
    fish_type = rng.choice(fish)
    day = rng.choice(days)
    unit = rng.choice(units)
    cur = rng.choice(currencies)
    
    w1 = int(rng.randint(40, int(80 * difficulty)))
    w2 = int(rng.randint(30, int(60 * difficulty)))
    w3 = int(rng.randint(20, int(40 * difficulty)))
    n = int(rng.randint(3, int(8 * difficulty)))
    price = round(rng.uniform(0.25, 2.5), 2)
    
    # Ensure result is integer
    while not ((w1 + w2) * price + (n-2) * w3 * price).is_integer():
        price = round(rng.uniform(0.25, 2.5), 2)

    result = generate_from_variables(name, fish_type, day, w1, w2, w3, n, unit, cur, price)
    
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
    return generate_from_variables("Deandre", "tuna", "Monday", 56, 46, 26, 3, 
                                 "kilogram", "$", 0.50)
