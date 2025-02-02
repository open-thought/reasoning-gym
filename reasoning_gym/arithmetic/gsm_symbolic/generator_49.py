from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, item1: str, item2: str, price1: int, price2: int, 
                          total: float, n1: int, percent: int, currency: str) -> Dict[str, Any]:
    change = total * percent / 100
    spent = total * (100 - percent) / 100
    cost_item1 = n1 * price1 / 100
    spent_item2 = spent - cost_item1
    n2 = int(spent_item2 / (price2 / 100))

    question = f"The vending machines sell {item1} for {price1} cents and {item2} for {price2} cents. {name} spent {currency}{total} and got {n1} bags of {item1} and had {percent}% of his money left. How many {item2} did he buy?"

    answer_cot = f"{name} got {currency}{change} in change because {total} x {percent}/100 = {change}\n" \
                 f"{name} spent {currency}{spent} because {total} - {change} = {spent}\n" \
                 f"{name} spent {currency}{cost_item1} on {item1} because {n1} x {price1/100} = {cost_item1}\n" \
                 f"{name} spent {spent_item2} on {item2} because {spent} - {cost_item1} = {spent_item2}\n" \
                 f"{name} bought {n2} {item2} because {spent_item2} / {price2/100} = {n2}\n" \
                 f"#### {n2}"

    return {
        'question': question,
        'answer': str(n2),
        'answer_cot': answer_cot,
        'answer_value': n2,
        'variables': {
            'name': name,
            'item1': item1,
            'item2': item2,
            'price1': price1,
            'price2': price2,
            'total_spent': total,
            'num_item1': n1,
            'num_item2': n2,
            'percent_change': percent,
            'currency': currency
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["George", "James", "John", "Robert", "Michael", "William"]
    items = ['pretzels', 'popcorn', 'gum', 'cookies', 'crackers']
    currencies = ['$', '£', '€']

    name = rng.choice(names)
    item1, item2 = rng.sample(items, 2)
    currency = rng.choice(currencies)

    price1 = int(rng.randrange(25, int(100 * difficulty), 5))
    price2 = int(rng.randrange(50, int(150 * difficulty), 5))
    while price2 <= price1:
        price2 = int(rng.randrange(50, int(150 * difficulty), 5))

    total = int(rng.randrange(500, int(1500 * difficulty), 100))
    n1 = int(rng.randint(1, int(10 * difficulty)))
    percent = int(rng.randint(1, int(10 * difficulty)))

    # Validate conditions
    while not (isinstance(total * percent / 100, int) and 
              isinstance((total * (100 - percent) / 100 - n1 * price1 / 100) / (price2 / 100), int) and
              (total * (100 - percent) / 100 - n1 * price1 / 100) % (price2 / 100) == 0):
        total = int(rng.randrange(500, int(1500 * difficulty), 100))
        n1 = int(rng.randint(1, int(10 * difficulty)))
        percent = int(rng.randint(1, int(10 * difficulty)))

    result = generate_from_variables(name, item1, item2, price1, price2, total, n1, percent, currency)

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
    return generate_from_variables("George", "chips", "candy bars", 40, 75, 5, 3, 1, "$")
