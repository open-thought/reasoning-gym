from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, product: str, location: str,
                          item1: str, item2: str, item3: str,
                          price1: float, price2: float, price3: float,
                          num1: int, num2: int, num3: int,
                          unit: str, currency: str) -> Dict[str, Any]:
    
    round_p1 = round(price1)
    round_p2 = round(price2) 
    round_p3 = round(price3)
    
    total = num1 * round_p1 + num2 * round_p2 + num3 * round_p3

    question = f"{name} has a {product} stand at the {location}. He sells three kinds of {product}s: {item1}, {item2} and {item3}. He usually sells {item1} for {currency}{price1:.2f} per {unit}, {item2} for {currency}{price2:.2f} per {unit} and {item3} for {currency}{price3:.2f} per {unit}. {name} has no change today, so he has decided to round all his prices to the nearest dollar. If {name} sells {num1} {unit}s of {item1}, {num2} {unit}s of {item2} and {num3} {unit}s of {item3}, how much will he make?"

    answer_cot = f"{name} will round his {item1} {'up' if round_p1 > price1 else 'down'} from {currency}{price1:.2f} to {currency}{round_p1}, since the number following the {int(price1)} is {'5 or higher' if round_p1 > price1 else 'less than 5'}.\n"
    answer_cot += f"{name} will round his {item2} {'up' if round_p2 > price2 else 'down'} from {currency}{price2:.2f} to {currency}{round_p2}, since the number following the {int(price2)} is {'5 or higher' if round_p2 > price2 else 'less than 5'}.\n"
    answer_cot += f"{name} will round his {item3} {'up' if round_p3 > price3 else 'down'} from {currency}{price3:.2f} to {currency}{round_p3}, since the number following the {int(price3)} is {'5 or higher' if round_p3 > price3 else 'less than 5'}.\n"
    answer_cot += f"{name} sells {num1} {item1} x {currency}{round_p1} = {currency}{num1*round_p1}\n"
    answer_cot += f"{name} sells {num2} {item2} x {currency}{round_p2} = {currency}{num2*round_p2}\n"
    answer_cot += f"{name} sells {num3} {item3} x {currency}{round_p3} = {currency}{num3*round_p3}\n"
    answer_cot += f"Altogether, {name} will make {currency}{num1*round_p1} + {currency}{num2*round_p2} + {currency}{num3*round_p3} = {currency}{total}\n#### {total}"

    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'name': name,
            'product': product,
            'location': location,
            'items': [item1, item2, item3],
            'original_prices': [price1, price2, price3],
            'rounded_prices': [round_p1, round_p2, round_p3],
            'quantities': [num1, num2, num3],
            'unit': unit,
            'currency': currency
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['John', 'Mike', 'David', 'James', 'Robert', 'William', 'Richard']
    products = ['vegetable', 'flower', 'herb', 'plant']
    locations = ['local fair', 'community market', 'street bazaar', 'town square']
    items = ['roses', 'daisies', 'tulips', 'lilies', 'sunflowers', 'orchids']
    units = ['bunch', 'basket', 'bouquet', 'bundle']
    currencies = ['$', '£', '€']

    name = rng.choice(names)
    product = rng.choice(products)
    location = rng.choice(locations)
    item1, item2, item3 = rng.sample(items, 3)
    unit = rng.choice(units)
    currency = rng.choice(currencies)

    # Scale prices by difficulty
    price1 = round(rng.uniform(1.26, 3.53 * difficulty), 2)
    price2 = round(rng.uniform(2.27, 5.53 * difficulty), 2)
    price3 = round(rng.uniform(4.85, 6.53 * difficulty), 2)

    num1 = int(rng.randint(5, int(21 * difficulty)))
    num2 = int(rng.randint(15, int(31 * difficulty)))
    num3 = int(rng.randint(35, int(41 * difficulty)))

    result = generate_from_variables(name, product, location, item1, item2, item3,
                                   price1, price2, price3, num1, num2, num3,
                                   unit, currency)

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
    return generate_from_variables('Artie', 'flower', 'Farmers Market',
                                 'marigolds', 'petunias', 'begonias',
                                 2.74, 1.87, 2.12,
                                 12, 9, 17,
                                 'pot', '$')
