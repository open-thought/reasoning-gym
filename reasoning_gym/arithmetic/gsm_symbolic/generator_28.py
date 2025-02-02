from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, item: str, price: float, percent: float, 
                          usage: int, extra_item: str, extra_price: float, 
                          currency: str, unit: str) -> Dict[str, Any]:
    price_increase = price * percent / 100
    new_price = price + price_increase
    weekly_usage = usage * 7
    coffee_cost = new_price * weekly_usage
    total_cost = coffee_cost + extra_price
    
    question = f"{name} goes to the store to buy some {item}. The normal brand of {item} he buys costs {currency}{price} per {unit}. He had to buy a more expensive brand that costs {int(percent)}% more since his favorite brand was sold out. He decides to buy a week's worth of {item} and he uses {usage} {unit} of {item} per day. He also decided to buy himself a {extra_item} for {currency}{extra_price}. How much did everything cost?"
    
    answer_cot = f"The {item} he bought was {price}*{percent/100}={price_increase} more expensive per {unit} than what he normally buys\nSo it cost {price}+{price_increase}={new_price} per {unit}\nHe goes through {usage}*7={weekly_usage} {unit}s of {item} a week\nSo he paid {new_price}*{weekly_usage}={coffee_cost} on {item}\nThat means his total bill was {coffee_cost}+{extra_price}={total_cost}\n#### {int(total_cost)}"

    return {
        'question': question,
        'answer': str(int(total_cost)),
        'answer_cot': answer_cot,
        'answer_value': int(total_cost),
        'variables': {
            'name': name,
            'item': item,
            'base_price': price,
            'percent_increase': percent,
            'usage_per_day': usage,
            'extra_item': extra_item,
            'extra_price': extra_price,
            'currency': currency,
            'unit': unit,
            'weekly_usage': weekly_usage,
            'total_cost': total_cost
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_male = ['John', 'Michael', 'David', 'James', 'Robert', 'William', 'Richard', 'Thomas']
    items = ['tea', 'sugar', 'flour', 'rice']
    currencies_sym = ['$', '£', '€']
    units = ['kilogram', 'kg']
    extra_items = ['cookie', 'muffin', 'bagel']
    
    name = rng.choice(names_male)
    item = rng.choice(items)
    currency = rng.choice(currencies_sym)
    unit = rng.choice(units)
    extra_item = rng.choice(extra_items)
    
    price = int(rng.randint(3, int(25 * difficulty)))
    percent = int(rng.randint(2, int(10 * difficulty))) * 5
    usage = int(rng.randint(1, int(3 * difficulty)))
    extra_price = int(rng.randint(1, int(5 * difficulty)))
    
    # Ensure price * percent / 100 is an integer
    while (price * percent / 100) != int(price * percent / 100):
        price = int(rng.randint(3, int(25 * difficulty)))
    
    result = generate_from_variables(name, item, price, percent, usage, 
                                   extra_item, extra_price, currency, unit)
    
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
    return generate_from_variables('Roger', 'coffee', 5, 20, 1, 'donut', 2, '$', 'pound')
