from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, fruit: str, food: str, d1: str, d2: str,
                          n1: int, n2: int, m1: int, m2: int, cn: int, cm: int, 
                          currency: str) -> Dict[str, Any]:
    
    gingerbread_sunday = n1 + n2
    total_gingerbread = n1 + gingerbread_sunday
    gingerbread_revenue = total_gingerbread * cn
    
    apple_pie_saturday = m2 - m1  
    total_apple_pie = m2 + apple_pie_saturday
    apple_pie_revenue = total_apple_pie * cm
    
    total_revenue = gingerbread_revenue + apple_pie_revenue

    question = f"{name} is selling {food} and {fruit} pie for a fundraiser. On {d1}, he sold {n1} boxes of {food} and {m1} fewer boxes of {fruit} pie, than on {d2}. On {d2}, he sold {n2} more boxes of {food} than on {d1} and {m2} boxes of {fruit} pie. If the {food} cost {currency}{cn} and the {fruit} pie cost {currency}{cm}, how much did {name} earn for two days?"

    answer_cot = f"He sold {n1} + {n2} = {gingerbread_sunday} boxes of {food} on {d2}.\nThe total number of boxes of {food}s that {name} sold is {n1} + {gingerbread_sunday} = {total_gingerbread}.\n{name} earned {total_gingerbread} x {currency}{cn} = {currency}{gingerbread_revenue} for selling {food}s.\nHe sold {m2} - {m1} = {apple_pie_saturday} boxes of {fruit} pie on {d1}.\nThe total number of boxes of {fruit} pie that {name} sold is {m2} + {apple_pie_saturday} = {total_apple_pie}.\nHe earned {total_apple_pie} x {currency}{cm} = {currency}{apple_pie_revenue} for selling {fruit} pie.\nSo, {name} earned {currency}{gingerbread_revenue} + {currency}{apple_pie_revenue} = {currency}{total_revenue} for two days.\n#### {total_revenue}"

    return {
        'question': question,
        'answer': str(total_revenue),
        'answer_cot': answer_cot,
        'answer_value': total_revenue,
        'variables': {
            'name': name,
            'fruit': fruit,
            'food': food,
            'day1': d1,
            'day2': d2,
            'gingerbread_day1': n1,
            'gingerbread_increase': n2,
            'apple_pie_difference': m1,
            'apple_pie_day2': m2,
            'gingerbread_price': cn,
            'apple_pie_price': cm,
            'currency': currency,
            'total_revenue': total_revenue
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['John', 'Michael', 'David', 'James', 'William', 'Robert']
    fruits = ['apple', 'cherry', 'blueberry', 'peach']
    foods = ['cookie', 'brownie', 'muffin', 'cupcake']
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    currencies = ['$', '£', '€']

    name = rng.choice(names)
    fruit = rng.choice(fruits)
    food = rng.choice(foods)
    d1, d2 = rng.sample(weekdays, 2)
    currency = rng.choice(currencies)

    n1 = int(rng.randint(21, int(30 * difficulty)))
    n2 = int(rng.randint(11, int(15 * difficulty)))
    m2 = int(rng.randint(21, int(30 * difficulty)))
    m1 = int(rng.randint(11, int(min(20, m2) * difficulty)))
    cn = int(rng.randint(7, int(13 * difficulty)))
    cm = int(rng.randint(20, int(33 * difficulty)))

    result = generate_from_variables(name, fruit, food, d1, d2, n1, n2, m1, m2, cn, cm, currency)
    
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
    return generate_from_variables('Sunny', 'apple', 'gingerbread', 'Saturday', 'Sunday',
                                 10, 5, 4, 15, 6, 15, '$')
