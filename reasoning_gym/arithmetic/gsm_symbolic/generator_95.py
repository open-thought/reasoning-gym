from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, relation: str, food: str, 
                          n1: int, n2: int, n3: int, time_unit: str,
                          time_period: str) -> Dict[str, Any]:
    daily_total = n1 + n2 + n3
    total = daily_total * (7 if time_period == "week" else 30)
    
    question = f"{name1} eats {n1} {food} per {time_unit}, {name2} eats {n2} {food} per {time_unit}, and their {relation} eats {n3} {food} per {time_unit}. How many {food} does this family eat in one {time_period}?"
    
    answer_cot = f"The number of {food} they eat in one {time_unit} is {n1} + {n2} + {n3} = {daily_total} {food}.\nThe number of {food} they eat in a {time_period} is {daily_total} * {7 if time_period == 'week' else 30} = {total} {food}.\n#### {total}"

    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'name1': name1,
            'name2': name2,
            'relation': relation,
            'food': food,
            'daily_servings1': n1,
            'daily_servings2': n2, 
            'daily_servings3': n3,
            'daily_total': daily_total,
            'time_unit': time_unit,
            'time_period': time_period
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    name1_options = ["A father", "A grandfather", "An uncle"]
    name2_options = ["his wife", "his partner", "his spouse"]
    relation_options = ["daughter", "son", "grandchild"]
    food_options = ["pizzas", "burritos", "tacos", "sushi rolls", "hamburgers"]
    
    name1 = rng.choice(name1_options)
    name2 = rng.choice(name2_options)
    relation = rng.choice(relation_options)
    food = rng.choice(food_options)
    
    n1 = int(rng.randint(2, int(9 * difficulty)))
    n2 = int(rng.randint(2, int(9 * difficulty)))
    n3 = int(rng.randint(2, int(9 * difficulty)))
    
    time_unit = "day"
    time_period = rng.choice(["week", "month"])

    result = generate_from_variables(name1, name2, relation, food, n1, n2, n3,
                                   time_unit, time_period)
    
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
    return generate_from_variables("A man", "his wife", "son", "sandwiches",
                                 5, 4, 2, "day", "week")
