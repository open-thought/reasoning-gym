from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, drink: str, sugar_ratio: int, water_ratio: int, 
                          total_items: int) -> Dict[str, Any]:
    total_ratio = sugar_ratio + water_ratio
    sugar_amount = (sugar_ratio * total_items) // total_ratio
    
    question = f"{name} makes {drink} using teaspoons of sugar and cups of water in the ratio of {sugar_ratio}:{water_ratio}. If she used a total of {total_items} teaspoons of sugar and cups of water, calculate the number of teaspoonfuls of sugar she used."
    
    answer_cot = f"The total ratio representing the ingredients she used to make the {drink} is {sugar_ratio}+{water_ratio} = {total_ratio}\nSince the fraction representing the number of teaspoons she used is {sugar_ratio}/{total_ratio}, she used {sugar_ratio}/{total_ratio}*{total_items} = {sugar_amount}\n#### {sugar_amount}"
    
    return {
        'question': question,
        'answer': str(sugar_amount),
        'answer_cot': answer_cot,
        'answer_value': sugar_amount,
        'variables': {
            'name': name,
            'drink': drink,
            'sugar_ratio': sugar_ratio,
            'water_ratio': water_ratio,
            'total_items': total_items,
            'sugar_amount': sugar_amount
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ['Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte']
    drinks = ['coffee', 'tea']
    
    name = rng.choice(names_female)
    drink = rng.choice(drinks)
    
    sugar_ratio = int(rng.randint(25, int(201 * difficulty)))
    water_ratio = int(rng.randint(5, int(101 * difficulty)))
    
    # Ensure total is divisible by ratio sum
    total_ratio = sugar_ratio + water_ratio
    num_multiples = rng.randint(1, int(10 * difficulty))
    total_items = total_ratio * num_multiples
    
    result = generate_from_variables(name, drink, sugar_ratio, water_ratio, total_items)
    
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
    return generate_from_variables("Katy", "coffee", 7, 13, 120)
