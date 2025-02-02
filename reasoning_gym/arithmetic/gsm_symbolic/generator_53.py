from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, sides: int, target: int, property: str) -> Dict[str, Any]:
    numbers_above = sides - target
    prob_above = (numbers_above / sides) * 100
    prob_two_in_row = 25  # probability of two even/odd in a row is always 25%
    difference = int(prob_above - prob_two_in_row)
    
    question = f"{name} is rolling a {sides}-sided die. How much more likely is it (expressed as a percentage) that he rolls a number greater than {target} than that he rolls two {property} numbers in a row?"
    
    answer_cot = f"There are {numbers_above} numbers greater than {target} on the dice, so the chances of rolling one of them are {numbers_above} / {sides} = {prob_above}%.\nThe chance of rolling one {property} number is 50%, so the chance of rolling two in a row is 50% * 50% = 25%.\nThe difference between these two probabilities is {prob_above}% - 25% = {difference}%.\n#### {difference}"

    return {
        'question': question,
        'answer': str(difference),
        'answer_cot': answer_cot,
        'answer_value': difference,
        'variables': {
            'name': name,
            'sides': sides,
            'target': target,
            'property': property,
            'numbers_above': numbers_above,
            'prob_above': prob_above,
            'prob_two_in_row': prob_two_in_row
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
    properties = ["even", "odd"]
    
    name = rng.choice(names)
    property = rng.choice(properties)
    
    dice_options = [4, 6, 8, 10, 12, 20]
    sides = rng.choice(dice_options)
    
    # Generate target ensuring conditions are met
    while True:
        target = rng.randint(1, sides-1)
        prob = ((sides-target)/sides) * 100
        if (sides-target) % target == 0 and prob.is_integer() and prob > 25:
            break
            
    result = generate_from_variables(name, sides, target, property)
    
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
    return generate_from_variables("Jerry", 6, 3, "even")
