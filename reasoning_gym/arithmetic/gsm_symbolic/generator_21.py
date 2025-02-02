from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, length: int, unit_length: str, plant_width: int,
                          space: float, owned: int, currency: str, cost: int) -> Dict[str, Any]:
    total_plants = int(length / space)
    plants_to_buy = total_plants - owned
    total_cost = plants_to_buy * cost

    question = f"{name} has a flower bed that is {length} {unit_length} long. {name} wants to fill her flower bed with plants. {name}'s flowers grow {plant_width} inches wide so she needs to leave {space} {unit_length} between every plant. {name} already owns {owned} flowers. Each flowering plant costs {currency}{cost} at the store, how much money will {name} spend at the store to fill up her flower bed?"
    
    answer_cot = f"{name}'s flower bed is {length} {unit_length} / {space} {unit_length} per plant = {total_plants} plants needed.\n{name} needs to buy {total_plants} plants - {owned} plants = {plants_to_buy} plants needed to purchase.\n{name} will spend {plants_to_buy} plants * {currency}{cost} = {currency}{total_cost}.\n#### {total_cost}"

    return {
        'question': question,
        'answer': str(total_cost),
        'answer_cot': answer_cot,
        'answer_value': total_cost,
        'variables': {
            'name': name,
            'bed_length': length,
            'unit': unit_length,
            'plant_width': plant_width,
            'plant_spacing': space,
            'owned_plants': owned,
            'currency': currency,
            'cost_per_plant': cost,
            'total_plants': total_plants,
            'plants_to_buy': plants_to_buy
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ['Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte']
    currencies = ['$', '£', '€']
    units = ['feet', 'meters']
    
    name = rng.choice(names_female)
    unit = rng.choice(units)
    currency = rng.choice(currencies)
    
    length = int(rng.randint(110, int(220 * difficulty)))
    plant_width = int(rng.randint(2, int(8 * difficulty)))
    space = round(rng.uniform(1.25, 2.0) * difficulty, 2)
    owned = int(rng.randint(10, int(30 * difficulty)))
    cost = int(rng.randint(3, int(15 * difficulty)))

    # Ensure conditions are met
    while not (plant_width * 3 < length and 
              plant_width < space and
              length % space == 0 and
              length / space > owned):
        length = int(rng.randint(110, int(220 * difficulty)))
        space = round(rng.uniform(1.25, 2.0) * difficulty, 2)
        owned = int(rng.randint(10, int(30 * difficulty)))

    result = generate_from_variables(name, length, unit, plant_width, space, owned, 
                                   currency, cost)
    
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
    return generate_from_variables('Pat', 111, 'feet', 12, 1.5, 17, '$', 6)
