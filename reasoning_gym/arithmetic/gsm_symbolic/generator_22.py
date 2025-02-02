from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, property_type: str, budget: int, price: int,
                          brokerage_fee: int, transfer_fee: int) -> Dict[str, Any]:
    brokerage_amount = int(price * brokerage_fee / 100)
    transfer_amount = int(price * transfer_fee / 100)
    total_price = price + brokerage_amount + transfer_amount
    difference = total_price - budget

    question = f"{name} is looking for a {property_type} that will not go beyond her ${budget:,} budget. She saw a property that has a selling price of ${price:,}. On top of that, the buyer has to pay a brokerage fee which is {brokerage_fee}% of the selling price, and also the transfer fee that is {transfer_fee}% of the selling price. How much more is the total price of the {property_type} than {name}'s budget?"

    answer_cot = f"The brokerage fee is ${price:,} x {brokerage_fee}/100 = ${brokerage_amount:,}.\nThe transfer fee is ${price:,} x {transfer_fee}/100 = ${transfer_amount:,}.\nThe total price of the {property_type} is ${price:,} + ${brokerage_amount:,} + ${transfer_amount:,} = ${total_price:,}.\nSo, it is ${total_price:,} - ${budget:,} = ${difference:,} more than {name}'s budget.\n#### {difference}"

    return {
        'question': question,
        'answer': f'{difference}',
        'answer_cot': answer_cot,
        'answer_value': difference,
        'variables': {
            'name': name,
            'property_type': property_type,
            'budget': budget,
            'price': price,
            'brokerage_fee': brokerage_fee,
            'transfer_fee': transfer_fee,
            'brokerage_amount': brokerage_amount,
            'transfer_amount': transfer_amount,
            'total_price': total_price
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['Mrs. Smith', 'Ms. Johnson', 'Dr. Patel', 'Mrs. Lee']
    property_types = ['house', 'apartment', 'condo', 'townhouse']
    
    name = rng.choice(names)
    property_type = rng.choice(property_types)
    
    # Scale ranges by difficulty while maintaining integer results
    budget = int(rng.randrange(300000, int(500000 * difficulty), 10000))
    price = int(rng.randrange(250000, budget, 10000))
    brokerage_fee = int(rng.randint(3, 8))
    transfer_fee = int(rng.randint(10, 15))
    
    # Verify conditions
    while True:
        total_cost = price * (1 + brokerage_fee/100 + transfer_fee/100)
        if (total_cost > budget + 1 and 
            price * brokerage_fee % 100 == 0 and 
            price * transfer_fee % 100 == 0):
            break
        price = int(rng.randrange(250000, budget, 10000))
    
    result = generate_from_variables(name, property_type, budget, price, 
                                   brokerage_fee, transfer_fee)
    
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
    return generate_from_variables('Mrs. Cruz', 'house', 400000, 350000, 5, 12)
