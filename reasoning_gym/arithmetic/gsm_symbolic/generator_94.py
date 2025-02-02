from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, num_shares: int, price_per_share: int, 
                          increase_pct: int, decrease_pct: int) -> Dict[str, Any]:
    initial_value = num_shares * price_per_share
    first_increase = initial_value * increase_pct/100
    value_after_increase = initial_value + first_increase 
    second_decrease = value_after_increase * decrease_pct/100
    final_value = value_after_increase - second_decrease
    
    question = f"{name} buys {num_shares} shares of a stock for ${price_per_share} each. The stock price increases {increase_pct}% the first year {name} holds it, then decreases {decrease_pct}% in the second year. What is the final value of all {name}'s shares?"
    
    answer_cot = f"First find the initial total value of {name}'s purchase: {num_shares} shares * ${price_per_share}/share = ${initial_value}\n" \
                 f"Then find the amount of the first price increase: ${initial_value} * {increase_pct/100} = ${int(first_increase)}\n" \
                 f"Add that amount to the initial value to find the value after the first year: ${initial_value} + ${int(first_increase)} = ${int(value_after_increase)}\n" \
                 f"Then multiply that amount by {decrease_pct}% to find the amount of the decrease in the second year: ${int(value_after_increase)} * {decrease_pct}% = ${int(second_decrease)}\n" \
                 f"Then subtract that amount from the value after the first year to find the final value: ${int(value_after_increase)} - ${int(second_decrease)} = ${int(final_value)}\n" \
                 f"#### {int(final_value)}"

    return {
        'question': question,
        'answer': str(int(final_value)),
        'answer_cot': answer_cot,
        'answer_value': int(final_value),
        'variables': {
            'name': name,
            'num_shares': num_shares,
            'price_per_share': price_per_share,
            'increase_pct': increase_pct,
            'decrease_pct': decrease_pct,
            'initial_value': initial_value,
            'final_value': int(final_value)
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['Maria', 'Sarah', 'Emma', 'Isabella', 'Sophia', 'Mia', 'Charlotte']
    name = rng.choice(names)
    
    num_shares = int(rng.randint(5, int(20 * difficulty)))
    price_per_share = int(rng.randint(5, int(100 * difficulty)))
    increase_pct = int(rng.randint(10, int(100 * difficulty)) // 5 * 5)
    decrease_pct = int(rng.randint(5, int(50 * difficulty)) // 5 * 5)
    
    # Ensure integer results
    while not (num_shares * price_per_share * increase_pct/100).is_integer() or \
          not (num_shares * price_per_share * (1 + increase_pct/100) * (1 - decrease_pct/100)).is_integer():
        num_shares = int(rng.randint(5, int(20 * difficulty)))
        price_per_share = int(rng.randint(5, int(100 * difficulty)))
        increase_pct = int(rng.randint(10, int(100 * difficulty)) // 5 * 5)
        decrease_pct = int(rng.randint(5, int(50 * difficulty)) // 5 * 5)
    
    result = generate_from_variables(name, num_shares, price_per_share, increase_pct, decrease_pct)
    
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
    return generate_from_variables("Maria", 8, 8, 50, 25)
