from random import Random
from typing import Dict, Any
from fractions import Fraction

def generate_from_variables(name: str, event: str, organization: str, fraction: str, 
                          current: int, total: int, currency: str) -> Dict[str, Any]:
    fraction = convert_fraction_word(fraction)
    fraction_val = Fraction(fraction)
    org_amount = int(total * fraction_val)
    covered_amount = org_amount + current
    missing_amount = total - covered_amount
    
    question = f"{name} is raising money for a {event}. He has applied for help from the {organization}, which has decided to cover {fraction} of the cost of the {event}. How much money is {name} missing if he has {currency}{current} and the {event} costs {currency}{total}?"
    
    answer_cot = f"{name}'s {organization} has decided to pay {total} * {fraction} = {currency}{org_amount} for his {event}.\nIn total {name} has covered {org_amount} + {current} = {currency}{covered_amount} for his {event}\nTherefore, {name} needs {total} - {covered_amount} = {currency}{missing_amount} more for the {event}.\n#### {missing_amount}"
    
    return {
        'question': question,
        'answer': str(missing_amount),
        'answer_cot': answer_cot,
        'answer_value': missing_amount,
        'variables': {
            'name': name,
            'event': event,
            'organization': organization,
            'fraction': fraction,
            'current_amount': current,
            'total_cost': total,
            'currency': currency,
            'org_contribution': org_amount,
            'covered_amount': covered_amount
        }
    }

def convert_fraction_word(fraction_str: str) -> str:
    """Convert word fractions to numeric form"""

    # Add fraction word mapping
    FRACTION_WORDS = {
    "half": "1/2",
    "one-half": "1/2",
    "quarter": "1/4",
    }
    return FRACTION_WORDS.get(fraction_str.lower(), fraction_str)

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["John", "Michael", "David", "James", "William", "Robert", "Joseph"]
    events = ["field trip", "sports tournament", "conference", "music festival", "science fair"]
    organizations = ["school", "community center", "local charity", "youth club", "parent association"]
    currencies = ["$", "€", "£"]
    fractions = ["half", "1/2", "one-half"]
    
    name = rng.choice(names)
    event = rng.choice(events)
    organization = rng.choice(organizations)
    currency = rng.choice(currencies)
    fraction = rng.choice(fractions)
    fraction = convert_fraction_word(fraction)
    
    # Scale ranges by difficulty but ensure results are integers
    current = int(rng.randint(10, int(200 * difficulty)) // 5 * 5)
    total = int(rng.randint(200, int(1000 * difficulty)) // 10 * 10)
    
    # Ensure conditions are met
    while current >= total or not float(total * Fraction(fraction)).is_integer():
        current = int(rng.randint(10, int(200 * difficulty)) // 5 * 5)
        total = int(rng.randint(200, int(1000 * difficulty)) // 10 * 10)
    
    result = generate_from_variables(name, event, organization, fraction, current, total, currency)
    
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
    return generate_from_variables("John", "school trip", "school", "half", 50, 300, "$")
