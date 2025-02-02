from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, task: str, profession: str, hours: int, 
                          work_type: str, rate: int, fee: int, currency: str) -> Dict[str, Any]:
    lost_income = hours * rate
    savings = lost_income - fee
    
    question = f"{name} is trying to decide whether to do {task} herself or hire an {profession}. If she does it herself, she'll be able to do {hours} fewer hours of {work_type} work, losing {currency}{rate}/hour in missed income. The {profession} charges {currency}{fee}. How much more money will she have if she hires the {profession}?"
    
    answer_cot = f"First find the total lost revenue if {name} does {task} herself: {currency}{rate}/hour * {hours} hours = {currency}{lost_income}\nThen subtract the {profession}'s charge to find how much money {name} saves: {currency}{lost_income} - {currency}{fee} = {currency}{savings}\n#### {savings}"

    return {
        'question': question,
        'answer': str(savings),
        'answer_cot': answer_cot,
        'answer_value': savings,
        'variables': {
            'name': name,
            'task': task,
            'profession': profession,
            'hours': hours,
            'work_type': work_type,
            'hourly_rate': rate,
            'fee': fee,
            'currency': currency,
            'lost_income': lost_income
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ['Emma', 'Sophia', 'Isabella', 'Olivia', 'Ava', 'Mia', 'Emily']
    tasks = ["her taxes", "her financial planning", "her business accounting"]
    professions = ["accountant", "financial advisor", "tax consultant", "bookkeeper"]
    work_types = ["freelance", "consulting", "part-time", "contract"]
    currencies = ['$', '€', '£']

    name = rng.choice(names_female)
    task = rng.choice(tasks)
    profession = rng.choice(professions)
    work_type = rng.choice(work_types)
    currency = rng.choice(currencies)
    
    hours = int(rng.randint(4, int(14 * difficulty)))
    rate = int(rng.randint(20, int(100 * difficulty)))
    fee = int(rng.randint(50, int(200 * difficulty)))
    
    # Ensure conditions are met
    while hours * rate <= fee:
        hours = int(rng.randint(4, int(14 * difficulty)))
        rate = int(rng.randint(20, int(100 * difficulty)))
        fee = int(rng.randint(50, int(200 * difficulty)))

    result = generate_from_variables(name, task, profession, hours, work_type, 
                                   rate, fee, currency)
    
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
    return generate_from_variables("Jackie", "her taxes", "accountant", 3, 
                                 "freelance", 35, 90, "$")
