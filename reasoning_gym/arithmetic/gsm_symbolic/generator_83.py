from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, total: int, diff: int, unit: str) -> Dict[str, Any]:
    amount1 = (total - diff) // 2  # Sam's amount
    amount2 = amount1 + diff       # Harry's amount
    
    question = f"If {name1} and {name2} have {total} {unit} of fence between them, and they agree to split it with {name2} getting {diff} {unit} more than {name1}, how much is left over for {name1}?"
    
    answer_cot = f"Let x be the amount of fence {name1} gets and y be the amount {name2} gets. We know that y = x + {diff}, and y + x = {total}.\nSubstituting the first equation into the second equation, we get 2x+{diff}={total}\nSubtracting the {diff} from both sides, we get 2x={total-diff}\nWe divide each side by two, leaving x={amount1}. This means {name1} has {amount1} {unit} of fence left over.\n#### {amount1}"

    return {
        'question': question,
        'answer': str(amount1),
        'answer_cot': answer_cot,
        'answer_value': amount1,
        'variables': {
            'name1': name1,
            'name2': name2,
            'total_fence': total,
            'difference': diff,
            'unit': unit,
            'amount1': amount1,
            'amount2': amount2
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['Sam', 'Harry', 'Tom', 'John', 'Mike', 'Dave', 'Steve', 'Bob']
    units = ['feet', 'yards', 'meters']
    
    name1, name2 = rng.sample(names, 2)
    unit = rng.choice(units)
    
    # Scale ranges by difficulty while maintaining integer division
    diff = int(rng.randrange(20, int(200 * difficulty), 10))
    total = int(rng.randrange(diff + 20, int(1000 * difficulty), 20))
    
    # Ensure conditions are met
    while total - diff <= 10 or (total - diff) % 2 != 0:
        diff = int(rng.randrange(20, int(200 * difficulty), 10))
        total = int(rng.randrange(diff + 20, int(1000 * difficulty), 20))
    
    result = generate_from_variables(name1, name2, total, diff, unit)
    
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
    return generate_from_variables('Sam', 'Harry', 100, 60, 'feet')
