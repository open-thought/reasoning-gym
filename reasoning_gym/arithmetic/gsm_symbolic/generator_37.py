from random import Random
from typing import Dict, Any

def generate_from_variables(worker: str, base: int, unit: str, tool1: str, 
                          tool2: str, tool3: str, mult1: int, mult2: int,
                          n: int, days: int) -> Dict[str, Any]:
    iron_amount = base * mult1
    steel_amount = int(iron_amount * (1 + mult2/100))
    daily_total = steel_amount * n
    total_amount = daily_total * days

    question = f"One {worker} can mine {base} {unit} of ore per day with {tool1}. He can mine {mult1} times as much with a {tool2} and {mult2}% more with a {tool3} than with a {tool2}. How many {unit} of ore can {n} {worker}s with {tool3}s mine in a month with {days} days?"
    
    answer_cot = f"First find how much ore a {worker} can mine with a {tool2}: {base} {unit}/day * {mult1} = {iron_amount} {unit}/day\n" \
                 f"Then multiply that amount by {100+mult2}% to find how much a {worker} can mine with a {tool3}: {iron_amount} {unit}/day * {100+mult2}% = {steel_amount} {unit}/day\n" \
                 f"Then multiply the amount one {worker} can mine in a day with a {tool3} by the number of {worker}s: {steel_amount} {unit}/day/{worker} * {n} {worker}s = {daily_total} {unit}/day\n" \
                 f"Then multiply the daily amount of ore by the number of days to find the total ore mined in a month: {daily_total} {unit}/day * {days} days = {total_amount} {unit}/day\n" \
                 f"#### {total_amount}"

    return {
        'question': question,
        'answer': str(total_amount),
        'answer_cot': answer_cot,
        'answer_value': total_amount,
        'variables': {
            'worker': worker,
            'base_amount': base,
            'unit': unit,
            'tool1': tool1,
            'tool2': tool2,
            'tool3': tool3,
            'mult1': mult1,
            'mult2': mult2,
            'num_workers': n,
            'num_days': days,
            'iron_amount': iron_amount,
            'steel_amount': steel_amount,
            'daily_total': daily_total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    workers = ["miner", "goblin", "gnome", "troll"] 
    tools1 = ["bare hands", "basic shovel", "wooden pickaxe"]
    units = ["pounds", "kgs"]
    tools2 = ["nickel pickaxe", "bronze pickaxe", "silver pickaxe"]
    tools3 = ["steel pickaxe", "diamond pickaxe", "mithril pickaxe", "titanium pickaxe"]

    worker = rng.choice(workers)
    tool1 = rng.choice(tools1)
    unit = rng.choice(units)
    tool2 = rng.choice(tools2)
    tool3 = rng.choice(tools3)

    base = int(rng.randint(5, int(20 * difficulty)))
    mult1 = int(rng.randint(2, int(4 * difficulty)))
    mult2 = int(rng.randint(30, int(80 * difficulty)) // 5 * 5)
    n = int(rng.randint(20, int(50 * difficulty)))
    days = int(rng.randint(28, 32))

    # Verify conditions
    while not (base * mult1 * (1 + mult2/100)).is_integer() or \
          int(base * mult1 * (1 + mult2/100) * n * days) >= 100000:
        base = int(rng.randint(5, int(20 * difficulty)))
        mult1 = int(rng.randint(2, int(4 * difficulty))) 
        mult2 = int(rng.randint(30, int(80 * difficulty)) // 5 * 5)
        n = int(rng.randint(20, int(50 * difficulty)))

    result = generate_from_variables(worker, base, unit, tool1, tool2, tool3, 
                                   mult1, mult2, n, days)

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
    return generate_from_variables("dwarf", 12, "pounds", "his bare hands",
                                 "iron pickaxe", "steel pickaxe", 2, 50, 40, 30)
