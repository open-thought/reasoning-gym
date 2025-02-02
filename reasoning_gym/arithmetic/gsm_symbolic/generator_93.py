from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, name3: str, name4: str,
                          creature: str, weapon1: str, weapon2: str, weapon3: str, 
                          weapon4: str, weapon5: str, n1: int, frac1: float,
                          mult1: int, frac2: float) -> Dict[str, Any]:
    
    kills_arthur = int(n1 * frac1)
    kills_walter = int(kills_arthur * mult1) 
    kills_bruce = int(kills_walter * frac2)

    question = f"{name1} slew {n1} {creature} with his mighty {weapon1}, while {name2}, using a {weapon2}, slew {frac1} as many {creature} as {name1}. Using a {weapon3}, {name3} slew {mult1} as many {creature} as {name2}. But {name4}, having forgot his {weapon4} at home, slew {frac2} as many {creature} as {name3} using a {weapon5}. How many {creature} has {name4} slain?"

    answer_cot = f"{name2} slew {frac1} as many {creature} as {name1}, or {n1}*{frac1}={kills_arthur} {creature}.\n{name3} slew {mult1} as many {creature} as {name2}, or {mult1}*{kills_arthur}={kills_walter} {creature}.\n{name4} slew {frac2} as many {creature} as {name3}, or {kills_walter}*{frac2}={kills_bruce} {creature}.\n#### {kills_bruce}"

    return {
        'question': question,
        'answer': str(kills_bruce),
        'answer_cot': answer_cot,
        'answer_value': kills_bruce,
        'variables': {
            'name1': name1,
            'name2': name2,
            'name3': name3, 
            'name4': name4,
            'creature': creature,
            'weapon1': weapon1,
            'weapon2': weapon2,
            'weapon3': weapon3,
            'weapon4': weapon4,
            'weapon5': weapon5,
            'initial_kills': n1,
            'fraction1': frac1,
            'multiplier': mult1,
            'fraction2': frac2,
            'kills_arthur': kills_arthur,
            'kills_walter': kills_walter,
            'kills_bruce': kills_bruce
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_male = ["Arthur", "Bruce", "Charles", "David", "Edward", "Frederick", "George", "Henry"]
    creatures = ["ogres", "trolls", "goblins", "orcs", "giants"]
    weapons1 = ["sword", "mace", "battle axe", "war hammer"]
    weapons2 = ["spear", "lance", "javelin", "halberd"]
    weapons3 = ["rusty iron axe", "wooden club", "stone hammer", "bone dagger"]
    weapons4 = ["sword", "axe", "mace", "hammer"]
    weapons5 = ["nail file", "butter knife", "wooden spoon", "feather"]
    fractions = [0.25, 0.5, 0.75]
    multipliers = [2, 3, 4]

    name1, name2, name3, name4 = rng.sample(names_male, 4)
    creature = rng.choice(creatures)
    weapon1 = rng.choice(weapons1)
    weapon2 = rng.choice(weapons2)
    weapon3 = rng.choice(weapons3)
    weapon4 = rng.choice(weapons4)
    weapon5 = rng.choice(weapons5)
    
    # Scale numbers by difficulty but ensure integer results
    n1 = int(rng.randint(50, int(500 * difficulty)) // 50 * 50)
    frac1 = rng.choice(fractions)
    mult1 = rng.choice(multipliers)
    frac2 = rng.choice(fractions)
    
    # Ensure all divisions result in integers
    while not (n1 * frac1).is_integer() or not (n1 * frac1 * mult1).is_integer() or \
          not (n1 * frac1 * mult1 * frac2).is_integer():
        n1 = int(rng.randint(50, int(500 * difficulty)) // 50 * 50)

    result = generate_from_variables(name1, name2, name3, name4, creature,
                                   weapon1, weapon2, weapon3, weapon4, weapon5,
                                   n1, frac1, mult1, frac2)

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
    return generate_from_variables("Prince Thaddeus", "Prince Arthur", "Prince Walter", "Prince Bruce",
                                 "dragons", "sword", "spear", "rusty iron axe", "sword", "nail file",
                                 100, 0.75, 2, 0.2)
