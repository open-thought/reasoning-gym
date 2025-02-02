from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, family: str, blocks: int, animals: int, 
                          rings: int, total: int) -> Dict[str, Any]:
    bouncy_balls = total - (blocks + animals + rings)
    
    question = f"When {name} watches her {family}, she gets out a variety of toys for him. The bag of building blocks has {blocks} blocks in it. The bin of stuffed animals has {animals} stuffed animals inside. The tower of stacking rings has {rings} multicolored rings on it. {name} recently bought a tube of bouncy balls, bringing her total number of toys for her {family} up to {total}. How many bouncy balls came in the tube?"
    
    answer_cot = f"Let T be the number of bouncy balls in the tube.\nAfter buying the tube of balls, {name} has {blocks} + {animals} + {rings} + T = {blocks + animals + rings} + T = {total} toys for her {family}.\nThus, T = {total} - {blocks + animals + rings} = {bouncy_balls} bouncy balls came in the tube.\n#### {bouncy_balls}"

    return {
        'question': question,
        'answer': str(bouncy_balls),
        'answer_cot': answer_cot,
        'answer_value': bouncy_balls,
        'variables': {
            'name': name,
            'family': family,
            'building_blocks': blocks,
            'stuffed_animals': animals,
            'stacking_rings': rings,
            'total_toys': total,
            'bouncy_balls': bouncy_balls
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ['Sophie', 'Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia']
    family_members = ['nephew', 'cousin', 'brother']
    
    name = rng.choice(names_female)
    family = rng.choice(family_members)
    
    blocks = int(rng.randint(70, int(75 * difficulty)))
    animals = int(rng.randint(35, int(50 * difficulty)))
    rings = int(rng.randint(20, int(35 * difficulty)))
    
    total = blocks + animals + rings + int(rng.randint(20, int(100 * difficulty)))
    
    result = generate_from_variables(name, family, blocks, animals, rings, total)
    
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
    return generate_from_variables('Sophie', 'nephew', 31, 8, 9, 62)
