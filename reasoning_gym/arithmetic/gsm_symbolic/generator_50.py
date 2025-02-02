from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, pieces1: int, pieces2: int) -> Dict[str, Any]:
    half_pieces1 = pieces1 // 2
    total_pieces = half_pieces1 + pieces2
    
    question = f"{name} finished half of a {pieces1} piece puzzle, and then started and finished another {pieces2} piece puzzle within an hour. How many puzzle pieces did {name} place during that hour?"
    
    answer_cot = f"{name} did 1/2 * {pieces1} pieces = {half_pieces1} pieces.\n{name} completed {half_pieces1} pieces + {pieces2} pieces = {total_pieces} pieces.\n#### {total_pieces}"
    
    return {
        'question': question,
        'answer': str(total_pieces),
        'answer_cot': answer_cot,
        'answer_value': total_pieces,
        'variables': {
            'name': name,
            'puzzle1_pieces': pieces1,
            'puzzle2_pieces': pieces2,
            'half_puzzle1': half_pieces1,
            'total_pieces': total_pieces
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Teddy", "Tommy", "Billy", "Jimmy", "Bobby", "Danny"]
    name = rng.choice(names)
    
    # Generate random puzzle sizes that are even numbers
    puzzle1 = int(rng.randint(100, int(500 * difficulty)) // 2 * 2)
    puzzle2 = int(rng.randint(300, int(1000 * difficulty)) // 2 * 2)
    
    result = generate_from_variables(name, puzzle1, puzzle2)
    
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
    return generate_from_variables("Teddy", 500, 500)
