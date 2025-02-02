from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, color1: str, color2: str, n1: int, n2: int, 
                          n3: int, n4: int) -> Dict[str, Any]:
    blue_spools = n1 + n2
    total_spools = n1 + n2 + n3 + n4
    percent_blue = int(100 * blue_spools / total_spools)
    
    question = f"{name} has {n1} light {color1} spools of thread, {n2} dark {color1} spools of thread, {n3} light {color2} spools of thread, and {n4} dark {color2} spools of thread. What percent of her spools are {color1}?"
    
    answer_cot = f"First find the number of {color1} spools: {n1} spools + {n2} spools = {blue_spools} spools\nThen find the total number of spools: {n3} spools + {n4} spools + {blue_spools} spools = {total_spools} spools\nThen divide the number of {color1} spools by the total number of spools and multiply by 100% to express the answer as a percentage: {blue_spools} spools / {total_spools} spools * 100% = {percent_blue}%\n#### {percent_blue}"
    
    return {
        'question': question,
        'answer': str(percent_blue),
        'answer_cot': answer_cot,
        'answer_value': percent_blue,
        'variables': {
            'name': name,
            'color1': color1,
            'color2': color2,
            'light_color1_spools': n1,
            'dark_color1_spools': n2,
            'light_color2_spools': n3,
            'dark_color2_spools': n4,
            'total_color1_spools': blue_spools,
            'total_spools': total_spools,
            'percent_color1': percent_blue
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Candy", "Sarah", "Emma", "Olivia", "Isabella", "Sophia", "Mia", "Charlotte"]
    colors = ["blue", "red", "green", "yellow", "purple", "orange"]
    
    name = rng.choice(names)
    color1, color2 = rng.sample(colors, 2)
    
    # Generate numbers ensuring integer percentage result
    n1 = int(rng.randint(15, int(45 * difficulty)))
    n2 = int(rng.randint(45, int(100 * difficulty)))
    n3 = int(rng.randint(20, int(80 * difficulty)))
    n4 = int(rng.randint(50, int(100 * difficulty)))
    
    # Ensure percentage is integer
    total = n1 + n2 + n3 + n4
    while ((n1 + n2) * 100) % total != 0:
        n4 += 1
        total = n1 + n2 + n3 + n4
    
    result = generate_from_variables(name, color1, color2, n1, n2, n3, n4)
    
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
    return generate_from_variables("Candy", "blue", "green", 15, 45, 40, 50)
