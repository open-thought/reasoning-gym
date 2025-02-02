from random import Random
from typing import Dict, Any

def generate_from_variables(group: str, n: int, n_1: int, n_2: int, 
                          hobby1: str, hobby2: str, hobby3: str, hobby4: str) -> Dict[str, Any]:
    n_4 = 2 * n_2  # number that like hobby4 (music)
    n_3 = n - (n_1 + n_2 + n_4)  # number that like hobby3 (video games)
    
    question = f"A {group} of {n} students has various hobbies. {n_1} like to {hobby1}, {n_2} like to play {hobby2}, and the rest like to either {hobby3} or {hobby4}. How many like to {hobby3} if the number that like to {hobby4} is twice the number that prefer playing {hobby2}?"
    
    answer_cot = f"The number of students that like to {hobby4} is twice as many as the number who like {hobby2}, so 2 * {n_2} = {n_4}\nThe number that like to {hobby3} is {n} total students - {n_1} {hobby1} - {n_2} {hobby2} - {n_4} {hobby4} = {n_3}\n#### {n_3}"
    
    return {
        'question': question,
        'answer': str(n_3),
        'answer_cot': answer_cot,
        'answer_value': n_3,
        'variables': {
            'group_type': group,
            'total_students': n,
            'hobby1_count': n_1,
            'hobby2_count': n_2,
            'hobby3_count': n_3,
            'hobby4_count': n_4,
            'hobby1': hobby1,
            'hobby2': hobby2,
            'hobby3': hobby3,
            'hobby4': hobby4
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    groups = ["group", "class"]
    hobbies = ['read', 'paint', 'hike', 'dance', 'bake', 'play video games', 'play music'] 
    sports = ['basketball', 'soccer', 'tennis', 'baseball', 'volleyball']
    
    group = rng.choice(groups)
    hobby2 = rng.choice(sports)
    hobby1, hobby3, hobby4 = rng.sample([h for h in hobbies if h not in [hobby2]], 3)
    
    # Generate numbers that satisfy conditions
    n = int(rng.randint(20, int(200 * difficulty)))
    n_2 = int(rng.randint(2, n//6))  # Keep n_2 small since we multiply by 2
    n_1 = int(rng.randint(2, n//3))
    
    # Verify n_1 + n_2 + (2*n_2) < n
    while n_1 + 3*n_2 >= n:
        n = int(rng.randint(20, int(200 * difficulty)))
        n_2 = int(rng.randint(2, n//6))
        n_1 = int(rng.randint(2, n//3))
        
    result = generate_from_variables(group, n, n_1, n_2, hobby1, hobby2, hobby3, hobby4)
    
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
    return generate_from_variables("class", 50, 10, 5, "bake", "basketball", 
                                 "play video games", "play music")
