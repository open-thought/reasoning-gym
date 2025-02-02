from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, name3: str, platform: str,
                          mult1: int, mult2: int, n: int) -> Dict[str, Any]:
    base_friends = n // mult1  # Dorothy's friends
    charlie_friends = n  # Charlie's friends 
    james_friends = base_friends * mult2  # James's friends
    
    question = f"{name1} has {mult1} times as many {platform} friends as {name2}. {name3} has {mult2} times as many friends on {platform} as {name2}. If {name1} has {n} friends on {platform}, how many {platform} friends does {name3} have?"
    
    answer_cot = f"{name2} has {n} / {mult1} = {base_friends} {platform} friends.\n{name3} has {mult2} * {base_friends} = {james_friends} {platform} friends.\n#### {james_friends}"
    
    return {
        'question': question,
        'answer': str(james_friends),
        'answer_cot': answer_cot,
        'answer_value': james_friends,
        'variables': {
            'name1': name1,
            'name2': name2,
            'name3': name3,
            'platform': platform,
            'mult1': mult1,
            'mult2': mult2,
            'base_friends': base_friends,
            'charlie_friends': charlie_friends,
            'james_friends': james_friends
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['Charlie', 'Dorothy', 'James', 'Sarah', 'Michael', 'Emily', 'David']
    platforms = ['Instagram', 'Twitter', 'LinkedIn', 'TikTok', 'Snapchat']
    
    name1, name2, name3 = rng.sample(names, 3)
    platform = rng.choice(platforms)
    
    # Generate multipliers that will be different
    mult1 = rng.randint(2, int(5 * difficulty))
    mult2 = rng.randint(2, int(5 * difficulty))
    while mult2 == mult1:
        mult2 = rng.randint(2, int(5 * difficulty))
        
    # Generate n that's divisible by mult1
    base = rng.randint(4, int(20 * difficulty))
    n = base * mult1
    
    result = generate_from_variables(name1, name2, name3, platform, mult1, mult2, n)
    
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
    return generate_from_variables('Charlie', 'Dorothy', 'James', 'Facebook', 
                                 3, 4, 12)
