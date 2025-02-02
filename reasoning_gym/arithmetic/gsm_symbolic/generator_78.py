from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, age_diff: int, age1: int) -> Dict[str, Any]:
    age2 = age1 + age_diff
    avg_age = (age1 + age2) // 2
    
    question = f"{name1} and {name2} are currently {age_diff} years apart in age. If {name1}, who is younger than {name2}, is {age1} years old, what's the average of their ages?"
    
    answer_cot = f"If {name1} is {age1} years old, {name2} is {age1}+{age_diff} = {age2} years old.\n" \
                 f"The sum of their ages is {age2}+{age1} = {age1+age2} years\n" \
                 f"The average age of the two is {age1+age2}/2 = {avg_age} years\n" \
                 f"#### {avg_age}"

    return {
        'question': question,
        'answer': str(avg_age),
        'answer_cot': answer_cot,
        'answer_value': avg_age,
        'variables': {
            'name1': name1,
            'name2': name2,
            'age_diff': age_diff,
            'age1': age1,
            'age2': age2,
            'avg_age': avg_age
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia"]
    name1, name2 = rng.sample(names, 2)
    
    age_diff = int(rng.randint(5, int(30 * difficulty)))
    age1 = int(rng.randint(15, int(75 * difficulty)))
    
    # Ensure average is an integer
    while (2 * age1 + age_diff) % 2 != 0:
        age1 = int(rng.randint(15, int(75 * difficulty)))
    
    result = generate_from_variables(name1, name2, age_diff, age1)
    
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
    return generate_from_variables("Mia", "Emma", 16, 40)
