from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, name3: str, name4: str, name5: str,
                          num_dozen: int, found_first: int, multiplier: float, 
                          less_amount: int, fraction: float) -> Dict[str, Any]:
    total_eggs = num_dozen * 12
    found_second = found_first * multiplier
    found_third = found_second - less_amount 
    found_fourth = found_third * fraction
    total_found = found_first + found_second + found_third + found_fourth
    remaining = total_eggs - total_found

    question = f"{name1} hid {num_dozen} dozen eggs in the yard for the Easter egg hunt. {name2} finds {found_first} eggs. {name3} finds {multiplier:.0f} times as many as {name2}. {name4} finds {less_amount} less than {name3}, and {name5} finds {fraction:.1f} as many as {name4}. How many eggs are still hidden in the yard?"

    answer_cot = f"{name1} hides {num_dozen} x 12 = {total_eggs} eggs.\n"
    answer_cot += f"{name2} finds {found_first} eggs.\n"
    answer_cot += f"{name3} finds {found_first} x {multiplier:.0f} = {found_second} eggs.\n"
    answer_cot += f"{name4} finds {found_second} - {less_amount} = {found_third} eggs.\n"
    answer_cot += f"{name5} finds {found_third} x {fraction:.1f} = {found_fourth} eggs.\n"
    answer_cot += f"The children find a total of {found_first} + {found_second} + {found_third} + {found_fourth} = {total_found} eggs.\n"
    answer_cot += f"The total number of hidden eggs still in the yard is {total_eggs} - {total_found} = {remaining} eggs.\n#### {remaining}"

    return {
        'question': question,
        'answer': str(remaining),
        'answer_cot': answer_cot,
        'answer_value': remaining,
        'variables': {
            'name1': name1,
            'name2': name2,
            'name3': name3,
            'name4': name4,
            'name5': name5,
            'num_dozen': num_dozen,
            'found_first': found_first,
            'multiplier': multiplier,
            'less_amount': less_amount,
            'fraction': fraction,
            'total_eggs': total_eggs,
            'total_found': total_found,
            'remaining': remaining
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Oliver", "Isabella", "William", 
             "Sophia", "James", "Charlotte", "Benjamin", "Mia", "Lucas", "Harper"]
    
    name1, name2, name3, name4, name5 = rng.sample(names, 5)
    
    num_dozen = int(rng.randint(2, int(10 * difficulty)))
    found_first = int(rng.randint(3, int(15 * difficulty)))
    multiplier = 2.0  # Using 'twice' as specified
    less_amount = int(rng.randint(1, int(5 * difficulty)))
    fraction = 0.5  # Using 'half' as specified
    
    # Ensure all conditions are met
    total = num_dozen * 12
    found_second = found_first * multiplier
    found_third = found_second - less_amount
    found_fourth = found_third * fraction
    total_found = found_first + found_second + found_third + found_fourth
    
    # Regenerate if conditions not met
    while (not found_third > 0 or not total > total_found or 
           not float(found_fourth).is_integer()):
        num_dozen = int(rng.randint(2, int(10 * difficulty)))
        found_first = int(rng.randint(3, int(15 * difficulty)))
        less_amount = int(rng.randint(1, int(5 * difficulty)))
        total = num_dozen * 12
        found_second = found_first * multiplier
        found_third = found_second - less_amount
        found_fourth = found_third * fraction
        total_found = found_first + found_second + found_third + found_fourth

    result = generate_from_variables(name1, name2, name3, name4, name5,
                                   num_dozen, found_first, multiplier, 
                                   less_amount, fraction)
    
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
    return generate_from_variables("Cole", "Lamar", "Stacy", "Charlie", "Mei",
                                 3, 5, 2.0, 2, 0.5)
