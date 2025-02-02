from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, alphabets: tuple, n1: str, frac: str) -> Dict[str, Any]:
    alphabet_name, alphabet_count = alphabets
    
    # Calculate intermediate values
    full_writes = n1 * alphabet_count
    half_write = int(alphabet_count * frac) 
    subtotal = full_writes + half_write
    final_total = subtotal * 2

    question = f"{name} is learning to write and decides to keep re-writing the {alphabet_name} until she knows it. She writes it in full {n1}, writes {frac} of it once, then re-writes everything she has already written. How many letters has {name} written in total?"

    answer_cot = f"{name} has written the {alphabet_name} {n1} time(s) which is a total of {alphabet_count} * {n1} = {full_writes} letters.\n" \
                 f"She then writes {frac} the {alphabet_name}, which is {alphabet_count} * {frac} = {half_write} letters.\n" \
                 f"So far, this is a total of {full_writes} + {half_write} = {subtotal} letters.\n" \
                 f"Writing this again means she has doubled the number of letters she has written, so she has written a total of {subtotal} * 2 = {final_total} letters.\n" \
                 f"#### {final_total}"

    return {
        'question': question,
        'answer': str(final_total),
        'answer_cot': answer_cot,
        'answer_value': final_total,
        'variables': {
            'name': name,
            'alphabet_name': alphabet_name,
            'alphabet_count': alphabet_count,
            'times_written': n1,
            'fraction': frac,
            'full_writes': full_writes,
            'half_write': half_write,
            'total': final_total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ["Emma", "Sophia", "Olivia", "Ava", "Isabella", "Mia", "Charlotte", "Amelia"]
    alphabets = [("alphabet", 26), ("hiragana (with 48 letters)", 48), 
                ("farsi alphabet (with 32 letters)", 32), ("arabic abjad (with 28 letters)", 28)]
    multi_times = ["twice", "three times", "four times"]
    fraction_alnum = ["half", "one-third", "one-fourth"]
    
    name = rng.choice(names_female)
    alphabet = rng.choice(alphabets)
    n1 = rng.choice(multi_times)
    frac = rng.choice(fraction_alnum)
    
    # Convert text numbers to numeric values
    n1_map = {"twice": 2, "three times": 3, "four times": 4}
    frac_map = {"half": 0.5, "one-third": 1/3, "one-fourth": 0.25}
    
    # Ensure division results in integer
    while not (alphabet[1] * frac_map[frac]).is_integer():
        alphabet = rng.choice(alphabets)
    
    result = generate_from_variables(name, alphabet, n1_map[n1], frac_map[frac])
    
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
    return generate_from_variables("Elise", ("alphabet", 26), 2, 0.5)
