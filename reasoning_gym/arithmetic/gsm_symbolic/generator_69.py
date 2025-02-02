from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, document: str, total_pages: int, fraction: str) -> Dict[str, Any]:
    frac_num = eval(fraction)
    pages_done = int(total_pages * frac_num)
    pages_remaining = total_pages - pages_done

    question = f"{name} is required to submit a {total_pages}-page {document}. She already finished writing {fraction} of the {document}. How many pages does she have left to write?"
    
    answer_cot = f"{name} has already written {fraction} of the {document} which is {total_pages} pages x {fraction} = {pages_done} pages.\nSo, she still needs to write {total_pages} pages - {pages_done} pages = {pages_remaining} pages.\n#### {pages_remaining}"

    return {
        'question': question,
        'answer': str(pages_remaining),
        'answer_cot': answer_cot,
        'answer_value': pages_remaining,
        'variables': {
            'name': name,
            'document': document,
            'total_pages': total_pages,
            'fraction': fraction,
            'pages_done': pages_done,
            'pages_remaining': pages_remaining
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia", "Harper", "Evelyn"]
    documents = ["essay", "report", "thesis", "dissertation", "assignment"]
    fractions = ["1/2", "1/3", "1/4", "2/3", "3/4"]

    name = rng.choice(names_female)
    document = rng.choice(documents)
    fraction = rng.choice(fractions)
    
    # Generate total pages ensuring it's divisible by denominator
    denominator = int(fraction.split('/')[1])
    max_pages = int(325 * difficulty)
    total_pages = denominator * rng.randint(1, max_pages // denominator)
    
    result = generate_from_variables(name, document, total_pages, fraction)
    
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
    return generate_from_variables("Shiela", "research paper", 15, "1/3")
