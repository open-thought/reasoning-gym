from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, initial_pals: int, lost_pals: int, 
                          letters_per_week: int, pages_per_letter: int,
                          minutes_per_page: int) -> Dict[str, Any]:
    current_pals = initial_pals - lost_pals
    letters_received = current_pals * letters_per_week
    pages_to_write = letters_received * pages_per_letter
    total_minutes = pages_to_write * minutes_per_page
    hours = total_minutes // 60

    question = f"{name} was a pen pal with {initial_pals} people. He stopped being penpals with {lost_pals} of them. They each send {letters_per_week} letters a week that are {pages_per_letter} pages long. He responds in kind. He can write a page every {minutes_per_page} minutes. How many hours does he spend writing a week?"

    answer_cot = f"{name} is penpals with {initial_pals}-{lost_pals}={current_pals} people\n" \
                 f"Thus he gets {current_pals}*{letters_per_week}={letters_received} letters a week\n" \
                 f"So he writes {letters_received}*{pages_per_letter}={pages_to_write} pages a week\n" \
                 f"So he writes for {pages_to_write}*{minutes_per_page}={total_minutes} minutes a week\n" \
                 f"So he writes {total_minutes}/60={hours} hours a week\n#### {hours}"

    return {
        'question': question,
        'answer': str(hours),
        'answer_cot': answer_cot,
        'answer_value': hours,
        'variables': {
            'name': name,
            'initial_penpals': initial_pals,
            'lost_penpals': lost_pals,
            'current_penpals': current_pals,
            'letters_per_week': letters_per_week,
            'pages_per_letter': pages_per_letter,
            'minutes_per_page': minutes_per_page,
            'total_minutes': total_minutes
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Mike", "John", "David", "James", "Robert", "William", "Richard"]
    name = rng.choice(names)
    
    initial_pals = int(rng.randint(5, int(15 * difficulty)))
    lost_pals = int(rng.randint(1, initial_pals - 1))
    letters_per_week = int(rng.randint(2, int(5 * difficulty)))
    pages_per_letter = int(rng.randint(5, int(12 * difficulty)))
    minutes_per_page = int(rng.randint(4, int(15 * difficulty)))
    
    # Ensure result is in whole hours
    while ((initial_pals - lost_pals) * letters_per_week * pages_per_letter * minutes_per_page) % 60 != 0:
        minutes_per_page = int(rng.randint(4, int(15 * difficulty)))

    result = generate_from_variables(name, initial_pals, lost_pals, 
                                   letters_per_week, pages_per_letter, minutes_per_page)
    
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
    return generate_from_variables("Mike", 5, 2, 2, 5, 6)
