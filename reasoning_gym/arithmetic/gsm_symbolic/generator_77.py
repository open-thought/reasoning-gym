from random import Random
from typing import Dict, Any

def generate_from_variables(event: str, m: int, w: int, t: str, frac: float, m_left: int, 
                          group1: str, group2: str) -> Dict[str, Any]:
    total = m + w
    left_count = int(total * frac)
    stayed = total - left_count
    w_left = stayed - m_left
    
    question = f"At the beginning of the {event}, there were {m} {group1} and {w} {group2}. After {t}, {frac} of the total number of people left. How many {group2} are left if {m_left} {group1} stayed at the {event}?"
    
    answer_cot = f"There were a total of {m} {group1} + {w} {group2} = {total} people who attended the {event}.\n" \
                 f"After {t}, {total} people * {frac} = {left_count} people left the {event}.\n" \
                 f"This means {total} people - {left_count} people = {stayed} people stayed.\n" \
                 f"Out of the {stayed} who stayed, {stayed} people - {m_left} {group1} = {w_left} were {group2}.\n" \
                 f"#### {w_left}"

    return {
        'question': question,
        'answer': str(w_left),
        'answer_cot': answer_cot,
        'answer_value': w_left,
        'variables': {
            'event': event,
            'men': m,
            'women': w, 
            'time': t,
            'fraction_left': frac,
            'men_stayed': m_left,
            'women_stayed': w_left,
            'group1': group1,
            'group2': group2
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    events = ['party', 'meeting', 'conference', 'gathering', 'celebration']
    groups = ["teachers", "doctors", "engineers", "nurses", "artists", "lawyers"]
    times = ['an hour', 'two hours', 'half an hour', '45 minutes']
    fractions = [0.25, 0.5, 0.75, 0.33, 0.67]
    
    event = rng.choice(events)
    group1, group2 = rng.sample(groups, 2)
    t = rng.choice(times)
    frac = rng.choice(fractions)
    
    m = int(rng.randint(20, int(75 * difficulty)))
    w = int(rng.randint(10, int(80 * difficulty)))
    total = m + w
    
    # Ensure fraction calculations result in integers
    while not (total * frac).is_integer():
        m = int(rng.randint(20, int(75 * difficulty)))
        w = int(rng.randint(10, int(80 * difficulty)))
        total = m + w
    
    stayed = total - int(total * frac)
    m_left = int(rng.randint(15, min(stayed-1, int(35 * difficulty))))
    
    result = generate_from_variables(event, m, w, t, frac, m_left, group1, group2)
    
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
    return generate_from_variables('party', 25, 15, 'an hour', 0.25, 22, 'men', 'women')
