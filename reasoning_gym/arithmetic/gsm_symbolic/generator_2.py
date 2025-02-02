from random import Random
from typing import Dict, Any

def generate_from_variables(teacher: str, total: int, p1: int, p2: int,
                          group1: str, group2: str, group3: str, 
                          event: str) -> Dict[str, Any]:
    group1_count = int(total * p1 / 100)
    remaining = total - group1_count
    group23_count = int(remaining * p2 / 100)
    total_leaving = group1_count + group23_count
    
    question = f"In {teacher}'s class of {total} students, {p1}% of the class are {group1}. Out of the remaining class, {p2}% of the students are {group2} or part of {group3}. These 3 groups of students will need to leave early today to travel to an away {event}. How many students are leaving early?"
    
    answer_cot = f"{p1}% of the {total} student class are {group1} so that's {p1/100}*{total} = {group1_count} students\n" \
                 f"There are {total} students and {group1_count} are {group1} so that leaves {total}-{group1_count} = {remaining} students\n" \
                 f"{p2}% of the remaining {remaining} students are part of {group3} or {group2} so that's {p2/100}*{remaining} = {group23_count} students\n" \
                 f"{group1_count} students are {group1} and {group23_count} are part of {group3}/{group2} so {group1_count}+{group23_count} = {total_leaving} students will be leaving early\n" \
                 f"#### {total_leaving}"

    return {
        'question': question, 
        'answer': str(total_leaving),
        'answer_cot': answer_cot,
        'answer_value': total_leaving,
        'variables': {
            'teacher': teacher,
            'total_students': total,
            'percent_group1': p1,
            'percent_group23': p2,
            'group1': group1,
            'group2': group2,
            'group3': group3,
            'event': event,
            'group1_count': group1_count,
            'group23_count': group23_count
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    teachers = ['Ms. Johnson', 'Mr. Smith', 'Dr. Lee', 'Mrs. Garcia']
    sports = ['soccer players', 'basketball players', 'volleyball players', 'swimmers'] 
    activities = ['dancers', 'choir members', 'debate team members', 'robotics club members']
    events = ['competition', 'tournament', 'performance', 'meet']

    teacher = rng.choice(teachers)
    group1 = rng.choice(sports)
    group2, group3 = rng.sample(activities, 2)
    event = rng.choice(events)

    total = int(rng.randint(20, int(150 * difficulty)))
    p1 = int(rng.randint(10, min(50, int(100 * difficulty))))
    p2 = int(rng.randint(15, min(45, int(100 * difficulty))))
    
    # Ensure conditions are met
    while not (p1 < 100 and p2 < 100 and
              (total * p1) % 100 == 0 and
              ((total - total * p1 / 100) * p2) % 100 == 0):
        total = int(rng.randint(20, int(150 * difficulty)))
        p1 = int(rng.randint(10, min(50, int(100 * difficulty))))
        p2 = int(rng.randint(15, min(45, int(100 * difficulty))))

    result = generate_from_variables(teacher, total, p1, p2, group1, group2, group3, event)
    
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
    return generate_from_variables('Mr. Roper', 30, 20, 25, 'football players',
                                 'cheerleaders', 'band', 'game')
