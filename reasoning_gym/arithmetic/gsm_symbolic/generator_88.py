from random import Random
from typing import Dict, Any

def generate_from_variables(school: str, venue: str, total: int, graduates: int, faculty: int) -> Dict[str, Any]:
    remaining_seats = total - (graduates + faculty)
    tickets_per_graduate = remaining_seats // graduates
    
    question = f"{school} is holding graduation in their {venue} this year which has space for {total} people. After accounting for {graduates} seats for graduates and {faculty} seats for faculty attending, how many tickets would each graduate receive to give to their friends and family if the tickets are split equally?"
    
    answer_cot = f"Add graduate and faculty seats together. {graduates} + {faculty} = {graduates+faculty} seats for faculty and graduates\nMinus seats for faculty and graduates from total seats allowed. {total} - {graduates+faculty} = {remaining_seats} remaining seats.\nDivide remaining seats by the number of graduates. {remaining_seats}/{graduates} = {tickets_per_graduate} tickets\n#### {tickets_per_graduate}"

    return {
        'question': question,
        'answer': str(tickets_per_graduate),
        'answer_cot': answer_cot,
        'answer_value': tickets_per_graduate,
        'variables': {
            'school': school,
            'venue': venue,
            'total_seats': total,
            'graduate_seats': graduates,
            'faculty_seats': faculty,
            'remaining_seats': remaining_seats,
            'tickets_per_graduate': tickets_per_graduate
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    schools = ["Oakwood High School", "Riverside Academy", "Sunnyside High", "Greenville High School"]
    venues = ["Auditorium", "Gymnasium", "Sports Arena", "Convention Center"]
    
    school = rng.choice(schools)
    venue = rng.choice(venues)
    
    graduates = int(rng.randint(500, int(1500 * difficulty)) // 50 * 50)
    faculty = int(rng.randint(100, int(500 * difficulty)) // 50 * 50)
    
    # Ensure total seats allow for integer division of remaining seats
    remaining_seats = rng.randint(2, int(10 * difficulty)) * graduates
    total = remaining_seats + graduates + faculty
    
    result = generate_from_variables(school, venue, total, graduates, faculty)
    
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
    return generate_from_variables("Apple High School", "Fine Arts Center", 6000, 950, 300)
