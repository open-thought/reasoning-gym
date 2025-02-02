from random import Random
from typing import Dict, Any

def generate_from_variables(total: int, grade: str, school_name: str, num_girls: int, 
                          day: str, absent_girls: int, absent_boys: int) -> Dict[str, Any]:
    num_boys = total - num_girls
    remaining_boys = num_boys - absent_boys
    
    question = f"There are {total} {grade}-graders at {school_name} School. {num_girls} of them are girls. On {day}, {absent_girls} {grade}-grade girls and {absent_boys} {grade}-grade boys were absent. How many {grade} grade boys were at {school_name} School on {day}?"
    
    answer_cot = f"Of the {total} {grade} graders, {num_girls} are girls, so {total} students - {num_girls} girls = {num_boys} boys.\nOn {day} there were {num_boys} boys - {absent_boys} absent = {remaining_boys} boys.\n#### {remaining_boys}"

    return {
        'question': question,
        'answer': str(remaining_boys),
        'answer_cot': answer_cot,
        'answer_value': remaining_boys,
        'variables': {
            'total_students': total,
            'grade': grade,
            'school_name': school_name,
            'num_girls': num_girls,
            'num_boys': num_boys,
            'day': day,
            'absent_girls': absent_girls,
            'absent_boys': absent_boys,
            'remaining_boys': remaining_boys
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    school_names = ["Maple Grove", "Sunny Hill", "Oak Ridge", "Pine Valley"]
    grades = ["first", "second", "third", "fourth", "fifth"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    school_name = rng.choice(school_names)
    grade = rng.choice(grades)
    day = rng.choice(days)
    
    total = int(rng.randint(50, int(200 * difficulty)))
    num_girls = int(rng.randint(20, total - 1))
    num_boys = total - num_girls
    
    absent_girls = int(rng.randint(2, min(num_girls, int(10 * difficulty))))
    absent_boys = int(rng.randint(2, min(num_boys, int(10 * difficulty))))
    
    result = generate_from_variables(total, grade, school_name, num_girls, 
                                   day, absent_girls, absent_boys)
    
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
    return generate_from_variables(96, "fourth", "Small Tree", 43, "Friday", 5, 4)
