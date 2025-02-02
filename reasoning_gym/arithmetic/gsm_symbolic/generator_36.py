from random import Random
from typing import Dict, Any

def generate_from_variables(n: int, p1: int, r1: int, name: str, s1: str, s2: str, s3: str) -> Dict[str, Any]:
    easy_questions = int(n * (p1/100))
    other_questions = int(n * (1-p1/100))
    easy_correct = int(easy_questions * (r1/100))
    other_correct = int(other_questions * 0.5)
    total_correct = easy_correct + other_correct

    question = f"In a {n}-item quiz, {p1}% of the questions are {s1}, and the rest are equally divided as {s2} and {s3} questions. If {name} is sure to get {r1}% of the {s1} questions, and half of the {s2} and {s3} questions correctly, how many points is she sure to get?"

    answer_cot = f"The {s2} and {s3} questions comprises 100% - {p1}% = {100-p1}% of the quiz.\n" \
                 f"There are {n} questions x {p1}/100 = {easy_questions} {s1} questions.\n" \
                 f"There are a total of {n} questions x {100-p1}/100 = {other_questions} {s2} and {s3} questions.\n" \
                 f"If {name} is sure to get {r1}% of the {s1} questions, then this means she is sure of her {easy_questions} questions x {r1}/100 = {easy_correct} points.\n" \
                 f"From the {s2} and {s3} questions, she is sure to get half of it correctly so that is {other_questions} questions * 0.5 = {other_correct} points.\n" \
                 f"Thus, she is sure of getting {easy_correct} points + {other_correct} points = {total_correct} points in her quiz.\n#### {total_correct}"

    return {
        'question': question,
        'answer': str(total_correct),
        'answer_cot': answer_cot,
        'answer_value': total_correct,
        'variables': {
            'total_questions': n,
            'easy_percent': p1,
            'easy_correct_percent': r1,
            'student_name': name,
            'easy_subject': s1,
            'medium_subject': s2,
            'hard_subject': s3,
            'easy_questions': easy_questions,
            'other_questions': other_questions,
            'easy_correct': easy_correct,
            'other_correct': other_correct
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    subjects = ["history", "geography", "biology", "chemistry", "physics", 
                "economics", "literature", "algebra", "geometry"]
    names = ["Emma", "Sophia", "Olivia", "Ava", "Isabella", "Mia", "Charlotte"]
    
    # Generate valid numbers ensuring integer results
    while True:
        n = int(rng.randrange(10, int(151 * difficulty), 10))
        p1 = int(rng.randrange(5, int(71 * difficulty), 5))
        r1 = int(rng.randrange(5, int(101 * difficulty), 5))
        
        # Check conditions
        if (n * (p1/100)).is_integer() and \
           (n * (p1/100) * (r1/100)).is_integer() and \
           (n*(1-(p1/100))*0.5).is_integer():
            break
    
    name = rng.choice(names)
    s1, s2, s3 = rng.sample(subjects, 3)
    
    result = generate_from_variables(n, p1, r1, name, s1, s2, s3)
    
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
    return generate_from_variables(60, 40, 75, "Aries", "easy", "average", "difficult")
