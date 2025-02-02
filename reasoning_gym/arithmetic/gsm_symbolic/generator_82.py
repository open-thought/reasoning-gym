from random import Random
from typing import Dict, Any
import math

def generate_from_variables(name: str, num_emails: int, no_response_percent: int, 
                          workdays: int) -> Dict[str, Any]:
    no_response = num_emails * no_response_percent // 100
    responds_to = num_emails - no_response
    total_responses = responds_to * workdays
    
    question = f"{name} gets {num_emails} emails a day. {no_response_percent}% of those emails don't require any response. {name} responds to the rest of them. How many emails does {name} respond to in a {workdays} day work week?"
    
    answer_cot = f"{name} receives {no_response}={no_response} emails that don't require a response\n" \
                 f"So {name} responds to {num_emails}-{no_response}={responds_to} emails per day\n" \
                 f"In a {workdays} day work week, {name} responds to {responds_to}*{workdays}={total_responses} emails\n" \
                 f"#### {total_responses}"

    return {
        'question': question,
        'answer': str(total_responses),
        'answer_cot': answer_cot,
        'answer_value': total_responses,
        'variables': {
            'name': name,
            'num_emails': num_emails,
            'no_response_percent': no_response_percent,
            'workdays': workdays,
            'no_response': no_response,
            'responds_to': responds_to,
            'total_responses': total_responses
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
    name = rng.choice(names)
    
    # Generate random values scaled by difficulty
    num_emails = int(rng.randint(50, int(200 * difficulty)))
    no_response_percent = int(rng.randint(5, int(40 * difficulty)))
    workdays = int(rng.randint(3, int(7 * difficulty)))
    
    # Ensure conditions are met
    while not (num_emails * no_response_percent % 100 == 0 and 
              num_emails * (100 - no_response_percent) % 100 == 0):
        num_emails = int(rng.randint(50, int(200 * difficulty)))
        no_response_percent = int(rng.randint(5, int(40 * difficulty)))

    result = generate_from_variables(name, num_emails, no_response_percent, workdays)
    
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
    return generate_from_variables("James", 80, 20, 5)
