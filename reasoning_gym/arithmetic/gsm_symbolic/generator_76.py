from random import Random
from typing import Dict, Any

def generate_from_variables(n: int, p1: int, p2: int, company: str, frac: float) -> Dict[str, Any]:
    interviews = int(n * (p1/100))
    offers = int(interviews * (p2/100))
    accepts = int(offers * frac)
    
    question = f"{n} people apply for a job at {company}. Of the people that apply, only {p1}% receive interviews. Of those who receive interviews, {p2}% receive a job offer. Of those who receive a job offer, {frac:.2%} of the people accept the position. How many people accept the position?"
    
    answer_cot = f"The number of people that receive interviews is {n} * {p1/100} = {interviews} people\n" \
                 f"The number of people that receive a job offer is {interviews} * {p2/100} = {offers} people\n" \
                 f"The number of people that accept the position is {offers} * {frac} = {accepts} people\n" \
                 f"#### {accepts}"
    
    return {
        'question': question,
        'answer': str(accepts),
        'answer_cot': answer_cot,
        'answer_value': accepts,
        'variables': {
            'total_applicants': n,
            'interview_percent': p1,
            'offer_percent': p2,
            'company': company,
            'acceptance_fraction': frac,
            'num_interviews': interviews,
            'num_offers': offers,
            'num_accepts': accepts
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    companies = ["Microsoft", "Apple", "Amazon", "Facebook", "Netflix", "Tesla", "Google"]
    fractions = {"a third": 1/3, "half": 1/2, "a quarter": 1/4, "two thirds": 2/3}
    
    company = rng.choice(companies)
    frac = fractions[rng.choice(list(fractions.keys()))]
    
    # Generate values ensuring all divisions result in integers
    n = int(rng.randint(201, int(1001 * difficulty)))
    p1 = int(rng.randint(10, int(51 * difficulty)))
    p2 = int(rng.randint(10, int(51 * difficulty)))
    
    # Ensure integer results
    while not (n * (p1/100)).is_integer() or \
          not (n * (p1/100) * (p2/100)).is_integer() or \
          not (n * (p1/100) * (p2/100) * frac).is_integer():
        n = int(rng.randint(201, int(1001 * difficulty)))
        p1 = int(rng.randint(10, int(51 * difficulty)))
        p2 = int(rng.randint(10, int(51 * difficulty)))
    
    result = generate_from_variables(n, p1, p2, company, frac)
    
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
    return generate_from_variables(100, 30, 20, "Google", 1/3)
