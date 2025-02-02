from random import Random
from typing import Dict, Any

def generate_from_variables(occupation: str, weeks_per_month: int, days_per_week: int, 
                          pay_per_day: int, currency: str) -> Dict[str, Any]:
    days_per_month = days_per_week * weeks_per_month
    monthly_pay = days_per_month * pay_per_day
    yearly_pay = monthly_pay * 12

    question = f"A {occupation} works for {weeks_per_month} weeks every month and for {days_per_week} days every week. If he gets paid {currency}{pay_per_day} every day, how much does he earn if he works for a year?"
    
    answer_cot = f"The {occupation} works for {days_per_week} days every week and works for {weeks_per_month} weeks every month so he works for {days_per_week} days/week * {weeks_per_month} weeks/month = {days_per_month} days/month\nIf he earns {currency}{pay_per_day} every day he then earns {currency}{pay_per_day}/day * {days_per_month} days/month = {currency}{monthly_pay}/month\nA year is equal to 12 months so every year he earns {currency}{monthly_pay}/month * 12 months/year = {currency}{yearly_pay}\n#### {yearly_pay}"

    return {
        'question': question,
        'answer': str(yearly_pay),
        'answer_cot': answer_cot,
        'answer_value': yearly_pay,
        'variables': {
            'occupation': occupation,
            'weeks_per_month': weeks_per_month,
            'days_per_week': days_per_week,
            'pay_per_day': pay_per_day,
            'currency': currency,
            'days_per_month': days_per_month,
            'monthly_pay': monthly_pay
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    occupations = ["plumber", "electrician", "painter", "carpenter", "landscaper"]
    currencies = ["$", "£", "€"]
    
    occupation = rng.choice(occupations)
    currency = rng.choice(currencies)
    
    weeks_per_month = int(rng.randint(2, int(5 * difficulty)))
    days_per_week = int(rng.randint(4, int(7 * difficulty)))
    pay_per_day = int(rng.randint(40, int(200 * difficulty)) // 5 * 5)
    
    result = generate_from_variables(occupation, weeks_per_month, days_per_week, 
                                   pay_per_day, currency)
    
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
    return generate_from_variables("builder", 4, 6, 50, "$")
