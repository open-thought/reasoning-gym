from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, hours: int, days: int, rate: int, 
                          bonus: int, month: str) -> Dict[str, Any]:
    daily_pay = hours * rate
    monthly_days = days * 4
    monthly_base = daily_pay * monthly_days
    monthly_bonus = bonus * 4
    total_pay = monthly_base + monthly_bonus

    question = f"{name} works a {hours}-hour shift each day, {days} days a week. He earns ${rate} per hour and gets a ${bonus} bonus each week if the company performs well. How much money did {name} make in {month} if the company performed very well for the whole month?"

    answer_cot = f"In a day, {name} makes {hours} * {rate} = ${daily_pay}\n" \
                 f"If he works {days} days a week, the total number of days for the whole month is {days} * 4= {monthly_days} days.\n" \
                 f"Since he makes ${daily_pay} per day, the total amount for the whole month is {monthly_days} * {daily_pay}= ${monthly_base}.\n" \
                 f"He also got a {bonus} * 4 = ${monthly_bonus} bonus because the company performed well in all the weeks of {month}.\n" \
                 f"At the end of {month}, he earned {monthly_base} + {monthly_bonus} = ${total_pay}.\n#### {total_pay}"

    return {
        'question': question,
        'answer': str(total_pay),
        'answer_cot': answer_cot,
        'answer_value': total_pay,
        'variables': {
            'name': name,
            'hours_per_day': hours,
            'days_per_week': days,
            'hourly_rate': rate,
            'weekly_bonus': bonus,
            'month': month,
            'daily_pay': daily_pay,
            'monthly_base': monthly_base,
            'monthly_bonus': monthly_bonus
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
    months = ["January", "February", "March", "April", "May", "June", "July", 
             "August", "September", "October", "November", "December"]
    
    name = rng.choice(names)
    month = rng.choice(months)
    
    hours = int(rng.randint(6, int(13 * difficulty)))
    days = int(rng.randint(3, int(7 * difficulty)))
    rate = int(rng.randint(8, int(31 * difficulty)))
    bonus = int(rng.randint(100, int(601 * difficulty)))

    # Ensure rate * hours is an integer
    while (hours * rate) % 1 != 0:
        rate = int(rng.randint(8, int(31 * difficulty)))

    result = generate_from_variables(name, hours, days, rate, bonus, month)

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
    return generate_from_variables("Watson", 10, 5, 10, 300, "April")
