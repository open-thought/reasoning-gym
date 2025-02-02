from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, periods: int, extra_classes: int, 
                          mins_per_class: int, days: int, weekend_fraction: float) -> Dict[str, Any]:
    total_classes = periods + extra_classes
    daily_mins = total_classes * mins_per_class  
    weekly_mins = daily_mins * days
    weekend_mins = int(weekly_mins * weekend_fraction)
    total_mins = weekly_mins + 2 * weekend_mins
    total_hours = total_mins // 60

    question = f"There are {periods} periods in the day for a normal student but {name} has to take {extra_classes} extra classes. Each class is {mins_per_class} minutes long. He goes to class for {days} days a week. He then spends {weekend_fraction} of his weekly minutes each on Saturday and Sunday as extra learning time. How many hours a week does he spend learning?"

    answer_cot = f"He takes {periods}+{extra_classes}={total_classes} classes a day\n" \
                 f"That means he spends {mins_per_class}*{total_classes}={daily_mins} minutes per day in class\n" \
                 f"So he spends {daily_mins}*{days}={weekly_mins} minutes a week\n" \
                 f"That means he spends {weekly_mins}*{weekend_fraction}={weekend_mins} minutes each on Saturday and Sunday\n" \
                 f"So he spends {weekly_mins}+{weekend_mins}+{weekend_mins}={total_mins} minutes per week\n" \
                 f"So he spends {total_mins}/60={total_hours} hours per week\n#### {total_hours}"

    return {
        'question': question,
        'answer': str(total_hours),
        'answer_cot': answer_cot,
        'answer_value': total_hours,
        'variables': {
            'name': name,
            'periods': periods,
            'extra_classes': extra_classes,
            'mins_per_class': mins_per_class,
            'days': days,
            'weekend_fraction': weekend_fraction,
            'total_classes': total_classes,
            'daily_mins': daily_mins,
            'weekly_mins': weekly_mins,
            'weekend_mins': weekend_mins,
            'total_mins': total_mins
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["John", "James", "William", "Michael", "David", "Richard", "Thomas"]
    fractions = ["1/16", "1/8", "1/4", "1/2"]
    
    name = rng.choice(names)
    periods = int(rng.randint(5, int(10 * difficulty)))
    extra_classes = int(rng.randint(1, int(5 * difficulty)))
    mins_per_class = int(rng.randint(30, int(61 * difficulty)) // 5 * 5)
    days = int(rng.randint(4, int(7 * difficulty)))
    weekend_fraction = float(eval(rng.choice(fractions)))
    
    # Ensure results are integers
    while not (((periods + extra_classes) * mins_per_class * days * weekend_fraction).is_integer() and
              (((periods + extra_classes) * mins_per_class * days + 
                2 * (periods + extra_classes) * mins_per_class * days * weekend_fraction) / 60).is_integer()):
        periods = int(rng.randint(5, int(10 * difficulty)))
        extra_classes = int(rng.randint(1, int(5 * difficulty)))
        mins_per_class = int(rng.randint(30, int(61 * difficulty)) // 5 * 5)
        days = int(rng.randint(4, int(7 * difficulty)))
    
    result = generate_from_variables(name, periods, extra_classes, mins_per_class, 
                                   days, weekend_fraction)
    
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
    return generate_from_variables("John", 6, 2, 40, 5, 1/16)
