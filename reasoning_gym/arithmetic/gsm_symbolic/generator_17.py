from random import Random
from typing import Dict, Any
from fractions import Fraction

def generate_from_variables(facility: str, total: int, item: str, frac: Fraction, 
                          event: str, daily: int, period: int) -> Dict[str, Any]:
    initial_occupied = int(total * frac)
    initial_empty = total - initial_occupied
    weekly_admitted = daily * 7
    total_admitted = weekly_admitted * period
    final_empty = initial_empty - total_admitted

    question = f"A {facility} has a capacity of {total} {item}s with {frac} occupied. Due to the {event}, {daily} patients are admitted into the {facility} each day. Calculate the total number of unoccupied {item}s in the {facility} after {period} weeks."
    
    answer_cot = f"If {frac} of the total capacity of the {facility} {item}s is occupied, it means {frac} * {total} = {initial_occupied} {item}s have patients using them.\nThe total number of {item}s in the {facility} without new admissions is {total} {item}s - {initial_occupied} {item}s = {initial_empty} {item}s.\nIf {daily} people are admitted each day, the total number of patients in the {facility} after one week is {daily} patients/day * 7 days/week = {weekly_admitted} patients.\nAfter {period} weeks, the total number of patients admitted into the {facility} is {weekly_admitted} patients/week * {period} weeks = {total_admitted} patients, who each use one {item}.\nIf there were {initial_empty} unoccupied {item}s in the {facility} before the new admissions, the total number is reduced to {initial_empty} {item}s - {total_admitted} {item}s = {final_empty} unoccupied {item}s.\n#### {final_empty}"

    return {
        'question': question,
        'answer': str(final_empty),
        'answer_cot': answer_cot,
        'answer_value': final_empty,
        'variables': {
            'facility': facility,
            'total_capacity': total,
            'item': item,
            'initial_fraction': frac,
            'event': event,
            'daily_patients': daily,
            'period_weeks': period,
            'initial_occupied': initial_occupied,
            'initial_empty': initial_empty,
            'total_admitted': total_admitted
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    facilities = ["hospital", "clinic", "medical center", "care facility"]
    items = ["bed", "room", "ward"] 
    events = ["flu season", "natural disaster", "major accident", "pandemic"]
    fractions = [Fraction(1,5), Fraction(1,4), Fraction(1,3), Fraction(1,2)]

    facility = rng.choice(facilities)
    item = rng.choice(items)
    event = rng.choice(events)
    frac = rng.choice(fractions)

    total = int(rng.randint(500, int(2000 * difficulty)) // 100 * 100)
    daily = int(rng.randint(20, int(100 * difficulty)) // 5 * 5)
    period = int(rng.randint(2, int(5 * difficulty)))

    # Ensure conditions are met
    while not (total * frac).is_integer() or total * frac + daily * period * 7 >= total:
        total = int(rng.randint(500, int(2000 * difficulty)) // 100 * 100)
        daily = int(rng.randint(20, int(100 * difficulty)) // 5 * 5)
        period = int(rng.randint(2, int(5 * difficulty)))

    result = generate_from_variables(facility, total, item, frac, event, daily, period)

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
    return generate_from_variables("hospital", 1000, "bed", Fraction(1,5),
                                 "coronavirus outbreak", 50, 2)
