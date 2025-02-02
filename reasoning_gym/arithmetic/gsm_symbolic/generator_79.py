from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, vehicle: str, start_time: int, end_time: int, 
                          free_hours: int, currency: str, first_hour_cost: int, 
                          multiplier: int) -> Dict[str, Any]:
    total_hours = end_time - start_time
    paid_hours = total_hours - free_hours
    other_hours = paid_hours - 1
    hourly_rate = first_hour_cost * multiplier
    other_hours_cost = other_hours * hourly_rate
    total_cost = first_hour_cost + other_hours_cost

    question = f"{name} hires a {vehicle} from {start_time} PM to {end_time} PM. He gets {free_hours} hour free. The first paid hour is {currency}{first_hour_cost} and each hour after that is {multiplier} times the cost. How much did he pay?"

    answer_cot = f"He got it for {end_time}-{start_time}={total_hours} hours\n" \
                 f"He pays for {total_hours}-{free_hours}={paid_hours} hours\n" \
                 f"The first hour cost 1*{first_hour_cost}={currency}{first_hour_cost}\n" \
                 f"The other {paid_hours}-1={other_hours} hours are more expensive\n" \
                 f"They cost {first_hour_cost}*{multiplier}={currency}{hourly_rate} per hour\n" \
                 f"So those {other_hours} hours cost {other_hours}*{hourly_rate}={currency}{other_hours_cost}\n" \
                 f"So he pays {other_hours_cost}+{first_hour_cost}={currency}{total_cost}\n" \
                 f"#### {total_cost}"

    return {
        'question': question,
        'answer': str(total_cost),
        'answer_cot': answer_cot,
        'answer_value': total_cost,
        'variables': {
            'name': name,
            'vehicle': vehicle,
            'start_time': start_time,
            'end_time': end_time,
            'free_hours': free_hours,
            'currency': currency,
            'first_hour_cost': first_hour_cost,
            'multiplier': multiplier,
            'total_hours': total_hours,
            'paid_hours': paid_hours,
            'total_cost': total_cost
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["James", "John", "Robert", "Michael", "William", "David", "Richard"]
    vehicles = ["limousine", "party bus", "boat", "luxury car"]
    currencies = ["$", "€", "£"]

    name = rng.choice(names)
    vehicle = rng.choice(vehicles)
    currency = rng.choice(currencies)
    
    start_time = int(rng.randint(1, int(8 * difficulty)))
    end_time = int(rng.randint(start_time + 2, int(12 * difficulty)))
    free_hours = int(rng.randint(1, min(3, end_time - start_time - 1)))
    first_hour_cost = int(rng.randint(10, int(51 * difficulty)))
    multiplier = 2

    # Verify conditions
    while not ((end_time - start_time > free_hours + 1) and 
              ((end_time - start_time - free_hours - 1) * first_hour_cost * multiplier).is_integer()):
        start_time = int(rng.randint(1, int(8 * difficulty)))
        end_time = int(rng.randint(start_time + 2, int(12 * difficulty)))
        free_hours = int(rng.randint(1, min(3, end_time - start_time - 1)))
        first_hour_cost = int(rng.randint(10, int(51 * difficulty)))

    result = generate_from_variables(name, vehicle, start_time, end_time, free_hours,
                                   currency, first_hour_cost, multiplier)
    
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
    return generate_from_variables("James", "horse-drawn carriage", 5, 9, 1, "$", 15, 2)
