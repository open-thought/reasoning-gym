from random import Random
from typing import Dict, Any
import math

def generate_from_variables(name: str, n: int, n_first: int, apartments_each: int, 
                          percent_bigger: int, freq: int, rate: float,
                          currency: str) -> Dict[str, Any]:
    
    first_two = n_first * apartments_each
    third_complex = int(first_two * percent_bigger/100)
    total_apartments = first_two + third_complex + first_two
    weekly_visits = total_apartments * freq
    weekly_earnings = int(weekly_visits * rate)
    
    question = f"{name} collects garbage from {n} different apartment complexes. The first {n_first} have {apartments_each} apartments each and the last one is {percent_bigger}% bigger than the other {n_first} combined. {name} collects garbage {freq} times a week from each place and he gets paid {currency}{rate:.2f} per collection for each apartment. How much money does he make in a week?"
    
    answer_cot = f"The first {n_first} complexes have {first_two} apartments\n" \
                 f"The third one has {first_two}*{percent_bigger/100}={third_complex} more apartments than those {n_first} combined\n" \
                 f"So in total, it has {first_two}+{third_complex}={first_two + third_complex} apartments\n" \
                 f"So he goes to {first_two + third_complex}+{first_two}={total_apartments} apartments each time\n" \
                 f"That means he visits {total_apartments}*{freq}={weekly_visits} apartments every week\n" \
                 f"So he makes {weekly_visits}*{currency}{rate:.2f}={currency}{weekly_earnings} every week\n" \
                 f"#### {weekly_earnings}"

    return {
        'question': question,
        'answer': str(weekly_earnings),
        'answer_cot': answer_cot, 
        'answer_value': weekly_earnings,
        'variables': {
            'name': name,
            'num_complexes': n,
            'first_complexes': n_first,
            'apartments_per_complex': apartments_each,
            'percent_increase': percent_bigger,
            'collections_per_week': freq,
            'rate_per_apartment': rate,
            'currency': currency,
            'total_apartments': total_apartments,
            'weekly_visits': weekly_visits
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['John', 'Michael', 'David', 'James', 'Robert', 'William']
    currencies = ['$', '£', '€']
    
    name = rng.choice(names)
    currency = rng.choice(currencies)
    
    n = rng.randint(3, max(3, int(8 * difficulty)))
    n_first = n - 1
    apartments = int(rng.randrange(100, int(500 * difficulty), 50))
    percent = rng.randrange(20, int(81 * difficulty), 5)
    freq = rng.randint(2, max(2, int(6 * difficulty)))
    rates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    rate = rng.choice(rates)

    # Ensure results are integers
    while not ((n-1)*apartments*percent/100).is_integer():
        apartments = int(rng.randrange(100, int(500 * difficulty), 50))
        percent = rng.randrange(20, int(81 * difficulty), 5)
    
    result = generate_from_variables(name, n, n_first, apartments, percent, freq, rate, currency)
    
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
    return generate_from_variables('John', 3, 2, 200, 60, 3, 0.40, '$')
