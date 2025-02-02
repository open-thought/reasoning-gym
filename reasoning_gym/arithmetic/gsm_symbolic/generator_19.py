from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, pan: str, initial_kernels: int, time_interval: int,
                          multiplier_2: int, multiplier_3: int) -> Dict[str, Any]:
    second_interval = multiplier_2 * initial_kernels
    third_interval = multiplier_3 * initial_kernels  
    fourth_interval = third_interval // 2
    residual = fourth_interval // 4
    total = initial_kernels + second_interval + third_interval + fourth_interval + residual

    question = f"{name} is popping popcorn for a snack. As the {pan} of kernels heats up, the kernels start popping faster. {initial_kernels} pop in the first {time_interval} seconds of cooking, then {multiplier_2} times that amount in the next {time_interval} seconds. The kernels increase to {multiplier_3} times the initial popping rate in the next {time_interval} seconds, but in the final {time_interval} seconds, the popping slows down to half the rate as the past {time_interval} seconds. After {name} takes the {pan} off the heat, a quarter of the number of kernels that popped in the final {time_interval} seconds of cooking also pop from the residual heat. How many pieces of popcorn does {name} have to eat?"

    answer_cot = f"In the second {time_interval} seconds of cooking, {multiplier_2} * {initial_kernels} = {second_interval} kernels pop.\nIn the third {time_interval} seconds, {multiplier_3} * {initial_kernels} = {third_interval} kernels pop.\nIn the final {time_interval} seconds, {third_interval} / 2 = {fourth_interval} kernels pop.\nAfter cooking, the residual heat makes {fourth_interval} / 4 = {residual} kernels pop.\nThus, {name} has {initial_kernels} + {second_interval} + {third_interval} + {fourth_interval} + {residual} = {total} pieces of popcorn to eat.\n#### {total}"

    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'name': name,
            'pan': pan,
            'initial_kernels': initial_kernels,
            'time_interval': time_interval,
            'multiplier_2': multiplier_2,
            'multiplier_3': multiplier_3,
            'second_interval': second_interval,
            'third_interval': third_interval,
            'fourth_interval': fourth_interval,
            'residual': residual,
            'total': total
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Garrett", "James", "Michael", "David", "John", "Robert", "William"]
    pans = ["pan", "pot", "skillet"]
    
    name = rng.choice(names)
    pan = rng.choice(pans)
    
    # Generate numbers ensuring divisibility conditions are met
    initial_kernels = int(rng.randint(10, int(101 * difficulty)) // 10 * 10)
    time_interval = int(rng.randint(10, int(31 * difficulty)) // 2 * 2)
    multiplier_2 = rng.randint(2, int(5 * difficulty))
    
    # Ensure multiplier_3 > multiplier_2 and results in clean division by 8
    while True:
        multiplier_3 = rng.randint(multiplier_2 + 1, int(8 * difficulty))
        if (multiplier_3 * initial_kernels) % 8 == 0:
            break
            
    result = generate_from_variables(name, pan, initial_kernels, time_interval,
                                   multiplier_2, multiplier_3)
    
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
    return generate_from_variables("Garrett", "pan", 20, 30, 3, 4)
