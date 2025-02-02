from random import Random
from typing import Dict, Any
import math

def generate_from_variables(name: str, mult_run: int, frac_skip: float, skip_speed: int, 
                          total_time: int, frac_run: float, frac_walk: float) -> Dict[str, Any]:
    run_speed = skip_speed / frac_skip
    walk_speed = run_speed / mult_run
    walk_hours = total_time * frac_walk
    run_hours = total_time * frac_run
    run_dist = run_hours * run_speed
    walk_dist = walk_hours * walk_speed
    total_dist = int(run_dist + walk_dist)

    question = f"{name} can run {mult_run} times faster than she can walk, but she can skip at a rate of speed that is {frac_skip:.1f} as fast as she can run. If she can skip at {skip_speed} miles per hour, how many miles can she travel in {total_time} hours if she spends {frac_run:.2f} of the time running and {frac_walk:.2f} of the time walking?"

    answer_cot = f"""If {name} can skip at {frac_skip:.1f} the speed she can run, then she can run at {skip_speed}*{1/frac_skip:.1f}={run_speed} miles per hour.
And since she can run at a speed that is {mult_run} times faster than she can walk, this means she can walk at {run_speed}/{mult_run}={walk_speed} miles per hour.
If {frac_walk:.2f} of the time is spent walking, then she walks for {total_time}*{frac_walk:.2f}={walk_hours} hours.
If {frac_run:.2f} of the time is spent running, then she runs for {total_time}-{walk_hours}={run_hours} hours.
Thus, she runs for {run_hours} hours at {run_speed} miles per hour, or {run_hours}*{run_speed}={run_dist} miles.
She walks for {walk_hours} hours at {walk_speed} miles per hour, or {walk_hours}*{walk_speed}={walk_dist} miles.
Thus, altogether, she travels {run_dist}+{walk_dist}={total_dist} miles.
#### {total_dist}"""

    return {
        'question': question,
        'answer': str(total_dist),
        'answer_cot': answer_cot,
        'answer_value': total_dist,
        'variables': {
            'name': name,
            'mult_run': mult_run,
            'frac_skip': frac_skip,
            'skip_speed': skip_speed,
            'total_time': total_time,
            'frac_run': frac_run,
            'frac_walk': frac_walk,
            'run_speed': run_speed,
            'walk_speed': walk_speed,
            'total_dist': total_dist
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Dana", "Emma", "Sarah", "Julia", "Sophie", "Maria"]
    name = rng.choice(names)
    
    mult_run = rng.randint(2, int(6 * difficulty))
    frac_skip = 0.5 # Keep simple fraction
    skip_speed = rng.randint(2, int(10 * difficulty))
    total_time = rng.randrange(4, int(12 * difficulty), 2)
    
    # Ensure fractions add to 1
    frac_run = 1/3  # Keep simple fraction
    frac_walk = 2/3 # Keep simple fraction

    # Validate conditions
    while not (skip_speed / frac_skip < 13 and 
              (total_time * frac_walk * (skip_speed / frac_skip / mult_run)).is_integer() and
              (skip_speed / frac_skip).is_integer() and
              (total_time * frac_run).is_integer() and 
              (total_time * frac_walk).is_integer()):
        skip_speed = rng.randint(2, int(10 * difficulty))
        total_time = rng.randrange(4, int(12 * difficulty), 2)

    result = generate_from_variables(name, mult_run, frac_skip, skip_speed, 
                                   total_time, frac_run, frac_walk)

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
    return generate_from_variables("Dana", 4, 0.5, 3, 6, 1/3, 2/3)
