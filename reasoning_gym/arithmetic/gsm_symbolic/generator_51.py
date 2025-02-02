from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, parent: str, activity1: str, activity2: str, 
                          activity3: str, cur: str, times: int, budget: int,
                          tokens: int, cost1: int, cost2: int) -> Dict[str, Any]:
    
    cost_per_ride = cost2 * times
    cost_per_person = tokens + cost1 + cost_per_ride
    total_people = budget // cost_per_person
    friends = total_people - 1

    question = f"{name}'s {parent} said that she had {cur}{budget} budgeted for her birthday party. She wants to make sure she and her friends all get to play one round of {activity1}, have {cur}{tokens} in {activity2} tokens, and get to ride the {activity3} {times}. A round of {activity1} is {cur}{cost1}. The {activity3} cost {cur}{cost2} a ride. How many friends can she invite?"

    answer_cot = f"The {activity3} will cost {cur}{cost_per_ride} per person because {cost2} x {times} = {cost_per_ride}\n" \
                 f"Each person costs {cur}{cost_per_person} because {tokens} + {cost1} + {cost_per_ride} = {cost_per_person}\n" \
                 f"{total_people} total people can attend because {budget} / {cost_per_person} = {total_people}\n" \
                 f"She can invite {friends} friends because {total_people} - 1 = {friends}\n" \
                 f"#### {friends}"

    return {
        'question': question,
        'answer': str(friends),
        'answer_cot': answer_cot,
        'answer_value': friends,
        'variables': {
            'name': name,
            'parent': parent,
            'activity1': activity1,
            'activity2': activity2,
            'activity3': activity3,
            'currency': cur,
            'times': times,
            'budget': budget,
            'tokens': tokens,
            'cost1': cost1,
            'cost2': cost2,
            'cost_per_ride': cost_per_ride,
            'cost_per_person': cost_per_person,
            'total_people': total_people
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ['Emma', 'Olivia', 'Sophia', 'Isabella', 'Mia', 'Charlotte']
    parents = ['mom', 'dad', 'aunt', 'uncle']
    activities1 = ['mini-golf', 'bowling', 'laser tag']
    activities2 = ['arcade', 'game room', 'pinball'] 
    activities3 = ['go-karts', 'bumper cars', 'roller coaster']
    currencies = ['$', '£', '€']
    times_options = [2, 3]

    name = rng.choice(names_female)
    parent = rng.choice(parents)
    activity1 = rng.choice(activities1)
    activity2 = rng.choice(activities2)
    activity3 = rng.choice(activities3)
    cur = rng.choice(currencies)
    times = rng.choice(times_options)
    
    tokens = int(rng.randint(3, int(11 * difficulty)))
    cost1 = int(rng.randint(3, int(11 * difficulty)))
    cost2 = int(rng.randint(5, int(21 * difficulty)))
    
    # Generate budget ensuring conditions are met
    cost_per_person = tokens + cost1 + (cost2 * times)
    num_people = rng.randint(2, int(10 * difficulty))
    budget = cost_per_person * num_people

    result = generate_from_variables(name, parent, activity1, activity2, activity3,
                                   cur, times, budget, tokens, cost1, cost2)
    
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
    return generate_from_variables('Morgan', 'dad', 'mini-golf', 'arcade', 'go-karts',
                                 '$', 2, 90, 5, 5, 10)
