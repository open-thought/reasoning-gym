from random import Random
from typing import Dict, Any

def generate_from_variables(name1: str, name2: str, total_time: int, 
                          library_time: int, station_time: int, 
                          location1: str, location2: str, location3: str) -> Dict[str, Any]:
    
    time_after_library = total_time - library_time
    remaining_time = time_after_library - station_time
    
    question = f"{name1} and {name2} have {total_time} minutes to walk to {location1} together. It takes them {library_time} minutes to get to the corner where the {location2} is. It takes them another {station_time} minutes to get to the {location3}. How much longer do they have to get to {location1} without being late?"
    
    answer_cot = f"{name1} and {name2} arrive at the {location2} with {total_time} - {library_time} = {time_after_library} minutes left to reach the {location1}.\nThey then arrive at the {location3} and have {time_after_library} - {station_time} = {remaining_time} minutes left to get to {location1} without being late.\n#### {remaining_time}"

    return {
        'question': question,
        'answer': str(remaining_time),
        'answer_cot': answer_cot,
        'answer_value': remaining_time,
        'variables': {
            'name1': name1,
            'name2': name2,
            'total_time': total_time,
            'library_time': library_time,
            'station_time': station_time,
            'location1': location1,
            'location2': location2,
            'location3': location3,
            'remaining_time': remaining_time
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ['John', 'Jack', 'James', 'William', 'Michael', 'David', 'Joseph']
    locations = ['cinema', 'mall', 'library', 'park', 'gym', 'bank', 'school']
    
    name1, name2 = rng.sample(names, 2)
    loc1, loc2, loc3 = rng.sample(locations, 3)
    
    # Generate times ensuring they're not divisible by 5
    library_time = int(rng.randint(10, int(30 * difficulty)))
    while library_time % 5 == 0:
        library_time = int(rng.randint(10, int(30 * difficulty)))
        
    station_time = int(rng.randint(10, int(70 * difficulty)))
    while station_time % 5 == 0:
        station_time = int(rng.randint(10, int(70 * difficulty)))
        
    # Ensure total time is greater than sum of other times
    min_total = library_time + station_time + 5
    total_time = int(rng.randint(min_total, int(140 * difficulty)))
    while total_time % 5 == 0:
        total_time = int(rng.randint(min_total, int(140 * difficulty)))

    result = generate_from_variables(name1, name2, total_time, library_time, 
                                   station_time, loc1, loc2, loc3)
    
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
    return generate_from_variables('John', 'Jack', 30, 6, 13, 
                                 'school', 'library', 'fire station')
