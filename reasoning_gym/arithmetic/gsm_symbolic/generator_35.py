from random import Random
from typing import Dict, Any

def generate_from_variables(job: str, building: str, room: str, num_rooms: int, 
                          num_days: int, time_per_room: int, hours_per_day: int) -> Dict[str, Any]:
    
    rooms_per_day = num_rooms // num_days
    minutes_per_day = rooms_per_day * time_per_room
    hours_cleaning = minutes_per_day / 60
    percentage = int(100 * hours_cleaning / hours_per_day)

    question = f"A {job} has to clean a {building} with {num_rooms} {room}s. They have {num_days} days to get it done. It takes them {time_per_room} minutes per {room}. If they work {hours_per_day} hour day, what percentage of their day, on average, is spent cleaning {room}s?"
    
    answer_cot = f"They have to clean {rooms_per_day} {room}s a day because {num_rooms} / {num_days} = {rooms_per_day}\n" \
                 f"They spend {minutes_per_day} minutes cleaning per day because {rooms_per_day} x {time_per_room} = {minutes_per_day}\n" \
                 f"They spend {hours_cleaning} hours a day because {minutes_per_day} / 60 = {hours_cleaning}\n" \
                 f"They spend {hours_cleaning/hours_per_day} of their day cleaning {room}s because {hours_cleaning} / {hours_per_day} = {hours_cleaning/hours_per_day}\n" \
                 f"They spend {percentage}% of their day cleaning {room}s because {hours_cleaning/hours_per_day} x 100 = {percentage}\n" \
                 f"#### {percentage}"

    return {
        'question': question,
        'answer': str(percentage),
        'answer_cot': answer_cot,
        'answer_value': percentage,
        'variables': {
            'job': job,
            'building': building,
            'room': room,
            'num_rooms': num_rooms,
            'num_days': num_days,
            'time_per_room': time_per_room,
            'hours_per_day': hours_per_day,
            'rooms_per_day': rooms_per_day,
            'minutes_per_day': minutes_per_day,
            'hours_cleaning': hours_cleaning,
            'percentage': percentage
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    jobs = ["janitor", "cleaner", "maintenance worker"]
    buildings = ["office building", "hospital", "university"]
    rooms = ["room", "floor"]
    
    job = rng.choice(jobs)
    building = rng.choice(buildings)
    room = rng.choice(rooms)

    num_days = int(rng.randint(3, int(12 * difficulty)))
    time_per_room = int(rng.randrange(10, int(46 * difficulty), 5))
    hours_per_day = int(rng.randint(6, int(17 * difficulty)))
    
    # Generate num_rooms ensuring divisibility by num_days
    rooms_per_day = rng.randint(5, int(20 * difficulty))
    num_rooms = rooms_per_day * num_days
    
    # Ensure conditions are met
    while (num_rooms/num_days) * time_per_room >= hours_per_day * 60:
        rooms_per_day = rng.randint(5, int(20 * difficulty))
        num_rooms = rooms_per_day * num_days

    result = generate_from_variables(job, building, room, num_rooms, num_days,
                                   time_per_room, hours_per_day)
    
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
    return generate_from_variables("custodian", "school", "classroom", 
                                 80, 5, 15, 8)
