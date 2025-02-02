from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, room_type: str, area: int, length: int,
                          unit1: str, unit2: str) -> Dict[str, Any]:
    conversion = 3 if unit1 == 'feet' and unit2 == 'yards' else 1
    length_converted = length * conversion
    width = area // length_converted
    perimeter = 2 * (width + length_converted)

    question = f"The area of {name}'s rectangular {room_type} is {area} square {unit1}. If the length of his room is {length} {unit2}, what is the perimeter of the room in {unit1}?"
    
    answer_cot = f"The length of the room is {length} {unit2} * ({conversion} {unit1} / 1 {unit2}) = {length_converted} {unit1}.\n" \
                 f"The width of the room is {area} square {unit1} / {length_converted} {unit1} = {width} {unit1}.\n" \
                 f"The room's perimeter is 2({width}+{length_converted}) = {perimeter}\n#### {perimeter}"

    return {
        'question': question,
        'answer': str(perimeter),
        'answer_cot': answer_cot,
        'answer_value': perimeter,
        'variables': {
            'name': name,
            'room_type': room_type,
            'area': area,
            'length': length,
            'unit1': unit1,
            'unit2': unit2,
            'width': width,
            'length_converted': length_converted
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["William", "James", "John", "Michael", "David", "Robert", "Thomas"]
    room_types = ["living room", "study", "office", "kitchen"]
    units = ["feet", "meters"]

    name = rng.choice(names)
    room_type = rng.choice(room_types)
    unit1 = rng.choice(units)
    unit2 = "yards" if unit1 == "feet" else "meters"
    
    length = int(rng.randint(5, int(44 * difficulty)))
    
    # Ensure width is larger than length and area calculation results in integer
    conversion = 3 if unit1 == 'feet' and unit2 == 'yards' else 1
    width = int(rng.randint(length * conversion + 1, int(100 * difficulty)))
    area = width * (length * conversion)

    result = generate_from_variables(name, room_type, area, length, unit1, unit2)
    
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
    return generate_from_variables("Billie", "bedroom", 360, 3, "feet", "yards")
