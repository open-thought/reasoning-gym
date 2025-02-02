from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, vehicle: str, weight_vehicle: int, 
                          item: str, weight_item: int, passenger_type: str,
                          num_passengers: int, weight_passenger: int,
                          unit: str, force_percent: int) -> Dict[str, Any]:
    
    total_passenger_weight = num_passengers * weight_passenger
    total_weight = weight_vehicle + weight_item + total_passenger_weight
    force_needed = int((total_weight * force_percent) / 100)
    
    question = f"{name}'s {vehicle} breaks down. The {vehicle} weighs {weight_vehicle} {unit} and he has {item} in it weighing {weight_item} {unit}. He also has his {num_passengers} young {passenger_type} who weigh {weight_passenger} {unit} each in it. If the force to move the {vehicle} is {force_percent}% of the weight, how much force does he need to push the {vehicle}?"
    
    answer_cot = f"His {num_passengers} {passenger_type} weigh {weight_passenger}*{num_passengers}={total_passenger_weight} {unit}\nSo the total weight of the {vehicle} and everything is {weight_vehicle}+{weight_item}+{total_passenger_weight}={total_weight} {unit}\nSo he needs to generate {total_weight}*{force_percent/100}={force_needed} {unit}\n#### {force_needed}"

    return {
        'question': question,
        'answer': str(force_needed),
        'answer_cot': answer_cot,
        'answer_value': force_needed,
        'variables': {
            'name': name,
            'vehicle': vehicle,
            'weight_vehicle': weight_vehicle,
            'item': item,
            'weight_item': weight_item,
            'passenger_type': passenger_type,
            'num_passengers': num_passengers,
            'weight_passenger': weight_passenger,
            'unit': unit,
            'force_percent': force_percent,
            'total_weight': total_weight
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["John", "Michael", "David", "James", "Robert", "William", "Richard"]
    vehicles = ['car', 'van', 'truck', 'SUV']
    items = ['luggage', 'groceries', 'equipment', 'furniture']
    passenger_types = ['children', 'friends', 'colleagues', 'teammates']
    units = ['pounds', 'kilograms']
    
    name = rng.choice(names)
    vehicle = rng.choice(vehicles)
    item = rng.choice(items)
    passenger_type = rng.choice(passenger_types)
    unit = rng.choice(units)
    
    weight_vehicle = int(rng.randint(1000, int(3000 * difficulty) // 50) * 50)
    weight_item = int(rng.randint(100, int(500 * difficulty) // 25) * 25)
    weight_passenger = int(rng.randint(50, int(100 * difficulty) // 5) * 5)
    num_passengers = int(rng.randint(2, int(5 * difficulty)))
    force_percent = int(rng.randint(1, int(6 * difficulty)))
    
    # Ensure force calculation results in integer
    total_weight = weight_vehicle + weight_item + (num_passengers * weight_passenger)
    while (total_weight * force_percent) % 100 != 0:
        force_percent = int(rng.randint(1, int(6 * difficulty)))
    
    result = generate_from_variables(name, vehicle, weight_vehicle, item, weight_item,
                                   passenger_type, num_passengers, weight_passenger,
                                   unit, force_percent)
    
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
    return generate_from_variables("John", "car", 1200, "luggage", 250,
                                 "children", 2, 75, "pounds", 1)
