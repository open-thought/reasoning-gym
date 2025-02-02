from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, fruit: str, area: str, 
                          field_size: int, density: int, months: int) -> Dict[str, Any]:
    fruits_per_harvest = field_size * density
    harvests_per_year = 12 // months
    total_fruits = fruits_per_harvest * harvests_per_year
    
    question = f"{name} has {field_size} {area}s of a {fruit} field. There are {density} {fruit}s per {area}. {name} can harvest his {fruit}s every {months} months. How many {fruit}s can {name} harvest within a year?"
    
    answer_cot = f"{name} has {density} x {field_size}= {fruits_per_harvest} {fruit}s on his field.\n{name} can harvest his {fruit}s 12 ÷ {months} = {harvests_per_year} times per year\nTherefore {name} can harvest {fruits_per_harvest} x {harvests_per_year} = {total_fruits} {fruit}s per year.\n#### {total_fruits}"
    
    return {
        'question': question,
        'answer': str(total_fruits),
        'answer_cot': answer_cot,
        'answer_value': total_fruits,
        'variables': {
            'name': name,
            'fruit': fruit,
            'area': area,
            'field_size': field_size,
            'density': density,
            'months_per_harvest': months,
            'fruits_per_harvest': fruits_per_harvest,
            'harvests_per_year': harvests_per_year
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["John", "Michael", "David", "James", "Robert", "William", "Richard"]
    fruits = ["pineapple", "mango", "banana", "papaya", "coconut"] 
    areas = ["hectare", "square yard", "square meter"]
    
    name = rng.choice(names)
    fruit = rng.choice(fruits)
    area = rng.choice(areas)
    
    field_size = int(rng.randint(5, int(100 * difficulty)) // 5 * 5)
    density = int(rng.randint(2, int(101 * difficulty)))
    months = rng.choice([1, 2, 3, 4, 6, 12])

    result = generate_from_variables(name, fruit, area, field_size, density, months)
    
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
    return generate_from_variables("John", "pineapple", "hectare", 10, 100, 3)
