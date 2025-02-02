from random import Random
from typing import Dict, Any
import math

def generate_from_variables(name: str, big_fish: str, length: int, num_remoras: int, 
                          remora_length: int) -> Dict[str, Any]:
    total_remora_length_inches = num_remoras * remora_length
    total_remora_length_feet = total_remora_length_inches / 12
    percentage = int((total_remora_length_feet / length) * 100)

    question = f"{name} saw a {length}-foot {big_fish} with {num_remoras} {remora_length}-inch remoras attached to it. What percentage of the {big_fish}'s body length is the combined length of the remoras?"
    
    answer_cot = f"First, find the combined length of the remoras in inches: {remora_length} inches/remora * {num_remoras} remoras = {total_remora_length_inches} inches\nThen divide that number by 12 to convert it to feet: {total_remora_length_inches} inches / 12 inches/foot = {total_remora_length_feet} foot\nThen divide the combined remora length in feet by the {big_fish}'s length and multiply by 100% to express the answer as a percentage: {total_remora_length_feet} foot / {length} feet * 100% = {percentage}%\n#### {percentage}"

    return {
        'question': question,
        'answer': str(percentage),
        'answer_cot': answer_cot,
        'answer_value': percentage,
        'variables': {
            'name': name,
            'big_fish': big_fish,
            'length_feet': length,
            'num_remoras': num_remoras,
            'remora_length_inches': remora_length,
            'total_remora_length_inches': total_remora_length_inches,
            'total_remora_length_feet': total_remora_length_feet,
            'percentage': percentage
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names = ["Benny", "Tommy", "Jimmy", "Billy", "Johnny", "Bobby"]
    big_fish = ["dolphin", "whale", "shark"]
    
    name = rng.choice(names)
    fish = rng.choice(big_fish)
    
    length = int(rng.randint(10, int(500 * difficulty)) // 10 * 10)
    num_remoras = int(rng.randint(2, int(10 * difficulty)))
    remora_length = int(rng.randint(2, int(100 * difficulty)))
    
    # Ensure conditions are met
    while (num_remoras * remora_length >= length * 12 or 
           (num_remoras * remora_length) % 12 != 0 or
           (length * 12) % (num_remoras * remora_length) != 0 or
           100 % int((num_remoras * remora_length)/(length * 12) * 100) != 0):
        length = int(rng.randint(10, int(500 * difficulty)) // 10 * 10)
        num_remoras = int(rng.randint(2, int(10 * difficulty)))
        remora_length = int(rng.randint(2, int(100 * difficulty)))

    result = generate_from_variables(name, fish, length, num_remoras, remora_length)
    
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
    return generate_from_variables("Benny", "shark", 10, 2, 6)
