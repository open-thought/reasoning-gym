from random import Random
from typing import Dict, Any
from fractions import Fraction

def generate_from_variables(food: str, calories: int, size: int, servings: int, 
                          total_target: int, consumed: int, unit: str) -> Dict[str, Any]:
    
    calories_left = total_target - consumed
    serving_fraction = Fraction(calories_left, calories)
    grams_per_serving = size // servings
    grams_allowed = grams_per_serving * serving_fraction

    question = f"According to its nutritional info, a bag of {food} has {calories} calories per serving. If a {size} {unit} bag has {servings} servings, how many {unit} can you eat if your daily calorie target is {total_target} and you have already consumed {consumed} calories?"

    answer_cot = f"If the total calorie target is {total_target} and I have consumed {consumed} calories then I have {total_target}-{consumed} = {calories_left} calories left to eat\n" \
                 f"If each serving of {food} has {calories} calories and I only have {calories_left} calories left to eat, then I can only eat {calories_left}/{calories} of a serving = {serving_fraction} of a serving\n" \
                 f"We also know that a {size} {unit} bag of {food} has {servings} servings, hence each serving has {size} {unit}/{servings} = {grams_per_serving} {unit}\n" \
                 f"If I can only eat {serving_fraction} of a serving, then I can eat only {grams_per_serving} * {serving_fraction} = {grams_allowed} {unit}\n" \
                 f"#### {float(grams_allowed)}"

    return {
        'question': question,
        'answer': str(float(grams_allowed)),
        'answer_cot': answer_cot,
        'answer_value': float(grams_allowed),
        'variables': {
            'food': food,
            'calories': calories,
            'size': size,
            'servings': servings,
            'total_target': total_target,
            'consumed': consumed,
            'unit': unit,
            'calories_left': calories_left,
            'grams_per_serving': grams_per_serving,
            'serving_fraction': serving_fraction
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    foods = ["popcorn", "breads", "cookies"]
    units = ["grams", "ounces", "oz"]
    
    food = rng.choice(foods)
    unit = rng.choice(units)
    
    calories = int(rng.randint(150, int(500 * difficulty)) // 25 * 25)
    size = int(rng.randint(100, int(400 * difficulty)) // 25 * 25)
    servings = int(rng.randint(4, int(8 * difficulty)))
    total_target = int(rng.randint(1900, int(2500 * difficulty)) // 5 * 5)
    consumed = int(rng.randint(600, int(1800 * difficulty)) // 25 * 25)

    # Ensure conditions are met
    while (consumed >= total_target or 
           not (size % servings == 0) or
           not ((size//servings) * Fraction(total_target-consumed, calories)).denominator == 1):
        calories = int(rng.randint(150, int(500 * difficulty)) // 25 * 25)
        size = int(rng.randint(100, int(400 * difficulty)) // 25 * 25)
        servings = int(rng.randint(4, int(8 * difficulty)))
        total_target = int(rng.randint(1900, int(2500 * difficulty)) // 5 * 5)
        consumed = int(rng.randint(600, int(1800 * difficulty)) // 25 * 25)

    result = generate_from_variables(food, calories, size, servings, total_target, consumed, unit)
    
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
    return generate_from_variables("chips", 250, 300, 5, 2000, 1800, "grams")
