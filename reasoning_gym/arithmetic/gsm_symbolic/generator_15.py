from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, sport: str, item1: str, item2: str, 
                          item3: str, item4: str, currency: str, price1: int,
                          price2: int, price3: int, price4: int, discount: int) -> Dict[str, Any]:
    
    shorts_price = price1 + price2
    shoes_price = price3 // 2
    socks_price = price4 - discount
    total = price1 + shorts_price + shoes_price + socks_price
    
    question = f"{name} qualified for a spot on the {sport} team, so she went shopping for some athletic gear. She bought a {item1} for {currency}{price1}, a pair of {sport} {item2} for {currency}{price2} more than the {item1} cost, and a pair of {item3} that were originally {currency}{price3} but were on sale for half price. She had a coupon for {currency}{discount} off the package of {currency}{price4} athletic {item4} that she also bought. How much did she spend on athletic gear?"

    answer_cot = f"The {item2} were {currency}{price2} more than the {item1}, so they cost {currency}{price2} + {currency}{price1} = {currency}{shorts_price}.\nHer {item3} were half the original {currency}{price3} price, so they cost {currency}{price3} / 2 = ${shoes_price}.\nWith her coupon, the {item4} were {currency}{price4} - {currency}{discount} = {currency}{socks_price}.\nThe {item1}, {item2}, {item3}, and {item4} together cost {currency}{price1} + {currency}{shorts_price} + {currency}{shoes_price} + {currency}{socks_price} = {currency}{total}.\n#### {total}"

    return {
        'question': question,
        'answer': str(total),
        'answer_cot': answer_cot,
        'answer_value': total,
        'variables': {
            'name': name,
            'sport': sport,
            'item1': item1,
            'item2': item2,
            'item3': item3, 
            'item4': item4,
            'currency': currency,
            'price1': price1,
            'price2': price2,
            'price3': price3,
            'price4': price4,
            'discount': discount
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia"]
    sports = ["swimming", "cycling", "basketball", "soccer", "volleyball"] 
    items1 = ["t-shirt", "jersey", "sports bra"]
    items2 = ["shorts", "leggings", "sweatpants"]
    items3 = ["sneakers", "cleats", "athletic shoes"]
    items4 = ["socks", "headbands", "wristbands"]
    currencies = ["$", "€", "£"]

    name = rng.choice(names_female)
    sport = rng.choice(sports)
    item1 = rng.choice(items1)
    item2 = rng.choice(items2)
    item3 = rng.choice(items3)
    item4 = rng.choice(items4)
    currency = rng.choice(currencies)

    price1 = int(rng.randint(8, int(25 * difficulty)))
    price2 = int(rng.randint(3, int(15 * difficulty)))
    price4 = int(rng.randint(5, int(15 * difficulty)))
    discount = int(rng.randint(1, min(5, price4)))

    # Ensure price3 is even for clean division by 2
    price3 = int(rng.randint(30, int(80 * difficulty)) // 2 * 2)

    result = generate_from_variables(name, sport, item1, item2, item3, item4, currency,
                                   price1, price2, price3, price4, discount)
    
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
    return generate_from_variables("Alison", "running", "tank top", "shorts", 
                                 "tennis shoes", "socks", "$", 10, 5, 48, 8, 2)
