from random import Random
from typing import Dict, Any

def generate_from_variables(name: str, count: int, child_type: str,
                          item1: str, item2: str, item3: str, item4: str, item5: str,
                          n1: int, n2: int, n3: int, n4: int, n5: int) -> Dict[str, Any]:
    
    skeins_per_child = n1 + n2 + n3 + n4 + n5
    total_skeins = count * skeins_per_child

    question = f"{name} is knitting winter wear for her {count} grandchildren. They're {child_type}, so they're all the same size. She wants to make a {item1}, {item2}, {item3}, {item4}, and {item5} for each of them. It takes {n1} skeins of wool to make a {item1}, {n2} for a {item2}, {n3} for a {item3}, {n4} for a pair of {item4}, and {n5} for a pair of {item5}. How many skeins of wool will she need to buy?"

    answer_cot = f"A full outfit for each child will require {n1} skeins per {item1} + {n2} skeins per {item2} + {n3} skeins per {item3} + {n4} skeins per pair of {item4} + {n5} skeins per pair of {item5} = {skeins_per_child} skeins of wool.\nSo to knit outfits for all of her grandchildren, she will need {count} * {skeins_per_child} = {total_skeins} skeins of wool.\n#### {total_skeins}"

    return {
        'question': question,
        'answer': str(total_skeins),
        'answer_cot': answer_cot,
        'answer_value': total_skeins,
        'variables': {
            'name': name,
            'count': count,
            'child_type': child_type,
            'items': [item1, item2, item3, item4, item5],
            'skeins_per_item': [n1, n2, n3, n4, n5],
            'skeins_per_child': skeins_per_child,
            'total_skeins': total_skeins
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    names_female = ["Martha", "Mary", "Elizabeth", "Susan", "Margaret", "Patricia"]
    clothing_items = ["sweater", "shawl", "hat", "cardigan", "poncho", "vest", "beanie", "tunic"]
    accessories = ["mittens", "booties", "socks", "leg warmers", "gloves"]
    children_types = [("twins", 2), ("triplets", 3), ("quadruplets", 4), ("quintuplets", 5)]

    name = rng.choice(names_female)
    child_type, count = rng.choice(children_types)
    item1, item2, item3 = rng.sample(clothing_items, 3)
    item4, item5 = rng.sample(accessories, 2)

    # Scale numbers based on difficulty
    n1 = int(rng.randint(3, int(19 * difficulty)))
    n2 = int(rng.randint(3, int(19 * difficulty)))
    n3 = int(rng.randint(3, int(19 * difficulty)))
    n4 = int(rng.randint(3, int(19 * difficulty)))
    n5 = int(rng.randint(3, int(19 * difficulty)))

    result = generate_from_variables(name, count, child_type,
                                   item1, item2, item3, item4, item5,
                                   n1, n2, n3, n4, n5)

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
    return generate_from_variables("Martha", 3, "triplets",
                                 "hat", "scarf", "sweater", "mittens", "socks",
                                 2, 4, 12, 1, 2)
