from random import Random
from typing import Dict, Any

def generate_from_variables(title: str, name: str, property_type: str, price: int,
                          fee1_name: str, fee1_percent: int, 
                          fee2_name: str, fee2_percent: int,
                          loan: int) -> Dict[str, Any]:
    
    fee1_amount = price * fee1_percent // 100
    fee2_amount = price * fee2_percent // 100
    total_fees = fee1_amount + fee2_amount + loan
    net_proceeds = price - total_fees

    question = f"{title} {name} sold his {property_type} for ${price}. He paid the {fee1_name} fees that amount to {fee1_percent}% of the selling price and also paid a {fee2_name} fee that is {fee2_percent}% of the selling price. If he also paid ${loan} for the remaining loan amount of the {property_type}, how much is {title} {name}'s net proceeds from selling the {property_type}?"

    answer_cot = f"{title} {name} paid ${price} x {fee1_percent}/100 = ${fee1_amount} for the {fee1_name} fees.\n" \
                 f"He paid ${price} x {fee2_percent}/100 = ${fee2_amount} for the {fee2_name} fee.\n" \
                 f"So, {title} {name} paid a total of ${fee1_amount} + ${fee2_amount} + ${loan} = ${total_fees} for the {fee1_name}, {fee2_name}, and loan fees.\n" \
                 f"Hence, {title} {name}'s net proceeds is ${price} - ${total_fees} = ${net_proceeds}.\n#### {net_proceeds}"

    return {
        'question': question,
        'answer': str(net_proceeds),
        'answer_cot': answer_cot,
        'answer_value': net_proceeds,
        'variables': {
            'title': title,
            'name': name,
            'property_type': property_type,
            'price': price,
            'fee1_name': fee1_name,
            'fee1_percent': fee1_percent,
            'fee2_name': fee2_name,
            'fee2_percent': fee2_percent,
            'loan': loan,
            'fee1_amount': fee1_amount,
            'fee2_amount': fee2_amount,
            'total_fees': total_fees,
            'net_proceeds': net_proceeds
        }
    }

def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
    titles = ['Mr.', 'Prof.', 'Dr.']
    names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    properties = ['house', 'apartment', 'condo', 'villa', 'cottage']
    fee1_names = ['transfer', 'registration', 'legal']
    fee2_names = ['brokerage', 'agent', 'realtor']

    title = rng.choice(titles)
    name = rng.choice(names)
    property_type = rng.choice(properties)
    fee1_name = rng.choice(fee1_names)
    fee2_name = rng.choice(fee2_names)

    price = int(rng.randint(200000, int(1000000 * difficulty)) // 10000 * 10000)
    fee1_percent = rng.randint(1, int(5 * difficulty))
    fee2_percent = rng.randint(2, int(7 * difficulty))
    loan = int(rng.randint(100000, int(700000 * difficulty)) // 10000 * 10000)

    # Ensure conditions are met
    while price <= loan or price - (price * (fee1_percent + fee2_percent) / 100 + loan) <= 1:
        price = int(rng.randint(200000, int(1000000 * difficulty)) // 10000 * 10000)
        loan = int(rng.randint(100000, int(700000 * difficulty)) // 10000 * 10000)

    result = generate_from_variables(title, name, property_type, price,
                                   fee1_name, fee1_percent,
                                   fee2_name, fee2_percent, loan)

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
    return generate_from_variables('Mr.', 'Tan', 'house', 400000,
                                 'transfer', 3,
                                 'brokerage', 5, 250000)
