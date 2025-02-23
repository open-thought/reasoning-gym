import random
from random import Random
from typing import Any

NUM_OF_PAIRS_GENERATED = 5


def create_random_list(rng: Random):
    length = rng.randint(3, 10)
    return [rng.randint(1, 100) for _ in range(length)]


def create_list_of_fives(rng: Random):
    length = rng.randint(1, 7)  # Random length between 1 and 7
    return [5] * length


def sort_integers(lst, order="ascending"):
    """
    Sorts a list of integers in ascending or descending order.

    Parameters:
        lst (list): The list of integers to sort.
        order (str): The order to sort in. Options are 'ascending' or 'descending'.

    Returns:
        list: The sorted list.
    """
    if order == "ascending":
        return sorted(lst)  # Sort in ascending order
    elif order == "descending":
        return sorted(lst, reverse=True)  # Sort in descending order
    else:
        raise ValueError("Invalid order. Use 'ascending' or 'descending'.")


def create_random_odd_numbers(count, start, end):
    """
    Generates a list of random odd numbers.

    Parameters:
        count (int): The number of odd numbers to generate.
        start (int): The lower bound of the range (inclusive).
        end (int): The upper bound of the range (inclusive).

    Returns:
        list: A list of random odd numbers.
    """
    odd_numbers = []
    while len(odd_numbers) < count:
        num = random.randint(start, end)  # Generate a random number
        if num % 2 != 0:  # Check if the number is odd
            odd_numbers.append(num)
    return odd_numbers


def create_numbers_divisible_by_five_or_ten(rng: Random):
    result = []
    for i in range(NUM_OF_PAIRS_GENERATED):
        if i % 2 == 0:
            num = create_random_odd_numbers(1, 1, 1000)[0] * 5  # Random multiple of 5
        else:
            num = rng.randint(1, 100) * 10  # Random multiple of 10
        result.append(num)
    return result


def generate_0(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where input remains unchanged"""
    pairs = {}

    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        input = str(input)
        output = input
        pairs[input] = output

    return pairs


def generate_1(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is a list of the third element
    after removing all other elements
    """
    pairs = {}

    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        target_idx = 2
        output = [input[target_idx]]
        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_2(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is a reversed list of the input"""
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        output = input[::-1]
        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_3(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is the sum of unique elements in the list less than 30"""
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        unique_input = list(set(input))

        total_sum = 0
        for num in unique_input:
            if num < 30:
                total_sum += num

        input = str(input)
        output = str([total_sum])
        pairs[input] = output

    return pairs


def generate_4(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is the count of elements equal to 5"""
    pairs = {}
    for i in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)

        if i % 2 == 0:
            input += create_list_of_fives(rng)

        # Shuffle the new input with fives
        rng.shuffle(input)

        total_count = 0
        for num in input:
            if num == 5:
                total_count += 1

        input = str(input)
        output = str([total_count])
        pairs[input] = output

    return pairs


def generate_5(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is a list of elements that are followed by an even number

    NOTE: This is suppose to be a relatively hard problem
    """
    pairs = {}
    for i in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        output = []
        for i in range(1, len(input)):

            # If the current element is an even number, we then add previous element into output
            if input[i] % 2 == 0:
                output.append(input[i - 1])

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_6(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is a list of elements where each element in input is added to its position(Using zero-indexing)"""
    pairs = {}
    for i in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        output = []
        for i, num in enumerate(input):
            element = i + num
            output.append(element)

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_7(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is a list of element whose position is indicated by the last element in the input

    EXAMPLE:
    1. [26, 88, 60, 1, 17, 75, 97, 89, 1] -> [88]
    2.  [49, 71, 2, 61, 3]: [61]
    """
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        # Create a chosen index between the bounds of the size of the input
        chosen_index = rng.randint(0, len(input) - 1)
        # Replace the last element with chosen_index
        input[-1] = chosen_index
        output = [input[chosen_index]]

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_8(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is count of elements in the input"""
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        output = [len(input)]

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_9(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is sum total of elements in the input"""
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        output = [sum(input)]

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_10(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is a list of the elements in ascending order"""
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        output = sort_integers(input, order="ascending")

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_11(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is a list of the elements in descending order"""
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        output = sort_integers(input, order="descending")

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_12(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is a list of the elements where the first and last element in input are replaced by their
    successor. Example, for an integer 4, successor is 5
    """
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        # Create successor for first and last element using a copy of input
        output = input.copy()
        output[0] += 1
        output[-1] += 1

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_13(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is [1] if list of input elements is in ascending order, [0] in descending order"""
    pairs = {}
    for i in range(NUM_OF_PAIRS_GENERATED):
        input = create_random_list(rng)
        if i % 2 == 0:
            input = sort_integers(input, order="ascending")
            output = [1]
        else:
            input = sort_integers(input, order="descending")
            output = [0]

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_14(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is [1] if input element is divisible by 10, [0] if divisible by 5"""
    pairs = {}

    nums = create_numbers_divisible_by_five_or_ten(rng)
    for num in nums:
        if num % 10 == 0:
            input = [num]
            output = [1]
        else:
            input = [num]
            output = [0]

        input = str(input)
        output = str(output)
        pairs[input] = output

    return pairs


def generate_15(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is a twice the amount of last element in the input"""
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        starter_input = create_random_list(rng)
        length = len(starter_input)
        first_element = rng.choice(starter_input)
        input = [first_element]

        for _ in range(1, length):
            prev = input[-1]
            input.append(prev * 2)

        # Create output here to prevent building on strings
        output = str([input[-1] * 2])
        input = str(input)
        pairs[input] = output

    return pairs


def generate_16(rng: Random) -> dict[str, Any]:
    """Generate input and output pairs where output is built from a function 2x - 4
    NOTE: This is suppose to be amazingly hard for the LLM.
    """
    pairs = {}
    for _ in range(NUM_OF_PAIRS_GENERATED):
        starter_input = create_random_list(rng)
        first_element = rng.choice(starter_input)
        output = (2 * first_element) - 4
        input = str([first_element])
        pairs[input] = str([output])

    return pairs
