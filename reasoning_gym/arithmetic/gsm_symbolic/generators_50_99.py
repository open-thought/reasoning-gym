from fractions import Fraction
from random import Random
from typing import Any

from reasoning_gym.utils import format_number, is_integer


def generate_50(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name: str, pieces1: int, pieces2: int) -> dict[str, Any]:
        half_pieces1 = pieces1 // 2
        total_pieces = half_pieces1 + pieces2

        question = f"{name} finished half of a {pieces1} piece puzzle, and then started and finished another {pieces2} piece puzzle within an hour. How many puzzle pieces did {name} place during that hour?"

        answer_cot = f"{name} did 1/2 * {pieces1} pieces = {half_pieces1} pieces.\n{name} completed {half_pieces1} pieces + {pieces2} pieces = {total_pieces} pieces.\n#### {total_pieces}"

        return {
            "question": question,
            "answer": format_number(total_pieces),
            "answer_cot": answer_cot,
            "answer_value": total_pieces,
            "variables": {
                "name": name,
                "puzzle1_pieces": pieces1,
                "puzzle2_pieces": pieces2,
                "half_puzzle1": half_pieces1,
                "total_pieces": total_pieces,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Teddy", "Tommy", "Billy", "Jimmy", "Bobby", "Danny"]
        name = rng.choice(names)

        # Generate random puzzle sizes that are even numbers
        puzzle1 = int(rng.randrange(100, int(500 * difficulty), 2))
        puzzle2 = int(rng.randrange(300, int(1000 * difficulty), 2))

        result = generate_from_variables(name, puzzle1, puzzle2)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_51(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        parent: str,
        activity1: str,
        activity2: str,
        activity3: str,
        cur: str,
        times: int,
        budget: int,
        tokens: int,
        cost1: int,
        cost2: int,
    ) -> dict[str, Any]:

        cost_per_ride = cost2 * times
        cost_per_person = tokens + cost1 + cost_per_ride
        total_people = budget // cost_per_person
        friends = total_people - 1

        question = f"{name}'s {parent} said that she had {cur}{budget} budgeted for her birthday party. She wants to make sure she and her friends all get to play one round of {activity1}, have {cur}{tokens} in {activity2} tokens, and get to ride the {activity3} {times}. A round of {activity1} is {cur}{cost1}. The {activity3} cost {cur}{cost2} a ride. How many friends can she invite?"

        answer_cot = (
            f"The {activity3} will cost {cur}{cost_per_ride} per person because {cost2} x {times} = {cost_per_ride}\n"
            f"Each person costs {cur}{cost_per_person} because {tokens} + {cost1} + {cost_per_ride} = {cost_per_person}\n"
            f"{total_people} total people can attend because {budget} / {cost_per_person} = {total_people}\n"
            f"She can invite {friends} friends because {total_people} - 1 = {friends}\n"
            f"#### {friends}"
        )

        return {
            "question": question,
            "answer": format_number(friends),
            "answer_cot": answer_cot,
            "answer_value": friends,
            "variables": {
                "name": name,
                "parent": parent,
                "activity1": activity1,
                "activity2": activity2,
                "activity3": activity3,
                "currency": cur,
                "times": times,
                "budget": budget,
                "tokens": tokens,
                "cost1": cost1,
                "cost2": cost2,
                "cost_per_ride": cost_per_ride,
                "cost_per_person": cost_per_person,
                "total_people": total_people,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Olivia", "Sophia", "Isabella", "Mia", "Charlotte"]
        parents = ["mom", "dad", "aunt", "uncle"]
        activities1 = ["mini-golf", "bowling", "laser tag"]
        activities2 = ["arcade", "game room", "pinball"]
        activities3 = ["go-karts", "bumper cars", "roller coaster"]
        currencies = ["$", "£", "€"]
        times_options = [2, 3]

        name = rng.choice(names_female)
        parent = rng.choice(parents)
        activity1 = rng.choice(activities1)
        activity2 = rng.choice(activities2)
        activity3 = rng.choice(activities3)
        cur = rng.choice(currencies)
        times = rng.choice(times_options)

        tokens = int(rng.randint(3, int(11 * difficulty)))
        cost1 = int(rng.randint(3, int(11 * difficulty)))
        cost2 = int(rng.randint(5, int(21 * difficulty)))

        # Generate budget ensuring conditions are met
        cost_per_person = tokens + cost1 + (cost2 * times)
        num_people = rng.randint(2, int(10 * difficulty))
        budget = cost_per_person * num_people

        result = generate_from_variables(
            name, parent, activity1, activity2, activity3, cur, times, budget, tokens, cost1, cost2
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_52(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name: str, alphabets: tuple, n1: str, frac: str) -> dict[str, Any]:
        alphabet_name, alphabet_count = alphabets

        # Calculate intermediate values
        full_writes = n1 * alphabet_count
        half_write = int(alphabet_count * frac)
        subtotal = full_writes + half_write
        final_total = subtotal * 2

        question = f"{name} is learning to write and decides to keep re-writing the {alphabet_name} until she knows it. She writes it in full {n1}, writes {frac} of it once, then re-writes everything she has already written. How many letters has {name} written in total?"

        answer_cot = (
            f"{name} has written the {alphabet_name} {n1} time(s) which is a total of {alphabet_count} * {n1} = {full_writes} letters.\n"
            f"She then writes {frac} the {alphabet_name}, which is {alphabet_count} * {frac} = {half_write} letters.\n"
            f"So far, this is a total of {full_writes} + {half_write} = {subtotal} letters.\n"
            f"Writing this again means she has doubled the number of letters she has written, so she has written a total of {subtotal} * 2 = {final_total} letters.\n"
            f"#### {final_total}"
        )

        return {
            "question": question,
            "answer": format_number(final_total),
            "answer_cot": answer_cot,
            "answer_value": final_total,
            "variables": {
                "name": name,
                "alphabet_name": alphabet_name,
                "alphabet_count": alphabet_count,
                "times_written": n1,
                "fraction": frac,
                "full_writes": full_writes,
                "half_write": half_write,
                "total": final_total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Sophia", "Olivia", "Ava", "Isabella", "Mia", "Charlotte", "Amelia"]
        alphabets = [
            ("alphabet", 26),
            ("hiragana (with 48 letters)", 48),
            ("farsi alphabet (with 32 letters)", 32),
            ("arabic abjad (with 28 letters)", 28),
        ]
        multi_times = ["twice", "three times", "four times"]
        fraction_alnum = ["half", "one-third", "one-fourth"]

        name = rng.choice(names_female)
        alphabet = rng.choice(alphabets)
        n1 = rng.choice(multi_times)
        frac = rng.choice(fraction_alnum)

        # Convert text numbers to numeric values
        n1_map = {"twice": 2, "three times": 3, "four times": 4}
        frac_map = {"half": 0.5, "one-third": 1 / 3, "one-fourth": 0.25}

        # Ensure division results in integer
        while not is_integer(alphabet[1] * frac_map[frac]):
            alphabet = rng.choice(alphabets)

        result = generate_from_variables(name, alphabet, n1_map[n1], frac_map[frac])

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_53(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name: str, sides: int, target: int, property: str) -> dict[str, Any]:
        numbers_above = sides - target
        prob_above = (numbers_above / sides) * 100
        prob_two_in_row = 25  # probability of two even/odd in a row is always 25%
        difference = int(prob_above - prob_two_in_row)

        question = f"{name} is rolling a {sides}-sided die. How much more likely is it (expressed as a percentage) that he rolls a number greater than {target} than that he rolls two {property} numbers in a row?"

        answer_cot = f"There are {numbers_above} numbers greater than {target} on the dice, so the chances of rolling one of them are {numbers_above} / {sides} = {prob_above}%.\nThe chance of rolling one {property} number is 50%, so the chance of rolling two in a row is 50% * 50% = 25%.\nThe difference between these two probabilities is {prob_above}% - 25% = {difference}%.\n#### {difference}"

        return {
            "question": question,
            "answer": format_number(difference),
            "answer_cot": answer_cot,
            "answer_value": difference,
            "variables": {
                "name": name,
                "sides": sides,
                "target": target,
                "property": property,
                "numbers_above": numbers_above,
                "prob_above": prob_above,
                "prob_two_in_row": prob_two_in_row,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
        properties = ["even", "odd"]

        name = rng.choice(names)
        property = rng.choice(properties)

        dice_options = [4, 6, 8, 10, 12, 20]
        sides = rng.choice(dice_options)

        # Generate target ensuring conditions are met
        while True:
            target = rng.randint(1, sides - 1)
            prob = ((sides - target) / sides) * 100
            if (sides - target) % target == 0 and prob.is_integer() and prob > 25:
                break

        result = generate_from_variables(name, sides, target, property)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_54(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name1: str,
        name2: str,
        total_time: int,
        library_time: int,
        station_time: int,
        location1: str,
        location2: str,
        location3: str,
    ) -> dict[str, Any]:

        time_after_library = total_time - library_time
        remaining_time = time_after_library - station_time

        question = f"{name1} and {name2} have {total_time} minutes to walk to {location1} together. It takes them {library_time} minutes to get to the corner where the {location2} is. It takes them another {station_time} minutes to get to the {location3}. How much longer do they have to get to {location1} without being late?"

        answer_cot = f"{name1} and {name2} arrive at the {location2} with {total_time} - {library_time} = {time_after_library} minutes left to reach the {location1}.\nThey then arrive at the {location3} and have {time_after_library} - {station_time} = {remaining_time} minutes left to get to {location1} without being late.\n#### {remaining_time}"

        return {
            "question": question,
            "answer": format_number(remaining_time),
            "answer_cot": answer_cot,
            "answer_value": remaining_time,
            "variables": {
                "name1": name1,
                "name2": name2,
                "total_time": total_time,
                "library_time": library_time,
                "station_time": station_time,
                "location1": location1,
                "location2": location2,
                "location3": location3,
                "remaining_time": remaining_time,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Jack", "James", "William", "Michael", "David", "Joseph"]
        locations = ["cinema", "mall", "library", "park", "gym", "bank", "school"]

        name1, name2 = rng.sample(names, 2)
        loc1, loc2, loc3 = rng.sample(locations, 3)

        # Generate times ensuring they're not divisible by 5
        library_time = int(rng.randint(10, int(30 * difficulty)))
        while library_time % 5 == 0:
            library_time = int(rng.randint(10, int(30 * difficulty)))

        station_time = int(rng.randint(10, int(70 * difficulty)))
        while station_time % 5 == 0:
            station_time = int(rng.randint(10, int(70 * difficulty)))

        # Ensure total time is greater than sum of other times
        min_total = library_time + station_time + 5
        total_time = int(rng.randint(min_total, int(140 * difficulty)))
        while total_time % 5 == 0:
            total_time = int(rng.randint(min_total, int(140 * difficulty)))

        result = generate_from_variables(name1, name2, total_time, library_time, station_time, loc1, loc2, loc3)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_55(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, place: str, fruit: str, location: str, insect1: str, insect2: str, n: int, frac: str
    ) -> dict[str, Any]:
        num_insect1 = int(n * 0.5)  # half as many bugs as ants
        total_insects = n + num_insect1

        question = f"{name} went to their {place} to pick some {fruit} and found {frac} as many {insect1} as {insect2} in the {location}. If there were {n} {insect2}, calculate the total number of insects in the {location}."

        answer_cot = f"If there were {n} {insect2}, the total number of {insect1} in the {location} is {frac} * {n} {insect2} = {num_insect1} {insect1}\nThe total number of insects in the {location} is {num_insect1} {insect1} + {n} {insect2} = {total_insects} insects\n#### {total_insects}"

        return {
            "question": question,
            "answer": format_number(total_insects),
            "answer_cot": answer_cot,
            "answer_value": total_insects,
            "variables": {
                "name": name,
                "place": place,
                "fruit": fruit,
                "location": location,
                "insect1": insect1,
                "insect2": insect2,
                "n": n,
                "frac": frac,
                "num_insect1": num_insect1,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Dax", "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Riley"]
        places = ["orchard", "backyard", "greenhouse", "allotment"]
        fruits = ["strawberries", "cherries", "blueberries", "raspberries"]
        locations = ["garden", "field", "plot", "patch"]
        insects = ["beetles", "ladybugs", "grasshoppers", "caterpillars", "bees", "wasps"]

        name = rng.choice(names)
        place = rng.choice(places)
        fruit = rng.choice(fruits)
        location = rng.choice(locations)
        insect1, insect2 = rng.sample(insects, 2)

        n = int(rng.randint(20, int(200 * difficulty)))
        # Ensure n is even for "half as many"
        if n % 2 == 1:
            n += 1

        result = generate_from_variables(name, place, fruit, location, insect1, insect2, n, "half")

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_56(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        family: str, item: str, total: int, n1: int, n2: int, flavor1: str, flavor2: str, flavor3: str
    ) -> dict[str, Any]:
        n3 = total - (n1 + n2)

        question = f"The {family} family is busy making {item}s. So far, they've made {total} {item}s. They have {n1} {flavor1} {item}s, {n2} {flavor2} {item}s, and some {flavor3} {item}s. How many {flavor3} {item}s have they made?"

        answer_cot = f"The total number of pieces of {flavor1} and {flavor2} {item}s is {n1} + {n2} = {n1+n2}.\nTherefore, they made {total} - {n1+n2} = {n3} {flavor3} {item}s.\n#### {n3}"

        return {
            "question": question,
            "answer": format_number(n3),
            "answer_cot": answer_cot,
            "answer_value": n3,
            "variables": {
                "family": family,
                "item": item,
                "total": total,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "flavor1": flavor1,
                "flavor2": flavor2,
                "flavor3": flavor3,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        families = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
        items = ["cupcake", "muffin", "brownie", "biscuit"]
        flavors = ["vanilla", "strawberry", "blueberry", "lemon", "peanut butter"]

        family = rng.choice(families)
        item = rng.choice(items)
        flavor1, flavor2, flavor3 = rng.sample(flavors, 3)

        total = int(rng.randrange(5000, int(10000 * difficulty), 25))
        n1 = int(rng.randint(1000, int(3000 * difficulty)))
        n2 = int(rng.randint(1000, int(3000 * difficulty)))

        while n1 + n2 >= total:
            n1 = int(rng.randint(1000, int(3000 * difficulty)))
            n2 = int(rng.randint(1000, int(3000 * difficulty)))

        result = generate_from_variables(family, item, total, n1, n2, flavor1, flavor2, flavor3)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_57(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        n1: int, sport1: str, sport2: str, sport3: str, n2: int, n3: int, multiplier: int
    ) -> dict[str, Any]:
        n_volleyball = n1 * multiplier
        n_soccer = n2 + n3
        total = n1 + n_volleyball + n_soccer

        question = f"There are {n1} students playing {sport1} and twice that number playing {sport2}. There are {n2} boys and {n3} girls playing {sport3}. If each student only participates in one group, how many students are there in total?"

        answer_cot = f"There are {n1} x {multiplier} = {n_volleyball} students playing {sport2}.\nThere are {n2} + {n3} = {n_soccer} students playing {sport3}.\nIn total there are {n1} + {n_volleyball} + {n_soccer} = {total} students.\n#### {total}"

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "tennis_players": n1,
                "volleyball_players": n_volleyball,
                "soccer_boys": n2,
                "soccer_girls": n3,
                "total_soccer": n_soccer,
                "total_students": total,
                "sports": [sport1, sport2, sport3],
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        sports = ["basketball", "badminton", "table tennis", "football", "volleyball"]
        sport1, sport2, sport3 = rng.sample(sports, 3)

        multiplier = 2  # "twice" that number

        # We need: n1 * multiplier + n2 + n3 <= 250
        # So: n1 * 2 + n2 + n3 <= 250
        # First generate n1 with consideration for leaving room for n2 and n3
        # Since n2, n3 >= 10 each, we need: n1 * 2 <= 230
        max_n1 = min(int(21 * difficulty), 115)  # 115 is floor(230/2)
        n1 = rng.randint(4, max_n1)

        # Now we know remaining space for n2 + n3
        remaining_total = 250 - (n1 * multiplier)

        # Calculate maximum for n2, ensuring space left for n3 (minimum 10)
        max_n2 = min(int(31 * difficulty), remaining_total - 10)
        if max_n2 < 10:
            # If ranges are too tight, adjust n1 down and recalculate
            n1 = max(4, n1 - 10)
            remaining_total = 250 - (n1 * multiplier)
            max_n2 = min(int(31 * difficulty), remaining_total - 10)

        n2 = rng.randint(10, max(11, max_n2))

        # Generate n3 with remaining space
        max_n3 = min(int(31 * difficulty), remaining_total - n2)
        n3 = rng.randint(10, max(11, max_n3))

        result = generate_from_variables(n1, sport1, sport2, sport3, n2, n3, multiplier)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_58(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, container: str, liquid: str, volume: int, unit: str, num_containers: int, calories: int
    ) -> dict[str, Any]:
        total_volume = volume * num_containers
        total_calories = total_volume * calories

        question = f"A {container} of {liquid} is {volume} {unit}s of {liquid}. {name} drinks {num_containers} {container}s of {liquid}. If {liquid} has {calories} calories per {unit} how many calories did he consume?"

        answer_cot = f"He drank {volume}*{num_containers}={total_volume} {unit}s of {liquid}.\nSo he drank {total_volume}*{calories}={total_calories} calories of {liquid}\n#### {total_calories}"

        return {
            "question": question,
            "answer": format_number(total_calories),
            "answer_cot": answer_cot,
            "answer_value": total_calories,
            "variables": {
                "name": name,
                "container": container,
                "liquid": liquid,
                "volume": volume,
                "unit": unit,
                "num_containers": num_containers,
                "calories": calories,
                "total_volume": total_volume,
                "total_calories": total_calories,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Mike", "James", "David", "Robert", "William"]
        containers = ["cup", "bottle", "carton"]
        liquids = ["juice", "soda", "sparkling water", "tea", "lemonade"]
        units = ["ounce", "mL", "cc", "oz"]

        name = rng.choice(names)
        container = rng.choice(containers)
        liquid = rng.choice(liquids)
        unit = rng.choice(units)

        volume = int(rng.randint(6, int(16 * difficulty)))
        num_containers = int(rng.randint(2, int(6 * difficulty)))
        calories = int(rng.randint(2, int(10 * difficulty)))

        result = generate_from_variables(name, container, liquid, volume, unit, num_containers, calories)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_59(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        time_per_interval: int, distance_per_interval: int, total_distance: int
    ) -> dict[str, Any]:
        intervals = total_distance // distance_per_interval
        total_time = intervals * time_per_interval

        question = f"A fog bank rolls in from the ocean to cover a city. It takes {time_per_interval} minutes to cover every {distance_per_interval} miles of the city. If the city is {total_distance} miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the whole city?"

        answer_cot = f"The city will be covered in {total_distance} / {distance_per_interval} = {intervals} intervals of {time_per_interval} minutes.\nThus, it will take {intervals} * {time_per_interval} = {total_time} minutes for the fog to cover the whole city.\n#### {total_time}"

        return {
            "question": question,
            "answer": format_number(total_time),
            "answer_cot": answer_cot,
            "answer_value": total_time,
            "variables": {
                "time_per_interval": time_per_interval,
                "distance_per_interval": distance_per_interval,
                "total_distance": total_distance,
                "intervals": intervals,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        # Start with total_distance limit and work backwards
        max_total_distance = 100

        # Generate num_intervals first
        min_intervals = 2
        max_intervals = int(20 * difficulty)

        # Calculate maximum allowed distance_per_interval based on constraints:
        # distance_per_interval * num_intervals <= max_total_distance
        min_distance = 2
        max_possible_distance = max_total_distance // min_intervals
        max_distance = min(int(100 * difficulty), max_possible_distance)

        if max_distance < min_distance:
            # Fallback if no valid solution exists
            distance_per_interval = min_distance
            num_intervals = max_total_distance // distance_per_interval
        else:
            distance_per_interval = rng.randint(min_distance, max_distance)
            # Calculate valid range for num_intervals based on chosen distance
            max_valid_intervals = min(max_intervals, max_total_distance // distance_per_interval)
            num_intervals = rng.randint(min_intervals, max_valid_intervals)

        total_distance = distance_per_interval * num_intervals

        # Generate time per interval independently since it has no constraints
        time_per_interval = int(rng.randint(2, int(500 * difficulty)))

        result = generate_from_variables(time_per_interval, distance_per_interval, total_distance)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_60(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, unit: str, total_dist: int, beach_dist: int, sidewalk_dist: int, speed_mult: int, beach_time: int
    ) -> dict[str, Any]:

        beach_rate = Fraction(beach_dist, beach_time)
        sidewalk_rate = beach_rate * speed_mult
        sidewalk_time = int(sidewalk_dist / sidewalk_rate)
        total_time = beach_time + sidewalk_time

        question = f"{name} walks {total_dist} {unit}s every day on her favorite walking trail, which includes {beach_dist} {unit}s of walking on the beach and {sidewalk_dist} {unit}s of walking on the sidewalk. On the sidewalk, {name} walks at twice the rate of speed that she does on the beach. If {beach_time} minutes of her walk is spent on the beach, how long does it take for her to complete the entire {total_dist}-{unit} walk, in minutes?"

        answer_cot = f"On the beach, {name} walks at a rate of {beach_dist} {unit}s per {beach_time} minutes, or {beach_dist}/{beach_time} = {beach_rate} {unit}s per minute.\nOn the sidewalk, she walks at {speed_mult} times the rate of speed as when she is on the sand, or {speed_mult} * {beach_rate} = {sidewalk_rate} {unit}s per minute.\nTo walk {sidewalk_dist} {unit}s on the sidewalk, it takes her {sidewalk_dist}÷{sidewalk_rate}={sidewalk_time} minutes.\nThus, in total, it takes {name} {beach_time}+{sidewalk_time}={total_time} minutes to walk her favorite route.\n#### {total_time}"

        return {
            "question": question,
            "answer": format_number(total_time),
            "answer_cot": answer_cot,
            "answer_value": total_time,
            "variables": {
                "name": name,
                "unit": unit,
                "total_distance": total_dist,
                "beach_distance": beach_dist,
                "sidewalk_distance": sidewalk_dist,
                "speed_multiplier": speed_mult,
                "beach_time": beach_time,
                "sidewalk_time": sidewalk_time,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Emma", "Sophia", "Isabella", "Olivia", "Ava", "Mia", "Emily"]
        units = ["mile", "kilometer", "block"]

        name = rng.choice(names)
        unit = rng.choice(units)
        speed_mult = 2  # Fixed as "twice" in question

        # Start with beach_dist since other values depend on it
        beach_dist = rng.randint(10, min(int(20 * difficulty), 15))

        # Calculate beach_time to ensure it's:
        # 1. Greater than beach_dist
        # 2. Greater than speed_mult * beach_dist
        # 3. Divisible by beach_dist
        min_time = max(40, (speed_mult * beach_dist + 1))  # Must be greater than 2 * beach_dist
        max_time = min(int(70 * difficulty), 65)

        # Find valid multiples of beach_dist within our range
        valid_times = []
        for multiplier in range((min_time + beach_dist - 1) // beach_dist, (max_time // beach_dist) + 1):
            candidate_time = multiplier * beach_dist
            if min_time <= candidate_time <= max_time:
                valid_times.append(candidate_time)

        if not valid_times:
            # Fallback: adjust beach_dist down and recalculate
            beach_dist = max(5, beach_dist - 5)
            base_multiplier = (min_time + beach_dist - 1) // beach_dist
            beach_time = beach_dist * base_multiplier
        else:
            beach_time = rng.choice(valid_times)

        # Generate sidewalk_dist last since it has fewest constraints
        sidewalk_dist = rng.randint(10, min(int(20 * difficulty), 15))
        total_dist = beach_dist + sidewalk_dist

        result = generate_from_variables(name, unit, total_dist, beach_dist, sidewalk_dist, speed_mult, beach_time)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_61(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        location: str,
        shop: str,
        item1: str,
        item2: str,
        item3: str,
        unit: str,
        cur: str,
        total: float,
        n1: int,
        n2: int,
        n12: int,
        k: int,
        n3: int,
        p1: float,
        p2: float,
        p3: float,
        discount: float,
    ) -> dict[str, Any]:

        # Calculate costs
        item1_cost = n1 * p1 + n2 * (1 - discount) * p1 + k * p1  # Cost of item1 with discount applied
        item2_cost = p2  # Cost of item2
        item3_cost = n3 * p3  # Cost of item3
        total_spent = int(item1_cost + item2_cost + item3_cost)
        money_left = total - total_spent

        question = f'{name} went to the {location} for vacation. Her parents gave her {cur}{total} to buy whatever she wanted. At the {shop}, {item1} was on sale for "Buy {n1} {unit}s at {cur}{p1} per {unit}, get {n2} {unit}s {discount} off." She scooped up {n12} {unit}s. She also bought a mixed bag of {item2} for {cur}{p2} and {n3} {item3} that were {cur}{p3} each. How much money does {name} have left?'

        answer_cot = f"{item1} is {n1} {unit}s for {cur}{p1} and gets {n2} {unit}s {discount} off. So {discount} off of {n2} {unit}s is {cur}{n2*discount}*{p1} = {cur}{n2*discount*p1}. The rest of {k} {unit}s does not have discount and come at {k*p1} so total is {n1}*{p1} + {n2}*{1-discount}*{p1} + {k}*{p1} = {item1_cost}\n{n3} {item3} at {cur}{p3} each is {n3}*{p3}={cur}{n3*p3}\nWhen you add all her purchases, {cur}{item1_cost}+{cur}{p2}+{cur}{n3*p3} = {cur}{total_spent}\nShe had {cur}{total} and spent {cur}{total_spent} so she had {cur}{total}-{cur}{total_spent} = {cur}{money_left} left over\n#### {money_left}"

        return {
            "question": question,
            "answer": format_number(money_left),
            "answer_cot": answer_cot,
            "answer_value": money_left,
            "variables": {
                "name": name,
                "location": location,
                "shop": shop,
                "item1": item1,
                "item2": item2,
                "item3": item3,
                "unit": unit,
                "currency": cur,
                "total_money": total,
                "n1": n1,
                "n2": n2,
                "n12": n12,
                "k": k,
                "n3": n3,
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "discount": discount,
                "total_spent": total_spent,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte"]
        locations = ["beach", "boardwalk", "pier", "coast"]
        shops = ["souvenir store", "gift shop", "beach shop", "seaside store"]
        items1 = ["fudge", "saltwater taffy", "rock candy", "cotton candy"]
        items2 = ["sand dollars", "starfish", "sea glass", "coral pieces"]
        items3 = ["postcards", "keychains", "stickers", "pins"]
        units = ["pound", "kilogram", "kg"]
        currencies = ["$", "£", "€"]
        fraction_nums = [0.25, 0.33, 0.5, 0.67, 0.75]

        # Choose random values for strings
        name = rng.choice(names_female)
        location = rng.choice(locations)
        shop = rng.choice(shops)
        item1 = rng.choice(items1)
        item2 = rng.choice(items2)
        item3 = rng.choice(items3)
        unit = rng.choice(units)
        cur = rng.choice(currencies)
        discount = rng.choice(fraction_nums[:4])

        # Generate fixed prices
        p2 = round(rng.uniform(11.25, 12.00), 2)
        p3 = round(rng.uniform(20.25, 21.25), 2)

        # Generate n1 first as it's the base for other values
        n1 = int(rng.randint(15, int(18 * difficulty)))

        # Generate n2 ensuring it's less than n1
        n2 = int(rng.randint(4, min(n1 - 1, int(10 * difficulty))))

        # Generate k ensuring 0 <= k < n1
        k = int(rng.randint(2, min(n1 - 1, int(5 * difficulty))))

        # Calculate n12
        n12 = n1 + n2 + k

        # Generate n3
        n3 = int(rng.randint(11, int(19 * difficulty)))

        # Generate p1 ensuring total cost is less than total budget
        min_p1 = 20
        max_p1 = int(24 * difficulty)

        # Calculate maximum p1 that keeps total cost under budget
        total = int(rng.randint(1200, int(1500 * difficulty)))

        # Function to calculate total cost
        def calc_total_cost(price):
            return n1 * price + n2 * (1 - discount) * price + k * price + p2 + n3 * p3

        # Find valid p1 values
        valid_p1 = []
        for p1 in range(min_p1, max_p1 + 1):
            cost = calc_total_cost(p1)
            if cost == int(cost) and cost < total:  # Ensure integer and under budget
                valid_p1.append(p1)

        if not valid_p1:
            # Fallback: adjust values to make it work
            p1 = min_p1
            # Adjust n values down if needed
            while calc_total_cost(p1) >= total:
                if n1 > 15:
                    n1 -= 1
                if n2 > 4:
                    n2 -= 1
                if k > 2:
                    k -= 1
                if n3 > 11:
                    n3 -= 1
                n12 = n1 + n2 + k
        else:
            p1 = rng.choice(valid_p1)

        result = generate_from_variables(
            name, location, shop, item1, item2, item3, unit, cur, total, n1, n2, n12, k, n3, p1, p2, p3, discount
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_62(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        item: str, num_slices: int, name1: str, name2: str, slices_per_day: int, multiplier: int, unit: str
    ) -> dict[str, Any]:

        second_person_slices = slices_per_day * multiplier
        total_daily_slices = slices_per_day + second_person_slices
        days_lasting = num_slices // total_daily_slices

        question = f"A {item} has {num_slices} {unit}. If {name1} can eat {slices_per_day} {unit} a day while {name2} can eat {multiplier} times as much, how many days will the {item} last?"

        answer_cot = f"{name2} can eat {slices_per_day} x {multiplier} = {second_person_slices} {unit} a day.\nTogether, {name1} and {name2} can eat {slices_per_day} + {second_person_slices} = {total_daily_slices} {unit} a day.\nSo, a {item} will last for {num_slices}/{total_daily_slices} = {days_lasting} days.\n#### {days_lasting}"

        return {
            "question": question,
            "answer": format_number(days_lasting),
            "answer_cot": answer_cot,
            "answer_value": days_lasting,
            "variables": {
                "item": item,
                "num_slices": num_slices,
                "name1": name1,
                "name2": name2,
                "slices_per_day": slices_per_day,
                "multiplier": multiplier,
                "second_person_slices": second_person_slices,
                "total_daily_slices": total_daily_slices,
                "unit": unit,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        items = ["pizza", "cake", "pie", "lasagna"]
        units = ["pieces", "portions", "servings"]
        names = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Elijah", "Charlotte", "James"]

        item = rng.choice(items)
        unit = rng.choice(units)
        name1, name2 = rng.sample(names, 2)

        slices_per_day = int(rng.randint(2, int(6 * difficulty)))
        multiplier = 2  # Using 'twice' as specified in original

        # Ensure total is divisible by daily consumption
        daily_total = slices_per_day + (slices_per_day * multiplier)
        num_days = rng.randint(2, int(8 * difficulty))
        num_slices = daily_total * num_days

        result = generate_from_variables(item, num_slices, name1, name2, slices_per_day, multiplier, unit)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_63(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name: str, hours: int, days: int, rate: int, bonus: int, month: str) -> dict[str, Any]:
        daily_pay = hours * rate
        monthly_days = days * 4
        monthly_base = daily_pay * monthly_days
        monthly_bonus = bonus * 4
        total_pay = monthly_base + monthly_bonus

        question = f"{name} works a {hours}-hour shift each day, {days} days a week. He earns ${rate} per hour and gets a ${bonus} bonus each week if the company performs well. How much money did {name} make in {month} if the company performed very well for the whole month?"

        answer_cot = (
            f"In a day, {name} makes {hours} * {rate} = ${daily_pay}\n"
            f"If he works {days} days a week, the total number of days for the whole month is {days} * 4= {monthly_days} days.\n"
            f"Since he makes ${daily_pay} per day, the total amount for the whole month is {monthly_days} * {daily_pay}= ${monthly_base}.\n"
            f"He also got a {bonus} * 4 = ${monthly_bonus} bonus because the company performed well in all the weeks of {month}.\n"
            f"At the end of {month}, he earned {monthly_base} + {monthly_bonus} = ${total_pay}.\n#### {total_pay}"
        )

        return {
            "question": question,
            "answer": format_number(total_pay),
            "answer_cot": answer_cot,
            "answer_value": total_pay,
            "variables": {
                "name": name,
                "hours_per_day": hours,
                "days_per_week": days,
                "hourly_rate": rate,
                "weekly_bonus": bonus,
                "month": month,
                "daily_pay": daily_pay,
                "monthly_base": monthly_base,
                "monthly_bonus": monthly_bonus,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        name = rng.choice(names)
        month = rng.choice(months)

        hours = int(rng.randint(6, int(13 * difficulty)))
        days = int(rng.randint(3, int(7 * difficulty)))
        rate = int(rng.randint(8, int(31 * difficulty)))
        bonus = int(rng.randint(100, int(601 * difficulty)))

        # Ensure rate * hours is an integer
        while (hours * rate) % 1 != 0:
            rate = int(rng.randint(8, int(31 * difficulty)))

        result = generate_from_variables(name, hours, days, rate, bonus, month)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_64(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name: str, n1: int, d1: int, n2: int, d2: int) -> dict[str, Any]:
        first_period = n1 * d1
        second_period = n2 * d2
        total_eggs = first_period + second_period
        dozens = total_eggs // 12

        question = f"If {name} eats {n1} eggs a day for {d1} days and then increases it to {n2} eggs a day for {d2} days, how many dozens of eggs will {name} need for {d1+d2} days?"

        answer_cot = (
            f"He starts off eating {n1} eggs a day for {d1} days for a total of {n1}*{d1} = {first_period} eggs\n"
            f"Then he increases it to {n2} eggs a day for {d2} days for a total of {n2}*{d2} = {second_period} eggs\n"
            f"All total he will eat {first_period}+{second_period} = {total_eggs} eggs\n"
            f"There are 12 eggs in 1 dozen and he will eat {total_eggs} eggs which is {total_eggs}/12 = {dozens} dozen eggs\n"
            f"#### {dozens}"
        )

        return {
            "question": question,
            "answer": format_number(dozens),
            "answer_cot": answer_cot,
            "answer_value": dozens,
            "variables": {
                "name": name,
                "eggs_per_day_first": n1,
                "days_first": d1,
                "eggs_per_day_second": n2,
                "days_second": d2,
                "total_eggs": total_eggs,
                "dozens": dozens,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Chester", "James", "John", "Robert", "Michael", "William", "David"]
        name = rng.choice(names)

        # Generate random values scaled by difficulty
        n1 = int(rng.randint(2, int(6 * difficulty)))
        n2 = int(rng.randint(4, int(8 * difficulty)))
        while n2 <= n1:
            n2 = int(rng.randint(4, int(8 * difficulty)))

        d1 = int(rng.randint(20, int(110 * difficulty)))
        d2 = int(rng.randint(20, int(110 * difficulty)))

        # Ensure results are divisible by 12
        while (n1 * d1 + n2 * d2) % 12 != 0:
            d1 = int(rng.randint(20, int(110 * difficulty)))
            d2 = int(rng.randint(20, int(110 * difficulty)))

        result = generate_from_variables(name, n1, d1, n2, d2)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_65(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, fish: str, day: str, w1: int, w2: int, w3: int, n: int, unit: str, cur: str, price: float
    ) -> dict[str, Any]:
        total = int((w1 + w2) * price + (n - 2) * w3 * price)

        question = f"{name} caught {n} {fish}s last {day}, the first {fish} he caught weighs {w1} {unit}s, the second {fish} he caught weighs {w2} {unit}s, and the last {fish} he caught weighs {w3} {unit}s. If a {unit} of {fish} costs {cur}{price:.2f}, how much will he earn after selling all the {fish}s to the market?"

        answer_cot = (
            f"{name} will earn {w1} x {cur}{price:.2f} = {cur}{w1*price:.2f} from the first {fish}.\n"
            f"He will earn {w2} x {cur}{price:.2f} = {cur}{w2*price:.2f} for the second {fish}.\n"
            f"The rest of the {fish}s are {n}-2 = {n-2}. He will earn {w3} x {cur}{price:.2f} = {cur}{w3*price:.2f} per each of them. So he will earn {n-2} * {w3*price:.2f} = {(n-2)*w3*price:.2f}\n"
            f"Therefore, the total amount he will earn for all the {fish}s is {cur}{w1*price:.2f} + {cur}{w2*price:.2f} + {cur}{(n-2)*w3*price:.2f}= {cur}{total}.\n#### {total}"
        )

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "name": name,
                "fish": fish,
                "day": day,
                "weight1": w1,
                "weight2": w2,
                "weight3": w3,
                "num_fish": n,
                "unit": unit,
                "currency": cur,
                "price": price,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Michael", "David", "James", "Robert", "William", "Richard"]
        fish = ["salmon", "cod", "trout", "steelhead"]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        units = ["kilogram", "pound", "kg"]
        currencies = ["$", "€", "£"]

        name = rng.choice(names)
        fish_type = rng.choice(fish)
        day = rng.choice(days)
        unit = rng.choice(units)
        cur = rng.choice(currencies)

        w1 = int(rng.randint(40, int(80 * difficulty)))
        w2 = int(rng.randint(30, int(60 * difficulty)))
        w3 = int(rng.randint(20, int(40 * difficulty)))
        n = int(rng.randint(3, int(8 * difficulty)))
        price = round(rng.uniform(0.25, 2.5), 2)

        # Ensure result is integer
        while not ((w1 + w2) * price + (n - 2) * w3 * price).is_integer():
            price = round(rng.uniform(0.25, 2.5), 2)

        result = generate_from_variables(name, fish_type, day, w1, w2, w3, n, unit, cur, price)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_66(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        weekdays: list,
        hour1: int,
        hour2: int,
        hour3: int,
        min1: int,
        min2: int,
        total_hours: int,
        num_wed_episodes: int,
    ) -> dict[str, Any]:

        mon, tue, wed, thu, fri = weekdays

        question = f"{name} watches TV after he finishes his homework every night. On {mon} and {tue}, he watched a {hour1}-hour episode of his favorite show each night. On {wed}, he watched a few episodes of a {min1}-minute show. On {thu}, he finished homework early and watched a {hour2}-hour episode and a {min2}-minute show. On {fri}, he got to stay up late for the weekend, so he watched two {hour3}-hour episodes. If he watched {total_hours} hours of TV in all, how many {min1}-minute episodes did he watch on {wed}?"

        answer_cot = (
            f"Let {wed[0]} be the number of episodes he watched on {wed}.\n"
            f"After {mon}, he had {total_hours} - {hour1} = {total_hours-hour1} hours of TV left.\n"
            f"After {tue}, he had {total_hours-hour1} - {hour1} = {total_hours-2*hour1} hours of TV left.\n"
            f"After {thu}, he had {total_hours-2*hour1} - {hour2} - {Fraction(min2,60)} = {total_hours-2*hour1-hour2-Fraction(min2,60)} hours of TV left.\n"
            f"After {fri}, he had {total_hours-2*hour1-hour2-Fraction(min2,60)} - {2*hour3} = {total_hours-2*hour1-hour2-Fraction(min2,60)-2*hour3} hours of TV left.\n"
            f"Each {min1}-minute episode is {Fraction(min1,60)} hour.\n"
            f"Thus, {wed[0]} = {num_wed_episodes} episodes.\n#### {num_wed_episodes}"
        )

        return {
            "question": question,
            "answer": format_number(num_wed_episodes),
            "answer_cot": answer_cot,
            "answer_value": num_wed_episodes,
            "variables": {
                "name": name,
                "weekdays": weekdays,
                "hour1": hour1,
                "hour2": hour2,
                "hour3": hour3,
                "min1": min1,
                "min2": min2,
                "total_hours": total_hours,
                "num_wed_episodes": num_wed_episodes,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        name = rng.choice(names)
        weekdays_sample = weekdays.copy()  # Keep original order for this problem

        hour1 = int(rng.randint(3, int(7 * difficulty)))
        hour2 = int(rng.randint(2, int(7 * difficulty)))
        hour3 = int(rng.randint(2, int(6 * difficulty)))

        min1 = int(rng.randint(1, int(12 * difficulty))) * 5  # Ensure divisible by 5
        min2 = int(rng.randint(1, int(11 * difficulty))) * 5  # Ensure divisible by 5

        # Calculate num_wed_episodes to ensure total_hours works out
        num_wed_episodes = int(rng.randint(1, int(8 * difficulty)))

        # Calculate total hours from all components
        total_hours = 2 * hour1 + hour2 + min2 / 60 + 2 * hour3 + (num_wed_episodes * min1 / 60)

        result = generate_from_variables(
            name, weekdays_sample, hour1, hour2, hour3, min1, min2, total_hours, num_wed_episodes
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_67(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, fruit: str, n1: int, n2: int, d1: str, d2: str, d3: str, mult: int
    ) -> dict[str, Any]:
        first_two_days = n1 + n2
        friday_amount = mult * n1
        total = first_two_days + friday_amount

        question = f"{name} picks {n1} {fruit}s on {d1}. Then he picks {n2} {fruit}s on {d2}. On {d3}, he picks {mult} times the number of {fruit}s he did on {d1}. How many {fruit}s does {name} have?"

        answer_cot = f"Combining {d1} and {d2}, {name} has {n1} {fruit}s + {n2} {fruit}s = {first_two_days} {fruit}s.\nOn {d3}, he picks {mult} * {n1} {fruit}s = {friday_amount} {fruit}s.\nAltogether, {name} has {first_two_days} {fruit}s + {friday_amount} {fruit}s = {total} {fruit}s.\n#### {total}"

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "name": name,
                "fruit": fruit,
                "day1_amount": n1,
                "day2_amount": n2,
                "day1": d1,
                "day2": d2,
                "day3": d3,
                "multiplier": mult,
                "day3_amount": friday_amount,
                "total": total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "James", "William", "Michael", "David", "Robert", "Thomas"]
        fruits = ["banana", "apple", "orange", "pear", "peach", "plum"]
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        multipliers = ["double", "triple", "quadruple"]
        mult_values = {"double": 2, "triple": 3, "quadruple": 4}

        name = rng.choice(names)
        fruit = rng.choice(fruits)
        d1, d2, d3 = rng.sample(weekdays, 3)
        mult_word = rng.choice(multipliers)
        mult = mult_values[mult_word]

        n1 = int(rng.randint(30, int(400 * difficulty)))
        n2 = int(rng.randint(50, int(400 * difficulty)))

        result = generate_from_variables(name, fruit, n1, n2, d1, d2, d3, mult)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_68(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(n0: int, r: int, d: int, disease: str) -> dict[str, Any]:
        # Calculate infected people after each day
        day1_new = n0 * r
        day1_total = n0 + day1_new

        day2_new = day1_total * r
        day2_total = day1_total + day2_new

        day3_new = day2_total * r
        day3_total = day2_total + day3_new

        question = f"A {disease} infects {n0} people. Every day, each infected person infects {r} others. How many people are infected after {d} days?"

        answer_cot = (
            f"On the first day, the original {n0} people infect {r} people each, so {n0} * {r} = {day1_new} more people are infected.\n"
            f"There are {n0} + {day1_new} = {day1_total} infected people after the first day.\n"
            f"On the second day, {day1_total} * {r} = {day2_new} more people are infected.\n"
            f"There are {day1_total} + {day2_new} = {day2_total} infected people after the second day.\n"
            f"On the third day, {day2_total} * {r} = {day3_new} more people are infected. Therefore, there are {day2_total} + {day3_new} = {day3_total} infected people after three days.\n"
            f"#### {day3_total}"
        )

        return {
            "question": question,
            "answer": format_number(day3_total),
            "answer_cot": answer_cot,
            "answer_value": day3_total,
            "variables": {"initial_infected": n0, "infection_rate": r, "days": d, "disease_type": disease},
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        diseases = ["virus", "bacteria", "parasite", "infection"]
        disease = rng.choice(diseases)
        d = 3  # Fixed at 3 days

        # We need: n0 * (r + 1)**3 < 20000
        # Work backwards to find maximum r for given n0
        # (r + 1)**3 < 20000/n0
        # r + 1 < (20000/n0)**(1/3)
        # r < (20000/n0)**(1/3) - 1

        # Start with n0 since it has simpler constraints
        max_n0 = min(int(21 * difficulty), 20)  # Cap at 20 to keep numbers manageable
        n0 = rng.randint(5, max_n0)

        # Calculate maximum r that satisfies our inequality
        max_possible_r = int((20000 / n0) ** (1 / 3) - 1)
        max_r = min(int(8 * difficulty), max_possible_r, 7)  # Cap at 7 for reasonable numbers

        if max_r < 2:
            # If range is too tight, adjust n0 down and recalculate
            n0 = 5  # Use minimum value
            max_possible_r = int((20000 / n0) ** (1 / 3) - 1)
            max_r = min(int(8 * difficulty), max_possible_r, 7)

        r = rng.randint(2, max(3, max_r))  # Ensure at least one valid choice

        result = generate_from_variables(n0, r, d, disease)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_69(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name: str, document: str, total_pages: int, fraction: str) -> dict[str, Any]:
        frac_num = eval(fraction)
        pages_done = int(total_pages * frac_num)
        pages_remaining = total_pages - pages_done

        question = f"{name} is required to submit a {total_pages}-page {document}. She already finished writing {fraction} of the {document}. How many pages does she have left to write?"

        answer_cot = f"{name} has already written {fraction} of the {document} which is {total_pages} pages x {fraction} = {pages_done} pages.\nSo, she still needs to write {total_pages} pages - {pages_done} pages = {pages_remaining} pages.\n#### {pages_remaining}"

        return {
            "question": question,
            "answer": format_number(pages_remaining),
            "answer_cot": answer_cot,
            "answer_value": pages_remaining,
            "variables": {
                "name": name,
                "document": document,
                "total_pages": total_pages,
                "fraction": fraction,
                "pages_done": pages_done,
                "pages_remaining": pages_remaining,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia", "Harper", "Evelyn"]
        documents = ["essay", "report", "thesis", "dissertation", "assignment"]
        fractions = ["1/2", "1/3", "1/4", "2/3", "3/4"]

        name = rng.choice(names_female)
        document = rng.choice(documents)
        fraction = rng.choice(fractions)

        # Generate total pages ensuring it's divisible by denominator
        denominator = int(fraction.split("/")[1])
        max_pages = int(325 * difficulty)
        total_pages = denominator * rng.randint(1, max_pages // denominator)

        result = generate_from_variables(name, document, total_pages, fraction)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_70(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, objects: str, n: int, obstacle: str, frac: float, k: int, fake_num: int, fake_object: str
    ) -> dict[str, Any]:

        dropped = int(n * frac)
        remaining = n - dropped
        found = k
        after_finding = remaining + found
        final = after_finding - fake_num

        question = f"{name} has a bag of {objects} with {n} inside. He tripped over {obstacle} while carrying it and dropped {dropped} of them. He scrambled to search for them but only came up with {k}. When he went back home, he inspected the {objects} further. {fake_num} of them he picked up wasn't a {objects}, but actually {fake_object} so he got rid of it. How many {objects} did {name} end up with?"

        answer_cot = (
            f"{name} dropped his {objects} and was left with {n}*{1-frac}={remaining} {objects}.\n"
            f"He searched and found some of his lost {objects}, getting him back to {remaining}+{k}={after_finding} {objects}.\n"
            f"He went home and removed {fake_object}, leaving him with {after_finding}-{fake_num}={final} {objects}.\n"
            f"#### {final}"
        )

        return {
            "question": question,
            "answer": format_number(final),
            "answer_cot": answer_cot,
            "answer_value": final,
            "variables": {
                "name": name,
                "objects": objects,
                "initial_count": n,
                "obstacle": obstacle,
                "fraction_dropped": frac,
                "found_count": k,
                "fake_count": fake_num,
                "fake_object": fake_object,
                "remaining": remaining,
                "after_finding": after_finding,
                "final_count": final,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
        objects = ["marbles", "coins", "buttons", "beads", "pebbles"]
        obstacles = ["rock", "stick", "toy", "root"]
        fake_objects = ["buttons", "coins", "pebbles", "beads"]
        fractions = [0.5, 0.25, 0.75]

        name = rng.choice(names)
        obj = rng.choice(objects)
        obstacle = rng.choice(obstacles)
        fake_object = rng.choice([x for x in fake_objects if x != obj])
        frac = rng.choice(fractions)

        # Start with n that's divisible by denominator of fraction
        # For 0.5: need multiple of 2
        # For 0.25: need multiple of 4
        # For 0.75: need multiple of 4
        if frac == 0.5:
            step = 2
        else:  # 0.25 or 0.75
            step = 4

        # Generate n as valid multiple
        min_n = 10
        max_n = int(101 * difficulty)
        valid_n = list(range(min_n + (step - min_n % step) % step, max_n, step))
        n = rng.choice(valid_n) if valid_n else min_n

        # Calculate n * frac (guaranteed to be integer due to our n selection)
        n_frac = int(n * frac)

        # Generate fake_num first (2 to min(10, n_frac))
        fake_num = int(rng.randint(2, min(10, n_frac)))

        # Generate k ensuring fake_num < k < n_frac
        if fake_num + 1 < n_frac:
            k = int(rng.randint(fake_num + 1, min(n_frac, int(20 * difficulty))))
        else:
            # If no valid range, adjust values to make it work
            k = fake_num + 1
            # Ensure n is large enough
            required_n = int((k + 1) / frac)
            n = required_n + (step - required_n % step) % step  # Round up to valid multiple
            n_frac = int(n * frac)

        result = generate_from_variables(name, obj, n, obstacle, frac, k, fake_num, fake_object)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_71(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        shop: str,
        item: str,
        item1: str,
        item2: str,
        item3: str,
        n1: int,
        n2: int,
        n3: int,
        p1: int,
        p2: int,
        p3: int,
    ) -> dict[str, Any]:
        cost1 = n1 * p1
        cost2 = n2 * p2
        cost3 = n3 * p3
        total_cost = cost1 + cost2 + cost3

        question = f"{name} went to the {shop} and bought various types of {item}. She bought {n1} dozen {item1} which cost ${p1} per dozen, {n2} dozen {item2} which cost ${p2} per dozen, and {n3} dozen {item3} for ${p3} per dozen. How much was the total cost?"

        answer_cot = f"The total charge for the {item1} was {n1} x ${p1} = ${cost1}.\nThe total charge for the {item2} was {n2} x ${p2} = ${cost2}.\nThe total charge for the {item3} was {n3} x ${p3} = ${p3*n3}.\nTherefore the total amount {name} paid for the {item} was ${cost1} + ${cost2} + ${cost3} = ${total_cost}.\n#### {total_cost}"

        return {
            "question": question,
            "answer": format_number(total_cost),
            "answer_cot": answer_cot,
            "answer_value": total_cost,
            "variables": {
                "name": name,
                "shop": shop,
                "item": item,
                "item1": item1,
                "item2": item2,
                "item3": item3,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "cost1": cost1,
                "cost2": cost2,
                "cost3": cost3,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Charlotte", "Mia", "Amelia"]
        shops = ["bakery", "patisserie", "confectionery", "cafe"]
        items = ["pastries", "baked goods", "desserts", "treats"]
        item1_options = ["donuts", "croissants", "eclairs", "danishes"]
        item2_options = ["mini cupcakes", "macarons", "cookies", "tarts"]
        item3_options = ["mini cheesecakes", "brownies", "muffins", "scones"]

        name = rng.choice(names_female)
        shop = rng.choice(shops)
        item = rng.choice(items)
        item1 = rng.choice(item1_options)
        item2 = rng.choice(item2_options)
        item3 = rng.choice(item3_options)

        n1 = int(rng.randint(1, int(10 * difficulty)))
        n2 = int(rng.randint(4, int(10 * difficulty)))
        n3 = int(rng.randint(2, int(10 * difficulty)))

        p1 = int(rng.randint(11, int(21 * difficulty)))
        p2 = int(rng.randint(73, int(90 * difficulty)))
        p3 = int(rng.randint(112, int(120 * difficulty)))

        result = generate_from_variables(name, shop, item, item1, item2, item3, n1, n2, n3, p1, p2, p3)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_72(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        structure: str, n1: int, color1: str, color2: str, color3: str, obj: str, mult: int, total: int
    ) -> dict[str, Any]:
        n2 = n1 * mult
        n3 = total - n1 - n2

        question = f"A {structure} is made out of {n1} {color1} {obj}s, {mult} times as many {color2} {obj}s, and an unknown number of {color3} {obj}s. If there are {total} {obj}s in the {structure} in total, how many {color3} {obj}s are there?"

        answer_cot = f"There are {n1}*{mult} = {n2} {color2} {obj}s in the {structure}.\nThere are {total}-{n1}-{n2} = {n3} {color3} {obj}s in the {structure}.\n#### {n3}"

        return {
            "question": question,
            "answer": format_number(n3),
            "answer_cot": answer_cot,
            "answer_value": n3,
            "variables": {
                "structure": structure,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "color1": color1,
                "color2": color2,
                "color3": color3,
                "obj": obj,
                "mult": mult,
                "total": total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        structures = ["building", "pyramid", "stack", "tower"]
        objects = ["brick", "cube", "tile", "block"]
        colors = ["green", "purple", "orange", "pink", "white", "black"]

        structure = rng.choice(structures)
        obj = rng.choice(objects)
        color1, color2, color3 = rng.sample(colors, 3)

        n1 = int(rng.randint(2, int(10 * difficulty)))
        mult = 2  # "twice" as specified in original
        n2 = n1 * mult

        # Ensure total is greater than n1 + n2
        min_total = n1 + n2 + 1
        total = int(rng.randint(min_total, min_total + int(20 * difficulty)))

        result = generate_from_variables(structure, n1, color1, color2, color3, obj, mult, total)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_73(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        fruit: str,
        food: str,
        d1: str,
        d2: str,
        n1: int,
        n2: int,
        m1: int,
        m2: int,
        cn: int,
        cm: int,
        currency: str,
    ) -> dict[str, Any]:

        gingerbread_sunday = n1 + n2
        total_gingerbread = n1 + gingerbread_sunday
        gingerbread_revenue = total_gingerbread * cn

        apple_pie_saturday = m2 - m1
        total_apple_pie = m2 + apple_pie_saturday
        apple_pie_revenue = total_apple_pie * cm

        total_revenue = gingerbread_revenue + apple_pie_revenue

        question = f"{name} is selling {food} and {fruit} pie for a fundraiser. On {d1}, he sold {n1} boxes of {food} and {m1} fewer boxes of {fruit} pie, than on {d2}. On {d2}, he sold {n2} more boxes of {food} than on {d1} and {m2} boxes of {fruit} pie. If the {food} cost {currency}{cn} and the {fruit} pie cost {currency}{cm}, how much did {name} earn for two days?"

        answer_cot = f"He sold {n1} + {n2} = {gingerbread_sunday} boxes of {food} on {d2}.\nThe total number of boxes of {food}s that {name} sold is {n1} + {gingerbread_sunday} = {total_gingerbread}.\n{name} earned {total_gingerbread} x {currency}{cn} = {currency}{gingerbread_revenue} for selling {food}s.\nHe sold {m2} - {m1} = {apple_pie_saturday} boxes of {fruit} pie on {d1}.\nThe total number of boxes of {fruit} pie that {name} sold is {m2} + {apple_pie_saturday} = {total_apple_pie}.\nHe earned {total_apple_pie} x {currency}{cm} = {currency}{apple_pie_revenue} for selling {fruit} pie.\nSo, {name} earned {currency}{gingerbread_revenue} + {currency}{apple_pie_revenue} = {currency}{total_revenue} for two days.\n#### {total_revenue}"

        return {
            "question": question,
            "answer": format_number(total_revenue),
            "answer_cot": answer_cot,
            "answer_value": total_revenue,
            "variables": {
                "name": name,
                "fruit": fruit,
                "food": food,
                "day1": d1,
                "day2": d2,
                "gingerbread_day1": n1,
                "gingerbread_increase": n2,
                "apple_pie_difference": m1,
                "apple_pie_day2": m2,
                "gingerbread_price": cn,
                "apple_pie_price": cm,
                "currency": currency,
                "total_revenue": total_revenue,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Michael", "David", "James", "William", "Robert"]
        fruits = ["apple", "cherry", "blueberry", "peach"]
        foods = ["cookie", "brownie", "muffin", "cupcake"]
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        currencies = ["$", "£", "€"]

        name = rng.choice(names)
        fruit = rng.choice(fruits)
        food = rng.choice(foods)
        d1, d2 = rng.sample(weekdays, 2)
        currency = rng.choice(currencies)

        n1 = int(rng.randint(21, int(30 * difficulty)))
        n2 = int(rng.randint(11, int(15 * difficulty)))
        m2 = int(rng.randint(21, int(30 * difficulty)))
        m1 = int(rng.randint(11, int(min(20, m2) * difficulty)))
        cn = int(rng.randint(7, int(13 * difficulty)))
        cm = int(rng.randint(20, int(33 * difficulty)))

        result = generate_from_variables(name, fruit, food, d1, d2, n1, n2, m1, m2, cn, cm, currency)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_74(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, big_fish: str, length: int, num_remoras: int, remora_length: int
    ) -> dict[str, Any]:
        total_remora_length_inches = num_remoras * remora_length
        total_remora_length_feet = total_remora_length_inches / 12
        percentage = int((total_remora_length_feet / length) * 100)

        question = f"{name} saw a {length}-foot {big_fish} with {num_remoras} {remora_length}-inch remoras attached to it. What percentage of the {big_fish}'s body length is the combined length of the remoras?"

        answer_cot = f"First, find the combined length of the remoras in inches: {remora_length} inches/remora * {num_remoras} remoras = {total_remora_length_inches} inches\nThen divide that number by 12 to convert it to feet: {total_remora_length_inches} inches / 12 inches/foot = {total_remora_length_feet} foot\nThen divide the combined remora length in feet by the {big_fish}'s length and multiply by 100% to express the answer as a percentage: {total_remora_length_feet} foot / {length} feet * 100% = {percentage}%\n#### {percentage}"

        return {
            "question": question,
            "answer": format_number(percentage),
            "answer_cot": answer_cot,
            "answer_value": percentage,
            "variables": {
                "name": name,
                "big_fish": big_fish,
                "length_feet": length,
                "num_remoras": num_remoras,
                "remora_length_inches": remora_length,
                "total_remora_length_inches": total_remora_length_inches,
                "total_remora_length_feet": total_remora_length_feet,
                "percentage": percentage,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Benny", "Tommy", "Jimmy", "Billy", "Johnny", "Bobby"]
        big_fish = ["dolphin", "whale", "shark"]

        name = rng.choice(names)
        fish = rng.choice(big_fish)

        # Start with remoras since they have the most constraints
        min_remoras = 2
        max_remoras = int(10 * difficulty)
        num_remoras = int(rng.randint(min_remoras, max_remoras))

        # Generate remora_length ensuring product will be divisible by 12
        # We need (num_remoras * remora_length) % 12 == 0
        min_remora_length = 2
        max_remora_length = int(100 * difficulty)

        # Find valid remora lengths that make the product divisible by 12
        valid_lengths = [l for l in range(min_remora_length, max_remora_length + 1) if (num_remoras * l) % 12 == 0]

        if not valid_lengths:
            # Adjust num_remoras to make it work with minimum length
            num_remoras = 12  # This ensures divisibility by 12
            remora_length = min_remora_length
        else:
            remora_length = rng.choice(valid_lengths)

        # Calculate total remora length
        total_remora_length = num_remoras * remora_length

        # Generate host fish length ensuring all conditions are met
        # We need:
        # 1. length * 12 > total_remora_length
        # 2. (length * 12) must be divisible by total_remora_length
        # 3. The ratio must give a percentage that divides 100

        min_length = max(10, (total_remora_length + 11) // 12)  # Round up division
        max_length = int(500 * difficulty)

        # Generate lengths that satisfy conditions
        valid_host_lengths = []
        for l in range(min_length, max_length + 1, 10):  # Step by 10 as per original
            if (l * 12) % total_remora_length == 0:  # Ensure clean division
                ratio = (total_remora_length * 100) // (l * 12)  # Calculate percentage
                if ratio > 0 and 100 % ratio == 0:  # Check if percentage divides 100
                    valid_host_lengths.append(l)

        if not valid_host_lengths:
            # Fallback: adjust values to make it work
            length = total_remora_length  # This ensures ratio is 1:1
        else:
            length = rng.choice(valid_host_lengths)

        result = generate_from_variables(name, fish, length, num_remoras, remora_length)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_75(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name1: str, name2: str, color1: str, color2: str, n1: int, n2: int, frac1: float, mult1: float
    ) -> dict[str, Any]:

        n1_result = int(n1 * frac1)
        n2_result = int(n2 * mult1)
        total = n1_result + n2_result

        question = f"{name1} has {n1} tubes of {color1} paint and {n2} tubes of {color2} paint. {name2} has half as many tubes of {color1} paint as {name1}, and three times as many tubes of {color2} paint as {name1}. How many tubes of paint does {name2} have?"

        answer_cot = (
            f"{name2} has {n1}*{frac1}={n1_result} tubes of {color1} paint\n"
            f"{name2} has {n2}*{mult1}={n2_result} tubes of {color2} paint\n"
            f"{name2} has a total of {n1_result}+{n2_result}={total} tubes of paint\n"
            f"#### {total}"
        )

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "name1": name1,
                "name2": name2,
                "color1": color1,
                "color2": color2,
                "n1": n1,
                "n2": n2,
                "frac1": frac1,
                "mult1": mult1,
                "n1_result": n1_result,
                "n2_result": n2_result,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Ben", "James", "John", "Michael", "William", "David", "Richard", "Joseph"]
        colors = ["blue", "red", "green", "yellow", "purple", "orange"]

        name1, name2 = rng.sample(names, 2)
        color1, color2 = rng.sample(colors, 2)

        # Generate numbers that ensure integer results
        n1 = int(rng.randint(2, int(20 * difficulty)))
        n2 = int(rng.randint(2, int(20 * difficulty)))
        frac1 = 0.5  # half
        mult1 = 3.0  # three times

        result = generate_from_variables(name1, name2, color1, color2, n1, n2, frac1, mult1)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_76(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(n: int, p1: int, p2: int, company: str, frac: float) -> dict[str, Any]:
        interviews = int(n * (p1 / 100))
        offers = int(interviews * (p2 / 100))
        accepts = int(offers * frac)

        question = f"{n} people apply for a job at {company}. Of the people that apply, only {p1}% receive interviews. Of those who receive interviews, {p2}% receive a job offer. Of those who receive a job offer, {frac:.2%} of the people accept the position. How many people accept the position?"

        answer_cot = (
            f"The number of people that receive interviews is {n} * {p1/100} = {interviews} people\n"
            f"The number of people that receive a job offer is {interviews} * {p2/100} = {offers} people\n"
            f"The number of people that accept the position is {offers} * {frac} = {accepts} people\n"
            f"#### {accepts}"
        )

        return {
            "question": question,
            "answer": format_number(accepts),
            "answer_cot": answer_cot,
            "answer_value": accepts,
            "variables": {
                "total_applicants": n,
                "interview_percent": p1,
                "offer_percent": p2,
                "company": company,
                "acceptance_fraction": frac,
                "num_interviews": interviews,
                "num_offers": offers,
                "num_accepts": accepts,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        companies = ["Microsoft", "Apple", "Amazon", "Facebook", "Netflix", "Tesla", "Google"]
        fractions = {"a third": 1 / 3, "half": 1 / 2, "a quarter": 1 / 4, "two thirds": 2 / 3}

        company = rng.choice(companies)
        frac_name = rng.choice(list(fractions.keys()))
        frac = fractions[frac_name]

        # To ensure integer results for all calculations:
        # 1. We need n * (p1/100) to be integer -> n should be multiple of 100
        # 2. We need that result * (p2/100) to be integer -> p1,p2 should be multiples of 5
        # 3. We need final result * frac to be integer -> scale n to work with fraction denominator

        # First determine base multiples based on fraction
        if frac == 1 / 3 or frac == 2 / 3:
            base_multiple = 300  # Works for thirds
        elif frac == 1 / 4:
            base_multiple = 400  # Works for quarters
        else:  # frac == 1/2
            base_multiple = 200  # Works for halves

        # Generate n as a multiple of our base to ensure clean division
        n_multiplier = rng.randint(1, min(int(3 * difficulty), 4))
        n = n_multiplier * base_multiple

        # Generate percentages as multiples of 5 to ensure clean division
        p1 = 5 * rng.randint(2, min(int(10 * difficulty), 10))  # 10 to 50 in steps of 5
        p2 = 5 * rng.randint(2, min(int(10 * difficulty), 10))  # 10 to 50 in steps of 5

        # These numbers guarantee:
        # 1. n * (p1/100) is integer (since n is multiple of 100 and p1 is multiple of 5)
        # 2. n * (p1/100) * (p2/100) is integer (same reason)
        # 3. n * (p1/100) * (p2/100) * frac is integer (due to base_multiple construction)

        result = generate_from_variables(n, p1, p2, company, frac)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_77(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        event: str, m: int, w: int, t: str, frac: float, m_left: int, group1: str, group2: str
    ) -> dict[str, Any]:
        total = m + w
        left_count = int(total * frac)
        stayed = total - left_count
        w_left = stayed - m_left

        question = f"At the beginning of the {event}, there were {m} {group1} and {w} {group2}. After {t}, {frac} of the total number of people left. How many {group2} are left if {m_left} {group1} stayed at the {event}?"

        answer_cot = (
            f"There were a total of {m} {group1} + {w} {group2} = {total} people who attended the {event}.\n"
            f"After {t}, {total} people * {frac} = {left_count} people left the {event}.\n"
            f"This means {total} people - {left_count} people = {stayed} people stayed.\n"
            f"Out of the {stayed} who stayed, {stayed} people - {m_left} {group1} = {w_left} were {group2}.\n"
            f"#### {w_left}"
        )

        return {
            "question": question,
            "answer": format_number(w_left),
            "answer_cot": answer_cot,
            "answer_value": w_left,
            "variables": {
                "event": event,
                "men": m,
                "women": w,
                "time": t,
                "fraction_left": frac,
                "men_stayed": m_left,
                "women_stayed": w_left,
                "group1": group1,
                "group2": group2,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        events = ["party", "meeting", "conference", "gathering", "celebration"]
        groups = ["teachers", "doctors", "engineers", "nurses", "artists", "lawyers"]
        times = ["an hour", "two hours", "half an hour", "45 minutes"]
        fractions = [0.25, 0.5, 0.75, 0.33, 0.67]

        event = rng.choice(events)
        group1, group2 = rng.sample(groups, 2)
        t = rng.choice(times)
        frac = rng.choice(fractions)

        # Calculate minimum total needed based on fraction and m_left requirements
        min_m_left = 15
        min_stayed = min_m_left + 1  # Need at least min_m_left + 1 people staying
        min_total = int(min_stayed / (1 - frac))  # Round up

        # Ensure min_total is large enough for m and w requirements
        min_total = max(min_total, 30)  # m >= 20 and w >= 10

        # Round up min_total to work with fraction
        if min_total * frac != int(min_total * frac):
            min_total = int((min_total + (1 / frac)) // (1 / frac) * (1 / frac))

        # Generate valid total
        max_total = int(155 * difficulty)  # max possible m + w
        valid_totals = [t for t in range(min_total, max_total + 1) if (t * frac).is_integer()]

        if not valid_totals:
            total = min_total
        else:
            total = rng.choice(valid_totals)

        # Calculate stayed amount
        stayed = total - int(total * frac)

        # Generate m_left first
        max_m_left = min(stayed - 1, int(35 * difficulty))
        m_left = rng.randint(15, max(15, max_m_left))

        # Now generate m ensuring it's at least m_left + int(total * frac)
        min_m = max(20, m_left + int(total * frac))
        max_m = min(int(75 * difficulty), total - 10)  # ensure w >= 10

        if min_m <= max_m:
            m = rng.randint(min_m, max_m)
        else:
            # Fallback case
            m = min_m
            total = m + 10  # minimum valid total

        w = total - m

        result = generate_from_variables(event, m, w, t, frac, m_left, group1, group2)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_78(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, age_diff: int, age1: int) -> dict[str, Any]:
        age2 = age1 + age_diff
        avg_age = (age1 + age2) // 2

        question = f"{name1} and {name2} are currently {age_diff} years apart in age. If {name1}, who is younger than {name2}, is {age1} years old, what's the average of their ages?"

        answer_cot = (
            f"If {name1} is {age1} years old, {name2} is {age1}+{age_diff} = {age2} years old.\n"
            f"The sum of their ages is {age2}+{age1} = {age1+age2} years\n"
            f"The average age of the two is {age1+age2}/2 = {avg_age} years\n"
            f"#### {avg_age}"
        )

        return {
            "question": question,
            "answer": format_number(avg_age),
            "answer_cot": answer_cot,
            "answer_value": avg_age,
            "variables": {
                "name1": name1,
                "name2": name2,
                "age_diff": age_diff,
                "age1": age1,
                "age2": age2,
                "avg_age": avg_age,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia"]
        name1, name2 = rng.sample(names, 2)

        # Generate age difference - ensure it's even to guarantee integer average
        age_diff = int(rng.randint(5, int(30 * difficulty)))
        if age_diff % 2 != 0:
            age_diff += 1  # Make it even if it's odd

        age1 = int(rng.randint(15, int(75 * difficulty)))

        result = generate_from_variables(name1, name2, age_diff, age1)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_79(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        vehicle: str,
        start_time: int,
        end_time: int,
        free_hours: int,
        currency: str,
        first_hour_cost: int,
        multiplier: int,
    ) -> dict[str, Any]:
        total_hours = end_time - start_time
        paid_hours = total_hours - free_hours
        other_hours = paid_hours - 1
        hourly_rate = first_hour_cost * multiplier
        other_hours_cost = other_hours * hourly_rate
        total_cost = first_hour_cost + other_hours_cost

        question = f"{name} hires a {vehicle} from {start_time} PM to {end_time} PM. He gets {free_hours} hour free. The first paid hour is {currency}{first_hour_cost} and each hour after that is {multiplier} times the cost. How much did he pay?"

        answer_cot = (
            f"He got it for {end_time}-{start_time}={total_hours} hours\n"
            f"He pays for {total_hours}-{free_hours}={paid_hours} hours\n"
            f"The first hour cost 1*{first_hour_cost}={currency}{first_hour_cost}\n"
            f"The other {paid_hours}-1={other_hours} hours are more expensive\n"
            f"They cost {first_hour_cost}*{multiplier}={currency}{hourly_rate} per hour\n"
            f"So those {other_hours} hours cost {other_hours}*{hourly_rate}={currency}{other_hours_cost}\n"
            f"So he pays {other_hours_cost}+{first_hour_cost}={currency}{total_cost}\n"
            f"#### {total_cost}"
        )

        return {
            "question": question,
            "answer": format_number(total_cost),
            "answer_cot": answer_cot,
            "answer_value": total_cost,
            "variables": {
                "name": name,
                "vehicle": vehicle,
                "start_time": start_time,
                "end_time": end_time,
                "free_hours": free_hours,
                "currency": currency,
                "first_hour_cost": first_hour_cost,
                "multiplier": multiplier,
                "total_hours": total_hours,
                "paid_hours": paid_hours,
                "total_cost": total_cost,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["James", "John", "Robert", "Michael", "William", "David", "Richard"]
        vehicles = ["limousine", "party bus", "boat", "luxury car"]
        currencies = ["$", "€", "£"]

        name = rng.choice(names)
        vehicle = rng.choice(vehicles)
        currency = rng.choice(currencies)

        start_time = int(rng.randint(1, int(8 * difficulty)))
        end_time = int(rng.randint(start_time + 2, int(12 * difficulty)))
        free_hours = int(rng.randint(1, min(3, end_time - start_time - 1)))
        first_hour_cost = int(rng.randint(10, int(51 * difficulty)))
        multiplier = 2

        # Verify conditions
        while not (
            (end_time - start_time > free_hours + 1)
            and is_integer((end_time - start_time - free_hours - 1) * first_hour_cost * multiplier)
        ):
            start_time = int(rng.randint(1, int(8 * difficulty)))
            end_time = int(rng.randint(start_time + 2, int(12 * difficulty)))
            free_hours = int(rng.randint(1, min(3, end_time - start_time - 1)))
            first_hour_cost = int(rng.randint(10, int(51 * difficulty)))

        result = generate_from_variables(
            name, vehicle, start_time, end_time, free_hours, currency, first_hour_cost, multiplier
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_80(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, color1: str, color2: str, n1: int, n2: int, n3: int, n4: int
    ) -> dict[str, Any]:
        blue_spools = n1 + n2
        total_spools = n1 + n2 + n3 + n4
        percent_blue = int(100 * blue_spools / total_spools)

        question = f"{name} has {n1} light {color1} spools of thread, {n2} dark {color1} spools of thread, {n3} light {color2} spools of thread, and {n4} dark {color2} spools of thread. What percent of her spools are {color1}?"

        answer_cot = f"First find the number of {color1} spools: {n1} spools + {n2} spools = {blue_spools} spools\nThen find the total number of spools: {n3} spools + {n4} spools + {blue_spools} spools = {total_spools} spools\nThen divide the number of {color1} spools by the total number of spools and multiply by 100% to express the answer as a percentage: {blue_spools} spools / {total_spools} spools * 100% = {percent_blue}%\n#### {percent_blue}"

        return {
            "question": question,
            "answer": format_number(percent_blue),
            "answer_cot": answer_cot,
            "answer_value": percent_blue,
            "variables": {
                "name": name,
                "color1": color1,
                "color2": color2,
                "light_color1_spools": n1,
                "dark_color1_spools": n2,
                "light_color2_spools": n3,
                "dark_color2_spools": n4,
                "total_color1_spools": blue_spools,
                "total_spools": total_spools,
                "percent_color1": percent_blue,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Candy", "Sarah", "Emma", "Olivia", "Isabella", "Sophia", "Mia", "Charlotte"]
        colors = ["blue", "red", "green", "yellow", "purple", "orange"]

        name = rng.choice(names)
        color1, color2 = rng.sample(colors, 2)

        # Generate numbers ensuring integer percentage result
        n1 = int(rng.randint(15, int(45 * difficulty)))
        n2 = int(rng.randint(45, int(100 * difficulty)))
        n3 = int(rng.randint(20, int(80 * difficulty)))
        n4 = int(rng.randint(50, int(100 * difficulty)))

        # Ensure percentage is integer
        total = n1 + n2 + n3 + n4
        while ((n1 + n2) * 100) % total != 0:
            n4 += 1
            total = n1 + n2 + n3 + n4

        result = generate_from_variables(name, color1, color2, n1, n2, n3, n4)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_81(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        occupation: str, weeks_per_month: int, days_per_week: int, pay_per_day: int, currency: str
    ) -> dict[str, Any]:
        days_per_month = days_per_week * weeks_per_month
        monthly_pay = days_per_month * pay_per_day
        yearly_pay = monthly_pay * 12

        question = f"A {occupation} works for {weeks_per_month} weeks every month and for {days_per_week} days every week. If he gets paid {currency}{pay_per_day} every day, how much does he earn if he works for a year?"

        answer_cot = f"The {occupation} works for {days_per_week} days every week and works for {weeks_per_month} weeks every month so he works for {days_per_week} days/week * {weeks_per_month} weeks/month = {days_per_month} days/month\nIf he earns {currency}{pay_per_day} every day he then earns {currency}{pay_per_day}/day * {days_per_month} days/month = {currency}{monthly_pay}/month\nA year is equal to 12 months so every year he earns {currency}{monthly_pay}/month * 12 months/year = {currency}{yearly_pay}\n#### {yearly_pay}"

        return {
            "question": question,
            "answer": format_number(yearly_pay),
            "answer_cot": answer_cot,
            "answer_value": yearly_pay,
            "variables": {
                "occupation": occupation,
                "weeks_per_month": weeks_per_month,
                "days_per_week": days_per_week,
                "pay_per_day": pay_per_day,
                "currency": currency,
                "days_per_month": days_per_month,
                "monthly_pay": monthly_pay,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        occupations = ["plumber", "electrician", "painter", "carpenter", "landscaper"]
        currencies = ["$", "£", "€"]

        occupation = rng.choice(occupations)
        currency = rng.choice(currencies)

        weeks_per_month = int(rng.randint(2, int(5 * difficulty)))
        days_per_week = int(rng.randint(4, int(7 * difficulty)))
        pay_per_day = int(rng.randrange(40, int(200 * difficulty), 5))

        result = generate_from_variables(occupation, weeks_per_month, days_per_week, pay_per_day, currency)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_82(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name: str, num_emails: int, no_response_percent: int, workdays: int) -> dict[str, Any]:
        no_response = num_emails * no_response_percent // 100
        responds_to = num_emails - no_response
        total_responses = responds_to * workdays

        question = f"{name} gets {num_emails} emails a day. {no_response_percent}% of those emails don't require any response. {name} responds to the rest of them. How many emails does {name} respond to in a {workdays} day work week?"

        answer_cot = (
            f"{name} receives {no_response}={no_response} emails that don't require a response\n"
            f"So {name} responds to {num_emails}-{no_response}={responds_to} emails per day\n"
            f"In a {workdays} day work week, {name} responds to {responds_to}*{workdays}={total_responses} emails\n"
            f"#### {total_responses}"
        )

        return {
            "question": question,
            "answer": format_number(total_responses),
            "answer_cot": answer_cot,
            "answer_value": total_responses,
            "variables": {
                "name": name,
                "num_emails": num_emails,
                "no_response_percent": no_response_percent,
                "workdays": workdays,
                "no_response": no_response,
                "responds_to": responds_to,
                "total_responses": total_responses,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
        name = rng.choice(names)

        # Generate random values scaled by difficulty
        num_emails = int(rng.randint(50, int(200 * difficulty)))
        no_response_percent = int(rng.randint(5, int(40 * difficulty)))
        workdays = int(rng.randint(3, int(7 * difficulty)))

        # Ensure conditions are met
        while not (num_emails * no_response_percent % 100 == 0 and num_emails * (100 - no_response_percent) % 100 == 0):
            num_emails = int(rng.randint(50, int(200 * difficulty)))
            no_response_percent = int(rng.randint(5, int(40 * difficulty)))

        result = generate_from_variables(name, num_emails, no_response_percent, workdays)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_83(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, total: int, diff: int, unit: str) -> dict[str, Any]:
        amount1 = (total - diff) // 2  # Sam's amount
        amount2 = amount1 + diff  # Harry's amount

        question = f"If {name1} and {name2} have {total} {unit} of fence between them, and they agree to split it with {name2} getting {diff} {unit} more than {name1}, how much is left over for {name1}?"

        answer_cot = f"Let x be the amount of fence {name1} gets and y be the amount {name2} gets. We know that y = x + {diff}, and y + x = {total}.\nSubstituting the first equation into the second equation, we get 2x+{diff}={total}\nSubtracting the {diff} from both sides, we get 2x={total-diff}\nWe divide each side by two, leaving x={amount1}. This means {name1} has {amount1} {unit} of fence left over.\n#### {amount1}"

        return {
            "question": question,
            "answer": format_number(amount1),
            "answer_cot": answer_cot,
            "answer_value": amount1,
            "variables": {
                "name1": name1,
                "name2": name2,
                "total_fence": total,
                "difference": diff,
                "unit": unit,
                "amount1": amount1,
                "amount2": amount2,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Sam", "Harry", "Tom", "John", "Mike", "Dave", "Steve", "Bob"]
        units = ["feet", "yards", "meters"]

        name1, name2 = rng.sample(names, 2)
        unit = rng.choice(units)

        # Start with the difference - must be divisible by 10 and between 20 and 180
        max_diff = min(int(200 * difficulty), 180)
        diff = 10 * rng.randint(2, max_diff // 10)  # This ensures diff is multiple of 10 between 20 and max_diff

        # For total: we need total > diff + 20 and (total - diff) must be even
        # Let's work backwards from our constraints
        min_total = diff + 22  # Adding 22 ensures total-diff > 10 and gives room for even adjustment
        max_total = min(int(1000 * difficulty), min_total + 200)  # Cap the maximum to avoid too large numbers

        # Ensure max_total is larger than min_total and generate valid number
        if max_total <= min_total:
            total = min_total
        else:
            # Generate total as min_total plus an even number
            total = min_total + (2 * rng.randint(0, (max_total - min_total) // 2))

        result = generate_from_variables(name1, name2, total, diff, unit)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_84(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, miles: str, time_cold: int, extra_time: int, multiplier: float, distance: int
    ) -> dict[str, Any]:
        time_warm = extra_time + multiplier * time_cold
        time_cold_total = distance * time_cold
        time_warm_total = distance * time_warm
        time_difference = time_warm_total - time_cold_total

        question = f"When the water is cold {name} swims a {miles} in {time_cold} minutes. When the water is warm {name} swims a {miles} in {extra_time} minutes more than {multiplier:.0f} times as long. How much longer does {name} take to swim {distance} {miles}s on a hot day than a cold day?"

        answer_cot = (
            f"Cold water {miles} = {time_cold} minutes\n"
            f"Warm water {miles} = {extra_time}+{multiplier:.0f}({time_cold})={time_warm} minutes\n"
            f"{distance} {miles}s in cold water: {distance}({time_cold})={time_cold_total} minutes\n"
            f"{distance} {miles}s in warm water: {distance}({time_warm})={time_warm_total} minutes\n"
            f"{name} takes {time_warm_total}-{time_cold_total}={time_difference} minutes longer\n"
            f"#### {time_difference}"
        )

        return {
            "question": question,
            "answer": format_number(time_difference),
            "answer_cot": answer_cot,
            "answer_value": time_difference,
            "variables": {
                "name": name,
                "unit": miles,
                "time_cold": time_cold,
                "extra_time": extra_time,
                "multiplier": multiplier,
                "distance": distance,
                "time_warm": time_warm,
                "time_cold_total": time_cold_total,
                "time_warm_total": time_warm_total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Ray", "Jim", "Bob", "Tom", "Mike", "John", "Steve"]
        units = ["mile", "kilometer"]

        name = rng.choice(names)
        unit = rng.choice(units)

        # First generate time_cold ensuring it's < 60 and reasonable with multiplier
        # Since multiplier is 2.0, time_cold needs to be < 30 to avoid total >= 60
        max_time_cold = min(25, int(50 * difficulty), 29)  # Cap at 29 to ensure multiplier works
        time_cold = rng.randint(10, max_time_cold)

        # Calculate maximum extra_time that keeps total under 60
        # extra_time + 2 * time_cold < 60
        # extra_time < 60 - 2 * time_cold
        max_extra = min(int(10 * difficulty), 60 - 2 * time_cold - 1)
        if max_extra < 1:
            # If we can't find valid extra_time, adjust time_cold down
            time_cold = 20  # Reset to safe value
            max_extra = 19  # 60 - 2*20 - 1
        extra_time = rng.randint(1, max_extra)

        # Fixed multiplier
        multiplier = 2.0

        # Calculate minimum distance needed to make the difference positive
        # distance * (extra_time + 2 * time_cold) - distance * time_cold > 0
        # distance * (extra_time + time_cold) > 0
        # Since extra_time is positive, this is always true for positive distance
        min_distance = 2
        max_distance = min(int(10 * difficulty), 8)  # Cap at 8 for reasonable numbers
        distance = rng.randint(min_distance, max_distance)

        result = generate_from_variables(name, unit, time_cold, extra_time, multiplier, distance)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_85(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, room_type: str, area: int, length: int, unit1: str, unit2: str
    ) -> dict[str, Any]:
        conversion = 3 if unit1 == "feet" and unit2 == "yards" else 1
        length_converted = length * conversion
        width = area // length_converted
        perimeter = 2 * (width + length_converted)

        question = f"The area of {name}'s rectangular {room_type} is {area} square {unit1}. If the length of his room is {length} {unit2}, what is the perimeter of the room in {unit1}?"

        answer_cot = (
            f"The length of the room is {length} {unit2} * ({conversion} {unit1} / 1 {unit2}) = {length_converted} {unit1}.\n"
            f"The width of the room is {area} square {unit1} / {length_converted} {unit1} = {width} {unit1}.\n"
            f"The room's perimeter is 2({width}+{length_converted}) = {perimeter}\n#### {perimeter}"
        )

        return {
            "question": question,
            "answer": format_number(perimeter),
            "answer_cot": answer_cot,
            "answer_value": perimeter,
            "variables": {
                "name": name,
                "room_type": room_type,
                "area": area,
                "length": length,
                "unit1": unit1,
                "unit2": unit2,
                "width": width,
                "length_converted": length_converted,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        def calculate_max_width(difficulty: float, length: int, conversion: int) -> int:
            # Cap the maximum width to avoid numbers getting too large
            theoretical_max = int(100 * difficulty)
            # Ensure we have room for at least length * conversion + 1
            min_required = length * conversion + 1
            return max(min_required + 1, min(theoretical_max, min_required + 50))

        names = ["William", "James", "John", "Michael", "David", "Robert", "Thomas"]
        room_types = ["living room", "study", "office", "kitchen"]
        units = ["feet", "meters"]

        name = rng.choice(names)
        room_type = rng.choice(room_types)
        unit1 = rng.choice(units)
        unit2 = "yards" if unit1 == "feet" else "meters"

        # First determine length, capped to avoid too large numbers
        max_length = min(int(44 * difficulty), 40)  # Cap at 40 to ensure room for width
        length = rng.randint(5, max_length)

        # Calculate conversion factor
        conversion = 3 if unit1 == "feet" and unit2 == "yards" else 1

        # Calculate width bounds
        min_width = length * conversion + 1
        max_width = calculate_max_width(difficulty, length, conversion)

        # If ranges are too tight, adjust length down
        if max_width <= min_width:
            length = max(5, length - 5)
            min_width = length * conversion + 1
            max_width = calculate_max_width(difficulty, length, conversion)

        # Generate width ensuring we have valid range
        width = rng.randint(min_width, max_width)
        area = width * (length * conversion)

        result = generate_from_variables(name, room_type, area, length, unit1, unit2)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_86(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        animals: str,
        unit: str,
        o1: str,
        o2: str,
        o3: str,
        o4: str,
        n1: int,
        n2: int,
        n3: int,
        n4: int,
        w1: int,
        w2: int,
        w3: int,
        w4: int,
        total: int,
    ) -> dict[str, Any]:
        # Calculate weights
        sugar_weight = n4 * w4
        carrot_weight = n3 * w3
        hay_weight = n1 * w1
        oat_weight = n2 * w2
        total_weight = sugar_weight + carrot_weight + hay_weight + oat_weight
        trips = total_weight // total

        question = f"A farmer is buying feed for his {animals}. He buys a variety of {o1}, {o2}, {o3} and {o4}. Since {o4} are a rare treat, he only buys {n4} {w4}-{unit} boxes of them for the whole stable. He only wants enough {o3} to feed the {animals} while the vegetables are fresh, so he buys {n3} {w3}-{unit} bags. {o1} is the main diet of his {animals}, so he buys {n1} {w1}-{unit} bales. {o2} are a staple to supplement the {o1}, so he buys {n2} {w2}-{unit} sacks. If his farm truck can carry {total} {unit}s at a time, how many trips does the farmer need to transport all the feed?"

        answer_cot = f"The farmer is buying {n4} * {w4} = {sugar_weight} {unit}s of {o4}.\nHe is buying {n3} * {w3} = {carrot_weight} {unit}s of {o3}.\nHe is buying {n1} * {w1} = {hay_weight} {unit}s of {o1}.\nHe is buying {n2} * {w2} = {oat_weight} {unit}s of {o2}.\nThe weight of all the feed is {sugar_weight} + {carrot_weight} + {hay_weight} + {oat_weight} = {total_weight} {unit}s.\nThus, the farmer needs {total_weight} / {total} = {trips} trips to transport all the feed in his farm truck.\n#### {trips}"

        return {
            "question": question,
            "answer": format_number(trips),
            "answer_cot": answer_cot,
            "answer_value": trips,
            "variables": {
                "animals": animals,
                "unit": unit,
                "feed_types": [o1, o2, o3, o4],
                "quantities": [n1, n2, n3, n4],
                "weights": [w1, w2, w3, w4],
                "truck_capacity": total,
                "total_weight": total_weight,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        animals = rng.choice(["horses", "cows", "sheep", "pigs", "alpacas"])
        unit = rng.choice(["pound", "kilogram"])
        feed_options = ["hay", "corn", "oats", "apples", "wheat"]
        o1, o2, o4 = rng.sample(feed_options, 3)
        o3 = rng.choice(["carrots", "beets", "cucumbers"])

        # Scale ranges by difficulty
        n4 = int(rng.randint(4, int(8 * difficulty)))
        n3 = int(rng.randint(11, int(15 * difficulty)))
        n2 = int(rng.randint(15, int(20 * difficulty)))
        n1 = int(rng.randint(31, int(35 * difficulty)))

        w4 = int(rng.randint(3, int(8 * difficulty)))
        w3 = int(rng.randint(5, int(10 * difficulty)))
        w2 = int(rng.randint(15, int(20 * difficulty)))
        w1 = int(rng.randint(35, int(45 * difficulty)))

        # Ensure weight conditions are met
        while not (w4 * n4 < w3 * n3 < w2 * n2 < w1 * n1):
            w4 = int(rng.randint(3, int(8 * difficulty)))
            w3 = int(rng.randint(5, int(10 * difficulty)))
            w2 = int(rng.randint(15, int(20 * difficulty)))
            w1 = int(rng.randint(35, int(45 * difficulty)))

        total_weight = n1 * w1 + n2 * w2 + n3 * w3 + n4 * w4
        # Find truck capacity that divides total weight
        total = total_weight // rng.randint(2, 4)

        result = generate_from_variables(animals, unit, o1, o2, o3, o4, n1, n2, n3, n4, w1, w2, w3, w4, total)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_87(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        unit: str,
        weight_large: int,
        weight_medium: int,
        weight_small: Fraction,
        num_large: int,
        num_medium: int,
        num_small: int,
        total_amount: int,
    ) -> dict[str, Any]:

        large_used = num_large * weight_large
        medium_used = num_medium * weight_medium
        small_used = float(num_small * weight_small)
        total_used = large_used + medium_used + small_used
        remaining = total_amount - total_used

        question = f"{name} wants to make different sized ice cubes with {total_amount} {unit}s of water. He can make giant cubes that use {weight_large} {unit}s per cube, medium cubes that use {weight_medium} {unit}s, and small cubes that use {weight_small} an {unit}. If he makes {num_large} giant cubes, {num_medium} medium cubes, and {num_small} small cubes, how many {unit}s of water does he have left?"

        answer_cot = (
            f"The giant cubes used up {large_used} {unit}s of water because {num_large} times {weight_large} equals {large_used}.\n"
            f"The medium cubes used up {medium_used} {unit}s of water because {num_medium} times {weight_medium} equals {medium_used}.\n"
            f"The small cubes used up {int(small_used)} {unit}s of water because {num_small} times {weight_small} equals {int(small_used)}.\n"
            f"This means that {name} has used up {int(total_used)} {unit}s of water because {large_used} plus {medium_used} plus {int(small_used)} equals {int(total_used)}.\n"
            f"{name} has {int(remaining)} {unit}s of water left because {total_amount} minus {int(total_used)} equals {int(remaining)}.\n"
            f"#### {int(remaining)}"
        )

        return {
            "question": question,
            "answer": format_number(int(remaining)),
            "answer_cot": answer_cot,
            "answer_value": int(remaining),
            "variables": {
                "name": name,
                "unit": unit,
                "weight_large": weight_large,
                "weight_medium": weight_medium,
                "weight_small": weight_small,
                "num_large": num_large,
                "num_medium": num_medium,
                "num_small": num_small,
                "total_amount": total_amount,
                "remaining": remaining,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Peter", "John", "Michael", "David", "James", "Robert", "William"]
        units = ["ounce", "gram", "milliliter"]

        name = rng.choice(names)
        unit = rng.choice(units)

        weight_large = int(rng.randint(7, int(14 * difficulty)))
        weight_medium = int(rng.randint(3, weight_large - 1))
        weight_small = Fraction(1, 2)

        num_large = int(rng.randint(2, int(8 * difficulty)))
        num_medium = int(rng.randint(4, int(12 * difficulty)))
        num_small = rng.choice([14, 24, 15])

        # Calculate needed total to ensure positive remainder
        used = num_large * weight_large + num_medium * weight_medium + float(num_small * weight_small)
        total_amount = int(used + rng.randint(1, int(10 * difficulty)))

        result = generate_from_variables(
            name, unit, weight_large, weight_medium, weight_small, num_large, num_medium, num_small, total_amount
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_88(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(school: str, venue: str, total: int, graduates: int, faculty: int) -> dict[str, Any]:
        remaining_seats = total - (graduates + faculty)
        tickets_per_graduate = remaining_seats // graduates

        question = f"{school} is holding graduation in their {venue} this year which has space for {total} people. After accounting for {graduates} seats for graduates and {faculty} seats for faculty attending, how many tickets would each graduate receive to give to their friends and family if the tickets are split equally?"

        answer_cot = f"Add graduate and faculty seats together. {graduates} + {faculty} = {graduates+faculty} seats for faculty and graduates\nMinus seats for faculty and graduates from total seats allowed. {total} - {graduates+faculty} = {remaining_seats} remaining seats.\nDivide remaining seats by the number of graduates. {remaining_seats}/{graduates} = {tickets_per_graduate} tickets\n#### {tickets_per_graduate}"

        return {
            "question": question,
            "answer": format_number(tickets_per_graduate),
            "answer_cot": answer_cot,
            "answer_value": tickets_per_graduate,
            "variables": {
                "school": school,
                "venue": venue,
                "total_seats": total,
                "graduate_seats": graduates,
                "faculty_seats": faculty,
                "remaining_seats": remaining_seats,
                "tickets_per_graduate": tickets_per_graduate,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        schools = ["Oakwood High School", "Riverside Academy", "Sunnyside High", "Greenville High School"]
        venues = ["Auditorium", "Gymnasium", "Sports Arena", "Convention Center"]

        school = rng.choice(schools)
        venue = rng.choice(venues)

        graduates = int(rng.randrange(500, int(1500 * difficulty), 50))
        faculty = int(rng.randrange(100, int(500 * difficulty), 50))

        # Ensure total seats allow for integer division of remaining seats
        remaining_seats = rng.randint(2, int(10 * difficulty)) * graduates
        total = remaining_seats + graduates + faculty

        result = generate_from_variables(school, venue, total, graduates, faculty)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_89(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name1: str,
        name2: str,
        name3: str,
        name4: str,
        name5: str,
        num_dozen: int,
        found_first: int,
        multiplier: float,
        less_amount: int,
        fraction: float,
    ) -> dict[str, Any]:
        total_eggs = num_dozen * 12
        found_second = found_first * multiplier
        found_third = found_second - less_amount
        found_fourth = found_third * fraction
        total_found = found_first + found_second + found_third + found_fourth
        remaining = total_eggs - total_found

        question = f"{name1} hid {num_dozen} dozen eggs in the yard for the Easter egg hunt. {name2} finds {found_first} eggs. {name3} finds {multiplier:.0f} times as many as {name2}. {name4} finds {less_amount} less than {name3}, and {name5} finds {fraction:.1f} as many as {name4}. How many eggs are still hidden in the yard?"

        answer_cot = f"{name1} hides {num_dozen} x 12 = {total_eggs} eggs.\n"
        answer_cot += f"{name2} finds {found_first} eggs.\n"
        answer_cot += f"{name3} finds {found_first} x {multiplier:.0f} = {found_second} eggs.\n"
        answer_cot += f"{name4} finds {found_second} - {less_amount} = {found_third} eggs.\n"
        answer_cot += f"{name5} finds {found_third} x {fraction:.1f} = {found_fourth} eggs.\n"
        answer_cot += f"The children find a total of {found_first} + {found_second} + {found_third} + {found_fourth} = {total_found} eggs.\n"
        answer_cot += f"The total number of hidden eggs still in the yard is {total_eggs} - {total_found} = {remaining} eggs.\n#### {remaining}"

        return {
            "question": question,
            "answer": format_number(remaining),
            "answer_cot": answer_cot,
            "answer_value": remaining,
            "variables": {
                "name1": name1,
                "name2": name2,
                "name3": name3,
                "name4": name4,
                "name5": name5,
                "num_dozen": num_dozen,
                "found_first": found_first,
                "multiplier": multiplier,
                "less_amount": less_amount,
                "fraction": fraction,
                "total_eggs": total_eggs,
                "total_found": total_found,
                "remaining": remaining,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = [
            "Emma",
            "Liam",
            "Olivia",
            "Noah",
            "Ava",
            "Oliver",
            "Isabella",
            "William",
            "Sophia",
            "James",
            "Charlotte",
            "Benjamin",
            "Mia",
            "Lucas",
            "Harper",
        ]

        name1, name2, name3, name4, name5 = rng.sample(names, 5)

        num_dozen = int(rng.randint(2, int(10 * difficulty)))
        found_first = int(rng.randint(3, int(15 * difficulty)))
        multiplier = 2.0  # Using 'twice' as specified
        less_amount = int(rng.randint(1, int(5 * difficulty)))
        fraction = 0.5  # Using 'half' as specified

        # Ensure all conditions are met
        total = num_dozen * 12
        found_second = found_first * multiplier
        found_third = found_second - less_amount
        found_fourth = found_third * fraction
        total_found = found_first + found_second + found_third + found_fourth

        # Regenerate if conditions not met
        while not found_third > 0 or not total > total_found or not float(found_fourth).is_integer():
            num_dozen = int(rng.randint(2, int(10 * difficulty)))
            found_first = int(rng.randint(3, int(15 * difficulty)))
            less_amount = int(rng.randint(1, int(5 * difficulty)))
            total = num_dozen * 12
            found_second = found_first * multiplier
            found_third = found_second - less_amount
            found_fourth = found_third * fraction
            total_found = found_first + found_second + found_third + found_fourth

        result = generate_from_variables(
            name1, name2, name3, name4, name5, num_dozen, found_first, multiplier, less_amount, fraction
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_90(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        device: str, currency: str, rate1: float, rate2: float, threshold: int, total_mins: int
    ) -> dict[str, Any]:
        first_period = threshold
        second_period = total_mins - threshold

        cost1 = first_period * rate1
        cost2 = second_period * rate2
        total_cost = int(cost1 + cost2)

        question = f"To make a call from a {device}, you must pay {currency}{rate1} for each minute of your call. After {threshold} minutes, that price drops to {currency}{rate2} per minute. How much would a {total_mins}-minute call cost?"

        answer_cot = f"First {threshold} minutes would be a cost of {threshold} * {rate1} = {currency}{int(cost1)}.\nAfter that, there are {total_mins} - {threshold} = {second_period} minutes of the call left.\nAnd these {second_period} minutes cost {second_period} * {rate2} = {currency}{int(cost2)}.\nSo in total, the {total_mins}-minute call would cost {int(cost1)} + {int(cost2)} = {currency}{total_cost}.\n#### {total_cost}"

        return {
            "question": question,
            "answer": format_number(total_cost),
            "answer_cot": answer_cot,
            "answer_value": total_cost,
            "variables": {
                "device": device,
                "currency": currency,
                "rate1": rate1,
                "rate2": rate2,
                "threshold": threshold,
                "total_mins": total_mins,
                "first_period_cost": int(cost1),
                "second_period_cost": int(cost2),
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        devices = ["payphone", "phone booth", "hotel room phone"]
        currencies = ["$", "£", "€"]

        device = rng.choice(devices)
        currency = rng.choice(currencies)

        # First generate threshold and total minutes
        threshold = int(rng.randint(10, min(int(50 * difficulty), 40)))  # Cap threshold
        total_mins = threshold + int(rng.randint(10, min(int(50 * difficulty), 30)))  # Generate relative to threshold

        # Generate rates working backwards from whole numbers
        # We'll generate integers first, then divide to get our rates
        rate1_base = rng.randint(20, min(int(50 * difficulty), 40))  # Will become 0.20 to 0.50
        rate2_base = rng.randint(10, rate1_base - 5)  # Will become 0.10 to rate1-0.05

        # Convert to actual rates ensuring they'll produce integer results
        rate1 = rate1_base / 100  # Convert to dollars
        rate2 = rate2_base / 100

        # These rates are guaranteed to:
        # 1. Be in the correct range (rate2 < rate1)
        # 2. Produce integer results when multiplied by threshold
        # 3. Be rounded to 2 decimal places
        rate1 = round(rate1, 2)
        rate2 = round(rate2, 2)

        result = generate_from_variables(device, currency, rate1, rate2, threshold, total_mins)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_91(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, fruit: str, area: str, field_size: int, density: int, months: int
    ) -> dict[str, Any]:
        fruits_per_harvest = field_size * density
        harvests_per_year = 12 // months
        total_fruits = fruits_per_harvest * harvests_per_year

        question = f"{name} has {field_size} {area}s of a {fruit} field. There are {density} {fruit}s per {area}. {name} can harvest his {fruit}s every {months} months. How many {fruit}s can {name} harvest within a year?"

        answer_cot = f"{name} has {density} x {field_size}= {fruits_per_harvest} {fruit}s on his field.\n{name} can harvest his {fruit}s 12 ÷ {months} = {harvests_per_year} times per year\nTherefore {name} can harvest {fruits_per_harvest} x {harvests_per_year} = {total_fruits} {fruit}s per year.\n#### {total_fruits}"

        return {
            "question": question,
            "answer": format_number(total_fruits),
            "answer_cot": answer_cot,
            "answer_value": total_fruits,
            "variables": {
                "name": name,
                "fruit": fruit,
                "area": area,
                "field_size": field_size,
                "density": density,
                "months_per_harvest": months,
                "fruits_per_harvest": fruits_per_harvest,
                "harvests_per_year": harvests_per_year,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Michael", "David", "James", "Robert", "William", "Richard"]
        fruits = ["pineapple", "mango", "banana", "papaya", "coconut"]
        areas = ["hectare", "square yard", "square meter"]

        name = rng.choice(names)
        fruit = rng.choice(fruits)
        area = rng.choice(areas)

        field_size = int(rng.randrange(5, int(100 * difficulty), 5))
        density = int(rng.randint(2, int(101 * difficulty)))
        months = rng.choice([1, 2, 3, 4, 6, 12])

        result = generate_from_variables(name, fruit, area, field_size, density, months)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_92(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        product: str,
        location: str,
        item1: str,
        item2: str,
        item3: str,
        price1: float,
        price2: float,
        price3: float,
        num1: int,
        num2: int,
        num3: int,
        unit: str,
        currency: str,
    ) -> dict[str, Any]:

        round_p1 = round(price1)
        round_p2 = round(price2)
        round_p3 = round(price3)

        total = num1 * round_p1 + num2 * round_p2 + num3 * round_p3

        question = f"{name} has a {product} stand at the {location}. He sells three kinds of {product}s: {item1}, {item2} and {item3}. He usually sells {item1} for {currency}{price1:.2f} per {unit}, {item2} for {currency}{price2:.2f} per {unit} and {item3} for {currency}{price3:.2f} per {unit}. {name} has no change today, so he has decided to round all his prices to the nearest dollar. If {name} sells {num1} {unit}s of {item1}, {num2} {unit}s of {item2} and {num3} {unit}s of {item3}, how much will he make?"

        answer_cot = f"{name} will round his {item1} {'up' if round_p1 > price1 else 'down'} from {currency}{price1:.2f} to {currency}{round_p1}, since the number following the {int(price1)} is {'5 or higher' if round_p1 > price1 else 'less than 5'}.\n"
        answer_cot += f"{name} will round his {item2} {'up' if round_p2 > price2 else 'down'} from {currency}{price2:.2f} to {currency}{round_p2}, since the number following the {int(price2)} is {'5 or higher' if round_p2 > price2 else 'less than 5'}.\n"
        answer_cot += f"{name} will round his {item3} {'up' if round_p3 > price3 else 'down'} from {currency}{price3:.2f} to {currency}{round_p3}, since the number following the {int(price3)} is {'5 or higher' if round_p3 > price3 else 'less than 5'}.\n"
        answer_cot += f"{name} sells {num1} {item1} x {currency}{round_p1} = {currency}{num1*round_p1}\n"
        answer_cot += f"{name} sells {num2} {item2} x {currency}{round_p2} = {currency}{num2*round_p2}\n"
        answer_cot += f"{name} sells {num3} {item3} x {currency}{round_p3} = {currency}{num3*round_p3}\n"
        answer_cot += f"Altogether, {name} will make {currency}{num1*round_p1} + {currency}{num2*round_p2} + {currency}{num3*round_p3} = {currency}{total}\n#### {total}"

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "name": name,
                "product": product,
                "location": location,
                "items": [item1, item2, item3],
                "original_prices": [price1, price2, price3],
                "rounded_prices": [round_p1, round_p2, round_p3],
                "quantities": [num1, num2, num3],
                "unit": unit,
                "currency": currency,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Mike", "David", "James", "Robert", "William", "Richard"]
        products = ["vegetable", "flower", "herb", "plant"]
        locations = ["local fair", "community market", "street bazaar", "town square"]
        items = ["roses", "daisies", "tulips", "lilies", "sunflowers", "orchids"]
        units = ["bunch", "basket", "bouquet", "bundle"]
        currencies = ["$", "£", "€"]

        name = rng.choice(names)
        product = rng.choice(products)
        location = rng.choice(locations)
        item1, item2, item3 = rng.sample(items, 3)
        unit = rng.choice(units)
        currency = rng.choice(currencies)

        # Scale prices by difficulty
        price1 = round(rng.uniform(1.26, 3.53 * difficulty), 2)
        price2 = round(rng.uniform(2.27, 5.53 * difficulty), 2)
        price3 = round(rng.uniform(4.85, 6.53 * difficulty), 2)

        num1 = int(rng.randint(5, int(21 * difficulty)))
        num2 = int(rng.randint(15, int(31 * difficulty)))
        num3 = int(rng.randint(35, int(41 * difficulty)))

        result = generate_from_variables(
            name, product, location, item1, item2, item3, price1, price2, price3, num1, num2, num3, unit, currency
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_93(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name1: str,
        name2: str,
        name3: str,
        name4: str,
        creature: str,
        weapon1: str,
        weapon2: str,
        weapon3: str,
        weapon4: str,
        weapon5: str,
        n1: int,
        frac1: float,
        mult1: int,
        frac2: float,
    ) -> dict[str, Any]:

        kills_arthur = int(n1 * frac1)
        kills_walter = int(kills_arthur * mult1)
        kills_bruce = int(kills_walter * frac2)

        question = f"{name1} slew {n1} {creature} with his mighty {weapon1}, while {name2}, using a {weapon2}, slew {frac1} as many {creature} as {name1}. Using a {weapon3}, {name3} slew {mult1} as many {creature} as {name2}. But {name4}, having forgot his {weapon4} at home, slew {frac2} as many {creature} as {name3} using a {weapon5}. How many {creature} has {name4} slain?"

        answer_cot = f"{name2} slew {frac1} as many {creature} as {name1}, or {n1}*{frac1}={kills_arthur} {creature}.\n{name3} slew {mult1} as many {creature} as {name2}, or {mult1}*{kills_arthur}={kills_walter} {creature}.\n{name4} slew {frac2} as many {creature} as {name3}, or {kills_walter}*{frac2}={kills_bruce} {creature}.\n#### {kills_bruce}"

        return {
            "question": question,
            "answer": format_number(kills_bruce),
            "answer_cot": answer_cot,
            "answer_value": kills_bruce,
            "variables": {
                "name1": name1,
                "name2": name2,
                "name3": name3,
                "name4": name4,
                "creature": creature,
                "weapon1": weapon1,
                "weapon2": weapon2,
                "weapon3": weapon3,
                "weapon4": weapon4,
                "weapon5": weapon5,
                "initial_kills": n1,
                "fraction1": frac1,
                "multiplier": mult1,
                "fraction2": frac2,
                "kills_arthur": kills_arthur,
                "kills_walter": kills_walter,
                "kills_bruce": kills_bruce,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_male = ["Arthur", "Bruce", "Charles", "David", "Edward", "Frederick", "George", "Henry"]
        creatures = ["ogres", "trolls", "goblins", "orcs", "giants"]
        weapons1 = ["sword", "mace", "battle axe", "war hammer"]
        weapons2 = ["spear", "lance", "javelin", "halberd"]
        weapons3 = ["rusty iron axe", "wooden club", "stone hammer", "bone dagger"]
        weapons4 = ["sword", "axe", "mace", "hammer"]
        weapons5 = ["nail file", "butter knife", "wooden spoon", "feather"]
        fractions = [0.25, 0.5, 0.75]
        multipliers = [2, 3, 4]

        name1, name2, name3, name4 = rng.sample(names_male, 4)
        creature = rng.choice(creatures)
        weapon1 = rng.choice(weapons1)
        weapon2 = rng.choice(weapons2)
        weapon3 = rng.choice(weapons3)
        weapon4 = rng.choice(weapons4)
        weapon5 = rng.choice(weapons5)

        # Scale numbers by difficulty but ensure integer results
        n1 = int(rng.randrange(50, int(500 * difficulty), 50))
        frac1 = rng.choice(fractions)
        mult1 = rng.choice(multipliers)
        frac2 = rng.choice(fractions)

        # Ensure all divisions result in integers
        while (
            not (n1 * frac1).is_integer()
            or not (n1 * frac1 * mult1).is_integer()
            or not (n1 * frac1 * mult1 * frac2).is_integer()
        ):
            n1 = int(rng.randrange(50, int(500 * difficulty), 50))

        result = generate_from_variables(
            name1, name2, name3, name4, creature, weapon1, weapon2, weapon3, weapon4, weapon5, n1, frac1, mult1, frac2
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_94(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, num_shares: int, price_per_share: int, increase_pct: int, decrease_pct: int
    ) -> dict[str, Any]:
        initial_value = num_shares * price_per_share
        first_increase = initial_value * increase_pct / 100
        value_after_increase = initial_value + first_increase
        second_decrease = value_after_increase * decrease_pct / 100
        final_value = value_after_increase - second_decrease

        question = f"{name} buys {num_shares} shares of a stock for ${price_per_share} each. The stock price increases {increase_pct}% the first year {name} holds it, then decreases {decrease_pct}% in the second year. What is the final value of all {name}'s shares?"

        answer_cot = (
            f"First find the initial total value of {name}'s purchase: {num_shares} shares * ${price_per_share}/share = ${initial_value}\n"
            f"Then find the amount of the first price increase: ${initial_value} * {increase_pct/100} = ${int(first_increase)}\n"
            f"Add that amount to the initial value to find the value after the first year: ${initial_value} + ${int(first_increase)} = ${int(value_after_increase)}\n"
            f"Then multiply that amount by {decrease_pct}% to find the amount of the decrease in the second year: ${int(value_after_increase)} * {decrease_pct}% = ${int(second_decrease)}\n"
            f"Then subtract that amount from the value after the first year to find the final value: ${int(value_after_increase)} - ${int(second_decrease)} = ${int(final_value)}\n"
            f"#### {int(final_value)}"
        )

        return {
            "question": question,
            "answer": format_number(int(final_value)),
            "answer_cot": answer_cot,
            "answer_value": int(final_value),
            "variables": {
                "name": name,
                "num_shares": num_shares,
                "price_per_share": price_per_share,
                "increase_pct": increase_pct,
                "decrease_pct": decrease_pct,
                "initial_value": initial_value,
                "final_value": int(final_value),
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Maria", "Sarah", "Emma", "Isabella", "Sophia", "Mia", "Charlotte"]
        name = rng.choice(names)

        num_shares = int(rng.randint(5, int(20 * difficulty)))
        price_per_share = int(rng.randint(5, int(100 * difficulty)))
        increase_pct = int(rng.randrange(10, int(100 * difficulty), 5))
        decrease_pct = int(rng.randrange(5, int(50 * difficulty), 5))

        # Ensure integer results
        while (
            not (num_shares * price_per_share * increase_pct / 100).is_integer()
            or not (num_shares * price_per_share * (1 + increase_pct / 100) * (1 - decrease_pct / 100)).is_integer()
        ):
            num_shares = int(rng.randint(5, int(20 * difficulty)))
            price_per_share = int(rng.randint(5, int(100 * difficulty)))
            increase_pct = int(rng.randrange(10, int(100 * difficulty), 5))
            decrease_pct = int(rng.randrange(5, int(50 * difficulty), 5))

        result = generate_from_variables(name, num_shares, price_per_share, increase_pct, decrease_pct)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_95(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name1: str, name2: str, relation: str, food: str, n1: int, n2: int, n3: int, time_unit: str, time_period: str
    ) -> dict[str, Any]:
        daily_total = n1 + n2 + n3
        total = daily_total * (7 if time_period == "week" else 30)

        question = f"{name1} eats {n1} {food} per {time_unit}, {name2} eats {n2} {food} per {time_unit}, and their {relation} eats {n3} {food} per {time_unit}. How many {food} does this family eat in one {time_period}?"

        answer_cot = f"The number of {food} they eat in one {time_unit} is {n1} + {n2} + {n3} = {daily_total} {food}.\nThe number of {food} they eat in a {time_period} is {daily_total} * {7 if time_period == 'week' else 30} = {total} {food}.\n#### {total}"

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "name1": name1,
                "name2": name2,
                "relation": relation,
                "food": food,
                "daily_servings1": n1,
                "daily_servings2": n2,
                "daily_servings3": n3,
                "daily_total": daily_total,
                "time_unit": time_unit,
                "time_period": time_period,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        name1_options = ["A father", "A grandfather", "An uncle"]
        name2_options = ["his wife", "his partner", "his spouse"]
        relation_options = ["daughter", "son", "grandchild"]
        food_options = ["pizzas", "burritos", "tacos", "sushi rolls", "hamburgers"]

        name1 = rng.choice(name1_options)
        name2 = rng.choice(name2_options)
        relation = rng.choice(relation_options)
        food = rng.choice(food_options)

        n1 = int(rng.randint(2, int(9 * difficulty)))
        n2 = int(rng.randint(2, int(9 * difficulty)))
        n3 = int(rng.randint(2, int(9 * difficulty)))

        time_unit = "day"
        time_period = rng.choice(["week", "month"])

        result = generate_from_variables(name1, name2, relation, food, n1, n2, n3, time_unit, time_period)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_96(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, food: str, animal1: str, animal2: str, n1: int, n2: int, k1: int, k2: int, unit: str
    ) -> dict[str, Any]:
        animal2_amount = 2 * n1 - n2  # Amount per sheep
        animal2_total = k2 * animal2_amount  # Total for sheep
        animal1_total = k1 * n1  # Total for goats
        total = animal1_total + animal2_total

        question = f"{name} is feeding his livestock {food}. Each {animal1} needs {n1} {unit}, and each {animal2} needs {n2} {unit} less than twice the amount each {animal1} needs. If there are {k1} {animal1}s and {k2} {animal2}s, how many {unit} of {food} does {name} need?"

        answer_cot = (
            f"First figure out how much {food} each {animal2} needs: {n1} {unit} * 2 - {n2} = {animal2_amount} {unit}/{animal2}\n"
            f"Now figure out how much {food} total the {animal2}s need: {animal2_amount} {unit}/{animal2} * {k2} {animal2} = {animal2_total} {unit}\n"
            f"Now figure out how much {food} total the {animal1}s need: {n1} {unit}/{animal1} * {k1} {animal1}s = {animal1_total} {unit}\n"
            f"Now add the two amounts of {food} to find the total: {animal2_total} {unit} + {animal1_total} {unit} = {total} {unit}\n#### {total}"
        )

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "name": name,
                "food": food,
                "animal1": animal1,
                "animal2": animal2,
                "n1": n1,
                "n2": n2,
                "k1": k1,
                "k2": k2,
                "unit": unit,
                "animal2_amount": animal2_amount,
                "animal2_total": animal2_total,
                "animal1_total": animal1_total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Michael", "David", "James", "Robert", "William", "Richard"]
        foods = ["hay", "grain", "feed", "silage"]
        animals = ["goat", "cow", "horse", "donkey", "llama", "alpaca", "pig", "turkey", "duck"]
        units = ["pounds", "kilograms", "kg"]

        name = rng.choice(names)
        food = rng.choice(foods)
        animal1, animal2 = rng.sample(animals, 2)
        unit = rng.choice(units)

        n1 = int(rng.randint(3, int(15 * difficulty)))
        n2 = int(rng.randint(1, int(10 * difficulty)))

        # Ensure 2*n1 - n2 > 0
        while 2 * n1 - n2 <= 0:
            n1 = int(rng.randint(3, int(15 * difficulty)))
            n2 = int(rng.randint(1, int(10 * difficulty)))

        k1 = int(rng.randint(10, int(50 * difficulty)))
        k2 = int(rng.randint(10, int(50 * difficulty)))

        result = generate_from_variables(name, food, animal1, animal2, n1, n2, k1, k2, unit)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_97(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, mult_run: int, frac_skip: float, skip_speed: int, total_time: int, frac_run: float, frac_walk: float
    ) -> dict[str, Any]:
        run_speed = skip_speed / frac_skip
        walk_speed = run_speed / mult_run
        walk_hours = total_time * frac_walk
        run_hours = total_time * frac_run
        run_dist = run_hours * run_speed
        walk_dist = walk_hours * walk_speed
        total_dist = int(run_dist + walk_dist)

        question = f"{name} can run {mult_run} times faster than she can walk, but she can skip at a rate of speed that is {frac_skip:.1f} as fast as she can run. If she can skip at {skip_speed} miles per hour, how many miles can she travel in {total_time} hours if she spends {frac_run:.2f} of the time running and {frac_walk:.2f} of the time walking?"

        answer_cot = f"""If {name} can skip at {frac_skip:.1f} the speed she can run, then she can run at {skip_speed}*{1/frac_skip:.1f}={run_speed} miles per hour.
        And since she can run at a speed that is {mult_run} times faster than she can walk, this means she can walk at {run_speed}/{mult_run}={walk_speed} miles per hour.
        If {frac_walk:.2f} of the time is spent walking, then she walks for {total_time}*{frac_walk:.2f}={walk_hours} hours.
        If {frac_run:.2f} of the time is spent running, then she runs for {total_time}-{walk_hours}={run_hours} hours.
        Thus, she runs for {run_hours} hours at {run_speed} miles per hour, or {run_hours}*{run_speed}={run_dist} miles.
        She walks for {walk_hours} hours at {walk_speed} miles per hour, or {walk_hours}*{walk_speed}={walk_dist} miles.
        Thus, altogether, she travels {run_dist}+{walk_dist}={total_dist} miles.
        #### {total_dist}"""

        return {
            "question": question,
            "answer": format_number(total_dist),
            "answer_cot": answer_cot,
            "answer_value": total_dist,
            "variables": {
                "name": name,
                "mult_run": mult_run,
                "frac_skip": frac_skip,
                "skip_speed": skip_speed,
                "total_time": total_time,
                "frac_run": frac_run,
                "frac_walk": frac_walk,
                "run_speed": run_speed,
                "walk_speed": walk_speed,
                "total_dist": total_dist,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Dana", "Emma", "Sarah", "Julia", "Sophie", "Maria"]
        name = rng.choice(names)

        # Keep simple fractions constant
        frac_skip = 0.5
        frac_run = 1 / 3
        frac_walk = 2 / 3

        # First generate skip_speed that ensures skip_speed/frac_skip is integer and < 13
        # Since frac_skip is 0.5, skip_speed should be a multiple of 0.5 and < 6.5
        valid_skip_speeds = [2, 3, 4, 5, 6]
        skip_speed = rng.choice(valid_skip_speeds)

        # Calculate skip_rate = skip_speed/frac_skip (will be integer and < 13)
        skip_rate = skip_speed / frac_skip

        # Generate multiplier that will create valid divisions
        mult_run = rng.randint(2, min(int(6 * difficulty), 5))

        # For total_time, we need:
        # - total_time * frac_run (1/3) to be integer
        # - total_time * frac_walk (2/3) to be integer
        # - total_time * frac_walk * (skip_rate/mult_run) to be integer
        # So total_time should be multiple of 6 to handle fractions
        # and should make the skip calculation work
        base_time = 6  # Multiple of both 3 and 2 for fractions
        max_time = min(int(12 * difficulty), 12)
        valid_times = [
            t
            for t in range(base_time, max_time + 1, base_time)
            if (t * frac_walk * (skip_rate / mult_run)).is_integer()
        ]
        if not valid_times:
            valid_times = [6]  # Fallback to smallest valid time
        total_time = rng.choice(valid_times)

        result = generate_from_variables(name, mult_run, frac_skip, skip_speed, total_time, frac_run, frac_walk)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_98(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        vehicle: str,
        weight_vehicle: int,
        item: str,
        weight_item: int,
        passenger_type: str,
        num_passengers: int,
        weight_passenger: int,
        unit: str,
        force_percent: int,
    ) -> dict[str, Any]:

        total_passenger_weight = num_passengers * weight_passenger
        total_weight = weight_vehicle + weight_item + total_passenger_weight
        force_needed = int((total_weight * force_percent) / 100)

        question = f"{name}'s {vehicle} breaks down. The {vehicle} weighs {weight_vehicle} {unit} and he has {item} in it weighing {weight_item} {unit}. He also has his {num_passengers} young {passenger_type} who weigh {weight_passenger} {unit} each in it. If the force to move the {vehicle} is {force_percent}% of the weight, how much force does he need to push the {vehicle}?"

        answer_cot = f"His {num_passengers} {passenger_type} weigh {weight_passenger}*{num_passengers}={total_passenger_weight} {unit}\nSo the total weight of the {vehicle} and everything is {weight_vehicle}+{weight_item}+{total_passenger_weight}={total_weight} {unit}\nSo he needs to generate {total_weight}*{force_percent/100}={force_needed} {unit}\n#### {force_needed}"

        return {
            "question": question,
            "answer": format_number(force_needed),
            "answer_cot": answer_cot,
            "answer_value": force_needed,
            "variables": {
                "name": name,
                "vehicle": vehicle,
                "weight_vehicle": weight_vehicle,
                "item": item,
                "weight_item": weight_item,
                "passenger_type": passenger_type,
                "num_passengers": num_passengers,
                "weight_passenger": weight_passenger,
                "unit": unit,
                "force_percent": force_percent,
                "total_weight": total_weight,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Michael", "David", "James", "Robert", "William", "Richard"]
        vehicles = ["car", "van", "truck", "SUV"]
        items = ["luggage", "groceries", "equipment", "furniture"]
        passenger_types = ["children", "friends", "colleagues", "teammates"]
        units = ["pounds", "kilograms"]

        name = rng.choice(names)
        vehicle = rng.choice(vehicles)
        item = rng.choice(items)
        passenger_type = rng.choice(passenger_types)
        unit = rng.choice(units)

        # Generate weights that are multiples of 100 to make force calculations easier
        # This ensures any percentage will result in integers
        weight_vehicle = int(rng.randrange(10, min(int(30 * difficulty), 25)) * 100)  # 1000-2500 in steps of 100
        weight_item = int(rng.randrange(1, min(int(5 * difficulty), 4)) * 100)  # 100-400 in steps of 100
        weight_passenger = int(rng.randrange(10, min(int(20 * difficulty), 15)) * 5)  # 50-75 in steps of 5
        num_passengers = int(rng.randint(2, min(int(5 * difficulty), 4)))

        # Calculate total weight - will be multiple of 5 due to construction
        total_weight = weight_vehicle + weight_item + (num_passengers * weight_passenger)

        # Since total_weight is multiple of 5, we can choose force_percent that guarantees integer result
        # We'll use multiples of 5 for force_percent to ensure clean division
        force_percent = 5 * rng.randint(1, min(int(1.2 * difficulty), 3))  # Will give 5%, 10%, 15%

        result = generate_from_variables(
            name,
            vehicle,
            weight_vehicle,
            item,
            weight_item,
            passenger_type,
            num_passengers,
            weight_passenger,
            unit,
            force_percent,
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)


def generate_99(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, currency: str, initial_amount: float, quantity: int, item: str, store_type: str, unit_price: float
    ) -> dict[str, Any]:
        total_cost = quantity * unit_price
        remaining = initial_amount - total_cost

        question = f"{name} has {currency}{initial_amount:.2f} and wants to buy {quantity} {item}s from a bin at the {store_type} store. Each {item} costs {currency}{unit_price:.2f}. How much money does {name} have left after paying for the {item}s?"

        answer_cot = f"{name} paid {quantity} * {currency}{unit_price:.2f} = {currency}{total_cost:.2f} for the {item}s.\n{name} has {currency}{initial_amount:.2f} - {currency}{total_cost:.2f} = {currency}{int(remaining)} left.\n#### {int(remaining)}"

        return {
            "question": question,
            "answer": format_number(int(remaining)),
            "answer_cot": answer_cot,
            "answer_value": int(remaining),
            "variables": {
                "name": name,
                "currency": currency,
                "initial_amount": initial_amount,
                "quantity": quantity,
                "item": item,
                "store_type": store_type,
                "unit_price": unit_price,
                "total_cost": total_cost,
                "remaining": remaining,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["David", "John", "Michael", "James", "William", "Robert"]
        currencies = ["$", "€", "£"]
        items = ["screw", "nail", "washer", "nut", "anchor"]
        store_types = ["hardware", "home improvement", "construction supply"]

        name = rng.choice(names)
        currency = rng.choice(currencies)
        item = rng.choice(items)
        store_type = rng.choice(store_types)

        # Generate values ensuring conditions are met
        quantity = int(rng.randint(15, int(60 * difficulty)))
        unit_price = round(rng.uniform(0.01, min(1.0, 1.0 * difficulty)), 2)

        # Ensure initial amount is sufficient and result is integer
        total_cost = quantity * unit_price
        remaining = rng.randint(1, int(100 * difficulty))
        initial_amount = total_cost + remaining

        result = generate_from_variables(name, currency, initial_amount, quantity, item, store_type, unit_price)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "difficulty": difficulty,
                "answer_value": result["answer_value"],
                "answer_cot": result["answer_cot"],
                "variables": result["variables"],
            },
        }

    return generate_example(rng, difficulty)
