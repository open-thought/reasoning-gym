import math
from fractions import Fraction
from random import Random
from typing import Any

from reasoning_gym.utils import format_number, is_integer


def generate_0(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, food: str, peel_rate: int, batch_size: int, time_per_batch: int, total_amount: int
    ) -> dict[str, Any]:

        peel_time = total_amount // peel_rate
        num_batches = total_amount // batch_size
        cook_time = num_batches * time_per_batch
        total_time = peel_time + cook_time

        question = f"{name} can peel {peel_rate} {food}s a minute and saute {batch_size} {food}s in {time_per_batch} minutes. How long will it take her to peel and saute {total_amount} {food}s?"

        answer_cot = (
            f"First find how long it takes {name} to peel the {food}: {total_amount} {food} / {peel_rate} {food}/minute = {peel_time} minutes\n"
            f"Then find how many batches of {food} she needs to cook: {total_amount} {food} / {batch_size} {food}/batch = {num_batches} batches\n"
            f"Then multiply the number of batches by the time per batch to find the total cook time: {num_batches} batches * {time_per_batch} minutes/batch = {cook_time} minutes\n"
            f"Then add the peeling time to find the total time {name} spends: {cook_time} minutes + {peel_time} minutes = {total_time} minutes\n"
            f"#### {total_time}"
        )

        return {
            "question": question,
            "answer": format_number(total_time),
            "answer_cot": answer_cot,
            "answer_value": total_time,
            "variables": {
                "name": name,
                "food": food,
                "peel_rate": peel_rate,
                "batch_size": batch_size,
                "time_per_batch": time_per_batch,
                "total_amount": total_amount,
                "peel_time": peel_time,
                "cook_time": cook_time,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emily", "Sarah", "Emma", "Sophia", "Olivia", "Ava", "Isabella", "Mia"]
        foods = ["shrimp", "onion", "carrot", "mushroom", "clam"]

        name = rng.choice(names_female)
        food = rng.choice(foods)

        peel_rate = int(rng.randint(4, int(15 * difficulty)))
        batch_size = int(rng.randrange(20, int(50 * difficulty), 5))
        time_per_batch = int(rng.randint(5, int(20 * difficulty)))

        # Ensure total is divisible by both peel_rate and batch_size
        lcm = peel_rate * batch_size // math.gcd(peel_rate, batch_size)
        num_lcm = rng.randint(1, int(4 * difficulty))
        total_amount = lcm * num_lcm

        result = generate_from_variables(name, food, peel_rate, batch_size, time_per_batch, total_amount)

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


def generate_1(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, family: str, blocks: int, animals: int, rings: int, total: int
    ) -> dict[str, Any]:
        bouncy_balls = total - (blocks + animals + rings)

        question = f"When {name} watches her {family}, she gets out a variety of toys for him. The bag of building blocks has {blocks} blocks in it. The bin of stuffed animals has {animals} stuffed animals inside. The tower of stacking rings has {rings} multicolored rings on it. {name} recently bought a tube of bouncy balls, bringing her total number of toys for her {family} up to {total}. How many bouncy balls came in the tube?"

        answer_cot = f"Let T be the number of bouncy balls in the tube.\nAfter buying the tube of balls, {name} has {blocks} + {animals} + {rings} + T = {blocks + animals + rings} + T = {total} toys for her {family}.\nThus, T = {total} - {blocks + animals + rings} = {bouncy_balls} bouncy balls came in the tube.\n#### {bouncy_balls}"

        return {
            "question": question,
            "answer": format_number(bouncy_balls),
            "answer_cot": answer_cot,
            "answer_value": bouncy_balls,
            "variables": {
                "name": name,
                "family": family,
                "building_blocks": blocks,
                "stuffed_animals": animals,
                "stacking_rings": rings,
                "total_toys": total,
                "bouncy_balls": bouncy_balls,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Sophie", "Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia"]
        family_members = ["nephew", "cousin", "brother"]

        name = rng.choice(names_female)
        family = rng.choice(family_members)

        blocks = int(rng.randint(70, int(75 * difficulty)))
        animals = int(rng.randint(35, int(50 * difficulty)))
        rings = int(rng.randint(20, int(35 * difficulty)))

        total = blocks + animals + rings + int(rng.randint(20, int(100 * difficulty)))

        result = generate_from_variables(name, family, blocks, animals, rings, total)

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


def generate_2(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        teacher: str, total: int, p1: int, p2: int, group1: str, group2: str, group3: str, event: str
    ) -> dict[str, Any]:
        group1_count = int(total * p1 / 100)
        remaining = total - group1_count
        group23_count = int(remaining * p2 / 100)
        total_leaving = group1_count + group23_count

        question = f"In {teacher}'s class of {total} students, {p1}% of the class are {group1}. Out of the remaining class, {p2}% of the students are {group2} or part of {group3}. These 3 groups of students will need to leave early today to travel to an away {event}. How many students are leaving early?"

        answer_cot = (
            f"{p1}% of the {total} student class are {group1} so that's {p1/100}*{total} = {group1_count} students\n"
            f"There are {total} students and {group1_count} are {group1} so that leaves {total}-{group1_count} = {remaining} students\n"
            f"{p2}% of the remaining {remaining} students are part of {group3} or {group2} so that's {p2/100}*{remaining} = {group23_count} students\n"
            f"{group1_count} students are {group1} and {group23_count} are part of {group3}/{group2} so {group1_count}+{group23_count} = {total_leaving} students will be leaving early\n"
            f"#### {total_leaving}"
        )

        return {
            "question": question,
            "answer": format_number(total_leaving),
            "answer_cot": answer_cot,
            "answer_value": total_leaving,
            "variables": {
                "teacher": teacher,
                "total_students": total,
                "percent_group1": p1,
                "percent_group23": p2,
                "group1": group1,
                "group2": group2,
                "group3": group3,
                "event": event,
                "group1_count": group1_count,
                "group23_count": group23_count,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        teachers = ["Ms. Johnson", "Mr. Smith", "Dr. Lee", "Mrs. Garcia"]
        sports = ["soccer players", "basketball players", "volleyball players", "swimmers"]
        activities = ["dancers", "choir members", "debate team members", "robotics club members"]
        events = ["competition", "tournament", "performance", "meet"]

        teacher = rng.choice(teachers)
        group1 = rng.choice(sports)
        group2, group3 = rng.sample(activities, 2)
        event = rng.choice(events)

        # Generate total first - keeping it as multiple of 100
        base_total = rng.randint(1, min(int(1.5 * difficulty), 2))
        total = base_total * 100  # Will be 100 or 200

        # Now generate p1 to ensure remaining is a multiple of 20
        # If total is 100, p1 should be a multiple of 20 (20, 40, 60, 80)
        # If total is 200, p1 should be a multiple of 10 (10, 20, 30, ..., 80)
        max_p1 = min(80, int(100 * difficulty))  # Cap at 80% to leave room for group2/3
        step = 20 if total == 100 else 10
        p1_choices = list(range(20, max_p1 + 1, step))
        p1 = rng.choice(p1_choices)

        # Calculate remaining - will be a multiple of 20
        remaining = total - (total * p1 // 100)

        # Now generate p2 as a multiple of remaining's divisor
        # If remaining is 60, p2 should be multiple of (100/60) to ensure clean division
        divisor = 100 // remaining if remaining > 0 else 1
        max_p2 = min(90, int(100 * difficulty))  # Cap at 90%
        p2_choices = list(range(20, max_p2 + 1, divisor))
        p2 = rng.choice(p2_choices) if p2_choices else 20

        result = generate_from_variables(teacher, total, p1, p2, group1, group2, group3, event)

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


def generate_3(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        initial_pals: int,
        lost_pals: int,
        letters_per_week: int,
        pages_per_letter: int,
        minutes_per_page: int,
    ) -> dict[str, Any]:
        current_pals = initial_pals - lost_pals
        letters_received = current_pals * letters_per_week
        pages_to_write = letters_received * pages_per_letter
        total_minutes = pages_to_write * minutes_per_page
        hours = total_minutes // 60

        question = f"{name} was a pen pal with {initial_pals} people. He stopped being penpals with {lost_pals} of them. They each send {letters_per_week} letters a week that are {pages_per_letter} pages long. He responds in kind. He can write a page every {minutes_per_page} minutes. How many hours does he spend writing a week?"

        answer_cot = (
            f"{name} is penpals with {initial_pals}-{lost_pals}={current_pals} people\n"
            f"Thus he gets {current_pals}*{letters_per_week}={letters_received} letters a week\n"
            f"So he writes {letters_received}*{pages_per_letter}={pages_to_write} pages a week\n"
            f"So he writes for {pages_to_write}*{minutes_per_page}={total_minutes} minutes a week\n"
            f"So he writes {total_minutes}/60={hours} hours a week\n#### {hours}"
        )

        return {
            "question": question,
            "answer": format_number(hours),
            "answer_cot": answer_cot,
            "answer_value": hours,
            "variables": {
                "name": name,
                "initial_penpals": initial_pals,
                "lost_penpals": lost_pals,
                "current_penpals": current_pals,
                "letters_per_week": letters_per_week,
                "pages_per_letter": pages_per_letter,
                "minutes_per_page": minutes_per_page,
                "total_minutes": total_minutes,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Mike", "John", "David", "James", "Robert", "William", "Richard"]
        name = rng.choice(names)

        initial_pals = int(rng.randint(5, int(15 * difficulty)))
        lost_pals = int(rng.randint(1, initial_pals - 1))
        letters_per_week = int(rng.randint(2, int(5 * difficulty)))
        pages_per_letter = int(rng.randint(5, int(12 * difficulty)))

        # Calculate minutes_per_page that ensures whole hours
        # Total minutes = (initial_pals - lost_pals) * letters_per_week * pages_per_letter * minutes_per_page
        # This should be divisible by 60

        multiplier = (initial_pals - lost_pals) * letters_per_week * pages_per_letter

        # Generate valid minutes that make total divisible by 60
        min_minutes = 4
        max_minutes = int(15 * difficulty)

        # Find numbers between min_minutes and max_minutes that make multiplier * minutes divisible by 60
        valid_minutes = [m for m in range(min_minutes, max_minutes + 1) if (multiplier * m) % 60 == 0]

        if not valid_minutes:
            # If no valid minutes found, adjust multiplier to make it work with a reasonable minutes value
            minutes_per_page = min_minutes
            # Round up pages_per_letter to make total minutes divisible by 60
            total_minutes = multiplier * minutes_per_page
            pages_per_letter = ((total_minutes + 59) // 60 * 60) // (multiplier // pages_per_letter)
        else:
            minutes_per_page = rng.choice(valid_minutes)

        result = generate_from_variables(
            name, initial_pals, lost_pals, letters_per_week, pages_per_letter, minutes_per_page
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


def generate_4(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, items: str, food: str, location: str, container: str, num_jars: int, per_jar: int, per_pan: int
    ) -> dict[str, Any]:
        total_items = num_jars * per_jar
        num_pans = total_items // per_pan

        question = f"{name} has {num_jars} jars of {items} in her {location}. Each jar of {items} can decorate {per_jar} {food}s. {name} wants to bake enough {food}s to use up all of her {items}. If each {container} holds {per_pan} {food}s, how many {container}s worth of {food}s should she bake?"

        answer_cot = f"She has enough {items} for {num_jars} * {per_jar} = {total_items} {food}s.\nShe needs {total_items} / {per_pan} = {num_pans} {container}s to bake all of the {food}s.\n#### {num_pans}"

        return {
            "question": question,
            "answer": format_number(num_pans),
            "answer_cot": answer_cot,
            "answer_value": num_pans,
            "variables": {
                "name": name,
                "items": items,
                "food": food,
                "location": location,
                "container": container,
                "num_jars": num_jars,
                "per_jar": per_jar,
                "per_pan": per_pan,
                "total_items": total_items,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Mary", "Sarah", "Emma", "Elizabeth", "Catherine"]
        items = ["sprinkles", "frosting", "icing", "chocolate chips"]
        foods = ["cupcake", "cookie", "brownie", "muffin"]
        locations = ["pantry", "cupboard", "kitchen cabinet", "storage room"]
        containers = ["pan", "tray", "baking sheet", "rack"]

        name = rng.choice(names_female)
        item = rng.choice(items)
        food = rng.choice(foods)
        location = rng.choice(locations)
        container = rng.choice(containers)

        # Start with per_jar, then make per_pan a multiple of it
        per_jar = int(rng.randint(6, int(20 * difficulty)))

        # Generate per_pan as a multiple of per_jar within range
        multiplier = rng.randint(1, max(1, int(24 * difficulty) // per_jar))
        per_pan = per_jar * multiplier

        # Ensure per_pan is within bounds
        per_pan = min(per_pan, int(24 * difficulty))

        num_jars = int(rng.randint(3, int(15 * difficulty)))

        result = generate_from_variables(name, item, food, location, container, num_jars, per_jar, per_pan)

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


def generate_5(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name1: str,
        name2: str,
        city: str,
        celebrity_type: str,
        vacation_type: str,
        n1: int,
        n2: int,
        s1: int,
        s2: int,
        goal: int,
    ) -> dict[str, Any]:
        signatures_collected = s1 + s2
        signatures_needed = goal - signatures_collected

        question = f"{name1} and {name2} are sisters from {city} who love collecting signatures from {celebrity_type}. During their {vacation_type} from school, the sisters spend every afternoon collecting signatures. After {n1} weeks, {name1} and {name2} compare their autograph books, counting up the number of signatures each sister has collected. {name1} has {s1} signatures in her book, and {name2} has {s2}. The sisters have {n2} more weeks of {vacation_type}, and they decide they want to reach {goal} signatures between them by the end of the summer. How many signatures do the sisters need to collect to reach their goal?"

        answer_cot = f"{name1} and {name2} have already collected {s1} + {s2} signatures = {signatures_collected} signatures.\nSince their goal is {goal}, they need to collect {goal} - {signatures_collected} signatures. {goal} - {signatures_collected} = {signatures_needed} signatures\n#### {signatures_needed}"

        return {
            "question": question,
            "answer": format_number(signatures_needed),
            "answer_cot": answer_cot,
            "answer_value": signatures_needed,
            "variables": {
                "name1": name1,
                "name2": name2,
                "city": city,
                "celebrity_type": celebrity_type,
                "vacation_type": vacation_type,
                "weeks_passed": n1,
                "weeks_remaining": n2,
                "signatures1": s1,
                "signatures2": s2,
                "goal": goal,
                "signatures_collected": signatures_collected,
                "signatures_needed": signatures_needed,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Carol", "Jennifer"]
        cities = ["Los Angeles", "New York", "Chicago", "Houston", "Phoenix"]
        celebrity_types = ["movie stars", "athletes", "musicians", "politicians", "authors"]
        vacation_types = ["winter break", "spring break", "summer break", "fall break"]

        name1, name2 = rng.sample(names_female, 2)
        city = rng.choice(cities)
        celebrity_type = rng.choice(celebrity_types)
        vacation_type = rng.choice(vacation_types)

        n1 = int(rng.randint(3, int(8 * difficulty)))
        n2 = int(rng.randint(2, int(5 * difficulty)))
        s1 = int(rng.randint(15, int(40 * difficulty)))
        s2 = int(rng.randint(30, int(60 * difficulty)))
        goal = int(rng.randrange(90, int(150 * difficulty), 5))

        # Ensure conditions are met
        while s1 + s2 >= goal:
            s1 = int(rng.randint(15, int(40 * difficulty)))
            s2 = int(rng.randint(30, int(60 * difficulty)))

        result = generate_from_variables(name1, name2, city, celebrity_type, vacation_type, n1, n2, s1, s2, goal)

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


def generate_6(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(n_girls: int, place: str, multiplier: int) -> dict[str, Any]:
        n_boys = n_girls * multiplier
        total_kids = n_girls + n_boys

        question = f"There are {n_girls} girls in the {place}. If there are {multiplier} times the number of boys in the {place}, how many kids are in the {place}?"

        answer_cot = f"There are {n_girls} girls x {multiplier} boys/girl = {n_boys} boys in the {place}.\nIn total there are {n_girls} girls + {n_boys} boys = {total_kids} kids in the {place}\n#### {total_kids}"

        return {
            "question": question,
            "answer": format_number(total_kids),
            "answer_cot": answer_cot,
            "answer_value": total_kids,
            "variables": {
                "n_girls": n_girls,
                "place": place,
                "multiplier": multiplier,
                "n_boys": n_boys,
                "total_kids": total_kids,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        places = ["park", "yard", "field", "playground", "garden"]
        multipliers = [2, 3, 4]  # twice, triple, quadruple

        place = rng.choice(places)
        multiplier = rng.choice(multipliers)

        # Scale n_girls with difficulty but ensure result is valid
        n_girls = int(rng.randint(5, int(50 * difficulty)))
        while n_girls * (multiplier + 1) > 200:
            n_girls = int(rng.randint(5, int(50 * difficulty)))

        result = generate_from_variables(n_girls, place, multiplier)

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


def generate_7(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, plants_received: int, plants_per_ledge: int, num_ledges: int, plants_to_give: int
    ) -> dict[str, Any]:

        initial_plants = plants_per_ledge * num_ledges
        total_plants = initial_plants + plants_received
        plants_given = num_ledges * plants_to_give
        remaining_plants = total_plants - plants_given

        question = f"{name} is an avid gardener. Yesterday, she received {plants_received} new potted plants from her favorite plant nursery. She already has {plants_per_ledge} potted plants on each of the {num_ledges} window ledges of her large country home. Feeling generous, she has decided that she will give {plants_to_give} potted plant from each ledge to friends and family tomorrow. How many potted plants will {name} remain with?"

        answer_cot = f"Yesterday, before receiving the plants, {name} had {num_ledges}*{plants_per_ledge} = {initial_plants} potted plants\nAfter receiving an additional {plants_received} plants, she therefore had a total of {initial_plants} + {plants_received} = {total_plants} potted plants\nTomorrow, {name}'s plant giveaway will be {num_ledges}*{plants_to_give} = {plants_given} potted plants.\nShe will therefore remain with {total_plants} - {plants_given} = {remaining_plants} potted plants.\n#### {remaining_plants}"

        return {
            "question": question,
            "answer": format_number(remaining_plants),
            "answer_cot": answer_cot,
            "answer_value": remaining_plants,
            "variables": {
                "name": name,
                "plants_received": plants_received,
                "plants_per_ledge": plants_per_ledge,
                "num_ledges": num_ledges,
                "plants_to_give": plants_to_give,
                "initial_plants": initial_plants,
                "total_plants": total_plants,
                "plants_given": plants_given,
                "remaining_plants": remaining_plants,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Mary", "Emma", "Sophia", "Isabella", "Olivia", "Ava", "Mia"]

        name = rng.choice(names)
        plants_received = int(rng.randint(20, int(50 * difficulty)))
        plants_per_ledge = int(rng.randint(7, int(13 * difficulty)))
        num_ledges = int(rng.randint(50, int(70 * difficulty)))
        plants_to_give = int(rng.randint(3, int(8 * difficulty)))

        # Ensure condition: w * r + x - w*n > 0
        while (num_ledges * plants_per_ledge + plants_received - num_ledges * plants_to_give) <= 0:
            plants_per_ledge = int(rng.randint(7, int(13 * difficulty)))
            plants_to_give = int(rng.randint(3, int(8 * difficulty)))

        result = generate_from_variables(name, plants_received, plants_per_ledge, num_ledges, plants_to_give)

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


def generate_8(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, drink: str, sugar_ratio: int, water_ratio: int, total_items: int
    ) -> dict[str, Any]:
        total_ratio = sugar_ratio + water_ratio
        sugar_amount = (sugar_ratio * total_items) // total_ratio

        question = f"{name} makes {drink} using teaspoons of sugar and cups of water in the ratio of {sugar_ratio}:{water_ratio}. If she used a total of {total_items} teaspoons of sugar and cups of water, calculate the number of teaspoonfuls of sugar she used."

        answer_cot = f"The total ratio representing the ingredients she used to make the {drink} is {sugar_ratio}+{water_ratio} = {total_ratio}\nSince the fraction representing the number of teaspoons she used is {sugar_ratio}/{total_ratio}, she used {sugar_ratio}/{total_ratio}*{total_items} = {sugar_amount}\n#### {sugar_amount}"

        return {
            "question": question,
            "answer": format_number(sugar_amount),
            "answer_cot": answer_cot,
            "answer_value": sugar_amount,
            "variables": {
                "name": name,
                "drink": drink,
                "sugar_ratio": sugar_ratio,
                "water_ratio": water_ratio,
                "total_items": total_items,
                "sugar_amount": sugar_amount,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte"]
        drinks = ["coffee", "tea"]

        name = rng.choice(names_female)
        drink = rng.choice(drinks)

        sugar_ratio = int(rng.randint(25, int(201 * difficulty)))
        water_ratio = int(rng.randint(5, int(101 * difficulty)))

        # Ensure total is divisible by ratio sum
        total_ratio = sugar_ratio + water_ratio
        num_multiples = rng.randint(1, int(10 * difficulty))
        total_items = total_ratio * num_multiples

        result = generate_from_variables(name, drink, sugar_ratio, water_ratio, total_items)

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


def generate_9(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        num_bills: int,
        bill_value: int,
        num_items1: int,
        price1: int,
        num_items2: int,
        price2: int,
        item1: str,
        item2: str,
        currency: str,
    ) -> dict[str, Any]:

        initial_amount = num_bills * bill_value
        spent_items1 = num_items1 * price1
        spent_items2 = num_items2 * price2
        total_spent = spent_items1 + spent_items2
        remaining = initial_amount - total_spent

        question = f"{name} has {num_bills} {currency}{bill_value} bills. He buys {num_items1} {item1}s for {currency}{price1} each. He also buys {num_items2} packs of {item2}s for {currency}{price2} each. How much money does he have left?"

        answer_cot = (
            f"{name} starts off with {num_bills} * {currency}{bill_value} = {currency}{initial_amount}.\n"
            f"{name} spends {num_items1} {item1}s * {currency}{price1} = {currency}{spent_items1} on {item1}s.\n"
            f"{name} spends {num_items2} packs of {item2}s * {currency}{price2} = {currency}{spent_items2} on {item2}s.\n"
            f"Total {name} has spent {currency}{spent_items1} + {currency}{spent_items2} = {currency}{total_spent}.\n"
            f"{name} has {currency}{initial_amount} - {currency}{total_spent} = {currency}{remaining} remaining.\n#### {remaining}"
        )

        return {
            "question": question,
            "answer": format_number(remaining),
            "answer_cot": answer_cot,
            "answer_value": remaining,
            "variables": {
                "name": name,
                "num_bills": num_bills,
                "bill_value": bill_value,
                "num_items1": num_items1,
                "price1": price1,
                "num_items2": num_items2,
                "price2": price2,
                "item1": item1,
                "item2": item2,
                "currency": currency,
                "initial_amount": initial_amount,
                "total_spent": total_spent,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Craig", "John", "Michael", "David", "James", "Robert", "William"]
        items1 = ["toy car", "action figure", "coloring book", "puzzle", "board game"]
        items2 = ["sticker", "candy bar", "trading card", "pencil", "eraser"]
        currencies = ["$", "€", "£"]
        bills = [(5, 5), (10, 10), (20, 20), (50, 50), (100, 100)]

        name = rng.choice(names)
        item1 = rng.choice(items1)
        item2 = rng.choice(items2)
        currency = rng.choice(currencies)
        bill_value = rng.choice(bills)[1]

        # First determine number of bills to establish total money available
        num_bills = max(2, int(rng.randint(1, int(10 * difficulty))))  # Ensure at least 2 bills
        total_money = num_bills * bill_value

        # Calculate maximum possible prices based on difficulty and ensure they leave room for quantities
        max_price1 = min(int(10 * difficulty), total_money // 8)  # Ensure room for at least 4 of each item
        max_price2 = min(int(10 * difficulty), total_money // 8)

        # Generate prices (minimum 1)
        price1 = max(1, min(max_price1, int(total_money // 6)))
        price2 = max(1, min(max_price2, int(total_money // 6)))

        # Calculate maximum possible quantities that fit within total_money
        max_items1 = min(int(15 * difficulty), (total_money // 2) // price1)
        max_items1 = max(3, max_items1)  # Ensure at least 3 possible items

        # Generate first quantity
        num_items1 = rng.randint(2, max_items1)

        # Calculate remaining money and second quantity
        remaining_money = total_money - (num_items1 * price1)
        max_items2 = min(int(10 * difficulty), remaining_money // price2)
        max_items2 = max(3, min(max_items2, remaining_money // price2))  # Ensure at least 3 possible items

        num_items2 = rng.randint(2, max_items2)

        result = generate_from_variables(
            name, num_bills, bill_value, num_items1, price1, num_items2, price2, item1, item2, currency
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


def generate_10(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name1: str, name2: str, age1: int, years: int, relation_type: str, mult: int
    ) -> dict[str, Any]:
        future_age = age1 * mult
        current_age = future_age - years

        question = f"{name1} is {age1} years old. In {years} years his {relation_type} {name2} will be {mult} times as old as {name1} is now. How old is {name2} right now?"

        answer_cot = f"{years} years from now {name2} will be {age1}*{mult}={future_age}.\nRight now {name2} is {future_age}-{years}={current_age} years old.\n#### {current_age}"

        return {
            "question": question,
            "answer": format_number(current_age),
            "answer_cot": answer_cot,
            "answer_value": current_age,
            "variables": {
                "name1": name1,
                "name2": name2,
                "age1": age1,
                "years": years,
                "relation_type": relation_type,
                "mult": mult,
                "future_age": future_age,
                "current_age": current_age,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_male = ["James", "John", "Robert", "Michael", "William", "David", "Richard"]
        names_female = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan"]
        relation_types = ["sister", "cousin"]

        name1 = rng.choice(names_male)
        name2 = rng.choice(names_female)
        relation_type = rng.choice(relation_types)

        age1 = int(rng.randint(8, int(25 * difficulty)))
        years = int(rng.randint(2, int(10 * difficulty)))
        mult = int(rng.randint(2, int(5 * difficulty)))

        # Ensure conditions are met
        while age1 * mult - years <= 0:
            age1 = int(rng.randint(8, int(25 * difficulty)))
            years = int(rng.randint(2, int(10 * difficulty)))
            mult = int(rng.randint(2, int(5 * difficulty)))

        result = generate_from_variables(name1, name2, age1, years, relation_type, mult)

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


def generate_11(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name: str, food1: str, food2: str, mult: int, n: int, m: int, k: int) -> dict[str, Any]:
        # Initial amounts
        initial_food2 = n  # Initial amount of food2
        initial_food1 = n * mult  # Initial amount of food1 (mult times food2)
        initial_total = initial_food1 + initial_food2

        # Bought amounts - m more food2 and k fewer food1 than the bought amount of food2
        bought_food2 = m  # Bought m more food2
        bought_food1 = m - k  # k fewer food1 than the bought amount of food2 (m)
        bought_total = bought_food1 + bought_food2

        # Final totals
        final_food1 = initial_food1 + bought_food1  # Total food1
        final_food2 = initial_food2 + bought_food2  # Total food2
        final_total = final_food1 + final_food2  # Total of both foods

        question = (
            f"At {name}'s house, there is {mult} times as much {food1} as {food2}. "
            f"He has a total of {n} {food2} in his house. "
            f"At the store, {name} bought {m} {food2}. He also bought some {food1}, "
            f"but he bought {k} fewer {food1} than the {m} {food2} he just bought. "
            f"What is the combined total of {food1} and {food2} that {name} now has in the house?"
        )

        answer_cot = (
            f"Let's solve this step by step:\n\n"
            f"1. Initial amounts:\n"
            f"   • {name} has {n} {food2}\n"
            f"   • He has {mult} times as many {food1}, so {n} × {mult} = {initial_food1} {food1}\n"
            f"   • Total initial items: {initial_food1} + {n} = {initial_total}\n\n"
            f"2. Bought at store:\n"
            f"   • Bought {m} new {food2}\n"
            f"   • For {food1}, bought {k} fewer than the {m} {food2}, so {m} - {k} = {bought_food1} {food1}\n"
            f"   • Total bought items: {bought_food1} + {m} = {bought_total}\n\n"
            f"3. Final amounts:\n"
            f"   • Total {food1}: {initial_food1} + {bought_food1} = {final_food1}\n"
            f"   • Total {food2}: {initial_food2} + {bought_food2} = {final_food2}\n"
            f"   • Combined total: {final_food1} + {final_food2} = {final_total}\n\n"
            f"Therefore, {name} now has {final_total} items in total.\n"
            f"#### {final_total}"
        )

        return {
            "question": question,
            "answer": format_number(final_total),
            "answer_cot": answer_cot,
            "answer_value": final_total,
            "variables": {
                "name": name,
                "food1": food1,
                "food2": food2,
                "multiplier": mult,
                "initial_amount": n,
                "bought_amount": m,
                "difference": k,
                "initial_total": initial_total,
                "bought_total": bought_total,
                "final_total": final_total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Allan", "John", "Michael", "David", "James", "Robert", "William"]
        foods = ["corn", "apple", "banana", "orange", "pear", "grape", "fig", "persimmon", "plum", "kiwi"]

        name = rng.choice(names)
        food1, food2 = rng.sample(foods, 2)
        mult = rng.randint(2, int(4 * difficulty))

        n = int(rng.randint(20, int(100 * difficulty)))
        m = int(rng.randint(30, int(100 * difficulty)))
        k = int(rng.randint(10, min(m, int(50 * difficulty))))

        result = generate_from_variables(name, food1, food2, mult, n, m, k)

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


def generate_12(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, game1: str, game2: str, period: str, time1: int, time2: int, num1: int, num2: int
    ) -> dict[str, Any]:
        total_time1 = time1 * num1
        total_time2 = time2 * num2
        total_time = total_time1 + total_time2

        question = f"It takes {name} {time1} minutes to finish a {game1} and {time2} minutes to finish a {game2}. Over the {period} she solved {num1} {game1}s and {num2} {game2}s. How much time did she spend playing these games?"

        answer_cot = (
            f"It takes {time1} minutes to complete a {game1} and she completed {num1} for a total of {time1}*{num1} = {total_time1} minutes\n"
            f"It takes {time2} minutes to complete a {game2} and she completed {num2} for a total of {time2}*{num2} = {total_time2} minutes\n"
            f"She spent {total_time1} minutes on {game1}s and {total_time2} minutes on {game2}s for a total of {total_time1}+{total_time2} = {total_time} minutes\n"
            f"#### {total_time}"
        )

        return {
            "question": question,
            "answer": format_number(total_time),
            "answer_cot": answer_cot,
            "answer_value": total_time,
            "variables": {
                "name": name,
                "game1": game1,
                "game2": game2,
                "period": period,
                "time1": time1,
                "time2": time2,
                "num1": num1,
                "num2": num2,
                "total_time1": total_time1,
                "total_time2": total_time2,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte"]
        games = ["word puzzle", "jigsaw puzzle", "chess puzzle", "riddle", "brain teaser"]
        periods = ["weekend", "vacation", "holiday", "day off", "free time"]

        name = rng.choice(names)
        game1, game2 = rng.sample(games, 2)
        period = rng.choice(periods)

        time1 = int(rng.randint(5, int(30 * difficulty)))
        time2 = int(rng.randint(3, int(20 * difficulty)))
        while time2 >= time1:  # ensure time1 > time2
            time1 = int(rng.randint(5, int(30 * difficulty)))
            time2 = int(rng.randint(3, int(20 * difficulty)))

        num1 = int(rng.randint(2, int(10 * difficulty)))
        num2 = int(rng.randint(4, int(15 * difficulty)))

        result = generate_from_variables(name, game1, game2, period, time1, time2, num1, num2)

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


def generate_13(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        park_name: str, unit: str, length1: int, length2: int, speed1: int, speed2: int
    ) -> dict[str, Any]:
        time1 = length1 // speed1
        time2 = length2 // speed2
        time_diff = time1 - time2

        question = f"The biggest waterslide at {park_name} is {length1} {unit} long, and people slide down at {speed1} {unit}/minute. The second biggest waterslide is {length2} {unit} long, but steeper, so people slide down at {speed2} {unit}/minute. How much longer does it take to ride the biggest slide compared to the second biggest slide?"

        answer_cot = f"First find the ride length of the biggest slide: {length1} {unit} / {speed1} {unit}/minute = {time1} minutes\nThen find the ride length of the second biggest slide: {length2} {unit} / {speed2} {unit}/minute = {time2} minutes\nThen subtract the ride length of the second longest slide from the longest slide to find the difference: {time1} minutes - {time2} minutes = {time_diff} minutes\n#### {time_diff}"

        return {
            "question": question,
            "answer": format_number(time_diff),
            "answer_cot": answer_cot,
            "answer_value": time_diff,
            "variables": {
                "park_name": park_name,
                "unit": unit,
                "length1": length1,
                "length2": length2,
                "speed1": speed1,
                "speed2": speed2,
                "time1": time1,
                "time2": time2,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        parks = ["Splash World", "Aqua Adventure", "Water Wonderland", "Neptunes Kingdom"]
        units = ["yards", "meters"]

        park_name = rng.choice(parks)
        unit = rng.choice(units)

        # First generate speeds ensuring speed2 > speed1
        # Generate speed1 as multiple of 5
        speed1 = 5 * rng.randint(8, min(int(16 * difficulty), 12))  # 40 to 60 in steps of 5

        # Generate speed2 as multiple of 5, but bigger than speed1
        min_speed2 = (speed1 // 5 + 2) * 5  # At least 10 more than speed1
        max_speed2 = min(int(100 * difficulty), 80)
        max_speed2 = (max_speed2 // 5) * 5  # Make it multiple of 5
        speed2 = 5 * rng.randint(min_speed2 // 5, max_speed2 // 5)

        # Now generate lengths that are multiples of both their speeds
        # This ensures length % speed == 0 for both
        # First generate number of time units (ensuring time1 > time2)
        time2 = rng.randint(3, 5)  # 3-5 time units for second slide
        time1 = time2 + rng.randint(1, 2)  # 1-2 more time units than slide 2

        # Calculate lengths based on speeds and times
        length2 = speed2 * time2  # Guarantees length2 % speed2 == 0
        length1 = speed1 * time1  # Guarantees length1 % speed1 == 0
        # This also guarantees length1/speed1 > length2/speed2 since we constructed time1 > time2

        # Adjust if length1 <= length2
        if length1 <= length2:
            # Add one more time unit to length1
            length1 = speed1 * (time1 + 1)

        result = generate_from_variables(park_name, unit, length1, length2, speed1, speed2)

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


def generate_14(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, day1: str, day2: str, day3: str, time1: int, time2: int, mult: int
    ) -> dict[str, Any]:
        combined_time = time1 + time2
        target_time = combined_time * mult

        question = f"On {day3}, {name} wants to exercise for {mult} the amount of time he did on {day2} and {day1} combined. On {day1} he exercised for {time1} minutes. On {day2} he exercised for {time2} minutes. How many minutes does he have to exercise on {day3} to reach his goal?"

        answer_cot = f"On {day1} and {day2} he exercised a total of {combined_time} minutes because {time1} + {time2} = {combined_time}\nOn {day3} he has to exercise for {target_time} minutes because {combined_time} x {mult} = {target_time}\n#### {target_time}"

        return {
            "question": question,
            "answer": format_number(target_time),
            "answer_cot": answer_cot,
            "answer_value": target_time,
            "variables": {
                "name": name,
                "day1": day1,
                "day2": day2,
                "day3": day3,
                "time1": time1,
                "time2": time2,
                "multiplier": mult,
                "combined_time": combined_time,
                "target_time": target_time,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Peter", "John", "Michael", "David", "James", "Robert", "William"]
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        multipliers = [2, 3, 4]

        name = rng.choice(names)
        day1, day2, day3 = rng.sample(weekdays, 3)
        mult = rng.choice(multipliers)

        # We need (time1 + time2) * mult / 60 < 14
        # So: time1 + time2 < (14 * 60) / mult
        max_total_time = int((14 * 60) / mult) - 1  # Subtract 1 for safety margin

        # Now we know max total time, generate individual times
        max_individual_time = min(int(60 * difficulty), max_total_time // 2)  # Ensure sum stays under max
        min_time = 10

        if max_individual_time <= min_time:
            # If ranges are too tight, use safe values
            time1 = min_time
            time2 = min_time
        else:
            # Generate times that sum to less than max_total_time
            time1 = rng.randint(min_time, max_individual_time)
            time2 = rng.randint(min_time, min(max_individual_time, max_total_time - time1))

        result = generate_from_variables(name, day1, day2, day3, time1, time2, mult)

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


def generate_15(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        sport: str,
        item1: str,
        item2: str,
        item3: str,
        item4: str,
        currency: str,
        price1: int,
        price2: int,
        price3: int,
        price4: int,
        discount: int,
    ) -> dict[str, Any]:

        shorts_price = price1 + price2
        shoes_price = price3 // 2
        socks_price = price4 - discount
        total = price1 + shorts_price + shoes_price + socks_price

        question = f"{name} qualified for a spot on the {sport} team, so she went shopping for some athletic gear. She bought a {item1} for {currency}{price1}, a pair of {sport} {item2} for {currency}{price2} more than the {item1} cost, and a pair of {item3} that were originally {currency}{price3} but were on sale for half price. She had a coupon for {currency}{discount} off the package of {currency}{price4} athletic {item4} that she also bought. How much did she spend on athletic gear?"

        answer_cot = f"The {item2} were {currency}{price2} more than the {item1}, so they cost {currency}{price2} + {currency}{price1} = {currency}{shorts_price}.\nHer {item3} were half the original {currency}{price3} price, so they cost {currency}{price3} / 2 = ${shoes_price}.\nWith her coupon, the {item4} were {currency}{price4} - {currency}{discount} = {currency}{socks_price}.\nThe {item1}, {item2}, {item3}, and {item4} together cost {currency}{price1} + {currency}{shorts_price} + {currency}{shoes_price} + {currency}{socks_price} = {currency}{total}.\n#### {total}"

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "name": name,
                "sport": sport,
                "item1": item1,
                "item2": item2,
                "item3": item3,
                "item4": item4,
                "currency": currency,
                "price1": price1,
                "price2": price2,
                "price3": price3,
                "price4": price4,
                "discount": discount,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
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

        result = generate_from_variables(
            name, sport, item1, item2, item3, item4, currency, price1, price2, price3, price4, discount
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


def generate_16(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name1: str, name2: str, name3: str, platform: str, mult1: int, mult2: int, n: int
    ) -> dict[str, Any]:
        base_friends = n // mult1  # Dorothy's friends
        charlie_friends = n  # Charlie's friends
        james_friends = base_friends * mult2  # James's friends

        question = f"{name1} has {mult1} times as many {platform} friends as {name2}. {name3} has {mult2} times as many friends on {platform} as {name2}. If {name1} has {n} friends on {platform}, how many {platform} friends does {name3} have?"

        answer_cot = f"{name2} has {n} / {mult1} = {base_friends} {platform} friends.\n{name3} has {mult2} * {base_friends} = {james_friends} {platform} friends.\n#### {james_friends}"

        return {
            "question": question,
            "answer": format_number(james_friends),
            "answer_cot": answer_cot,
            "answer_value": james_friends,
            "variables": {
                "name1": name1,
                "name2": name2,
                "name3": name3,
                "platform": platform,
                "mult1": mult1,
                "mult2": mult2,
                "base_friends": base_friends,
                "charlie_friends": charlie_friends,
                "james_friends": james_friends,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Charlie", "Dorothy", "James", "Sarah", "Michael", "Emily", "David"]
        platforms = ["Instagram", "Twitter", "LinkedIn", "TikTok", "Snapchat"]

        name1, name2, name3 = rng.sample(names, 3)
        platform = rng.choice(platforms)

        # Generate multipliers that will be different
        mult1 = rng.randint(2, int(5 * difficulty))
        mult2 = rng.randint(2, int(5 * difficulty))
        while mult2 == mult1:
            mult2 = rng.randint(2, int(5 * difficulty))

        # Generate n that's divisible by mult1
        base = rng.randint(4, int(20 * difficulty))
        n = base * mult1

        result = generate_from_variables(name1, name2, name3, platform, mult1, mult2, n)

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


def generate_17(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        facility: str, total: int, item: str, frac: Fraction, event: str, daily: int, period: int
    ) -> dict[str, Any]:
        initial_occupied = int(total * frac)
        initial_empty = total - initial_occupied
        weekly_admitted = daily * 7
        total_admitted = weekly_admitted * period
        final_empty = initial_empty - total_admitted

        question = f"A {facility} has a capacity of {total} {item}s with {frac} occupied. Due to the {event}, {daily} patients are admitted into the {facility} each day. Calculate the total number of unoccupied {item}s in the {facility} after {period} weeks."

        answer_cot = f"If {frac} of the total capacity of the {facility} {item}s is occupied, it means {frac} * {total} = {initial_occupied} {item}s have patients using them.\nThe total number of {item}s in the {facility} without new admissions is {total} {item}s - {initial_occupied} {item}s = {initial_empty} {item}s.\nIf {daily} people are admitted each day, the total number of patients in the {facility} after one week is {daily} patients/day * 7 days/week = {weekly_admitted} patients.\nAfter {period} weeks, the total number of patients admitted into the {facility} is {weekly_admitted} patients/week * {period} weeks = {total_admitted} patients, who each use one {item}.\nIf there were {initial_empty} unoccupied {item}s in the {facility} before the new admissions, the total number is reduced to {initial_empty} {item}s - {total_admitted} {item}s = {final_empty} unoccupied {item}s.\n#### {final_empty}"

        return {
            "question": question,
            "answer": format_number(final_empty),
            "answer_cot": answer_cot,
            "answer_value": final_empty,
            "variables": {
                "facility": facility,
                "total_capacity": total,
                "item": item,
                "initial_fraction": frac,
                "event": event,
                "daily_patients": daily,
                "period_weeks": period,
                "initial_occupied": initial_occupied,
                "initial_empty": initial_empty,
                "total_admitted": total_admitted,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        facilities = ["hospital", "clinic", "medical center", "care facility"]
        items = ["bed", "room", "ward"]
        events = ["flu season", "natural disaster", "major accident", "pandemic"]
        fractions = [Fraction(1, 5), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)]

        facility = rng.choice(facilities)
        item = rng.choice(items)
        event = rng.choice(events)
        frac = rng.choice(fractions)

        # First, generate total as a multiple of the fraction's denominator * 100
        # This ensures total * frac will be integer
        denominator = frac.denominator
        base_total = denominator * 100  # This ensures clean division
        total_multiplier = rng.randint(5, min(int(20 * difficulty), 15))
        total = base_total * total_multiplier  # Will be multiple of both 100 and denominator

        # Calculate maximum daily and period values that won't exceed capacity
        # We need: total * frac + daily * period * 7 < total
        # So: daily * period * 7 < total * (1 - frac)
        remaining_capacity = total * (1 - frac)
        max_weeks = min(int(5 * difficulty), 4)  # Cap period at 4 weeks for reasonable numbers
        period = rng.randint(2, max_weeks)

        # Now calculate maximum daily rate that won't exceed capacity
        # daily < remaining_capacity / (period * 7)
        max_daily = int(remaining_capacity / (period * 7))
        max_daily = (max_daily // 5) * 5  # Round down to multiple of 5
        min_daily = 20  # Minimum daily rate

        if max_daily <= min_daily:
            # If our range is too tight, adjust period down
            period = 2
            max_daily = int(remaining_capacity / (period * 7))
            max_daily = (max_daily // 5) * 5

        daily = 5 * rng.randint(min_daily // 5, max_daily // 5)

        result = generate_from_variables(facility, total, item, frac, event, daily, period)

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


def generate_18(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, game: str, n1: int, n2: int, frac: float) -> dict[str, Any]:
        score2 = int(frac * n1 + n2)
        total = n1 + score2

        question = f"{name1} scored {n1} points in one game of {game}. {name2} scored {n2} more than {frac:.0%} as many as {name1}. How many points did {name1} and {name2} have in total?"

        answer_cot = f"{name1} = {n1} points\n{name2} = {frac} * {n1} + {n2} = {score2} points\n{n1} + {score2} = {total} points\nTogether, {name1} and {name2} scored {total} points.\n#### {total}"

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "name1": name1,
                "name2": name2,
                "game": game,
                "score1": n1,
                "bonus": n2,
                "fraction": frac,
                "score2": score2,
                "total_score": total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = [
            "James",
            "John",
            "Robert",
            "Michael",
            "William",
            "David",
            "Richard",
            "Joseph",
            "Thomas",
            "Charles",
            "Mary",
            "Patricia",
            "Jennifer",
            "Linda",
            "Elizabeth",
            "Barbara",
            "Susan",
            "Jessica",
            "Sarah",
            "Karen",
        ]
        games = ["bowling", "darts", "archery", "basketball", "tennis"]
        fractions = [0.5]  # Could add more fractions if needed

        name1, name2 = rng.sample(names, 2)
        game = rng.choice(games)
        frac = rng.choice(fractions)

        n1 = int(rng.randint(200, int(500 * difficulty)))
        n2 = int(rng.randint(5, int(50 * difficulty)))

        # Ensure fraction calculation results in integer
        while not is_integer(frac * n1):
            n1 = int(rng.randint(200, int(500 * difficulty)))

        result = generate_from_variables(name1, name2, game, n1, n2, frac)

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


def generate_19(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, pan: str, initial_kernels: int, time_interval: int, multiplier_2: int, multiplier_3: int
    ) -> dict[str, Any]:
        second_interval = multiplier_2 * initial_kernels
        third_interval = multiplier_3 * initial_kernels
        fourth_interval = third_interval // 2
        residual = fourth_interval // 4
        total = initial_kernels + second_interval + third_interval + fourth_interval + residual

        question = f"{name} is popping popcorn for a snack. As the {pan} of kernels heats up, the kernels start popping faster. {initial_kernels} pop in the first {time_interval} seconds of cooking, then {multiplier_2} times that amount in the next {time_interval} seconds. The kernels increase to {multiplier_3} times the initial popping rate in the next {time_interval} seconds, but in the final {time_interval} seconds, the popping slows down to half the rate as the past {time_interval} seconds. After {name} takes the {pan} off the heat, a quarter of the number of kernels that popped in the final {time_interval} seconds of cooking also pop from the residual heat. How many pieces of popcorn does {name} have to eat?"

        answer_cot = f"In the second {time_interval} seconds of cooking, {multiplier_2} * {initial_kernels} = {second_interval} kernels pop.\nIn the third {time_interval} seconds, {multiplier_3} * {initial_kernels} = {third_interval} kernels pop.\nIn the final {time_interval} seconds, {third_interval} / 2 = {fourth_interval} kernels pop.\nAfter cooking, the residual heat makes {fourth_interval} / 4 = {residual} kernels pop.\nThus, {name} has {initial_kernels} + {second_interval} + {third_interval} + {fourth_interval} + {residual} = {total} pieces of popcorn to eat.\n#### {total}"

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "name": name,
                "pan": pan,
                "initial_kernels": initial_kernels,
                "time_interval": time_interval,
                "multiplier_2": multiplier_2,
                "multiplier_3": multiplier_3,
                "second_interval": second_interval,
                "third_interval": third_interval,
                "fourth_interval": fourth_interval,
                "residual": residual,
                "total": total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Garrett", "James", "Michael", "David", "John", "Robert", "William"]
        pans = ["pan", "pot", "skillet"]

        name = rng.choice(names)
        pan = rng.choice(pans)

        # Generate numbers ensuring divisibility conditions are met
        initial_kernels = int(rng.randrange(10, int(101 * difficulty), 10))
        time_interval = int(rng.randrange(10, int(31 * difficulty), 2))
        multiplier_2 = rng.randint(2, int(5 * difficulty))

        # Ensure multiplier_3 > multiplier_2 and results in clean division by 8
        while True:
            multiplier_3 = rng.randint(multiplier_2 + 1, int(8 * difficulty))
            if (multiplier_3 * initial_kernels) % 8 == 0:
                break

        result = generate_from_variables(name, pan, initial_kernels, time_interval, multiplier_2, multiplier_3)

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


def generate_20(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, obj: str, surface: str, capacity: int, total: int, num_trays: int
    ) -> dict[str, Any]:
        max_capacity = capacity * num_trays
        leftover = total - max_capacity

        question = f"{name} places {obj}s on the {surface}. Each {surface} can hold {capacity} {obj}s. If he has {total} {obj}s and {num_trays} {surface}s, how many {obj}s won't he be able to place on the {surface}?"

        answer_cot = f"{name} will be able to place a total of {capacity} x {num_trays} = {max_capacity} {obj}s.\nTherefore, there are {total} - {max_capacity} = {leftover} {obj}s that he won't be able to place on the {surface}.\n#### {leftover}"

        return {
            "question": question,
            "answer": format_number(leftover),
            "answer_cot": answer_cot,
            "answer_value": leftover,
            "variables": {
                "name": name,
                "obj": obj,
                "surface": surface,
                "capacity_per_tray": capacity,
                "total_items": total,
                "num_trays": num_trays,
                "max_capacity": max_capacity,
                "leftover": leftover,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["James", "John", "Robert", "Michael", "William", "David", "Richard"]
        objects = ["olive", "almond", "cookie", "cracker", "banana"]
        surfaces = ["plate", "table", "bowl", "tray", "basket"]

        name = rng.choice(names)
        obj = rng.choice(objects)
        surface = rng.choice(surfaces)

        capacity = int(rng.randint(20, int(51 * difficulty)))
        num_trays = int(rng.randint(2, int(7 * difficulty)))

        # Ensure total is greater than max capacity
        max_capacity = capacity * num_trays
        total = max_capacity + int(rng.randint(1, int(20 * difficulty)))

        result = generate_from_variables(name, obj, surface, capacity, total, num_trays)

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


def generate_21(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, length: int, unit_length: str, plant_width: int, space: float, owned: int, currency: str, cost: int
    ) -> dict[str, Any]:
        total_plants = int(length / space)
        plants_to_buy = total_plants - owned
        total_cost = plants_to_buy * cost

        question = (
            f"{name} has a flower bed that is {length} {unit_length} long. "
            f"{name} wants to fill her flower bed with plants. "
            f"Each plant, including the space it needs around it, requires {space} {unit_length} of space. "
            f"{name} already owns {owned} flowers. "
            f"Each flowering plant costs {currency}{cost} at the store, how much money will {name} spend "
            f"at the store to fill up her flower bed?"
        )

        answer_cot = (
            f"Let's solve this step by step:\n"
            f"\n"
            f"1. Calculate how many plants fit in the flower bed:\n"
            f"   - Each plant needs {space} {unit_length} of total space\n"
            f"   - The flower bed is {length} {unit_length} long\n"
            f"   - So {length} {unit_length} / {space} {unit_length} per plant = {total_plants} plants total\n"
            f"\n"
            f"2. Calculate how many plants need to be bought:\n"
            f"   - {name} already has {owned} plants\n"
            f"   - So {total_plants} plants needed - {owned} owned = {plants_to_buy} plants to buy\n"
            f"\n"
            f"3. Calculate total cost:\n"
            f"   - {plants_to_buy} plants × {currency}{cost} per plant = {currency}{total_cost}\n"
            f"\n"
            f"#### {total_cost}"
        )

        return {
            "question": question,
            "answer": format_number(total_cost),
            "answer_cot": answer_cot,
            "answer_value": total_cost,
            "variables": {
                "name": name,
                "bed_length": length,
                "unit": unit_length,
                "plant_width": plant_width,
                "plant_spacing": space,
                "owned_plants": owned,
                "currency": currency,
                "cost_per_plant": cost,
                "total_plants": total_plants,
                "plants_to_buy": plants_to_buy,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte"]
        currencies = ["$", "£", "€"]
        units = ["feet", "meters"]

        name = rng.choice(names_female)
        unit = rng.choice(units)
        currency = rng.choice(currencies)

        # Start with space - we only have a few valid options
        space = rng.choice([1.25, 1.5, 1.75, 2.0])

        # Choose number of total plants first (this is what we really care about)
        total_plants = rng.randint(80, min(160, int(120 * difficulty)))

        # Calculate length to ensure it's divisible by space
        length = total_plants * space

        # Choose owned plants ensuring it's less than total
        owned = rng.randint(10, min(total_plants - 1, int(30 * difficulty)))

        # Simple cost scaling
        cost = rng.randint(3, min(15, int(8 * difficulty)))

        # Plant width is just for display, not used in calculation
        plant_width = rng.randint(2, 8)

        result = generate_from_variables(name, length, unit, plant_width, space, owned, currency, cost)

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


def generate_22(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, property_type: str, budget: int, price: int, brokerage_fee: int, transfer_fee: int
    ) -> dict[str, Any]:
        brokerage_amount = int(price * brokerage_fee / 100)
        transfer_amount = int(price * transfer_fee / 100)
        total_price = price + brokerage_amount + transfer_amount
        difference = total_price - budget

        question = f"{name} is looking for a {property_type} that will not go beyond her ${budget:,} budget. She saw a property that has a selling price of ${price:,}. On top of that, the buyer has to pay a brokerage fee which is {brokerage_fee}% of the selling price, and also the transfer fee that is {transfer_fee}% of the selling price. How much more is the total price of the {property_type} than {name}'s budget?"

        answer_cot = f"The brokerage fee is ${price:,} x {brokerage_fee}/100 = ${brokerage_amount:,}.\nThe transfer fee is ${price:,} x {transfer_fee}/100 = ${transfer_amount:,}.\nThe total price of the {property_type} is ${price:,} + ${brokerage_amount:,} + ${transfer_amount:,} = ${total_price:,}.\nSo, it is ${total_price:,} - ${budget:,} = ${difference:,} more than {name}'s budget.\n#### {difference}"

        return {
            "question": question,
            "answer": f"{difference}",
            "answer_cot": answer_cot,
            "answer_value": difference,
            "variables": {
                "name": name,
                "property_type": property_type,
                "budget": budget,
                "price": price,
                "brokerage_fee": brokerage_fee,
                "transfer_fee": transfer_fee,
                "brokerage_amount": brokerage_amount,
                "transfer_amount": transfer_amount,
                "total_price": total_price,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Mrs. Smith", "Ms. Johnson", "Dr. Patel", "Mrs. Lee"]
        property_types = ["house", "apartment", "condo", "townhouse"]

        name = rng.choice(names)
        property_type = rng.choice(property_types)

        # Scale ranges by difficulty while maintaining integer results
        budget = int(rng.randrange(300000, int(500000 * difficulty), 10000))
        price = int(rng.randrange(250000, budget, 10000))
        brokerage_fee = int(rng.randint(3, 8))
        transfer_fee = int(rng.randint(10, 15))

        # Verify conditions
        while True:
            total_cost = price * (1 + brokerage_fee / 100 + transfer_fee / 100)
            if total_cost > budget + 1 and price * brokerage_fee % 100 == 0 and price * transfer_fee % 100 == 0:
                break
            price = int(rng.randrange(250000, budget, 10000))

        result = generate_from_variables(name, property_type, budget, price, brokerage_fee, transfer_fee)

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


def generate_23(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, task: str, profession: str, hours: int, work_type: str, rate: int, fee: int, currency: str
    ) -> dict[str, Any]:
        lost_income = hours * rate
        savings = lost_income - fee

        question = f"{name} is trying to decide whether to do {task} herself or hire an {profession}. If she does it herself, she'll be able to do {hours} fewer hours of {work_type} work, losing {currency}{rate}/hour in missed income. The {profession} charges {currency}{fee}. How much more money will she have if she hires the {profession}?"

        answer_cot = f"First find the total lost revenue if {name} does {task} herself: {currency}{rate}/hour * {hours} hours = {currency}{lost_income}\nThen subtract the {profession}'s charge to find how much money {name} saves: {currency}{lost_income} - {currency}{fee} = {currency}{savings}\n#### {savings}"

        return {
            "question": question,
            "answer": format_number(savings),
            "answer_cot": answer_cot,
            "answer_value": savings,
            "variables": {
                "name": name,
                "task": task,
                "profession": profession,
                "hours": hours,
                "work_type": work_type,
                "hourly_rate": rate,
                "fee": fee,
                "currency": currency,
                "lost_income": lost_income,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Sophia", "Isabella", "Olivia", "Ava", "Mia", "Emily"]
        tasks = ["her taxes", "her financial planning", "her business accounting"]
        professions = ["accountant", "financial advisor", "tax consultant", "bookkeeper"]
        work_types = ["freelance", "consulting", "part-time", "contract"]
        currencies = ["$", "€", "£"]

        name = rng.choice(names_female)
        task = rng.choice(tasks)
        profession = rng.choice(professions)
        work_type = rng.choice(work_types)
        currency = rng.choice(currencies)

        hours = int(rng.randint(4, int(14 * difficulty)))
        rate = int(rng.randint(20, int(100 * difficulty)))
        fee = int(rng.randint(50, int(200 * difficulty)))

        # Ensure conditions are met
        while hours * rate <= fee:
            hours = int(rng.randint(4, int(14 * difficulty)))
            rate = int(rng.randint(20, int(100 * difficulty)))
            fee = int(rng.randint(50, int(200 * difficulty)))

        result = generate_from_variables(name, task, profession, hours, work_type, rate, fee, currency)

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


def generate_24(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        comet_name: str, name: str, relative: str, orbit_period: int, relative_age: int, multiple: int
    ) -> dict[str, Any]:
        second_viewing_age = relative_age * multiple
        first_viewing_age = second_viewing_age - orbit_period

        question = f"Comet {comet_name} orbits the sun every {orbit_period} years. {name}'s {relative} saw the Comet when he was {relative_age} years old. {name} saw the comet a second time when he was {multiple} times the age his {relative} was when he saw the Comet. How old was {name} when he saw the Comet for the first time?"

        answer_cot = f"{name} saw the Comet for the second time when he was {relative_age} years * {multiple}= {second_viewing_age} years old.\nComet {comet_name} can be seen every {orbit_period} years, so {name} saw the comet for the first time when he was {second_viewing_age} years - {orbit_period} years = {first_viewing_age} years old.\n#### {first_viewing_age}"

        return {
            "question": question,
            "answer": format_number(first_viewing_age),
            "answer_cot": answer_cot,
            "answer_value": first_viewing_age,
            "variables": {
                "comet_name": comet_name,
                "name": name,
                "relative": relative,
                "orbit_period": orbit_period,
                "relative_age": relative_age,
                "multiple": multiple,
                "second_viewing_age": second_viewing_age,
                "first_viewing_age": first_viewing_age,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        comets = ["Halley", "Hale-Bopp", "Hyakutake", "Encke"]
        names = ["William", "James", "John", "Robert", "Michael", "David"]
        relatives = ["dad", "father", "uncle", "grandfather"]
        multiples = ["two", "three", "four"]

        comet_name = rng.choice(comets)
        name = rng.choice(names)
        relative = rng.choice(relatives)
        multiple = rng.choice(multiples)
        multiple_num = {"two": 2, "three": 3, "four": 4}[multiple]

        orbit_period = int(rng.randrange(50, int(101 * difficulty), 5))

        # Calculate valid range for relative_age based on constraints:
        # 1. multiple_num * relative_age < 100 (upper bound)
        # 2. multiple_num * relative_age > orbit_period (lower bound)
        # 3. relative_age must be between 20 and 51*difficulty

        min_age = max(20, (orbit_period + 1) // multiple_num)
        max_age = min(int(51 * difficulty), 99 // multiple_num)

        if min_age <= max_age:
            relative_age = rng.randint(min_age, max_age)
        else:
            # If no valid solution in range, adjust orbit_period down
            orbit_period = (min_age * multiple_num) - 1
            relative_age = min_age

        result = generate_from_variables(comet_name, name, relative, orbit_period, relative_age, multiple_num)

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


def generate_25(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        food: str, calories: int, size: int, servings: int, total_target: int, consumed: int, unit: str
    ) -> dict[str, Any]:

        calories_left = total_target - consumed
        serving_fraction = Fraction(calories_left, calories)
        grams_per_serving = size // servings
        grams_allowed = grams_per_serving * serving_fraction

        question = f"According to its nutritional info, a bag of {food} has {calories} calories per serving. If a {size} {unit} bag has {servings} servings, how many {unit} can you eat if your daily calorie target is {total_target} and you have already consumed {consumed} calories?"

        answer_cot = (
            f"If the total calorie target is {total_target} and I have consumed {consumed} calories then I have {total_target}-{consumed} = {calories_left} calories left to eat\n"
            f"If each serving of {food} has {calories} calories and I only have {calories_left} calories left to eat, then I can only eat {calories_left}/{calories} of a serving = {serving_fraction} of a serving\n"
            f"We also know that a {size} {unit} bag of {food} has {servings} servings, hence each serving has {size} {unit}/{servings} = {grams_per_serving} {unit}\n"
            f"If I can only eat {serving_fraction} of a serving, then I can eat only {grams_per_serving} * {serving_fraction} = {grams_allowed} {unit}\n"
            f"#### {float(grams_allowed)}"
        )

        return {
            "question": question,
            "answer": format_number(float(grams_allowed)),
            "answer_cot": answer_cot,
            "answer_value": float(grams_allowed),
            "variables": {
                "food": food,
                "calories": calories,
                "size": size,
                "servings": servings,
                "total_target": total_target,
                "consumed": consumed,
                "unit": unit,
                "calories_left": calories_left,
                "grams_per_serving": grams_per_serving,
                "serving_fraction": serving_fraction,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        foods = ["popcorn", "breads", "cookies"]
        units = ["grams", "ounces", "oz"]

        food = rng.choice(foods)
        unit = rng.choice(units)

        # First generate servings (will be used as divisor)
        servings = rng.randint(4, min(int(8 * difficulty), 6))

        # Generate size as multiple of servings to ensure clean division
        base_size = rng.randint(4, min(int(16 * difficulty), 12))  # Will be multiplied by servings
        size = base_size * servings  # Guarantees size % servings == 0

        # Generate calories as multiple of 25
        calories = 25 * rng.randint(6, min(int(20 * difficulty), 16))  # 150 to 400 in steps of 25

        # Generate total_target and consumed in a way that ensures integer servings needed
        # First generate consumed as multiple of 25
        consumed = 25 * rng.randint(24, min(int(72 * difficulty), 60))  # 600 to 1500 in steps of 25

        # Calculate what total_target needs to be to ensure integer servings needed
        # We need (total_target - consumed) / calories to be integer when multiplied by (size/servings)
        serving_size = size // servings
        required_multiplier = rng.randint(1, 3)  # How many servings needed
        calories_needed = required_multiplier * calories

        # Calculate total_target to satisfy our conditions
        total_target = consumed + calories_needed
        # Adjust if too low
        if total_target < 1900:
            total_target = 1900 + (calories_needed - (1900 - total_target))
        # Round to multiple of 5
        total_target = 5 * (total_target // 5)

        result = generate_from_variables(food, calories, size, servings, total_target, consumed, unit)

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


def generate_26(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(n: int, ball_type: str, color: str, frac_1: float, frac_2: float) -> dict[str, Any]:
        first_calc = int(n * frac_1)
        final_calc = int(first_calc * frac_2)

        question = f"A juggler can juggle {n} balls. {frac_1:.0%} of the balls are {ball_type} balls, and {frac_2:.0%} of the {ball_type} balls are {color}. How many {color} {ball_type} balls are there?"

        answer_cot = f"{ball_type} balls:{n} * {frac_1}={first_calc}\n{color} {ball_type} balls:{first_calc}*{frac_2}={final_calc} balls\n#### {final_calc}"

        return {
            "question": question,
            "answer": format_number(final_calc),
            "answer_cot": answer_cot,
            "answer_value": final_calc,
            "variables": {
                "total_balls": n,
                "ball_type": ball_type,
                "color": color,
                "fraction_first": frac_1,
                "fraction_second": frac_2,
                "first_calculation": first_calc,
                "final_calculation": final_calc,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        ball_types = ["golf", "tennis"]
        colors = ["blue", "red", "green", "yellow", "white"]
        fractions = [0.5, 0.25, 0.75]

        ball_type = rng.choice(ball_types)
        color = rng.choice(colors)
        frac_1 = rng.choice(fractions)
        frac_2 = rng.choice(fractions)

        # Generate n that ensures integer results
        n = int(rng.randint(10, int(100 * difficulty)))
        while not is_integer(n * frac_1) or not is_integer(n * frac_1 * frac_2):
            n = int(rng.randint(10, int(100 * difficulty)))

        result = generate_from_variables(n, ball_type, color, frac_1, frac_2)

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


def generate_27(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        n: int,
        n_first: int,
        apartments_each: int,
        percent_bigger: int,
        freq: int,
        rate: float,
        currency: str,
    ) -> dict[str, Any]:

        first_two = n_first * apartments_each
        third_complex = int(first_two * percent_bigger / 100)
        total_apartments = first_two + third_complex + first_two
        weekly_visits = total_apartments * freq
        weekly_earnings = weekly_visits * rate

        question = f"{name} collects garbage from {n} different apartment complexes. The first {n_first} have {apartments_each} apartments each and the last one is {percent_bigger}% bigger than the other {n_first} combined. {name} collects garbage {freq} times a week from each place and he gets paid {currency}{rate:.2f} per collection for each apartment. How much money does he make in a week?"

        answer_cot = (
            f"The first {n_first} complexes have {first_two} apartments\n"
            f"The third one has {first_two}*{percent_bigger/100}={third_complex} more apartments than those {n_first} combined\n"
            f"So in total, it has {first_two}+{third_complex}={first_two + third_complex} apartments\n"
            f"So he goes to {first_two + third_complex}+{first_two}={total_apartments} apartments each time\n"
            f"That means he visits {total_apartments}*{freq}={weekly_visits} apartments every week\n"
            f"So he makes {weekly_visits}*{currency}{rate:.2f}={currency}{weekly_earnings} every week\n"
            f"#### {weekly_earnings}"
        )

        return {
            "question": question,
            "answer": format_number(weekly_earnings),
            "answer_cot": answer_cot,
            "answer_value": weekly_earnings,
            "variables": {
                "name": name,
                "num_complexes": n,
                "first_complexes": n_first,
                "apartments_per_complex": apartments_each,
                "percent_increase": percent_bigger,
                "collections_per_week": freq,
                "rate_per_apartment": rate,
                "currency": currency,
                "total_apartments": total_apartments,
                "weekly_visits": weekly_visits,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Michael", "David", "James", "Robert", "William"]
        currencies = ["$", "£", "€"]

        name = rng.choice(names)
        currency = rng.choice(currencies)

        n = rng.randint(3, max(3, int(8 * difficulty)))
        n_first = n - 1
        apartments = int(rng.randrange(100, int(500 * difficulty), 50))
        percent = rng.randrange(20, int(81 * difficulty), 5)
        freq = rng.randint(2, max(2, int(6 * difficulty)))
        rates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        rate = rng.choice(rates)

        # Ensure results are integers
        while not is_integer((n - 1) * apartments * percent / 100):
            apartments = int(rng.randrange(100, int(500 * difficulty), 50))
            percent = rng.randrange(20, int(81 * difficulty), 5)

        result = generate_from_variables(name, n, n_first, apartments, percent, freq, rate, currency)

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


def generate_28(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        item: str,
        price: float,
        percent: float,
        usage: int,
        extra_item: str,
        extra_price: float,
        currency: str,
        unit: str,
    ) -> dict[str, Any]:
        price_increase = price * percent / 100
        new_price = price + price_increase
        weekly_usage = usage * 7
        coffee_cost = new_price * weekly_usage
        total_cost = coffee_cost + extra_price

        question = f"{name} goes to the store to buy some {item}. The normal brand of {item} he buys costs {currency}{price} per {unit}. He had to buy a more expensive brand that costs {int(percent)}% more since his favorite brand was sold out. He decides to buy a week's worth of {item} and he uses {usage} {unit} of {item} per day. He also decided to buy himself a {extra_item} for {currency}{extra_price}. How much did everything cost?"

        answer_cot = f"The {item} he bought was {price}*{percent/100}={price_increase} more expensive per {unit} than what he normally buys\nSo it cost {price}+{price_increase}={new_price} per {unit}\nHe goes through {usage}*7={weekly_usage} {unit}s of {item} a week\nSo he paid {new_price}*{weekly_usage}={coffee_cost} on {item}\nThat means his total bill was {coffee_cost}+{extra_price}={total_cost}\n#### {int(total_cost)}"

        return {
            "question": question,
            "answer": format_number(int(total_cost)),
            "answer_cot": answer_cot,
            "answer_value": int(total_cost),
            "variables": {
                "name": name,
                "item": item,
                "base_price": price,
                "percent_increase": percent,
                "usage_per_day": usage,
                "extra_item": extra_item,
                "extra_price": extra_price,
                "currency": currency,
                "unit": unit,
                "weekly_usage": weekly_usage,
                "total_cost": total_cost,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_male = ["John", "Michael", "David", "James", "Robert", "William", "Richard", "Thomas"]
        items = ["tea", "sugar", "flour", "rice"]
        currencies_sym = ["$", "£", "€"]
        units = ["kilogram", "kg"]
        extra_items = ["cookie", "muffin", "bagel"]

        name = rng.choice(names_male)
        item = rng.choice(items)
        currency = rng.choice(currencies_sym)
        unit = rng.choice(units)
        extra_item = rng.choice(extra_items)

        price = int(rng.randint(3, int(25 * difficulty)))
        percent = int(rng.randint(2, int(10 * difficulty))) * 5
        usage = int(rng.randint(1, int(3 * difficulty)))
        extra_price = int(rng.randint(1, int(5 * difficulty)))

        # Ensure price * percent / 100 is an integer
        while (price * percent / 100) != int(price * percent / 100):
            price = int(rng.randint(3, int(25 * difficulty)))

        result = generate_from_variables(name, item, price, percent, usage, extra_item, extra_price, currency, unit)

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


def generate_29(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, n1: int, n2: int, k1: int, k2: int) -> dict[str, Any]:
        total_puppies = n1 + n2
        spotted_puppies = k1 + k2
        percentage = int(100 * spotted_puppies / total_puppies)

        question = f"{name1}'s dog has {n1} puppies, {k1} of which have spots. {name2}'s dog has {n2} puppies, {k2} of which have spots. What percentage of all the puppies have spots?"

        answer_cot = (
            f"First find the total number of puppies: {n1} puppies + {n2} puppies = {total_puppies} puppies\n"
            f"Then find the total number of puppies with spots: {k1} puppies + {k2} puppies = {spotted_puppies} puppies\n"
            f"Then divide the number of spotted puppies by the total number of puppies and multiply by 100% to find the percentage of puppies with spots: {spotted_puppies} puppies / {total_puppies} puppies * 100% = {percentage}%\n"
            f"#### {percentage}"
        )

        return {
            "question": question,
            "answer": format_number(percentage),
            "answer_cot": answer_cot,
            "answer_value": percentage,
            "variables": {
                "name1": name1,
                "name2": name2,
                "puppies1": n1,
                "puppies2": n2,
                "spotted1": k1,
                "spotted2": k2,
                "total_puppies": total_puppies,
                "total_spotted": spotted_puppies,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = [
            "Jennifer",
            "Michael",
            "Christopher",
            "Jessica",
            "Matthew",
            "Ashley",
            "Joshua",
            "Amanda",
            "Daniel",
            "David",
            "James",
            "Robert",
            "John",
            "Joseph",
        ]

        name1, name2 = rng.sample(names, 2)

        # First generate k1 and k2 as multiples of 10 to make division cleaner
        k1 = 10 * rng.randint(17, min(int(29 * difficulty), 25))  # 170 to 250 in steps of 10
        k2 = 10 * rng.randint(12, min(int(17 * difficulty), 15))  # 120 to 150 in steps of 10

        # Calculate total k and ensure it's a factor of the total n we'll generate
        total_k = k1 + k2  # Will be multiple of 10

        # We want total_n to be between 1350 and 1700
        # But let's make it a bit larger to ensure room for division
        min_total_n = max(1350, total_k * 3)  # Ensure at least 3 times total_k
        max_total_n = 1700

        # Generate multiplier that will give us a valid total_n
        min_multiplier = (min_total_n + total_k - 1) // total_k  # Round up
        max_multiplier = max_total_n // total_k

        if max_multiplier < min_multiplier:
            # If our ranges are too tight, adjust multiplier to ensure valid range
            min_multiplier = 3  # Guarantee at least 3x total_k
            max_multiplier = 4  # But not too large

        multiplier = rng.randint(min_multiplier, max_multiplier)
        total_n = total_k * multiplier

        # Now split total_n into n1 and n2
        # n1 should be larger but not too much larger
        # Instead of using percentages, use fixed ranges
        min_n1 = max(950, total_n // 2)  # At least half of total
        max_n1 = min(int(1050 * difficulty), total_n - 400)  # Leave at least 400 for n2

        # Ensure the range is valid
        if min_n1 >= max_n1:
            # If range is invalid, just do an even split
            n1 = total_n // 2 + 50  # Slightly more than half
        else:
            n1 = 5 * (rng.randint(min_n1 // 5, max_n1 // 5))  # Make it multiple of 5

        n2 = total_n - n1  # Remainder goes to n2

        result = generate_from_variables(name1, name2, n1, n2, k1, k2)

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


def generate_30(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        group: str, n: int, n_1: int, n_2: int, hobby1: str, hobby2: str, hobby3: str, hobby4: str
    ) -> dict[str, Any]:
        n_4 = 2 * n_2  # number that like hobby4 (music)
        n_3 = n - (n_1 + n_2 + n_4)  # number that like hobby3 (video games)

        question = f"A {group} of {n} students has various hobbies. {n_1} like to {hobby1}, {n_2} like to play {hobby2}, and the rest like to either {hobby3} or {hobby4}. How many like to {hobby3} if the number that like to {hobby4} is twice the number that prefer playing {hobby2}?"

        answer_cot = f"The number of students that like to {hobby4} is twice as many as the number who like {hobby2}, so 2 * {n_2} = {n_4}\nThe number that like to {hobby3} is {n} total students - {n_1} {hobby1} - {n_2} {hobby2} - {n_4} {hobby4} = {n_3}\n#### {n_3}"

        return {
            "question": question,
            "answer": format_number(n_3),
            "answer_cot": answer_cot,
            "answer_value": n_3,
            "variables": {
                "group_type": group,
                "total_students": n,
                "hobby1_count": n_1,
                "hobby2_count": n_2,
                "hobby3_count": n_3,
                "hobby4_count": n_4,
                "hobby1": hobby1,
                "hobby2": hobby2,
                "hobby3": hobby3,
                "hobby4": hobby4,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        groups = ["group", "class"]
        hobbies = ["read", "paint", "hike", "dance", "bake", "play video games", "play music"]
        sports = ["basketball", "soccer", "tennis", "baseball", "volleyball"]

        group = rng.choice(groups)
        hobby2 = rng.choice(sports)
        hobby1, hobby3, hobby4 = rng.sample([h for h in hobbies if h not in [hobby2]], 3)

        # Generate numbers that satisfy conditions
        n = int(rng.randint(20, int(200 * difficulty)))
        n_2 = int(rng.randint(2, n // 6))  # Keep n_2 small since we multiply by 2
        n_1 = int(rng.randint(2, n // 3))

        # Verify n_1 + n_2 + (2*n_2) < n
        while n_1 + 3 * n_2 >= n:
            n = int(rng.randint(20, int(200 * difficulty)))
            n_2 = int(rng.randint(2, n // 6))
            n_1 = int(rng.randint(2, n // 3))

        result = generate_from_variables(group, n, n_1, n_2, hobby1, hobby2, hobby3, hobby4)

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


def generate_31(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, fruit: str, total: int, n1: int, n2: int, n3: int, sibling1: str, sibling2: str
    ) -> dict[str, Any]:
        slice2 = n1 + n2
        slice3 = slice2 + n3
        total_eaten = n1 + slice2 + slice3

        question = f"{name} sliced an {fruit} into {total} pieces. She ate {n1} slice, her {sibling1} ate {n2} more than her, and her {sibling2} ate {n3} more than her {sibling1}. How many slices of {fruit} did they all eat?"

        answer_cot = f"Her {sibling1} ate {n1} + {n2} = {slice2} slices.\nHer {sibling2} ate {slice2} + {n3} = {slice3} slices.\nThey ate a total of {n1} + {slice2} + {slice3} = {total_eaten} slices.\n#### {total_eaten}"

        return {
            "question": question,
            "answer": format_number(total_eaten),
            "answer_cot": answer_cot,
            "answer_value": total_eaten,
            "variables": {
                "name": name,
                "fruit": fruit,
                "total_slices": total,
                "first_person_slices": n1,
                "second_person_extra": n2,
                "third_person_extra": n3,
                "sibling1": sibling1,
                "sibling2": sibling2,
                "total_eaten": total_eaten,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Doxa"]
        fruits = ["orange", "pear", "peach", "mango", "kiwi", "apple"]
        siblings = ["brother", "sister", "cousin", "friend"]

        name = rng.choice(names_female)
        fruit = rng.choice(fruits)
        sibling1, sibling2 = rng.sample(siblings, 2)

        # Start with minimum values
        min_n1, max_n1 = 3, int(15 * difficulty)
        min_n2, max_n2 = 5, int(13 * difficulty)
        min_n3, max_n3 = 3, int(14 * difficulty)

        # Calculate minimum required total based on minimum values
        min_total = 3 * min_n1 + 2 * min_n2 + min_n3

        # Generate total that's large enough
        total = int(rng.randint(min_total, int(33 * difficulty)))

        # Generate n1
        n1 = min_n1  # Start with minimum
        remaining = total - (3 * n1)

        # Generate n2 with remaining space
        max_possible_n2 = min(max_n2, remaining // 2)
        n2 = rng.randint(min_n2, max(min_n2, max_possible_n2))
        remaining -= 2 * n2

        # Generate n3 with final remaining space
        max_possible_n3 = min(max_n3, remaining)
        n3 = rng.randint(min_n3, max(min_n3, max_possible_n3))

        result = generate_from_variables(name, fruit, total, n1, n2, n3, sibling1, sibling2)

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


def generate_32(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, periods: int, extra_classes: int, mins_per_class: int, days: int, weekend_fraction: float
    ) -> dict[str, Any]:
        total_classes = periods + extra_classes
        daily_mins = total_classes * mins_per_class
        weekly_mins = daily_mins * days
        weekend_mins = int(weekly_mins * weekend_fraction)
        total_mins = weekly_mins + 2 * weekend_mins
        total_hours = total_mins // 60

        question = f"There are {periods} periods in the day for a normal student but {name} has to take {extra_classes} extra classes. Each class is {mins_per_class} minutes long. He goes to class for {days} days a week. He then spends {weekend_fraction} of his weekday class time minutes each on Saturday and Sunday as extra learning time. How many hours a week does he spend learning?"

        answer_cot = (
            f"He takes {periods}+{extra_classes}={total_classes} classes a day\n"
            f"That means he spends {mins_per_class}*{total_classes}={daily_mins} minutes per day in class\n"
            f"So he spends {daily_mins}*{days}={weekly_mins} minutes a week\n"
            f"That means he spends {weekly_mins}*{weekend_fraction}={weekend_mins} minutes each on Saturday and Sunday\n"
            f"So he spends {weekly_mins}+{weekend_mins}+{weekend_mins}={total_mins} minutes per week\n"
            f"So he spends {total_mins}/60={total_hours} hours per week\n#### {total_hours}"
        )

        return {
            "question": question,
            "answer": format_number(total_hours),
            "answer_cot": answer_cot,
            "answer_value": total_hours,
            "variables": {
                "name": name,
                "periods": periods,
                "extra_classes": extra_classes,
                "mins_per_class": mins_per_class,
                "days": days,
                "weekend_fraction": weekend_fraction,
                "total_classes": total_classes,
                "daily_mins": daily_mins,
                "weekly_mins": weekly_mins,
                "weekend_mins": weekend_mins,
                "total_mins": total_mins,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "James", "William", "Michael", "David", "Richard", "Thomas"]
        fractions = ["1/16", "1/8", "1/4", "1/2"]

        name = rng.choice(names)
        periods = int(rng.randint(5, int(10 * difficulty)))
        extra_classes = int(rng.randint(1, int(5 * difficulty)))
        mins_per_class = int(rng.randrange(30, int(61 * difficulty), 5))
        days = int(rng.randint(4, int(7 * difficulty)))
        weekend_fraction = float(eval(rng.choice(fractions)))

        # Ensure results are integers
        while not (
            ((periods + extra_classes) * mins_per_class * days * weekend_fraction).is_integer()
            and (
                (
                    (periods + extra_classes) * mins_per_class * days
                    + 2 * (periods + extra_classes) * mins_per_class * days * weekend_fraction
                )
                / 60
            ).is_integer()
        ):
            periods = int(rng.randint(5, int(10 * difficulty)))
            extra_classes = int(rng.randint(1, int(5 * difficulty)))
            mins_per_class = int(rng.randrange(30, int(61 * difficulty), 5))
            days = int(rng.randint(4, int(7 * difficulty)))

        result = generate_from_variables(name, periods, extra_classes, mins_per_class, days, weekend_fraction)

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


def generate_33(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, mult: int, n: int) -> dict[str, Any]:
        n_mult = n * mult
        daily_total = n + n_mult
        weekly_total = daily_total * 7

        question = f"{name1} operates the cash register exactly {mult} times as fast as her less-experienced colleague {name2}. Daily, {name2} processes {n} customers. What is the total weekly production for the two if they work all days of the week?"

        answer_cot = (
            f"While {name2} is processing {n} orders in a day, {name1} processes {n} orders/day * {mult} = {n_mult} orders/day.\n"
            f"In a day, they process {n_mult} orders/day + {n} orders/day = {daily_total} orders together.\n"
            f"The total number of orders the two processes in a week is {daily_total} orders/day * 7 days/week = {weekly_total} orders\n"
            f"#### {weekly_total}"
        )

        return {
            "question": question,
            "answer": format_number(weekly_total),
            "answer_cot": answer_cot,
            "answer_value": weekly_total,
            "variables": {
                "name1": name1,
                "name2": name2,
                "multiplier": mult,
                "base_rate": n,
                "fast_rate": n_mult,
                "daily_total": daily_total,
                "weekly_total": weekly_total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names_female = ["Julie", "Sarah", "Emma", "Sophia", "Olivia", "Isabella", "Mia", "Charlotte"]
        multi_times = [2, 3, 4]

        name1, name2 = rng.sample(names_female, 2)
        mult = rng.choice(multi_times)

        # Generate n that satisfies both conditions:
        # 1. n * mult must be integer
        # 2. (n + n * mult) * 7 must be integer
        # Since we're using int() already, first condition is always met
        # For second condition: (n + n * mult) = n(1 + mult) must be divisible by 7

        # Find valid numbers between 30 and 100*difficulty that work
        min_n = 30
        max_n = int(100 * difficulty)

        # Generate numbers that when multiplied by (1 + mult) are divisible by 7
        valid_n = []
        for potential_n in range(min_n, max_n + 1):
            if ((potential_n * (1 + mult)) % 7) == 0:
                valid_n.append(potential_n)

        if not valid_n:
            # If no valid numbers found, adjust n to nearest valid number
            n = ((min_n // 7) * 7) + 7  # Round up to next multiple of 7
        else:
            n = rng.choice(valid_n)

        result = generate_from_variables(name1, name2, mult, n)

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


def generate_34(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(event: str, item: str, family: str, n: int, m: int, total: int) -> dict[str, Any]:
        twins_total = 2 * n
        remaining = total - twins_total
        friends_found = remaining - m

        question = f"The {event} team hid {total} {item}. The {family} twins each found {n} {item}. All the other {item} except {m} were found by their friends. How many {item} did the friends find?"

        answer_cot = f"The {family} twins found, {n} * 2 = {twins_total} {item}.\nThe number that remained hidden was {total} - {twins_total} = {remaining} {item}\nSince {m} {item} were not found, this means the friends found {remaining} - {m} = {friends_found} {item}\n#### {friends_found}"

        return {
            "question": question,
            "answer": format_number(friends_found),
            "answer_cot": answer_cot,
            "answer_value": friends_found,
            "variables": {
                "event": event,
                "item": item,
                "family": family,
                "items_per_twin": n,
                "unfound_items": m,
                "total_items": total,
                "twins_total": twins_total,
                "friends_found": friends_found,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        events = ["Halloween candy hunt", "Treasure hunt", "Scavenger hunt", "Charity fundraiser"]
        items = ["eggs", "treats", "toys", "coins", "tokens", "balls", "candies", "goodies"]
        families = ["Johnson", "Williams", "Mirzakhani", "Lopez", "Garcia", "Lee"]

        event = rng.choice(events)
        item = rng.choice(items)
        family = rng.choice(families)

        total = int(rng.randrange(50, int(201 * difficulty), 10))
        n = int(rng.randint(10, int(51 * difficulty)))
        m = int(rng.randint(5, int(21 * difficulty)))

        # Ensure conditions are met
        while 2 * n + m >= total:
            n = int(rng.randint(10, int(51 * difficulty)))
            m = int(rng.randint(5, int(21 * difficulty)))

        result = generate_from_variables(event, item, family, n, m, total)

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


def generate_35(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        job: str, building: str, room: str, num_rooms: int, num_days: int, time_per_room: int, hours_per_day: int
    ) -> dict[str, Any]:

        # Calculate values ensuring integer percentage
        rooms_per_day = num_rooms // num_days  # Integer division for rooms per day
        minutes_per_day = rooms_per_day * time_per_room  # Calculate minutes from rooms
        hours_cleaning = minutes_per_day / 60
        percentage = int(100 * hours_cleaning / hours_per_day)  # Convert to integer percentage

        question = f"A {job} has to clean a {building} with {num_rooms} {room}s. They have {num_days} days to get it done. It takes them {time_per_room} minutes per {room}. If they work {hours_per_day} hour day, what percentage of their day, on average, is spent cleaning {room}s?"

        answer_cot = (
            f"They have to clean {rooms_per_day} {room}s a day because {num_rooms} / {num_days} = {rooms_per_day}\n"
            f"They spend {minutes_per_day} minutes cleaning per day because {rooms_per_day} x {time_per_room} = {minutes_per_day}\n"
            f"They spend {hours_cleaning} hours a day because {minutes_per_day} / 60 = {hours_cleaning}\n"
            f"They spend {hours_cleaning/hours_per_day} of their day cleaning {room}s because {hours_cleaning} / {hours_per_day} = {hours_cleaning/hours_per_day}\n"
            f"They spend {percentage}% of their day cleaning {room}s because {hours_cleaning/hours_per_day} x 100 = {percentage}\n"
            f"#### {percentage}"
        )

        return {
            "question": question,
            "answer": format_number(percentage),
            "answer_cot": answer_cot,
            "answer_value": percentage,
            "variables": {
                "job": job,
                "building": building,
                "room": room,
                "num_rooms": num_rooms,
                "num_days": num_days,
                "time_per_room": time_per_room,
                "hours_per_day": hours_per_day,
                "rooms_per_day": rooms_per_day,
                "minutes_per_day": minutes_per_day,
                "hours_cleaning": hours_cleaning,
                "percentage": percentage,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        jobs = ["janitor", "cleaner", "maintenance worker"]
        buildings = ["office building", "hospital", "university"]
        rooms = ["room", "floor"]

        job = rng.choice(jobs)
        building = rng.choice(buildings)
        room = rng.choice(rooms)

        # Start with hours_per_day and target percentage
        hours_per_day = rng.randint(6, 17)
        target_percentage = rng.randint(20, 80)

        # Choose rooms_per_day first
        rooms_per_day = rng.randint(2, 8)

        # Calculate exact time_per_room needed
        # If p = 100 * (rooms_per_day * time_per_room / 60) / hours_per_day
        # Then time_per_room = (p * hours_per_day * 60) / (100 * rooms_per_day)
        time_per_room = (target_percentage * hours_per_day * 60) // (100 * rooms_per_day)

        # Adjust time_per_room up if needed to hit target percentage exactly
        while True:
            minutes_per_day = rooms_per_day * time_per_room
            hours_cleaning = minutes_per_day / 60
            actual_percentage = int(100 * hours_cleaning / hours_per_day)
            if actual_percentage == target_percentage:
                break
            time_per_room += 1
            if time_per_room > 300:  # Sanity check
                raise ValueError("Could not find valid time_per_room")

        # Choose num_days and calculate total rooms
        num_days = rng.randint(3, 12)
        num_rooms = rooms_per_day * num_days

        result = generate_from_variables(job, building, room, num_rooms, num_days, time_per_room, hours_per_day)

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


def generate_36(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(n: int, p1: int, r1: int, name: str, s1: str, s2: str, s3: str) -> dict[str, Any]:
        easy_questions = int(n * (p1 / 100))
        other_questions = int(n * (1 - p1 / 100))
        easy_correct = int(easy_questions * (r1 / 100))
        other_correct = int(other_questions * 0.5)
        total_correct = easy_correct + other_correct

        question = f"In a {n}-item quiz, {p1}% of the questions are {s1}, and the rest are equally divided as {s2} and {s3} questions. If {name} is sure to get {r1}% of the {s1} questions, and half of the {s2} and {s3} questions correctly, how many points is she sure to get?"

        answer_cot = (
            f"The {s2} and {s3} questions comprises 100% - {p1}% = {100-p1}% of the quiz.\n"
            f"There are {n} questions x {p1}/100 = {easy_questions} {s1} questions.\n"
            f"There are a total of {n} questions x {100-p1}/100 = {other_questions} {s2} and {s3} questions.\n"
            f"If {name} is sure to get {r1}% of the {s1} questions, then this means she is sure of her {easy_questions} questions x {r1}/100 = {easy_correct} points.\n"
            f"From the {s2} and {s3} questions, she is sure to get half of it correctly so that is {other_questions} questions * 0.5 = {other_correct} points.\n"
            f"Thus, she is sure of getting {easy_correct} points + {other_correct} points = {total_correct} points in her quiz.\n#### {total_correct}"
        )

        return {
            "question": question,
            "answer": format_number(total_correct),
            "answer_cot": answer_cot,
            "answer_value": total_correct,
            "variables": {
                "total_questions": n,
                "easy_percent": p1,
                "easy_correct_percent": r1,
                "student_name": name,
                "easy_subject": s1,
                "medium_subject": s2,
                "hard_subject": s3,
                "easy_questions": easy_questions,
                "other_questions": other_questions,
                "easy_correct": easy_correct,
                "other_correct": other_correct,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        subjects = [
            "history",
            "geography",
            "biology",
            "chemistry",
            "physics",
            "economics",
            "literature",
            "algebra",
            "geometry",
        ]
        names = ["Emma", "Sophia", "Olivia", "Ava", "Isabella", "Mia", "Charlotte"]

        # Generate valid numbers ensuring integer results
        while True:
            n = int(rng.randrange(10, int(151 * difficulty), 10))
            p1 = int(rng.randrange(5, int(71 * difficulty), 5))
            r1 = int(rng.randrange(5, int(101 * difficulty), 5))

            # Check conditions
            if (
                is_integer(n * (p1 / 100))
                and is_integer(n * (p1 / 100) * (r1 / 100))
                and is_integer(n * (1 - (p1 / 100)) * 0.5)
            ):
                break

        name = rng.choice(names)
        s1, s2, s3 = rng.sample(subjects, 3)

        result = generate_from_variables(n, p1, r1, name, s1, s2, s3)

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


def generate_37(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        worker: str, base: int, unit: str, tool1: str, tool2: str, tool3: str, mult1: int, mult2: int, n: int, days: int
    ) -> dict[str, Any]:
        iron_amount = base * mult1
        steel_amount = int(iron_amount * (1 + mult2 / 100))
        daily_total = steel_amount * n
        total_amount = daily_total * days

        question = f"One {worker} can mine {base} {unit} of ore per day with {tool1}. He can mine {mult1} times as much with a {tool2} and {mult2}% more with a {tool3} than with a {tool2}. How many {unit} of ore can {n} {worker}s with {tool3}s mine in a month with {days} days?"

        answer_cot = (
            f"First find how much ore a {worker} can mine with a {tool2}: {base} {unit}/day * {mult1} = {iron_amount} {unit}/day\n"
            f"Then multiply that amount by {100+mult2}% to find how much a {worker} can mine with a {tool3}: {iron_amount} {unit}/day * {100+mult2}% = {steel_amount} {unit}/day\n"
            f"Then multiply the amount one {worker} can mine in a day with a {tool3} by the number of {worker}s: {steel_amount} {unit}/day/{worker} * {n} {worker}s = {daily_total} {unit}/day\n"
            f"Then multiply the daily amount of ore by the number of days to find the total ore mined in a month: {daily_total} {unit}/day * {days} days = {total_amount} {unit}/day\n"
            f"#### {int(total_amount)}"
        )

        return {
            "question": question,
            "answer": format_number(int(total_amount)),
            "answer_cot": answer_cot,
            "answer_value": total_amount,
            "variables": {
                "worker": worker,
                "base_amount": base,
                "unit": unit,
                "tool1": tool1,
                "tool2": tool2,
                "tool3": tool3,
                "mult1": mult1,
                "mult2": mult2,
                "num_workers": n,
                "num_days": days,
                "iron_amount": iron_amount,
                "steel_amount": steel_amount,
                "daily_total": daily_total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        workers = ["miner", "goblin", "gnome", "troll"]
        tools1 = ["bare hands", "basic shovel", "wooden pickaxe"]
        units = ["pounds", "kgs"]
        tools2 = ["nickel pickaxe", "bronze pickaxe", "silver pickaxe"]
        tools3 = ["steel pickaxe", "diamond pickaxe", "mithril pickaxe", "titanium pickaxe"]

        worker = rng.choice(workers)
        tool1 = rng.choice(tools1)
        unit = rng.choice(units)
        tool2 = rng.choice(tools2)
        tool3 = rng.choice(tools3)

        # Generate mult2 first as multiple of 5
        mult2 = 5 * rng.randint(6, min(int(16 * difficulty), 12))  # 30 to 60 in steps of 5

        # Generate mult1
        mult1 = rng.randint(2, min(int(4 * difficulty), 3))  # 2 or 3 to keep numbers manageable

        # For base value, we need base * mult1 * (1 + mult2/100) to be integer
        # Since mult2 is multiple of 5, we need base * mult1 * (100 + mult2) / 100 to be integer
        # Let's generate base that ensures this
        base_candidates = []
        for b in range(5, min(int(20 * difficulty), 15)):
            if is_integer(b * mult1 * (1 + mult2 / 100)):
                base_candidates.append(b)
        if not base_candidates:
            # Fallback: use value that works with our multipliers
            base = 4  # Works with most combinations since it's highly divisible
        else:
            base = rng.choice(base_candidates)

        # Calculate intermediate result
        intermediate = base * mult1 * (1 + mult2 / 100)

        # Now calculate maximum n that keeps total under 100,000
        max_possible_n = 100000 // (intermediate * 31)  # Use 31 for days to be safe
        max_n = min(int(50 * difficulty), max_possible_n, 40)  # Cap at 40 to keep reasonable
        n = rng.randint(20, max(21, max_n))

        # Days can be any value since we calculated n to work with worst case
        days = rng.randint(28, 32)

        result = generate_from_variables(worker, base, unit, tool1, tool2, tool3, mult1, mult2, n, days)

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


def generate_38(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str,
        count: int,
        child_type: str,
        item1: str,
        item2: str,
        item3: str,
        item4: str,
        item5: str,
        n1: int,
        n2: int,
        n3: int,
        n4: int,
        n5: int,
    ) -> dict[str, Any]:

        skeins_per_child = n1 + n2 + n3 + n4 + n5
        total_skeins = count * skeins_per_child

        question = f"{name} is knitting winter wear for her {count} grandchildren. They're {child_type}, so they're all the same size. She wants to make a {item1}, {item2}, {item3}, {item4}, and {item5} for each of them. It takes {n1} skeins of wool to make a {item1}, {n2} for a {item2}, {n3} for a {item3}, {n4} for a pair of {item4}, and {n5} for a pair of {item5}. How many skeins of wool will she need to buy?"

        answer_cot = f"A full outfit for each child will require {n1} skeins per {item1} + {n2} skeins per {item2} + {n3} skeins per {item3} + {n4} skeins per pair of {item4} + {n5} skeins per pair of {item5} = {skeins_per_child} skeins of wool.\nSo to knit outfits for all of her grandchildren, she will need {count} * {skeins_per_child} = {total_skeins} skeins of wool.\n#### {total_skeins}"

        return {
            "question": question,
            "answer": format_number(total_skeins),
            "answer_cot": answer_cot,
            "answer_value": total_skeins,
            "variables": {
                "name": name,
                "count": count,
                "child_type": child_type,
                "items": [item1, item2, item3, item4, item5],
                "skeins_per_item": [n1, n2, n3, n4, n5],
                "skeins_per_child": skeins_per_child,
                "total_skeins": total_skeins,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
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

        result = generate_from_variables(name, count, child_type, item1, item2, item3, item4, item5, n1, n2, n3, n4, n5)

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


def generate_39(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        total: int, grade: str, school_name: str, num_girls: int, day: str, absent_girls: int, absent_boys: int
    ) -> dict[str, Any]:
        num_boys = total - num_girls
        remaining_boys = num_boys - absent_boys

        question = f"There are {total} {grade}-graders at {school_name} School. {num_girls} of them are girls. On {day}, {absent_girls} {grade}-grade girls and {absent_boys} {grade}-grade boys were absent. How many {grade} grade boys were at {school_name} School on {day}?"

        answer_cot = f"Of the {total} {grade} graders, {num_girls} are girls, so {total} students - {num_girls} girls = {num_boys} boys.\nOn {day} there were {num_boys} boys - {absent_boys} absent = {remaining_boys} boys.\n#### {remaining_boys}"

        return {
            "question": question,
            "answer": format_number(remaining_boys),
            "answer_cot": answer_cot,
            "answer_value": remaining_boys,
            "variables": {
                "total_students": total,
                "grade": grade,
                "school_name": school_name,
                "num_girls": num_girls,
                "num_boys": num_boys,
                "day": day,
                "absent_girls": absent_girls,
                "absent_boys": absent_boys,
                "remaining_boys": remaining_boys,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        school_names = ["Maple Grove", "Sunny Hill", "Oak Ridge", "Pine Valley"]
        grades = ["first", "second", "third", "fourth", "fifth"]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        school_name = rng.choice(school_names)
        grade = rng.choice(grades)
        day = rng.choice(days)

        # First generate total ensuring it's large enough for minimum requirements
        # We need at least 4 students (2 of each gender) for absences
        min_total = 40  # Minimum to ensure reasonable gender split and absences
        max_total = min(int(200 * difficulty), 150)  # Cap maximum for reasonable numbers
        total = int(rng.randint(min_total, max_total))

        # Ensure num_girls leaves enough boys (at least 20)
        min_girls = 20  # Minimum girls to allow absences
        max_girls = total - 20  # Leave at least 20 boys
        num_girls = int(rng.randint(min_girls, max_girls))
        num_boys = total - num_girls

        # Calculate absences ensuring ranges are valid
        max_absent_girls = min(num_girls // 4, int(10 * difficulty), 8)  # Cap at 8 or 25% of total
        max_absent_boys = min(num_boys // 4, int(10 * difficulty), 8)  # Cap at 8 or 25% of total

        # Ensure minimum of 2 absences and valid ranges
        absent_girls = int(rng.randint(2, max(3, max_absent_girls)))
        absent_boys = int(rng.randint(2, max(3, max_absent_boys)))

        result = generate_from_variables(total, grade, school_name, num_girls, day, absent_girls, absent_boys)

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


def generate_40(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(item: str, n1: int, c1: str, c2: str, c3: str, p: int) -> dict[str, Any]:
        more_cards = int(p / 100 * n1)
        n2 = n1 + more_cards
        n3 = n1 + n2
        total = n3 + n3

        question = f"In a set of {item}'s cards, there are {n1} {c1} cards, and {p}% more {c2} cards. {c3} cards are as many as the sum of {c1} and {c2} cards. How many cards of all mentioned colors are there?"

        answer_cot = (
            f"There are {p}/100 * {n1} = {more_cards} more {c2} cards than {c1} cards.\n"
            f"Which means there are {n1} + {more_cards} = {n2} {c2} cards.\n"
            f"{c3} cards make up to {n1} + {n2} = {n3} cards.\n"
            f"So in total, there are {n3} + {n3} = {total} cards of different colors.\n"
            f"#### {total}"
        )

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "item": item,
                "n1": n1,
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "p": p,
                "more_cards": more_cards,
                "total": total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        items = ["magician", "artist", "chef", "scientist", "athlete"]
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]

        item = rng.choice(items)
        c1, c2, c3 = rng.sample(colors, 3)

        # Keep generating n1 until we find one that has valid factors
        MAX_ATTEMPTS = 100
        attempts = 0

        while attempts < MAX_ATTEMPTS:
            attempts += 1
            n1 = int(rng.randint(20, int(81 * difficulty)) // 1 * 1)
            factors = [p for p in range(20, min(90, int(100 * difficulty)) + 1) if (p * n1) % 100 == 0]
            if factors:
                break

        if not factors:
            raise ValueError("Could not find valid factors after maximum attempts")

        p = rng.choice(factors)

        result = generate_from_variables(item, n1, c1, c2, c3, p)

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


def generate_41(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, event: str, organization: str, fraction: str, current: int, total: int, currency: str
    ) -> dict[str, Any]:
        fraction = convert_fraction_word(fraction)
        fraction_val = Fraction(fraction)
        org_amount = int(total * fraction_val)
        covered_amount = org_amount + current
        missing_amount = total - covered_amount

        question = f"{name} is raising money for a {event}. He has applied for help from the {organization}, which has decided to cover {fraction} of the cost of the {event}. How much money is {name} missing if he has {currency}{current} and the {event} costs {currency}{total}?"

        answer_cot = f"{name}'s {organization} has decided to pay {total} * {fraction} = {currency}{org_amount} for his {event}.\nIn total {name} has covered {org_amount} + {current} = {currency}{covered_amount} for his {event}\nTherefore, {name} needs {total} - {covered_amount} = {currency}{missing_amount} more for the {event}.\n#### {missing_amount}"

        return {
            "question": question,
            "answer": format_number(missing_amount),
            "answer_cot": answer_cot,
            "answer_value": missing_amount,
            "variables": {
                "name": name,
                "event": event,
                "organization": organization,
                "fraction": fraction,
                "current_amount": current,
                "total_cost": total,
                "currency": currency,
                "org_contribution": org_amount,
                "covered_amount": covered_amount,
            },
        }

    def convert_fraction_word(fraction_str: str) -> str:
        """Convert word fractions to numeric form"""

        # Add fraction word mapping
        FRACTION_WORDS = {
            "half": "1/2",
            "one-half": "1/2",
            "quarter": "1/4",
        }
        return FRACTION_WORDS.get(fraction_str.lower(), fraction_str)

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Michael", "David", "James", "William", "Robert", "Joseph"]
        events = ["field trip", "sports tournament", "conference", "music festival", "science fair"]
        organizations = ["school", "community center", "local charity", "youth club", "parent association"]
        currencies = ["$", "€", "£"]
        fractions = ["half", "1/2", "one-half"]

        name = rng.choice(names)
        event = rng.choice(events)
        organization = rng.choice(organizations)
        currency = rng.choice(currencies)
        fraction = rng.choice(fractions)
        fraction = convert_fraction_word(fraction)
        frac_val = Fraction(fraction)

        # Generate total first
        total = int(rng.randrange(200, int(1000 * difficulty), 10))

        # Calculate organization contribution
        org_contribution = int(total * frac_val)

        # Generate current ensuring total contribution doesn't exceed total
        max_current = total - org_contribution - 50  # Leave buffer
        if max_current < 10:  # If not enough room, adjust total up
            total = int((org_contribution + 60) * 1.5)  # Ensure enough space
            org_contribution = int(total * frac_val)
            max_current = total - org_contribution - 50

        current = int(rng.randrange(10, min(int(200 * difficulty), max_current), 5))

        # Verify conditions
        while not is_integer(total * frac_val) or (org_contribution + current >= total):
            total = int(rng.randrange(200, int(1000 * difficulty), 10))
            org_contribution = int(total * frac_val)
            max_current = total - org_contribution - 50
            if max_current < 10:
                total = int((org_contribution + 60) * 1.5)
                org_contribution = int(total * frac_val)
                max_current = total - org_contribution - 50
            current = int(rng.randrange(10, min(int(200 * difficulty), max_current), 5))

        result = generate_from_variables(name, event, organization, fraction, current, total, currency)

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


def generate_42(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        title: str,
        name: str,
        property_type: str,
        price: int,
        fee1_name: str,
        fee1_percent: int,
        fee2_name: str,
        fee2_percent: int,
        loan: int,
    ) -> dict[str, Any]:

        fee1_amount = price * fee1_percent // 100
        fee2_amount = price * fee2_percent // 100
        total_fees = fee1_amount + fee2_amount + loan
        net_proceeds = price - total_fees

        question = f"{title} {name} sold his {property_type} for ${price}. He paid the {fee1_name} fees that amount to {fee1_percent}% of the selling price and also paid a {fee2_name} fee that is {fee2_percent}% of the selling price. If he also paid ${loan} for the remaining loan amount of the {property_type}, how much is {title} {name}'s net proceeds from selling the {property_type}?"

        answer_cot = (
            f"{title} {name} paid ${price} x {fee1_percent}/100 = ${fee1_amount} for the {fee1_name} fees.\n"
            f"He paid ${price} x {fee2_percent}/100 = ${fee2_amount} for the {fee2_name} fee.\n"
            f"So, {title} {name} paid a total of ${fee1_amount} + ${fee2_amount} + ${loan} = ${total_fees} for the {fee1_name}, {fee2_name}, and loan fees.\n"
            f"Hence, {title} {name}'s net proceeds is ${price} - ${total_fees} = ${net_proceeds}.\n#### {net_proceeds}"
        )

        return {
            "question": question,
            "answer": format_number(net_proceeds),
            "answer_cot": answer_cot,
            "answer_value": net_proceeds,
            "variables": {
                "title": title,
                "name": name,
                "property_type": property_type,
                "price": price,
                "fee1_name": fee1_name,
                "fee1_percent": fee1_percent,
                "fee2_name": fee2_name,
                "fee2_percent": fee2_percent,
                "loan": loan,
                "fee1_amount": fee1_amount,
                "fee2_amount": fee2_amount,
                "total_fees": total_fees,
                "net_proceeds": net_proceeds,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        titles = ["Mr.", "Prof.", "Dr."]
        names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        properties = ["house", "apartment", "condo", "villa", "cottage"]
        fee1_names = ["transfer", "registration", "legal"]
        fee2_names = ["brokerage", "agent", "realtor"]

        title = rng.choice(titles)
        name = rng.choice(names)
        property_type = rng.choice(properties)
        fee1_name = rng.choice(fee1_names)
        fee2_name = rng.choice(fee2_names)

        # Generate fee percentages first
        fee1_percent = rng.randint(1, min(int(5 * difficulty), 10))
        fee2_percent = rng.randint(2, min(int(7 * difficulty), 12))
        total_fee_percent = fee1_percent + fee2_percent

        # Generate loan first, capped to maintain reasonable numbers
        max_loan = min(int(700000 * difficulty), 2000000)  # Cap at 2M for very high difficulty
        loan = 10000 * rng.randint(10, max_loan // 10000)  # Multiples of 10000

        # Calculate minimum price needed to satisfy conditions
        # We need:
        # 1. price > loan
        # 2. price - (price * total_fee_percent/100 + loan) > 1
        # Solving for price:
        # price * (1 - total_fee_percent/100) > loan + 1
        # price > (loan + 1)/(1 - total_fee_percent/100)
        min_price_from_loan = loan + 100000  # Ensure significant gap
        min_price_from_fees = int((loan + 1000) / (1 - total_fee_percent / 100))  # Add buffer
        min_price = max(200000, min_price_from_loan, min_price_from_fees)

        # Generate price that satisfies constraints
        max_price = min(int(1000000 * difficulty), 3000000)  # Cap at 3M for very high difficulty
        if max_price <= min_price:
            max_price = min_price + 500000  # Ensure valid range

        # Round to nearest 10000
        min_price = 10000 * (min_price // 10000)
        max_price = 10000 * (max_price // 10000)

        price = 10000 * rng.randint(min_price // 10000, max_price // 10000)

        result = generate_from_variables(
            title, name, property_type, price, fee1_name, fee1_percent, fee2_name, fee2_percent, loan
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


def generate_43(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(person1: str, item: str, n: int, relation: str, k: int) -> dict[str, Any]:
        other_amount = n - k
        total = n + other_amount

        question = f"A {person1} has {n} {item}s. His {relation} has {k} fewer {item}s than he has. How many {item}s do they have together?"

        answer_cot = f"His {relation} has {n} - {k} = {other_amount} {item}s.\nTogether, they have {n} + {other_amount} = {total} {item}s.\n#### {total}"

        return {
            "question": question,
            "answer": format_number(total),
            "answer_cot": answer_cot,
            "answer_value": total,
            "variables": {
                "person1": person1,
                "item": item,
                "first_amount": n,
                "relation": relation,
                "difference": k,
                "second_amount": other_amount,
                "total": total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        people = ["student", "boy", "child", "kid"]
        items = ["marble", "sticker", "toy", "book", "pencil"]
        relations = ["sister", "brother", "friend", "cousin"]

        person1 = rng.choice(people)
        item = rng.choice(items)
        relation = rng.choice(relations)

        n = int(rng.randint(5, int(21 * difficulty)))
        k = int(rng.randint(2, min(n - 1, int(10 * difficulty))))

        result = generate_from_variables(person1, item, n, relation, k)

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


def generate_44(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        store: str,
        color1: str,
        color2: str,
        color3: str,
        n1: int,
        n2: int,
        n3: int,
        p1: int,
        p2: int,
        p3: int,
        currency: str,
    ) -> dict[str, Any]:
        total1 = n1 * p1
        total2 = n2 * p2
        total3 = n3 * p3
        grand_total = total1 + total2 + total3

        question = f"There are currently {n1} {color1} balls, {n2} {color2} balls, and {n3} {color3} balls in the {store}. {color1} balls cost {currency}{p1}, {color2} balls cost {currency}{p2} and {color3} balls cost {currency}{p3}. How much will the {store} have received after all the balls are sold?"

        answer_cot = f"For the {color1} balls, {n1} balls * {currency}{p1}/ball = {currency}{total1}.\nFor the {color2} balls, {n2} balls * {currency}{p2}/ball = {currency}{total2}.\nFor the {color3} balls, {n3} balls * {currency}{p3}/ball = {currency}{total3}.\nFor all balls, {currency}{total1} + {currency}{total2} + {currency}{total3} = {currency}{grand_total}.\n#### {grand_total}"

        return {
            "question": question,
            "answer": format_number(grand_total),
            "answer_cot": answer_cot,
            "answer_value": grand_total,
            "variables": {
                "store": store,
                "colors": [color1, color2, color3],
                "quantities": [n1, n2, n3],
                "prices": [p1, p2, p3],
                "currency": currency,
                "subtotals": [total1, total2, total3],
                "total": grand_total,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        stores = ["store", "shop", "market", "warehouse"]
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink"]
        currencies = ["$", "€", "£"]

        store = rng.choice(stores)
        color1, color2, color3 = rng.sample(colors, 3)
        currency = rng.choice(currencies)

        n1 = int(rng.randint(3, int(20 * difficulty)))
        n2 = int(rng.randint(5, int(30 * difficulty)))
        n3 = int(rng.randint(15, int(50 * difficulty)))

        p1 = int(rng.randint(5, int(15 * difficulty)))
        p2 = int(rng.randint(3, int(10 * difficulty)))
        p3 = int(rng.randint(2, int(8 * difficulty)))

        result = generate_from_variables(store, color1, color2, color3, n1, n2, n3, p1, p2, p3, currency)

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


def generate_45(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, event: str, food: str, obj: str, package_husband: int, used_items: int, total_remaining: int
    ) -> dict[str, Any]:

        total_items = total_remaining + used_items
        package_size = total_items - package_husband

        question = f"{name} was preparing for a {event} at her house, where she intended to serve {food}. She noticed that she was out of plastic {obj}, so she bought a new package of {obj}. Later, her husband also bought a package of {package_husband} new {obj} and gave them to {name}. While {name} was making the {food}, she used {used_items} of the {obj} to sample her {food}. Later, when she went to set the table, she had a total of {total_remaining} {obj}. How many {obj} were in the package that {name} bought?"

        answer_cot = f"The total number of {obj} from {name} and her husband was {total_remaining}+{used_items}={total_items} {obj}.\nSince the husband bought a package of {package_husband} {obj}, then {name}'s package contained {total_items}-{package_husband}={package_size} {obj}.\n#### {package_size}"

        return {
            "question": question,
            "answer": format_number(package_size),
            "answer_cot": answer_cot,
            "answer_value": package_size,
            "variables": {
                "name": name,
                "event": event,
                "food": food,
                "obj": obj,
                "husband_package": package_husband,
                "used_items": used_items,
                "remaining": total_remaining,
                "total": total_items,
                "package_size": package_size,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["Emma", "Olivia", "Sophia", "Isabella", "Ava", "Mia", "Charlotte"]
        events = ["lunch party", "birthday party", "potluck party", "baby shower", "game night"]
        foods = [
            "roast chicken",
            "grilled salmon",
            "beef stew",
            "vegetable lasagna",
            "stuffed peppers",
            "shrimp scampi",
            "creme brulee",
        ]
        objects = ["spoons", "forks", "plates"]

        name = rng.choice(names)
        event = rng.choice(events)
        food = rng.choice(foods)
        obj = rng.choice(objects)

        package_husband = int(rng.randint(5, int(20 * difficulty)))
        package_size = int(rng.randint(10, int(30 * difficulty)))
        total_items = package_size + package_husband

        # Ensure used_items is less than total_items
        max_used = total_items - 1  # Leave at least 1 item
        used_items = int(rng.randint(3, min(max_used, int(10 * difficulty))))
        total_remaining = total_items - used_items

        # Regenerate if conditions not met
        while total_remaining <= 0:
            package_size = int(rng.randint(10, int(30 * difficulty)))
            total_items = package_size + package_husband
            max_used = total_items - 1
            used_items = int(rng.randint(3, min(max_used, int(10 * difficulty))))
            total_remaining = total_items - used_items

        result = generate_from_variables(name, event, food, obj, package_husband, used_items, total_remaining)

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


def generate_46(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        fruit1: str, fruit2: str, n1: int, n2: int, frac1: float, frac2: float, spill: int, total: int
    ) -> dict[str, Any]:
        n1_after_spill = n1 - spill
        water_fruit1 = n1_after_spill * frac1
        water_fruit2 = n2 * frac2
        total_water = water_fruit1 + water_fruit2

        question = f"I have {n1} liters of {fruit1} drink that are {frac1} water and I wish to add it to {n2} liters of {fruit2} drink that is {frac2} water. But as I pour it, I spill {spill} liter of the {fruit1} drink. How much water is in the remaining {total} liters?"

        answer_cot = f"There are {n2} x {frac2} = {water_fruit2} liters of water from the {n2} liters {fruit2} drink.\nAfter {spill} liter of {fruit1} drink was spilled, there were {n1} - {spill} = {n1_after_spill} liters of {fruit1} drink left.\nOut of the {n1_after_spill} liters, {n1_after_spill} x {frac1} = {water_fruit1} liters are water.\nThus, there are a total of {water_fruit2} + {water_fruit1} = {total_water} liters of water out of the {total} liters.\n#### {int(total_water)}"

        return {
            "question": question,
            "answer": format_number(int(total_water)),
            "answer_cot": answer_cot,
            "answer_value": int(total_water),
            "variables": {
                "fruit1": fruit1,
                "fruit2": fruit2,
                "initial_amount1": n1,
                "initial_amount2": n2,
                "water_fraction1": frac1,
                "water_fraction2": frac2,
                "spilled_amount": spill,
                "total_volume": total,
                "water_content1": water_fruit1,
                "water_content2": water_fruit2,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        fruits = ["apple", "grape", "mango", "peach", "lemon"]
        fractions = {
            "two-thirds": 2 / 3,
            "three-fifths": 3 / 5,
            "three-quarters": 3 / 4,
            "one-half": 1 / 2,
            "four-fifths": 4 / 5,
        }

        fruit1, fruit2 = rng.sample(fruits, 2)
        frac1 = rng.choice(list(fractions.values()))
        frac2 = rng.choice(list(fractions.values()))

        n1 = int(rng.randint(9, int(21 * difficulty)))
        n2 = int(rng.randint(12, int(31 * difficulty)))
        spill = int(rng.randint(3, min(7, n1)))

        # Ensure conditions are met
        while not (n1 + n2 - spill > 0 and is_integer(n2 * frac2) and is_integer((n1 - spill) * frac1)):
            n1 = int(rng.randint(9, int(21 * difficulty)))
            n2 = int(rng.randint(12, int(31 * difficulty)))
            spill = int(rng.randint(3, min(7, n1)))

        total = n1 + n2 - spill

        result = generate_from_variables(fruit1, fruit2, n1, n2, frac1, frac2, spill, total)

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


def generate_47(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, n1: int, c1: float, n2: int, c2: float, c3: int, obj1: str, obj2: str, currency: str
    ) -> dict[str, Any]:
        cost1 = n1 * c1
        cost2 = n2 * c2
        total_cost = cost1 + cost2 + c3

        question = f"{name} went to buy some school supplies. He bought {n1} {obj1} which cost {currency}{c1} each, {n2} {obj2} which cost {currency}{c2} each, and a rim of bond paper which cost {currency}{c3}. How much did {name} spend on everything?"

        answer_cot = (
            f"{name} spent {n1} x {currency}{c1} = {currency}{int(cost1)} for the {obj1}.\n"
            f"He also spent {n2} x {currency}{c2} = {currency}{int(cost2)} for the {obj2}.\n"
            f"Therefore, {name} spent a total of {currency}{int(cost1)} + {currency}{int(cost2)} + {currency}{c3} = {currency}{int(total_cost)} for the school supplies.\n"
            f"#### {int(total_cost)}"
        )

        return {
            "question": question,
            "answer": format_number(int(total_cost)),
            "answer_cot": answer_cot,
            "answer_value": int(total_cost),
            "variables": {
                "name": name,
                "items1_count": n1,
                "item1_cost": c1,
                "items2_count": n2,
                "item2_cost": c2,
                "paper_cost": c3,
                "item1": obj1,
                "item2": obj2,
                "currency": currency,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["John", "Michael", "David", "James", "William", "Robert", "Thomas"]
        items = ["notebooks", "pencils", "erasers", "crayons", "colored pencils", "markers", "rulers", "folders"]
        currencies = ["$", "€", "£"]

        name = rng.choice(names)
        obj1, obj2 = rng.sample(items, 2)
        currency = rng.choice(currencies)

        n1 = int(rng.randrange(6, int(25 * difficulty), 2))
        c1 = round(rng.uniform(2.25, 11.5 * difficulty) * 4) / 4  # Round to nearest 0.25
        n2 = int(rng.randrange(4, int(15 * difficulty), 2))
        c2 = round(rng.uniform(8.25, 19.5 * difficulty) * 4) / 4
        c3 = int(rng.randint(10, int(25 * difficulty)))

        # Ensure conditions are met
        while not is_integer(n1 * c1) or not is_integer(n2 * c2):
            c1 = round(rng.uniform(2.25, 11.5 * difficulty) * 4) / 4
            c2 = round(rng.uniform(8.25, 19.5 * difficulty) * 4) / 4

        result = generate_from_variables(name, n1, c1, n2, c2, c3, obj1, obj2, currency)

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


def generate_48(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        item1: str, item2: str, shop: str, currency: str, price1: int, price2: int, n1: int, n2: int
    ) -> dict[str, Any]:
        total1 = n1 * price1
        total2 = n2 * price2
        diff = total1 - total2

        question = f"A loaf of {item1} at the {shop} costs {currency}{price1}. {item2}s cost {currency}{price2} each. How much more do {n1} loaves of {item1} cost than {n2} {item2}s?"

        answer_cot = (
            f"{n1} loaves of {item1} cost {n1} * {currency}{price1} = {currency}{total1}.\n"
            f"{n2} {item2}s cost {n2} * {currency}{price2} = {currency}{total2}.\n"
            f"The loaves of {item1} cost {currency}{total1} - {currency}{total2} = {currency}{diff} more than the {item2}s.\n"
            f"#### {diff}"
        )

        return {
            "question": question,
            "answer": format_number(diff),
            "answer_cot": answer_cot,
            "answer_value": diff,
            "variables": {
                "item1": item1,
                "item2": item2,
                "shop": shop,
                "currency": currency,
                "price1": price1,
                "price2": price2,
                "n1": n1,
                "n2": n2,
                "total1": total1,
                "total2": total2,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        items1 = ["bread", "sourdough"]
        items2 = ["bagel", "muffin", "croissant", "biscuit"]
        shops = ["bakery", "cafe", "store", "market"]
        currencies = ["$", "£", "€"]

        item1 = rng.choice(items1)
        item2 = rng.choice(items2)
        shop = rng.choice(shops)
        currency = rng.choice(currencies)

        price1 = int(rng.randint(2, int(10 * difficulty)))
        price2 = int(rng.randint(1, int(5 * difficulty)))
        n1 = int(rng.randint(2, int(10 * difficulty)))
        n2 = int(rng.randint(2, int(10 * difficulty)))

        # Ensure condition: n1 * price1 > n2 * price2
        while n1 * price1 <= n2 * price2:
            n1 = int(rng.randint(2, int(10 * difficulty)))
            n2 = int(rng.randint(2, int(10 * difficulty)))

        result = generate_from_variables(item1, item2, shop, currency, price1, price2, n1, n2)

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


def generate_49(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:

    def generate_from_variables(
        name: str, item1: str, item2: str, price1: int, price2: int, total: float, n1: int, percent: int, currency: str
    ) -> dict[str, Any]:
        spent = total * (100 - percent) / 100  # Amount spent
        cost_item1 = n1 * price1 / 100  # Cost of item1
        spent_item2 = spent - cost_item1  # Amount spent on item2
        n2 = int(spent_item2 / (price2 / 100))  # Number of item2 bought

        question = f"The vending machines sell {item1} for {price1} cents and {item2} for {price2} cents. {name} spent {currency}{total} and got {n1} bags of {item1} and had {percent}% of his money left. How many {item2} did he buy?"

        answer_cot = (
            f"{name} spent {currency}{spent} because {total} * {(100-percent)/100} = {spent}\n"
            f"{name} spent {currency}{cost_item1} on {item1} because {n1} x {price1/100} = {cost_item1}\n"
            f"{name} spent {spent_item2} on {item2} because {spent} - {cost_item1} = {spent_item2}\n"
            f"{name} bought {n2} {item2} because {spent_item2} / {price2/100} = {n2}\n"
            f"#### {n2}"
        )

        return {
            "question": question,
            "answer": format_number(n2),
            "answer_cot": answer_cot,
            "answer_value": n2,
            "variables": {
                "name": name,
                "item1": item1,
                "item2": item2,
                "price1": price1,
                "price2": price2,
                "total_spent": total,
                "num_item1": n1,
                "num_item2": n2,
                "percent_change": percent,
                "currency": currency,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> dict[str, Any]:
        names = ["George", "James", "John", "Robert", "Michael", "William"]
        items = ["pretzels", "popcorn", "gum", "cookies", "crackers"]
        currencies = ["$", "£", "€"]

        name = rng.choice(names)
        item1, item2 = rng.sample(items, 2)
        currency = rng.choice(currencies)

        # Generate prices ensuring price2 > price1
        price1 = int(rng.randrange(25, int(100 * difficulty), 5))
        min_price2 = ((price1 + 5) // 5) * 5  # Round up to next multiple of 5
        price2 = int(rng.randrange(min_price2, int(150 * difficulty), 5))

        # Generate percent first (1-10)
        percent = int(rng.randint(1, int(10 * difficulty)))

        # Generate n1 (1-10)
        n1 = int(rng.randint(1, int(10 * difficulty)))

        # Calculate total that satisfies all conditions:
        # 1. total * percent / 100 must be integer
        # 2. (total * (100 - percent) / 100 - n1 * price1 / 100) / (price2 / 100) must be integer

        # First, make total divisible by 100/gcd(percent,100) to satisfy condition 1
        from math import gcd

        # Calculate minimum total that satisfies all conditions
        multiplier = 100 // gcd(100, percent)  # This ensures total * percent / 100 is integer

        # Start with minimum total of 500 rounded up to valid multiple
        min_total = 500
        min_total = ((min_total + multiplier - 1) // multiplier) * multiplier

        # Generate valid total that satisfies both conditions
        valid_totals = []
        max_total = int(1500 * difficulty)

        for t in range(min_total, max_total + 1, multiplier):
            spent_on_item1 = n1 * price1
            remaining_after_percent = t * (100 - percent)
            if (
                remaining_after_percent % 100 == 0
                and (remaining_after_percent - spent_on_item1) % price2 == 0  # Ensure clean division
            ):  # Ensure clean division for item2
                valid_totals.append(t)

        if not valid_totals:
            # Fallback: adjust values to make it work
            total = 1000  # Use a reasonable default
            percent = 10  # Use values that work well with 100
            n1 = 1
        else:
            total = rng.choice(valid_totals)

        result = generate_from_variables(name, item1, item2, price1, price2, total, n1, percent, currency)

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
