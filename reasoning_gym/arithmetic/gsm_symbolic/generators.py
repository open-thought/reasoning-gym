import math
from fractions import Fraction
from random import Random
from typing import Any, Dict


def generate_0(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, food: str, peel_rate: int, batch_size: int, time_per_batch: int, total_amount: int
    ) -> Dict[str, Any]:

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
            "answer": str(total_time),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_1(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, family: str, blocks: int, animals: int, rings: int, total: int
    ) -> Dict[str, Any]:
        bouncy_balls = total - (blocks + animals + rings)

        question = f"When {name} watches her {family}, she gets out a variety of toys for him. The bag of building blocks has {blocks} blocks in it. The bin of stuffed animals has {animals} stuffed animals inside. The tower of stacking rings has {rings} multicolored rings on it. {name} recently bought a tube of bouncy balls, bringing her total number of toys for her {family} up to {total}. How many bouncy balls came in the tube?"

        answer_cot = f"Let T be the number of bouncy balls in the tube.\nAfter buying the tube of balls, {name} has {blocks} + {animals} + {rings} + T = {blocks + animals + rings} + T = {total} toys for her {family}.\nThus, T = {total} - {blocks + animals + rings} = {bouncy_balls} bouncy balls came in the tube.\n#### {bouncy_balls}"

        return {
            "question": question,
            "answer": str(bouncy_balls),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_2(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        teacher: str, total: int, p1: int, p2: int, group1: str, group2: str, group3: str, event: str
    ) -> Dict[str, Any]:
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
            "answer": str(total_leaving),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        teachers = ["Ms. Johnson", "Mr. Smith", "Dr. Lee", "Mrs. Garcia"]
        sports = ["soccer players", "basketball players", "volleyball players", "swimmers"]
        activities = ["dancers", "choir members", "debate team members", "robotics club members"]
        events = ["competition", "tournament", "performance", "meet"]

        teacher = rng.choice(teachers)
        group1 = rng.choice(sports)
        group2, group3 = rng.sample(activities, 2)
        event = rng.choice(events)

        total = int(rng.randint(20, int(150 * difficulty)))
        p1 = int(rng.randint(10, min(50, int(100 * difficulty))))
        p2 = int(rng.randint(15, min(45, int(100 * difficulty))))

        # Ensure conditions are met
        while not (p1 < 100 and p2 < 100 and (total * p1) % 100 == 0 and ((total - total * p1 / 100) * p2) % 100 == 0):
            total = int(rng.randint(20, int(150 * difficulty)))
            p1 = int(rng.randint(10, min(50, int(100 * difficulty))))
            p2 = int(rng.randint(15, min(45, int(100 * difficulty))))

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


def generate_3(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str,
        initial_pals: int,
        lost_pals: int,
        letters_per_week: int,
        pages_per_letter: int,
        minutes_per_page: int,
    ) -> Dict[str, Any]:
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
            "answer": str(hours),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["Mike", "John", "David", "James", "Robert", "William", "Richard"]
        name = rng.choice(names)

        initial_pals = int(rng.randint(5, int(15 * difficulty)))
        lost_pals = int(rng.randint(1, initial_pals - 1))
        letters_per_week = int(rng.randint(2, int(5 * difficulty)))
        pages_per_letter = int(rng.randint(5, int(12 * difficulty)))
        minutes_per_page = int(rng.randint(4, int(15 * difficulty)))

        # Ensure result is in whole hours
        while ((initial_pals - lost_pals) * letters_per_week * pages_per_letter * minutes_per_page) % 60 != 0:
            minutes_per_page = int(rng.randint(4, int(15 * difficulty)))

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


def generate_4(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, items: str, food: str, location: str, container: str, num_jars: int, per_jar: int, per_pan: int
    ) -> Dict[str, Any]:
        total_items = num_jars * per_jar
        num_pans = total_items // per_pan

        question = f"{name} has {num_jars} jars of {items} in her {location}. Each jar of {items} can decorate {per_jar} {food}s. {name} wants to bake enough {food}s to use up all of her {items}. If each {container} holds {per_pan} {food}s, how many {container}s worth of {food}s should she bake?"

        answer_cot = f"She has enough {items} for {num_jars} * {per_jar} = {total_items} {food}s.\nShe needs {total_items} / {per_pan} = {num_pans} {container}s to bake all of the {food}s.\n#### {num_pans}"

        return {
            "question": question,
            "answer": str(num_pans),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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

        # Generate numbers ensuring divisibility
        per_pan = int(rng.randint(6, int(24 * difficulty)))
        per_jar = int(rng.randint(6, int(20 * difficulty)))
        num_jars = int(rng.randint(3, int(15 * difficulty)))

        # Ensure total is divisible by per_pan
        total = num_jars * per_jar
        while total % per_pan != 0:
            num_jars = int(rng.randint(3, int(15 * difficulty)))
            total = num_jars * per_jar

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


def generate_5(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:
        signatures_collected = s1 + s2
        signatures_needed = goal - signatures_collected

        question = f"{name1} and {name2} are sisters from {city} who love collecting signatures from {celebrity_type}. During their {vacation_type} from school, the sisters spend every afternoon collecting signatures. After {n1} weeks, {name1} and {name2} compare their autograph books, counting up the number of signatures each sister has collected. {name1} has {s1} signatures in her book, and {name2} has {s2}. The sisters have {n2} more weeks of {vacation_type}, and they decide they want to reach {goal} signatures between them by the end of the summer. How many signatures do the sisters need to collect to reach their goal?"

        answer_cot = f"{name1} and {name2} have already collected {s1} + {s2} signatures = {signatures_collected} signatures.\nSince their goal is {goal}, they need to collect {goal} - {signatures_collected} signatures. {goal} - {signatures_collected} = {signatures_needed} signatures\n#### {signatures_needed}"

        return {
            "question": question,
            "answer": str(signatures_needed),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_6(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(n_girls: int, place: str, multiplier: int) -> Dict[str, Any]:
        n_boys = n_girls * multiplier
        total_kids = n_girls + n_boys

        question = f"There are {n_girls} girls in the {place}. If there are {multiplier} times the number of boys in the {place}, how many kids are in the {place}?"

        answer_cot = f"There are {n_girls} girls x {multiplier} boys/girl = {n_boys} boys in the {place}.\nIn total there are {n_girls} girls + {n_boys} boys = {total_kids} kids in the {place}\n#### {total_kids}"

        return {
            "question": question,
            "answer": str(total_kids),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_7(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, plants_received: int, plants_per_ledge: int, num_ledges: int, plants_to_give: int
    ) -> Dict[str, Any]:

        initial_plants = plants_per_ledge * num_ledges
        total_plants = initial_plants + plants_received
        plants_given = num_ledges * plants_to_give
        remaining_plants = total_plants - plants_given

        question = f"{name} is an avid gardener. Yesterday, she received {plants_received} new potted plants from her favorite plant nursery. She already has {plants_per_ledge} potted plants on each of the {num_ledges} window ledges of her large country home. Feeling generous, she has decided that she will give {plants_to_give} potted plant from each ledge to friends and family tomorrow. How many potted plants will {name} remain with?"

        answer_cot = f"Yesterday, before receiving the plants, {name} had {num_ledges}*{plants_per_ledge} = {initial_plants} potted plants\nAfter receiving an additional {plants_received} plants, she therefore had a total of {initial_plants} + {plants_received} = {total_plants} potted plants\nTomorrow, {name}'s plant giveaway will be {num_ledges}*{plants_to_give} = {plants_given} potted plants.\nShe will therefore remain with {total_plants} - {plants_given} = {remaining_plants} potted plants.\n#### {remaining_plants}"

        return {
            "question": question,
            "answer": str(remaining_plants),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_8(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, drink: str, sugar_ratio: int, water_ratio: int, total_items: int
    ) -> Dict[str, Any]:
        total_ratio = sugar_ratio + water_ratio
        sugar_amount = (sugar_ratio * total_items) // total_ratio

        question = f"{name} makes {drink} using teaspoons of sugar and cups of water in the ratio of {sugar_ratio}:{water_ratio}. If she used a total of {total_items} teaspoons of sugar and cups of water, calculate the number of teaspoonfuls of sugar she used."

        answer_cot = f"The total ratio representing the ingredients she used to make the {drink} is {sugar_ratio}+{water_ratio} = {total_ratio}\nSince the fraction representing the number of teaspoons she used is {sugar_ratio}/{total_ratio}, she used {sugar_ratio}/{total_ratio}*{total_items} = {sugar_amount}\n#### {sugar_amount}"

        return {
            "question": question,
            "answer": str(sugar_amount),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_9(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

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
            "answer": str(remaining),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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

        num_bills = int(rng.randint(1, int(10 * difficulty)))
        num_items1 = int(rng.randint(2, int(15 * difficulty)))
        num_items2 = int(rng.randint(2, int(10 * difficulty)))
        price1 = int(rng.randint(1, int(10 * difficulty)))
        price2 = int(rng.randint(1, int(10 * difficulty)))

        # Ensure total cost doesn't exceed available money
        while (num_items1 * price1 + num_items2 * price2) > (num_bills * bill_value):
            num_items1 = int(rng.randint(2, int(15 * difficulty)))
            num_items2 = int(rng.randint(2, int(10 * difficulty)))

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


def generate_10(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name1: str, name2: str, age1: int, years: int, relation_type: str, mult: int
    ) -> Dict[str, Any]:
        future_age = age1 * mult
        current_age = future_age - years

        question = f"{name1} is {age1} years old. In {years} years his {relation_type} {name2} will be {mult} times as old as {name1} is now. How old is {name2} right now?"

        answer_cot = f"{years} years from now {name2} will be {age1}*{mult}={future_age}.\nRight now {name2} is {future_age}-{years}={current_age} years old.\n#### {current_age}"

        return {
            "question": question,
            "answer": str(current_age),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_11(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name: str, food1: str, food2: str, mult: int, n: int, m: int, k: int) -> Dict[str, Any]:
        initial_food1 = n * mult  # initial corn
        initial_total = initial_food1 + n  # initial total
        bought_food1 = m - k  # bought corn
        bought_total = bought_food1 + m  # total bought
        final_total = initial_total + bought_total  # final total

        question = f"At {name}'s house, there is {mult} times as much {food1} as {food2}. He has a total of {n} {food2} in his house. {name} bought {m} more {food2} at the store and {k} fewer {food1} than the number of {food2}. Find the combined total of the number of {food1} and {food2} {name} has in the house?"

        answer_cot = f"Before buying any {food1} and {food2}, {name} had {mult} times as many {food1} as {food2}, which is {n} {food2} * {mult} {food1}/{food2} = {initial_food1} {food1}\nThe total number of {food1} and {food2} that {name} had before is {initial_food1} {food1} + {n} {food2} = {initial_total} items\nWhen he bought {k} fewer {food1} than {food2}, he bought {m} {food1} - {k} {food1} = {bought_food1} {food1}\nIn total, he bought {bought_food1} {food1} + {m} {food2} = {bought_total} items\nAfter the purchases, {name} has {initial_total} items + {bought_total} items = {final_total} total {food1} and {food2} combined.\n#### {final_total}"

        return {
            "question": question,
            "answer": str(final_total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["Allan", "John", "Michael", "David", "James", "Robert", "William"]
        foods = ["corn", "apple", "banana", "orange", "pear", "grape", "fig", "persimmon", "plum", "kiwi"]
        multipliers = ["twice", "three times", "four times"]

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


def generate_12(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, game1: str, game2: str, period: str, time1: int, time2: int, num1: int, num2: int
    ) -> Dict[str, Any]:
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
            "answer": str(total_time),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_13(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        park_name: str, unit: str, length1: int, length2: int, speed1: int, speed2: int
    ) -> Dict[str, Any]:
        time1 = length1 // speed1
        time2 = length2 // speed2
        time_diff = time1 - time2

        question = f"The biggest waterslide at {park_name} is {length1} {unit} long, and people slide down at {speed1} {unit}/minute. The second biggest waterslide is {length2} {unit} long, but steeper, so people slide down at {speed2} {unit}/minute. How much longer does it take to ride the biggest slide compared to the second biggest slide?"

        answer_cot = f"First find the ride length of the biggest slide: {length1} {unit} / {speed1} {unit}/minute = {time1} minutes\nThen find the ride length of the second biggest slide: {length2} {unit} / {speed2} {unit}/minute = {time2} minutes\nThen subtract the ride length of the second longest slide from the longest slide to find the difference: {time1} minutes - {time2} minutes = {time_diff} minutes\n#### {time_diff}"

        return {
            "question": question,
            "answer": str(time_diff),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        parks = ["Splash World", "Aqua Adventure", "Water Wonderland", "Neptunes Kingdom"]
        units = ["yards", "meters"]

        park_name = rng.choice(parks)
        unit = rng.choice(units)

        length1 = int(rng.randrange(250, int(401 * difficulty), 10))
        length2 = int(rng.randrange(200, int(301 * difficulty), 10))
        speed1 = int(rng.randrange(40, int(81 * difficulty), 5))
        speed2 = int(rng.randrange(60, int(101 * difficulty), 5))

        # Ensure conditions are met
        while (
            length1 <= length2
            or speed2 <= speed1
            or length1 % speed1 != 0
            or length2 % speed2 != 0
            or (length1 // speed1) <= (length2 // speed2)
        ):
            length1 = int(rng.randrange(250, int(401 * difficulty), 10))
            length2 = int(rng.randrange(200, int(301 * difficulty), 10))
            speed1 = int(rng.randrange(40, int(81 * difficulty), 5))
            speed2 = int(rng.randrange(60, int(101 * difficulty), 5))

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


def generate_14(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, day1: str, day2: str, day3: str, time1: int, time2: int, mult: int
    ) -> Dict[str, Any]:
        combined_time = time1 + time2
        target_time = combined_time * mult

        question = f"On {day3}, {name} wants to exercise for {mult} the amount of time he did on {day2} and {day1} combined. On {day1} he exercised for {time1} minutes. On {day2} he exercised for {time2} minutes. How many minutes does he have to exercise on {day3} to reach his goal?"

        answer_cot = f"On {day1} and {day2} he exercised a total of {combined_time} minutes because {time1} + {time2} = {combined_time}\nOn {day3} he has to exercise for {target_time} minutes because {combined_time} x {mult} = {target_time}\n#### {target_time}"

        return {
            "question": question,
            "answer": str(target_time),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["Peter", "John", "Michael", "David", "James", "Robert", "William"]
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        multipliers = [2, 3, 4]

        name = rng.choice(names)
        day1, day2, day3 = rng.sample(weekdays, 3)
        mult = rng.choice(multipliers)

        time1 = int(rng.randint(10, int(60 * difficulty)))
        time2 = int(rng.randint(10, int(60 * difficulty)))

        # Check conditions
        while (time1 + time2) <= 0 or ((time1 + time2) * mult / 60) >= 14:
            time1 = int(rng.randint(10, int(60 * difficulty)))
            time2 = int(rng.randint(10, int(60 * difficulty)))

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


def generate_15(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

        shorts_price = price1 + price2
        shoes_price = price3 // 2
        socks_price = price4 - discount
        total = price1 + shorts_price + shoes_price + socks_price

        question = f"{name} qualified for a spot on the {sport} team, so she went shopping for some athletic gear. She bought a {item1} for {currency}{price1}, a pair of {sport} {item2} for {currency}{price2} more than the {item1} cost, and a pair of {item3} that were originally {currency}{price3} but were on sale for half price. She had a coupon for {currency}{discount} off the package of {currency}{price4} athletic {item4} that she also bought. How much did she spend on athletic gear?"

        answer_cot = f"The {item2} were {currency}{price2} more than the {item1}, so they cost {currency}{price2} + {currency}{price1} = {currency}{shorts_price}.\nHer {item3} were half the original {currency}{price3} price, so they cost {currency}{price3} / 2 = ${shoes_price}.\nWith her coupon, the {item4} were {currency}{price4} - {currency}{discount} = {currency}{socks_price}.\nThe {item1}, {item2}, {item3}, and {item4} together cost {currency}{price1} + {currency}{shorts_price} + {currency}{shoes_price} + {currency}{socks_price} = {currency}{total}.\n#### {total}"

        return {
            "question": question,
            "answer": str(total),
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


def generate_16(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name1: str, name2: str, name3: str, platform: str, mult1: int, mult2: int, n: int
    ) -> Dict[str, Any]:
        base_friends = n // mult1  # Dorothy's friends
        charlie_friends = n  # Charlie's friends
        james_friends = base_friends * mult2  # James's friends

        question = f"{name1} has {mult1} times as many {platform} friends as {name2}. {name3} has {mult2} times as many friends on {platform} as {name2}. If {name1} has {n} friends on {platform}, how many {platform} friends does {name3} have?"

        answer_cot = f"{name2} has {n} / {mult1} = {base_friends} {platform} friends.\n{name3} has {mult2} * {base_friends} = {james_friends} {platform} friends.\n#### {james_friends}"

        return {
            "question": question,
            "answer": str(james_friends),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_17(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        facility: str, total: int, item: str, frac: Fraction, event: str, daily: int, period: int
    ) -> Dict[str, Any]:
        initial_occupied = int(total * frac)
        initial_empty = total - initial_occupied
        weekly_admitted = daily * 7
        total_admitted = weekly_admitted * period
        final_empty = initial_empty - total_admitted

        question = f"A {facility} has a capacity of {total} {item}s with {frac} occupied. Due to the {event}, {daily} patients are admitted into the {facility} each day. Calculate the total number of unoccupied {item}s in the {facility} after {period} weeks."

        answer_cot = f"If {frac} of the total capacity of the {facility} {item}s is occupied, it means {frac} * {total} = {initial_occupied} {item}s have patients using them.\nThe total number of {item}s in the {facility} without new admissions is {total} {item}s - {initial_occupied} {item}s = {initial_empty} {item}s.\nIf {daily} people are admitted each day, the total number of patients in the {facility} after one week is {daily} patients/day * 7 days/week = {weekly_admitted} patients.\nAfter {period} weeks, the total number of patients admitted into the {facility} is {weekly_admitted} patients/week * {period} weeks = {total_admitted} patients, who each use one {item}.\nIf there were {initial_empty} unoccupied {item}s in the {facility} before the new admissions, the total number is reduced to {initial_empty} {item}s - {total_admitted} {item}s = {final_empty} unoccupied {item}s.\n#### {final_empty}"

        return {
            "question": question,
            "answer": str(final_empty),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        facilities = ["hospital", "clinic", "medical center", "care facility"]
        items = ["bed", "room", "ward"]
        events = ["flu season", "natural disaster", "major accident", "pandemic"]
        fractions = [Fraction(1, 5), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)]

        facility = rng.choice(facilities)
        item = rng.choice(items)
        event = rng.choice(events)
        frac = rng.choice(fractions)

        total = int(rng.randrange(500, int(2000 * difficulty), 100))
        daily = int(rng.randrange(20, int(100 * difficulty), 5))
        period = int(rng.randint(2, int(5 * difficulty)))

        # Ensure conditions are met
        while not (total * frac).is_integer() or total * frac + daily * period * 7 >= total:
            total = int(rng.randrange(500, int(2000 * difficulty), 100))
            daily = int(rng.randrange(20, int(100 * difficulty), 5))
            period = int(rng.randint(2, int(5 * difficulty)))

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


def generate_18(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, game: str, n1: int, n2: int, frac: float) -> Dict[str, Any]:
        score2 = int(frac * n1 + n2)
        total = n1 + score2

        question = f"{name1} scored {n1} points in one game of {game}. {name2} scored {n2} more than {frac:.0%} as many as {name1}. How many points did {name1} and {name2} have in total?"

        answer_cot = f"{name1} = {n1} points\n{name2} = {frac} * {n1} + {n2} = {score2} points\n{n1} + {score2} = {total} points\nTogether, {name1} and {name2} scored {total} points.\n#### {total}"

        return {
            "question": question,
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
        while not float(frac * n1).is_integer():
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


def generate_19(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, pan: str, initial_kernels: int, time_interval: int, multiplier_2: int, multiplier_3: int
    ) -> Dict[str, Any]:
        second_interval = multiplier_2 * initial_kernels
        third_interval = multiplier_3 * initial_kernels
        fourth_interval = third_interval // 2
        residual = fourth_interval // 4
        total = initial_kernels + second_interval + third_interval + fourth_interval + residual

        question = f"{name} is popping popcorn for a snack. As the {pan} of kernels heats up, the kernels start popping faster. {initial_kernels} pop in the first {time_interval} seconds of cooking, then {multiplier_2} times that amount in the next {time_interval} seconds. The kernels increase to {multiplier_3} times the initial popping rate in the next {time_interval} seconds, but in the final {time_interval} seconds, the popping slows down to half the rate as the past {time_interval} seconds. After {name} takes the {pan} off the heat, a quarter of the number of kernels that popped in the final {time_interval} seconds of cooking also pop from the residual heat. How many pieces of popcorn does {name} have to eat?"

        answer_cot = f"In the second {time_interval} seconds of cooking, {multiplier_2} * {initial_kernels} = {second_interval} kernels pop.\nIn the third {time_interval} seconds, {multiplier_3} * {initial_kernels} = {third_interval} kernels pop.\nIn the final {time_interval} seconds, {third_interval} / 2 = {fourth_interval} kernels pop.\nAfter cooking, the residual heat makes {fourth_interval} / 4 = {residual} kernels pop.\nThus, {name} has {initial_kernels} + {second_interval} + {third_interval} + {fourth_interval} + {residual} = {total} pieces of popcorn to eat.\n#### {total}"

        return {
            "question": question,
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_20(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, obj: str, surface: str, capacity: int, total: int, num_trays: int
    ) -> Dict[str, Any]:
        max_capacity = capacity * num_trays
        leftover = total - max_capacity

        question = f"{name} places {obj}s on the {surface}. Each {surface} can hold {capacity} {obj}s. If he has {total} {obj}s and {num_trays} {surface}s, how many {obj}s won't he be able to place on the {surface}?"

        answer_cot = f"{name} will be able to place a total of {capacity} x {num_trays} = {max_capacity} {obj}s.\nTherefore, there are {total} - {max_capacity} = {leftover} {obj}s that he won't be able to place on the {surface}.\n#### {leftover}"

        return {
            "question": question,
            "answer": str(leftover),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_21(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, length: int, unit_length: str, plant_width: int, space: float, owned: int, currency: str, cost: int
    ) -> Dict[str, Any]:
        total_plants = int(length / space)
        plants_to_buy = total_plants - owned
        total_cost = plants_to_buy * cost

        question = f"{name} has a flower bed that is {length} {unit_length} long. {name} wants to fill her flower bed with plants. {name}'s flowers grow {plant_width} inches wide so she needs to leave {space} {unit_length} between every plant. {name} already owns {owned} flowers. Each flowering plant costs {currency}{cost} at the store, how much money will {name} spend at the store to fill up her flower bed?"

        answer_cot = f"{name}'s flower bed is {length} {unit_length} / {space} {unit_length} per plant = {total_plants} plants needed.\n{name} needs to buy {total_plants} plants - {owned} plants = {plants_to_buy} plants needed to purchase.\n{name} will spend {plants_to_buy} plants * {currency}{cost} = {currency}{total_cost}.\n#### {total_cost}"

        return {
            "question": question,
            "answer": str(total_cost),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte"]
        currencies = ["$", "£", "€"]
        units = ["feet", "meters"]

        name = rng.choice(names_female)
        unit = rng.choice(units)
        currency = rng.choice(currencies)

        length = int(rng.randint(110, int(220 * difficulty)))
        plant_width = int(rng.randint(2, int(8 * difficulty)))
        space = round(rng.uniform(1.25, 2.0) * difficulty, 2)
        owned = int(rng.randint(10, int(30 * difficulty)))
        cost = int(rng.randint(3, int(15 * difficulty)))

        # Ensure conditions are met
        while not (plant_width * 3 < length and plant_width < space and length % space == 0 and length / space > owned):
            length = int(rng.randint(110, int(220 * difficulty)))
            space = round(rng.uniform(1.25, 2.0) * difficulty, 2)
            owned = int(rng.randint(10, int(30 * difficulty)))

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


def generate_22(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, property_type: str, budget: int, price: int, brokerage_fee: int, transfer_fee: int
    ) -> Dict[str, Any]:
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_23(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, task: str, profession: str, hours: int, work_type: str, rate: int, fee: int, currency: str
    ) -> Dict[str, Any]:
        lost_income = hours * rate
        savings = lost_income - fee

        question = f"{name} is trying to decide whether to do {task} herself or hire an {profession}. If she does it herself, she'll be able to do {hours} fewer hours of {work_type} work, losing {currency}{rate}/hour in missed income. The {profession} charges {currency}{fee}. How much more money will she have if she hires the {profession}?"

        answer_cot = f"First find the total lost revenue if {name} does {task} herself: {currency}{rate}/hour * {hours} hours = {currency}{lost_income}\nThen subtract the {profession}'s charge to find how much money {name} saves: {currency}{lost_income} - {currency}{fee} = {currency}{savings}\n#### {savings}"

        return {
            "question": question,
            "answer": str(savings),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_24(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        comet_name: str, name: str, relative: str, orbit_period: int, relative_age: int, multiple: int
    ) -> Dict[str, Any]:
        second_viewing_age = relative_age * multiple
        first_viewing_age = second_viewing_age - orbit_period

        question = f"Comet {comet_name} orbits the sun every {orbit_period} years. {name}'s {relative} saw the Comet when he was {relative_age} years old. {name} saw the comet a second time when he was {multiple} times the age his {relative} was when he saw the Comet. How old was {name} when he saw the Comet for the first time?"

        answer_cot = f"{name} saw the Comet for the second time when he was {relative_age} years * {multiple}= {second_viewing_age} years old.\nComet {comet_name} can be seen every {orbit_period} years, so {name} saw the comet for the first time when he was {second_viewing_age} years - {orbit_period} years = {first_viewing_age} years old.\n#### {first_viewing_age}"

        return {
            "question": question,
            "answer": str(first_viewing_age),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
        relative_age = int(rng.randint(20, int(51 * difficulty)))

        # Ensure conditions are met
        while (
            multiple_num * relative_age >= 100
            or multiple_num * relative_age <= orbit_period
            or (multiple_num * relative_age - orbit_period) % 1 != 0
        ):
            relative_age = int(rng.randint(20, int(51 * difficulty)))

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


def generate_25(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        food: str, calories: int, size: int, servings: int, total_target: int, consumed: int, unit: str
    ) -> Dict[str, Any]:

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
            "answer": str(float(grams_allowed)),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        foods = ["popcorn", "breads", "cookies"]
        units = ["grams", "ounces", "oz"]

        food = rng.choice(foods)
        unit = rng.choice(units)

        calories = int(rng.randrange(150, int(500 * difficulty), 25))
        size = int(rng.randrange(100, int(400 * difficulty), 25))
        servings = int(rng.randint(4, int(8 * difficulty)))
        total_target = int(rng.randrange(1900, int(2500 * difficulty), 5))
        consumed = int(rng.randrange(600, int(1800 * difficulty), 25))

        # Ensure conditions are met
        while (
            consumed >= total_target
            or not (size % servings == 0)
            or not ((size // servings) * Fraction(total_target - consumed, calories)).denominator == 1
        ):
            calories = int(rng.randrange(150, int(500 * difficulty), 25))
            size = int(rng.randrange(100, int(400 * difficulty), 25))
            servings = int(rng.randint(4, int(8 * difficulty)))
            total_target = int(rng.randrange(1900, int(2500 * difficulty), 5))
            consumed = int(rng.randrange(600, int(1800 * difficulty), 25))

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


def generate_26(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(n: int, ball_type: str, color: str, frac_1: float, frac_2: float) -> Dict[str, Any]:
        first_calc = int(n * frac_1)
        final_calc = int(first_calc * frac_2)

        question = f"A juggler can juggle {n} balls. {frac_1:.0%} of the balls are {ball_type} balls, and {frac_2:.0%} of the {ball_type} balls are {color}. How many {color} {ball_type} balls are there?"

        answer_cot = f"{ball_type} balls:{n} * {frac_1}={first_calc}\n{color} {ball_type} balls:{first_calc}*{frac_2}={final_calc} balls\n#### {final_calc}"

        return {
            "question": question,
            "answer": str(final_calc),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        ball_types = ["golf", "tennis"]
        colors = ["blue", "red", "green", "yellow", "white"]
        fractions = [0.5, 0.25, 0.75]

        ball_type = rng.choice(ball_types)
        color = rng.choice(colors)
        frac_1 = rng.choice(fractions)
        frac_2 = rng.choice(fractions)

        # Generate n that ensures integer results
        n = int(rng.randint(10, int(100 * difficulty)))
        while not (n * frac_1).is_integer() or not (n * frac_1 * frac_2).is_integer():
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


def generate_27(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str,
        n: int,
        n_first: int,
        apartments_each: int,
        percent_bigger: int,
        freq: int,
        rate: float,
        currency: str,
    ) -> Dict[str, Any]:

        first_two = n_first * apartments_each
        third_complex = int(first_two * percent_bigger / 100)
        total_apartments = first_two + third_complex + first_two
        weekly_visits = total_apartments * freq
        weekly_earnings = int(weekly_visits * rate)

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
            "answer": str(weekly_earnings),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
        while not ((n - 1) * apartments * percent / 100).is_integer():
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


def generate_28(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:
        price_increase = price * percent / 100
        new_price = price + price_increase
        weekly_usage = usage * 7
        coffee_cost = new_price * weekly_usage
        total_cost = coffee_cost + extra_price

        question = f"{name} goes to the store to buy some {item}. The normal brand of {item} he buys costs {currency}{price} per {unit}. He had to buy a more expensive brand that costs {int(percent)}% more since his favorite brand was sold out. He decides to buy a week's worth of {item} and he uses {usage} {unit} of {item} per day. He also decided to buy himself a {extra_item} for {currency}{extra_price}. How much did everything cost?"

        answer_cot = f"The {item} he bought was {price}*{percent/100}={price_increase} more expensive per {unit} than what he normally buys\nSo it cost {price}+{price_increase}={new_price} per {unit}\nHe goes through {usage}*7={weekly_usage} {unit}s of {item} a week\nSo he paid {new_price}*{weekly_usage}={coffee_cost} on {item}\nThat means his total bill was {coffee_cost}+{extra_price}={total_cost}\n#### {int(total_cost)}"

        return {
            "question": question,
            "answer": str(int(total_cost)),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_29(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, n1: int, n2: int, k1: int, k2: int) -> Dict[str, Any]:
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
            "answer": str(percentage),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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

        # Scale ranges by difficulty but ensure values remain integers
        n1 = int(rng.randrange(950, int(1050 * difficulty), 5))
        n2 = int(rng.randrange(400, int(650 * difficulty), 5))
        k1 = int(rng.randrange(170, int(290 * difficulty), 10))
        k2 = int(rng.randrange(120, int(170 * difficulty), 10))

        # Ensure conditions are met
        while (k1 + k2) >= (n1 + n2) or (n1 + n2) % (k1 + k2) != 0:
            n1 = int(rng.randrange(950, int(1050 * difficulty), 5))
            n2 = int(rng.randrange(400, int(650 * difficulty), 5))
            k1 = int(rng.randrange(170, int(290 * difficulty), 10))
            k2 = int(rng.randrange(120, int(170 * difficulty), 10))

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


def generate_30(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        group: str, n: int, n_1: int, n_2: int, hobby1: str, hobby2: str, hobby3: str, hobby4: str
    ) -> Dict[str, Any]:
        n_4 = 2 * n_2  # number that like hobby4 (music)
        n_3 = n - (n_1 + n_2 + n_4)  # number that like hobby3 (video games)

        question = f"A {group} of {n} students has various hobbies. {n_1} like to {hobby1}, {n_2} like to play {hobby2}, and the rest like to either {hobby3} or {hobby4}. How many like to {hobby3} if the number that like to {hobby4} is twice the number that prefer playing {hobby2}?"

        answer_cot = f"The number of students that like to {hobby4} is twice as many as the number who like {hobby2}, so 2 * {n_2} = {n_4}\nThe number that like to {hobby3} is {n} total students - {n_1} {hobby1} - {n_2} {hobby2} - {n_4} {hobby4} = {n_3}\n#### {n_3}"

        return {
            "question": question,
            "answer": str(n_3),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_31(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, fruit: str, total: int, n1: int, n2: int, n3: int, sibling1: str, sibling2: str
    ) -> Dict[str, Any]:
        slice2 = n1 + n2
        slice3 = slice2 + n3
        total_eaten = n1 + slice2 + slice3

        question = f"{name} sliced an {fruit} into {total} pieces. She ate {n1} slice, her {sibling1} ate {n2} more than her, and her {sibling2} ate {n3} more than her {sibling1}. How many slices of {fruit} did they all eat?"

        answer_cot = f"Her {sibling1} ate {n1} + {n2} = {slice2} slices.\nHer {sibling2} ate {slice2} + {n3} = {slice3} slices.\nThey ate a total of {n1} + {slice2} + {slice3} = {total_eaten} slices.\n#### {total_eaten}"

        return {
            "question": question,
            "answer": str(total_eaten),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Doxa"]
        fruits = ["orange", "pear", "peach", "mango", "kiwi", "apple"]
        siblings = ["brother", "sister", "cousin", "friend"]

        name = rng.choice(names_female)
        fruit = rng.choice(fruits)
        sibling1, sibling2 = rng.sample(siblings, 2)

        total = int(rng.randint(6, int(33 * difficulty)))
        n1 = int(rng.randint(3, int(15 * difficulty)))
        n2 = int(rng.randint(5, int(13 * difficulty)))
        n3 = int(rng.randint(3, int(14 * difficulty)))

        # Ensure conditions are met
        while n1 + (n1 + n2) + (n1 + n2 + n3) > total:
            n1 = int(rng.randint(3, int(15 * difficulty)))
            n2 = int(rng.randint(5, int(13 * difficulty)))
            n3 = int(rng.randint(3, int(14 * difficulty)))

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


def generate_32(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, periods: int, extra_classes: int, mins_per_class: int, days: int, weekend_fraction: float
    ) -> Dict[str, Any]:
        total_classes = periods + extra_classes
        daily_mins = total_classes * mins_per_class
        weekly_mins = daily_mins * days
        weekend_mins = int(weekly_mins * weekend_fraction)
        total_mins = weekly_mins + 2 * weekend_mins
        total_hours = total_mins // 60

        question = f"There are {periods} periods in the day for a normal student but {name} has to take {extra_classes} extra classes. Each class is {mins_per_class} minutes long. He goes to class for {days} days a week. He then spends {weekend_fraction} of his weekly minutes each on Saturday and Sunday as extra learning time. How many hours a week does he spend learning?"

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
            "answer": str(total_hours),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_33(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, mult: int, n: int) -> Dict[str, Any]:
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
            "answer": str(weekly_total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names_female = ["Julie", "Sarah", "Emma", "Sophia", "Olivia", "Isabella", "Mia", "Charlotte"]
        multi_times = [2, 3, 4]

        name1, name2 = rng.sample(names_female, 2)
        mult = rng.choice(multi_times)
        n = int(rng.randint(30, int(100 * difficulty)))

        # Ensure conditions are met
        while not (n * mult).is_integer() or not ((n + n * mult) * 7).is_integer():
            n = int(rng.randint(30, int(100 * difficulty)))

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


def generate_34(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(event: str, item: str, family: str, n: int, m: int, total: int) -> Dict[str, Any]:
        twins_total = 2 * n
        remaining = total - twins_total
        friends_found = remaining - m

        question = f"The {event} team hid {total} {item}. The {family} twins each found {n} {item}. All the other {item} except {m} were found by their friends. How many {item} did the friends find?"

        answer_cot = f"The {family} twins found, {n} * 2 = {twins_total} {item}.\nThe number that remained hidden was {total} - {twins_total} = {remaining} {item}\nSince {m} {item} were not found, this means the friends found {remaining} - {m} = {friends_found} {item}\n#### {friends_found}"

        return {
            "question": question,
            "answer": str(friends_found),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_35(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        job: str, building: str, room: str, num_rooms: int, num_days: int, time_per_room: int, hours_per_day: int
    ) -> Dict[str, Any]:

        rooms_per_day = num_rooms // num_days
        minutes_per_day = rooms_per_day * time_per_room
        hours_cleaning = minutes_per_day / 60
        percentage = int(100 * hours_cleaning / hours_per_day)

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
            "answer": str(percentage),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        jobs = ["janitor", "cleaner", "maintenance worker"]
        buildings = ["office building", "hospital", "university"]
        rooms = ["room", "floor"]

        job = rng.choice(jobs)
        building = rng.choice(buildings)
        room = rng.choice(rooms)

        num_days = int(rng.randint(3, int(12 * difficulty)))
        time_per_room = int(rng.randrange(10, int(46 * difficulty), 5))
        hours_per_day = int(rng.randint(6, int(17 * difficulty)))

        # Generate num_rooms ensuring divisibility by num_days
        rooms_per_day = rng.randint(5, int(20 * difficulty))
        num_rooms = rooms_per_day * num_days

        # Ensure conditions are met
        while (num_rooms / num_days) * time_per_room >= hours_per_day * 60:
            rooms_per_day = rng.randint(5, int(20 * difficulty))
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


def generate_36(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(n: int, p1: int, r1: int, name: str, s1: str, s2: str, s3: str) -> Dict[str, Any]:
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
            "answer": str(total_correct),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
                (n * (p1 / 100)).is_integer()
                and (n * (p1 / 100) * (r1 / 100)).is_integer()
                and (n * (1 - (p1 / 100)) * 0.5).is_integer()
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


def generate_37(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        worker: str, base: int, unit: str, tool1: str, tool2: str, tool3: str, mult1: int, mult2: int, n: int, days: int
    ) -> Dict[str, Any]:
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
            f"#### {total_amount}"
        )

        return {
            "question": question,
            "answer": str(total_amount),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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

        base = int(rng.randint(5, int(20 * difficulty)))
        mult1 = int(rng.randint(2, int(4 * difficulty)))
        mult2 = int(rng.randrange(30, int(80 * difficulty), 5))
        n = int(rng.randint(20, int(50 * difficulty)))
        days = int(rng.randint(28, 32))

        # Verify conditions
        while (
            not (base * mult1 * (1 + mult2 / 100)).is_integer()
            or int(base * mult1 * (1 + mult2 / 100) * n * days) >= 100000
        ):
            base = int(rng.randint(5, int(20 * difficulty)))
            mult1 = int(rng.randint(2, int(4 * difficulty)))
            mult2 = int(rng.randrange(30, int(80 * difficulty), 5))
            n = int(rng.randint(20, int(50 * difficulty)))

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


def generate_38(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

        skeins_per_child = n1 + n2 + n3 + n4 + n5
        total_skeins = count * skeins_per_child

        question = f"{name} is knitting winter wear for her {count} grandchildren. They're {child_type}, so they're all the same size. She wants to make a {item1}, {item2}, {item3}, {item4}, and {item5} for each of them. It takes {n1} skeins of wool to make a {item1}, {n2} for a {item2}, {n3} for a {item3}, {n4} for a pair of {item4}, and {n5} for a pair of {item5}. How many skeins of wool will she need to buy?"

        answer_cot = f"A full outfit for each child will require {n1} skeins per {item1} + {n2} skeins per {item2} + {n3} skeins per {item3} + {n4} skeins per pair of {item4} + {n5} skeins per pair of {item5} = {skeins_per_child} skeins of wool.\nSo to knit outfits for all of her grandchildren, she will need {count} * {skeins_per_child} = {total_skeins} skeins of wool.\n#### {total_skeins}"

        return {
            "question": question,
            "answer": str(total_skeins),
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


def generate_39(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        total: int, grade: str, school_name: str, num_girls: int, day: str, absent_girls: int, absent_boys: int
    ) -> Dict[str, Any]:
        num_boys = total - num_girls
        remaining_boys = num_boys - absent_boys

        question = f"There are {total} {grade}-graders at {school_name} School. {num_girls} of them are girls. On {day}, {absent_girls} {grade}-grade girls and {absent_boys} {grade}-grade boys were absent. How many {grade} grade boys were at {school_name} School on {day}?"

        answer_cot = f"Of the {total} {grade} graders, {num_girls} are girls, so {total} students - {num_girls} girls = {num_boys} boys.\nOn {day} there were {num_boys} boys - {absent_boys} absent = {remaining_boys} boys.\n#### {remaining_boys}"

        return {
            "question": question,
            "answer": str(remaining_boys),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        school_names = ["Maple Grove", "Sunny Hill", "Oak Ridge", "Pine Valley"]
        grades = ["first", "second", "third", "fourth", "fifth"]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        school_name = rng.choice(school_names)
        grade = rng.choice(grades)
        day = rng.choice(days)

        total = int(rng.randint(50, int(200 * difficulty)))
        num_girls = int(rng.randint(20, total - 1))
        num_boys = total - num_girls

        absent_girls = int(rng.randint(2, min(num_girls, int(10 * difficulty))))
        absent_boys = int(rng.randint(2, min(num_boys, int(10 * difficulty))))

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


def generate_40(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(item: str, n1: int, c1: str, c2: str, c3: str, p: int) -> Dict[str, Any]:
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
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        items = ["magician", "artist", "chef", "scientist", "athlete"]
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]

        item = rng.choice(items)
        c1, c2, c3 = rng.sample(colors, 3)

        n1 = int(rng.randint(20, int(81 * difficulty)))

        # Generate p ensuring division results in integer
        while True:
            p = int(rng.randint(20, min(90, int(100 * difficulty))))
            if (p / 100 * n1).is_integer():
                break

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


def generate_41(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, event: str, organization: str, fraction: str, current: int, total: int, currency: str
    ) -> Dict[str, Any]:
        fraction = convert_fraction_word(fraction)
        fraction_val = Fraction(fraction)
        org_amount = int(total * fraction_val)
        covered_amount = org_amount + current
        missing_amount = total - covered_amount

        question = f"{name} is raising money for a {event}. He has applied for help from the {organization}, which has decided to cover {fraction} of the cost of the {event}. How much money is {name} missing if he has {currency}{current} and the {event} costs {currency}{total}?"

        answer_cot = f"{name}'s {organization} has decided to pay {total} * {fraction} = {currency}{org_amount} for his {event}.\nIn total {name} has covered {org_amount} + {current} = {currency}{covered_amount} for his {event}\nTherefore, {name} needs {total} - {covered_amount} = {currency}{missing_amount} more for the {event}.\n#### {missing_amount}"

        return {
            "question": question,
            "answer": str(missing_amount),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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

        # Scale ranges by difficulty but ensure results are integers
        current = int(rng.randrange(10, int(200 * difficulty), 5))
        total = int(rng.randrange(200, int(1000 * difficulty), 10))

        # Ensure conditions are met
        while current >= total or not float(total * Fraction(fraction)).is_integer():
            current = int(rng.randrange(10, int(200 * difficulty), 5))
            total = int(rng.randrange(200, int(1000 * difficulty), 10))

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


def generate_42(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

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
            "answer": str(net_proceeds),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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

        price = int(rng.randrange(200000, int(1000000 * difficulty), 10000))
        fee1_percent = rng.randint(1, int(5 * difficulty))
        fee2_percent = rng.randint(2, int(7 * difficulty))
        loan = int(rng.randrange(100000, int(700000 * difficulty), 10000))

        # Ensure conditions are met
        while price <= loan or price - (price * (fee1_percent + fee2_percent) / 100 + loan) <= 1:
            price = int(rng.randrange(200000, int(1000000 * difficulty), 10000))
            loan = int(rng.randrange(100000, int(700000 * difficulty), 10000))

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


def generate_43(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(person1: str, item: str, n: int, relation: str, k: int) -> Dict[str, Any]:
        other_amount = n - k
        total = n + other_amount

        question = f"A {person1} has {n} {item}s. His {relation} has {k} fewer {item}s than he has. How many {item}s do they have together?"

        answer_cot = f"His {relation} has {n} - {k} = {other_amount} {item}s.\nTogether, they have {n} + {other_amount} = {total} {item}s.\n#### {total}"

        return {
            "question": question,
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_44(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:
        total1 = n1 * p1
        total2 = n2 * p2
        total3 = n3 * p3
        grand_total = total1 + total2 + total3

        question = f"There are currently {n1} {color1} balls, {n2} {color2} balls, and {n3} {color3} balls in the {store}. {color1} balls cost {currency}{p1}, {color2} balls cost {currency}{p2} and {color3} balls cost {currency}{p3}. How much will the {store} have received after all the balls are sold?"

        answer_cot = f"For the {color1} balls, {n1} balls * {currency}{p1}/ball = {currency}{total1}.\nFor the {color2} balls, {n2} balls * {currency}{p2}/ball = {currency}{total2}.\nFor the {color3} balls, {n3} balls * {currency}{p3}/ball = {currency}{total3}.\nFor all balls, {currency}{total1} + {currency}{total2} + {currency}{total3} = {currency}{grand_total}.\n#### {grand_total}"

        return {
            "question": question,
            "answer": str(grand_total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_45(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, event: str, food: str, obj: str, package_husband: int, used_items: int, total_remaining: int
    ) -> Dict[str, Any]:

        total_items = total_remaining + used_items
        package_size = total_items - package_husband

        question = f"{name} was preparing for a {event} at her house, where she intended to serve {food}. She noticed that she was out of plastic {obj}, so she bought a new package of {obj}. Later, her husband also bought a package of {package_husband} new {obj} and gave them to {name}. While {name} was making the {food}, she used {used_items} of the {obj} to sample her {food}. Later, when she went to set the table, she had a total of {total_remaining} {obj}. How many {obj} were in the package that {name} bought?"

        answer_cot = f"The total number of {obj} from {name} and her husband was {total_remaining}+{used_items}={total_items} {obj}.\nSince the husband bought a package of {package_husband} {obj}, then {name}'s package contained {total_items}-{package_husband}={package_size} {obj}.\n#### {package_size}"

        return {
            "question": question,
            "answer": str(package_size),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
        used_items = int(rng.randint(3, int(10 * difficulty)))

        # Calculate total_remaining to satisfy conditions
        package_size = int(rng.randint(10, int(30 * difficulty)))
        total_items = package_size + package_husband
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


def generate_46(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        fruit1: str, fruit2: str, n1: int, n2: int, frac1: float, frac2: float, spill: int, total: int
    ) -> Dict[str, Any]:
        n1_after_spill = n1 - spill
        water_fruit1 = n1_after_spill * frac1
        water_fruit2 = n2 * frac2
        total_water = water_fruit1 + water_fruit2

        question = f"I have {n1} liters of {fruit1} drink that are {frac1} water and I wish to add it to {n2} liters of {fruit2} drink that is {frac2} water. But as I pour it, I spill {spill} liter of the {fruit1} drink. How much water is in the remaining {total} liters?"

        answer_cot = f"There are {n2} x {frac2} = {water_fruit2} liters of water from the {n2} liters {fruit2} drink.\nAfter {spill} liter of {fruit1} drink was spilled, there were {n1} - {spill} = {n1_after_spill} liters of {fruit1} drink left.\nOut of the {n1_after_spill} liters, {n1_after_spill} x {frac1} = {water_fruit1} liters are water.\nThus, there are a total of {water_fruit2} + {water_fruit1} = {total_water} liters of water out of the {total} liters.\n#### {int(total_water)}"

        return {
            "question": question,
            "answer": str(int(total_water)),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
        while not (n1 + n2 - spill > 0 and (n2 * frac2).is_integer() and ((n1 - spill) * frac1).is_integer()):
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


def generate_47(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, n1: int, c1: float, n2: int, c2: float, c3: int, obj1: str, obj2: str, currency: str
    ) -> Dict[str, Any]:
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
            "answer": str(int(total_cost)),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
        while not (n1 * c1).is_integer() or not (n2 * c2).is_integer():
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


def generate_48(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        item1: str, item2: str, shop: str, currency: str, price1: int, price2: int, n1: int, n2: int
    ) -> Dict[str, Any]:
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
            "answer": str(diff),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_49(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, item1: str, item2: str, price1: int, price2: int, total: float, n1: int, percent: int, currency: str
    ) -> Dict[str, Any]:
        change = total * percent / 100
        spent = total * (100 - percent) / 100
        cost_item1 = n1 * price1 / 100
        spent_item2 = spent - cost_item1
        n2 = int(spent_item2 / (price2 / 100))

        question = f"The vending machines sell {item1} for {price1} cents and {item2} for {price2} cents. {name} spent {currency}{total} and got {n1} bags of {item1} and had {percent}% of his money left. How many {item2} did he buy?"

        answer_cot = (
            f"{name} got {currency}{change} in change because {total} x {percent}/100 = {change}\n"
            f"{name} spent {currency}{spent} because {total} - {change} = {spent}\n"
            f"{name} spent {currency}{cost_item1} on {item1} because {n1} x {price1/100} = {cost_item1}\n"
            f"{name} spent {spent_item2} on {item2} because {spent} - {cost_item1} = {spent_item2}\n"
            f"{name} bought {n2} {item2} because {spent_item2} / {price2/100} = {n2}\n"
            f"#### {n2}"
        )

        return {
            "question": question,
            "answer": str(n2),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["George", "James", "John", "Robert", "Michael", "William"]
        items = ["pretzels", "popcorn", "gum", "cookies", "crackers"]
        currencies = ["$", "£", "€"]

        name = rng.choice(names)
        item1, item2 = rng.sample(items, 2)
        currency = rng.choice(currencies)

        price1 = int(rng.randrange(25, int(100 * difficulty), 5))
        price2 = int(rng.randrange(50, int(150 * difficulty), 5))
        while price2 <= price1:
            price2 = int(rng.randrange(50, int(150 * difficulty), 5))

        total = int(rng.randrange(500, int(1500 * difficulty), 100))
        n1 = int(rng.randint(1, int(10 * difficulty)))
        percent = int(rng.randint(1, int(10 * difficulty)))

        # Validate conditions
        while not (
            isinstance(total * percent / 100, int)
            and isinstance((total * (100 - percent) / 100 - n1 * price1 / 100) / (price2 / 100), int)
            and (total * (100 - percent) / 100 - n1 * price1 / 100) % (price2 / 100) == 0
        ):
            total = int(rng.randrange(500, int(1500 * difficulty), 100))
            n1 = int(rng.randint(1, int(10 * difficulty)))
            percent = int(rng.randint(1, int(10 * difficulty)))

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


def generate_50(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name: str, pieces1: int, pieces2: int) -> Dict[str, Any]:
        half_pieces1 = pieces1 // 2
        total_pieces = half_pieces1 + pieces2

        question = f"{name} finished half of a {pieces1} piece puzzle, and then started and finished another {pieces2} piece puzzle within an hour. How many puzzle pieces did {name} place during that hour?"

        answer_cot = f"{name} did 1/2 * {pieces1} pieces = {half_pieces1} pieces.\n{name} completed {half_pieces1} pieces + {pieces2} pieces = {total_pieces} pieces.\n#### {total_pieces}"

        return {
            "question": question,
            "answer": str(total_pieces),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_51(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

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
            "answer": str(friends),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_52(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name: str, alphabets: tuple, n1: str, frac: str) -> Dict[str, Any]:
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
            "answer": str(final_total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
        while not (alphabet[1] * frac_map[frac]).is_integer():
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


def generate_53(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name: str, sides: int, target: int, property: str) -> Dict[str, Any]:
        numbers_above = sides - target
        prob_above = (numbers_above / sides) * 100
        prob_two_in_row = 25  # probability of two even/odd in a row is always 25%
        difference = int(prob_above - prob_two_in_row)

        question = f"{name} is rolling a {sides}-sided die. How much more likely is it (expressed as a percentage) that he rolls a number greater than {target} than that he rolls two {property} numbers in a row?"

        answer_cot = f"There are {numbers_above} numbers greater than {target} on the dice, so the chances of rolling one of them are {numbers_above} / {sides} = {prob_above}%.\nThe chance of rolling one {property} number is 50%, so the chance of rolling two in a row is 50% * 50% = 25%.\nThe difference between these two probabilities is {prob_above}% - 25% = {difference}%.\n#### {difference}"

        return {
            "question": question,
            "answer": str(difference),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_54(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name1: str,
        name2: str,
        total_time: int,
        library_time: int,
        station_time: int,
        location1: str,
        location2: str,
        location3: str,
    ) -> Dict[str, Any]:

        time_after_library = total_time - library_time
        remaining_time = time_after_library - station_time

        question = f"{name1} and {name2} have {total_time} minutes to walk to {location1} together. It takes them {library_time} minutes to get to the corner where the {location2} is. It takes them another {station_time} minutes to get to the {location3}. How much longer do they have to get to {location1} without being late?"

        answer_cot = f"{name1} and {name2} arrive at the {location2} with {total_time} - {library_time} = {time_after_library} minutes left to reach the {location1}.\nThey then arrive at the {location3} and have {time_after_library} - {station_time} = {remaining_time} minutes left to get to {location1} without being late.\n#### {remaining_time}"

        return {
            "question": question,
            "answer": str(remaining_time),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_55(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, place: str, fruit: str, location: str, insect1: str, insect2: str, n: int, frac: str
    ) -> Dict[str, Any]:
        num_insect1 = int(n * 0.5)  # half as many bugs as ants
        total_insects = n + num_insect1

        question = f"{name} went to their {place} to pick some {fruit} and found {frac} as many {insect1} as {insect2} in the {location}. If there were {n} {insect2}, calculate the total number of insects in the {location}."

        answer_cot = f"If there were {n} {insect2}, the total number of {insect1} in the {location} is {frac} * {n} {insect2} = {num_insect1} {insect1}\nThe total number of insects in the {location} is {num_insect1} {insect1} + {n} {insect2} = {total_insects} insects\n#### {total_insects}"

        return {
            "question": question,
            "answer": str(total_insects),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_56(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        family: str, item: str, total: int, n1: int, n2: int, flavor1: str, flavor2: str, flavor3: str
    ) -> Dict[str, Any]:
        n3 = total - (n1 + n2)

        question = f"The {family} family is busy making {item}s. So far, they've made {total} {item}s. They have {n1} {flavor1} {item}s, {n2} {flavor2} {item}s, and some {flavor3} {item}s. How many {flavor3} {item}s have they made?"

        answer_cot = f"The total number of pieces of {flavor1} and {flavor2} {item}s is {n1} + {n2} = {n1+n2}.\nTherefore, they made {total} - {n1+n2} = {n3} {flavor3} {item}s.\n#### {n3}"

        return {
            "question": question,
            "answer": str(n3),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_57(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        n1: int, sport1: str, sport2: str, sport3: str, n2: int, n3: int, multiplier: int
    ) -> Dict[str, Any]:
        n_volleyball = n1 * multiplier
        n_soccer = n2 + n3
        total = n1 + n_volleyball + n_soccer

        question = f"There are {n1} students playing {sport1} and twice that number playing {sport2}. There are {n2} boys and {n3} girls playing {sport3}. If each student only participates in one group, how many students are there in total?"

        answer_cot = f"There are {n1} x {multiplier} = {n_volleyball} students playing {sport2}.\nThere are {n2} + {n3} = {n_soccer} students playing {sport3}.\nIn total there are {n1} + {n_volleyball} + {n_soccer} = {total} students.\n#### {total}"

        return {
            "question": question,
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        sports = ["basketball", "badminton", "table tennis", "football", "volleyball"]
        sport1, sport2, sport3 = rng.sample(sports, 3)

        # Generate numbers based on difficulty
        n1 = int(rng.randint(4, int(21 * difficulty)))
        n2 = int(rng.randint(10, int(31 * difficulty)))
        n3 = int(rng.randint(10, int(31 * difficulty)))
        multiplier = 2  # "twice" that number

        # Check condition
        while n1 * multiplier + n2 + n3 > 250:
            n1 = int(rng.randint(4, int(21 * difficulty)))
            n2 = int(rng.randint(10, int(31 * difficulty)))
            n3 = int(rng.randint(10, int(31 * difficulty)))

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


def generate_58(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, container: str, liquid: str, volume: int, unit: str, num_containers: int, calories: int
    ) -> Dict[str, Any]:
        total_volume = volume * num_containers
        total_calories = total_volume * calories

        question = f"A {container} of {liquid} is {volume} {unit}s of {liquid}. {name} drinks {num_containers} {container}s of {liquid}. If {liquid} has {calories} calories per {unit} how many calories did he consume?"

        answer_cot = f"He drank {volume}*{num_containers}={total_volume} {unit}s of {liquid}.\nSo he drank {total_volume}*{calories}={total_calories} calories of {liquid}\n#### {total_calories}"

        return {
            "question": question,
            "answer": str(total_calories),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_59(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        time_per_interval: int, distance_per_interval: int, total_distance: int
    ) -> Dict[str, Any]:
        intervals = total_distance // distance_per_interval
        total_time = intervals * time_per_interval

        question = f"A fog bank rolls in from the ocean to cover a city. It takes {time_per_interval} minutes to cover every {distance_per_interval} miles of the city. If the city is {total_distance} miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the whole city?"

        answer_cot = f"The city will be covered in {total_distance} / {distance_per_interval} = {intervals} intervals of {time_per_interval} minutes.\nThus, it will take {intervals} * {time_per_interval} = {total_time} minutes for the fog to cover the whole city.\n#### {total_time}"

        return {
            "question": question,
            "answer": str(total_time),
            "answer_cot": answer_cot,
            "answer_value": total_time,
            "variables": {
                "time_per_interval": time_per_interval,
                "distance_per_interval": distance_per_interval,
                "total_distance": total_distance,
                "intervals": intervals,
            },
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        # Generate random values scaled by difficulty
        distance_per_interval = int(rng.randint(2, int(100 * difficulty)))
        time_per_interval = int(rng.randint(2, int(500 * difficulty)))

        # Ensure total distance is divisible by distance_per_interval
        num_intervals = rng.randint(2, int(20 * difficulty))
        total_distance = distance_per_interval * num_intervals

        # Ensure total_distance is in valid range
        while total_distance > 100:
            num_intervals = rng.randint(2, int(20 * difficulty))
            total_distance = distance_per_interval * num_intervals

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


def generate_60(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, unit: str, total_dist: int, beach_dist: int, sidewalk_dist: int, speed_mult: int, beach_time: int
    ) -> Dict[str, Any]:

        beach_rate = Fraction(beach_dist, beach_time)
        sidewalk_rate = beach_rate * speed_mult
        sidewalk_time = int(sidewalk_dist / sidewalk_rate)
        total_time = beach_time + sidewalk_time

        question = f"{name} walks {total_dist} {unit}s every day on her favorite walking trail, which includes {beach_dist} {unit}s of walking on the beach and {sidewalk_dist} {unit}s of walking on the sidewalk. On the sidewalk, {name} walks at twice the rate of speed that she does on the beach. If {beach_time} minutes of her walk is spent on the beach, how long does it take for her to complete the entire {total_dist}-{unit} walk, in minutes?"

        answer_cot = f"On the beach, {name} walks at a rate of {beach_dist} {unit}s per {beach_time} minutes, or {beach_dist}/{beach_time} = {beach_rate} {unit}s per minute.\nOn the sidewalk, she walks at {speed_mult} times the rate of speed as when she is on the sand, or {speed_mult} * {beach_rate} = {sidewalk_rate} {unit}s per minute.\nTo walk {sidewalk_dist} {unit}s on the sidewalk, it takes her {sidewalk_dist}÷{sidewalk_rate}={sidewalk_time} minutes.\nThus, in total, it takes {name} {beach_time}+{sidewalk_time}={total_time} minutes to walk her favorite route.\n#### {total_time}"

        return {
            "question": question,
            "answer": str(total_time),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["Emma", "Sophia", "Isabella", "Olivia", "Ava", "Mia", "Emily"]
        units = ["mile", "kilometer", "block"]

        name = rng.choice(names)
        unit = rng.choice(units)
        speed_mult = 2  # Fixed as "twice" in question

        beach_time = int(rng.randint(40, int(70 * difficulty)))
        beach_dist = int(rng.randint(10, int(20 * difficulty)))
        sidewalk_dist = int(rng.randint(10, int(20 * difficulty)))
        total_dist = beach_dist + sidewalk_dist

        # Ensure mathematical consistency
        while not (beach_dist < beach_time and speed_mult * beach_dist < beach_time and beach_time % beach_dist == 0):
            beach_time = int(rng.randint(40, int(70 * difficulty)))
            beach_dist = int(rng.randint(10, int(20 * difficulty)))
            sidewalk_dist = int(rng.randint(10, int(20 * difficulty)))
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


def generate_61(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

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
            "answer": str(money_left),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names_female = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte"]
        locations = ["beach", "boardwalk", "pier", "coast"]
        shops = ["souvenir store", "gift shop", "beach shop", "seaside store"]
        items1 = ["fudge", "saltwater taffy", "rock candy", "cotton candy"]
        items2 = ["sand dollars", "starfish", "sea glass", "coral pieces"]
        items3 = ["postcards", "keychains", "stickers", "pins"]
        units = ["pound", "kilogram", "kg"]
        currencies = ["$", "£", "€"]
        fraction_nums = [0.25, 0.33, 0.5, 0.67, 0.75]

        name = rng.choice(names_female)
        location = rng.choice(locations)
        shop = rng.choice(shops)
        item1 = rng.choice(items1)
        item2 = rng.choice(items2)
        item3 = rng.choice(items3)
        unit = rng.choice(units)
        cur = rng.choice(currencies)

        total = int(rng.randint(1200, int(1500 * difficulty)))
        n1 = int(rng.randint(15, int(18 * difficulty)))
        n2 = int(rng.randint(4, int(10 * difficulty)))
        k = int(rng.randint(2, int(5 * difficulty)))
        n12 = n1 + n2 + k
        n3 = int(rng.randint(11, int(19 * difficulty)))
        p1 = int(rng.randint(20, int(24 * difficulty)))
        p2 = round(rng.uniform(11.25, 12.00), 2)
        p3 = round(rng.uniform(20.25, 21.25), 2)
        discount = rng.choice(fraction_nums[:4])

        # Ensure conditions are met
        while not (
            n2 < n1
            and n12 == n1 + n2 + k
            and 0 <= k < n1
            and int(n1 * p1 + n2 * (1 - discount) * p1 + k * p1 + p2 + n3 * p3)
            == n1 * p1 + n2 * (1 - discount) * p1 + k * p1 + p2 + n3 * p3
            and n1 * p1 + n2 * (1 - discount) * p1 + k * p1 + p2 + n3 * p3 < total
        ):
            n1 = int(rng.randint(15, int(18 * difficulty)))
            n2 = int(rng.randint(4, int(10 * difficulty)))
            k = int(rng.randint(2, int(5 * difficulty)))
            n12 = n1 + n2 + k
            p1 = int(rng.randint(20, int(24 * difficulty)))

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


def generate_62(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        item: str, num_slices: int, name1: str, name2: str, slices_per_day: int, multiplier: int, unit: str
    ) -> Dict[str, Any]:

        second_person_slices = slices_per_day * multiplier
        total_daily_slices = slices_per_day + second_person_slices
        days_lasting = num_slices // total_daily_slices

        question = f"A {item} has {num_slices} {unit}. If {name1} can eat {slices_per_day} {unit} a day while {name2} can eat {multiplier} times as much, how many days will the {item} last?"

        answer_cot = f"{name2} can eat {slices_per_day} x {multiplier} = {second_person_slices} {unit} a day.\nTogether, {name1} and {name2} can eat {slices_per_day} + {second_person_slices} = {total_daily_slices} {unit} a day.\nSo, a {item} will last for {num_slices}/{total_daily_slices} = {days_lasting} days.\n#### {days_lasting}"

        return {
            "question": question,
            "answer": str(days_lasting),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_63(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name: str, hours: int, days: int, rate: int, bonus: int, month: str) -> Dict[str, Any]:
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
            "answer": str(total_pay),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_64(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name: str, n1: int, d1: int, n2: int, d2: int) -> Dict[str, Any]:
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
            "answer": str(dozens),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_65(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, fish: str, day: str, w1: int, w2: int, w3: int, n: int, unit: str, cur: str, price: float
    ) -> Dict[str, Any]:
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
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_66(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

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
            "answer": str(num_wed_episodes),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_67(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, fruit: str, n1: int, n2: int, d1: str, d2: str, d3: str, mult: int
    ) -> Dict[str, Any]:
        first_two_days = n1 + n2
        friday_amount = mult * n1
        total = first_two_days + friday_amount

        question = f"{name} picks {n1} {fruit}s on {d1}. Then he picks {n2} {fruit}s on {d2}. On {d3}, he picks {mult} times the number of {fruit}s he did on {d1}. How many {fruit}s does {name} have?"

        answer_cot = f"Combining {d1} and {d2}, {name} has {n1} {fruit}s + {n2} {fruit}s = {first_two_days} {fruit}s.\nOn {d3}, he picks {mult} * {n1} {fruit}s = {friday_amount} {fruit}s.\nAltogether, {name} has {first_two_days} {fruit}s + {friday_amount} {fruit}s = {total} {fruit}s.\n#### {total}"

        return {
            "question": question,
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_68(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(n0: int, r: int, d: int, disease: str) -> Dict[str, Any]:
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
            "answer": str(day3_total),
            "answer_cot": answer_cot,
            "answer_value": day3_total,
            "variables": {"initial_infected": n0, "infection_rate": r, "days": d, "disease_type": disease},
        }

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        diseases = ["virus", "bacteria", "parasite", "infection"]

        disease = rng.choice(diseases)
        n0 = int(rng.randint(5, int(21 * difficulty)))
        r = int(rng.randint(2, int(8 * difficulty)))
        d = 3  # Fixed at 3 days per problem description

        # Check condition: n0 * (r + 1)**d < 20000
        while n0 * (r + 1) ** d >= 20000:
            n0 = int(rng.randint(5, int(21 * difficulty)))
            r = int(rng.randint(2, int(8 * difficulty)))

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


def generate_69(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name: str, document: str, total_pages: int, fraction: str) -> Dict[str, Any]:
        frac_num = eval(fraction)
        pages_done = int(total_pages * frac_num)
        pages_remaining = total_pages - pages_done

        question = f"{name} is required to submit a {total_pages}-page {document}. She already finished writing {fraction} of the {document}. How many pages does she have left to write?"

        answer_cot = f"{name} has already written {fraction} of the {document} which is {total_pages} pages x {fraction} = {pages_done} pages.\nSo, she still needs to write {total_pages} pages - {pages_done} pages = {pages_remaining} pages.\n#### {pages_remaining}"

        return {
            "question": question,
            "answer": str(pages_remaining),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_70(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, objects: str, n: int, obstacle: str, frac: float, k: int, fake_num: int, fake_object: str
    ) -> Dict[str, Any]:

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
            "answer": str(final),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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

        n = int(rng.randrange(10, int(101 * difficulty), 2))
        fake_num = int(rng.randint(2, min(10, int(n * frac))))
        k = int(rng.randint(fake_num + 1, min(int(n * frac), int(20 * difficulty))))

        # Ensure conditions are met
        while not (isinstance(n * frac, int) and k < n * frac and k > fake_num):
            n = int(rng.randrange(10, int(101 * difficulty), 2))
            k = int(rng.randint(fake_num + 1, min(int(n * frac), int(20 * difficulty))))

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


def generate_71(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:
        cost1 = n1 * p1
        cost2 = n2 * p2
        cost3 = n3 * p3
        total_cost = cost1 + cost2 + cost3

        question = f"{name} went to the {shop} and bought various types of {item}. She bought {n1} dozen {item1} which cost ${p1} per dozen, {n2} dozen {item2} which cost ${p2} per dozen, and {n3} dozen {item3} for ${p3} per dozen. How much was the total cost?"

        answer_cot = f"The total charge for the {item1} was {n1} x ${p1} = ${cost1}.\nThe total charge for the {item2} was {n2} x ${p2} = ${cost2}.\nThe total charge for the {item3} was {n3} x ${p3} = ${p3*n3}.\nTherefore the total amount {name} paid for the {item} was ${cost1} + ${cost2} + ${cost3} = ${total_cost}.\n#### {total_cost}"

        return {
            "question": question,
            "answer": str(total_cost),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_72(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        structure: str, n1: int, color1: str, color2: str, color3: str, obj: str, mult: int, total: int
    ) -> Dict[str, Any]:
        n2 = n1 * mult
        n3 = total - n1 - n2

        question = f"A {structure} is made out of {n1} {color1} {obj}s, {mult} times as many {color2} {obj}s, and an unknown number of {color3} {obj}s. If there are {total} {obj}s in the {structure} in total, how many {color3} {obj}s are there?"

        answer_cot = f"There are {n1}*{mult} = {n2} {color2} {obj}s in the {structure}.\nThere are {total}-{n1}-{n2} = {n3} {color3} {obj}s in the {structure}.\n#### {n3}"

        return {
            "question": question,
            "answer": str(n3),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_73(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

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
            "answer": str(total_revenue),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_74(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, big_fish: str, length: int, num_remoras: int, remora_length: int
    ) -> Dict[str, Any]:
        total_remora_length_inches = num_remoras * remora_length
        total_remora_length_feet = total_remora_length_inches / 12
        percentage = int((total_remora_length_feet / length) * 100)

        question = f"{name} saw a {length}-foot {big_fish} with {num_remoras} {remora_length}-inch remoras attached to it. What percentage of the {big_fish}'s body length is the combined length of the remoras?"

        answer_cot = f"First, find the combined length of the remoras in inches: {remora_length} inches/remora * {num_remoras} remoras = {total_remora_length_inches} inches\nThen divide that number by 12 to convert it to feet: {total_remora_length_inches} inches / 12 inches/foot = {total_remora_length_feet} foot\nThen divide the combined remora length in feet by the {big_fish}'s length and multiply by 100% to express the answer as a percentage: {total_remora_length_feet} foot / {length} feet * 100% = {percentage}%\n#### {percentage}"

        return {
            "question": question,
            "answer": str(percentage),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["Benny", "Tommy", "Jimmy", "Billy", "Johnny", "Bobby"]
        big_fish = ["dolphin", "whale", "shark"]

        name = rng.choice(names)
        fish = rng.choice(big_fish)

        length = int(rng.randrange(10, int(500 * difficulty), 10))
        num_remoras = int(rng.randint(2, int(10 * difficulty)))
        remora_length = int(rng.randint(2, int(100 * difficulty)))

        # Ensure conditions are met
        while (
            num_remoras * remora_length >= length * 12
            or (num_remoras * remora_length) % 12 != 0
            or (length * 12) % (num_remoras * remora_length) != 0
            or 100 % int((num_remoras * remora_length) / (length * 12) * 100) != 0
        ):
            length = int(rng.randrange(10, int(500 * difficulty), 10))
            num_remoras = int(rng.randint(2, int(10 * difficulty)))
            remora_length = int(rng.randint(2, int(100 * difficulty)))

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


def generate_75(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name1: str, name2: str, color1: str, color2: str, n1: int, n2: int, frac1: float, mult1: float
    ) -> Dict[str, Any]:

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
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_76(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(n: int, p1: int, p2: int, company: str, frac: float) -> Dict[str, Any]:
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
            "answer": str(accepts),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        companies = ["Microsoft", "Apple", "Amazon", "Facebook", "Netflix", "Tesla", "Google"]
        fractions = {"a third": 1 / 3, "half": 1 / 2, "a quarter": 1 / 4, "two thirds": 2 / 3}

        company = rng.choice(companies)
        frac = fractions[rng.choice(list(fractions.keys()))]

        # Generate values ensuring all divisions result in integers
        n = int(rng.randint(201, int(1001 * difficulty)))
        p1 = int(rng.randint(10, int(51 * difficulty)))
        p2 = int(rng.randint(10, int(51 * difficulty)))

        # Ensure integer results
        while (
            not (n * (p1 / 100)).is_integer()
            or not (n * (p1 / 100) * (p2 / 100)).is_integer()
            or not (n * (p1 / 100) * (p2 / 100) * frac).is_integer()
        ):
            n = int(rng.randint(201, int(1001 * difficulty)))
            p1 = int(rng.randint(10, int(51 * difficulty)))
            p2 = int(rng.randint(10, int(51 * difficulty)))

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


def generate_77(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        event: str, m: int, w: int, t: str, frac: float, m_left: int, group1: str, group2: str
    ) -> Dict[str, Any]:
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
            "answer": str(w_left),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        events = ["party", "meeting", "conference", "gathering", "celebration"]
        groups = ["teachers", "doctors", "engineers", "nurses", "artists", "lawyers"]
        times = ["an hour", "two hours", "half an hour", "45 minutes"]
        fractions = [0.25, 0.5, 0.75, 0.33, 0.67]

        event = rng.choice(events)
        group1, group2 = rng.sample(groups, 2)
        t = rng.choice(times)
        frac = rng.choice(fractions)

        m = int(rng.randint(20, int(75 * difficulty)))
        w = int(rng.randint(10, int(80 * difficulty)))
        total = m + w

        # Ensure fraction calculations result in integers
        while not (total * frac).is_integer():
            m = int(rng.randint(20, int(75 * difficulty)))
            w = int(rng.randint(10, int(80 * difficulty)))
            total = m + w

        stayed = total - int(total * frac)
        m_left = int(rng.randint(15, min(stayed - 1, int(35 * difficulty))))

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


def generate_78(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, age_diff: int, age1: int) -> Dict[str, Any]:
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
            "answer": str(avg_age),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia"]
        name1, name2 = rng.sample(names, 2)

        age_diff = int(rng.randint(5, int(30 * difficulty)))
        age1 = int(rng.randint(15, int(75 * difficulty)))

        # Ensure average is an integer
        while (2 * age1 + age_diff) % 2 != 0:
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


def generate_79(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str,
        vehicle: str,
        start_time: int,
        end_time: int,
        free_hours: int,
        currency: str,
        first_hour_cost: int,
        multiplier: int,
    ) -> Dict[str, Any]:
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
            "answer": str(total_cost),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
            and ((end_time - start_time - free_hours - 1) * first_hour_cost * multiplier).is_integer()
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


def generate_80(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, color1: str, color2: str, n1: int, n2: int, n3: int, n4: int
    ) -> Dict[str, Any]:
        blue_spools = n1 + n2
        total_spools = n1 + n2 + n3 + n4
        percent_blue = int(100 * blue_spools / total_spools)

        question = f"{name} has {n1} light {color1} spools of thread, {n2} dark {color1} spools of thread, {n3} light {color2} spools of thread, and {n4} dark {color2} spools of thread. What percent of her spools are {color1}?"

        answer_cot = f"First find the number of {color1} spools: {n1} spools + {n2} spools = {blue_spools} spools\nThen find the total number of spools: {n3} spools + {n4} spools + {blue_spools} spools = {total_spools} spools\nThen divide the number of {color1} spools by the total number of spools and multiply by 100% to express the answer as a percentage: {blue_spools} spools / {total_spools} spools * 100% = {percent_blue}%\n#### {percent_blue}"

        return {
            "question": question,
            "answer": str(percent_blue),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_81(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        occupation: str, weeks_per_month: int, days_per_week: int, pay_per_day: int, currency: str
    ) -> Dict[str, Any]:
        days_per_month = days_per_week * weeks_per_month
        monthly_pay = days_per_month * pay_per_day
        yearly_pay = monthly_pay * 12

        question = f"A {occupation} works for {weeks_per_month} weeks every month and for {days_per_week} days every week. If he gets paid {currency}{pay_per_day} every day, how much does he earn if he works for a year?"

        answer_cot = f"The {occupation} works for {days_per_week} days every week and works for {weeks_per_month} weeks every month so he works for {days_per_week} days/week * {weeks_per_month} weeks/month = {days_per_month} days/month\nIf he earns {currency}{pay_per_day} every day he then earns {currency}{pay_per_day}/day * {days_per_month} days/month = {currency}{monthly_pay}/month\nA year is equal to 12 months so every year he earns {currency}{monthly_pay}/month * 12 months/year = {currency}{yearly_pay}\n#### {yearly_pay}"

        return {
            "question": question,
            "answer": str(yearly_pay),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_82(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name: str, num_emails: int, no_response_percent: int, workdays: int) -> Dict[str, Any]:
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
            "answer": str(total_responses),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_83(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(name1: str, name2: str, total: int, diff: int, unit: str) -> Dict[str, Any]:
        amount1 = (total - diff) // 2  # Sam's amount
        amount2 = amount1 + diff  # Harry's amount

        question = f"If {name1} and {name2} have {total} {unit} of fence between them, and they agree to split it with {name2} getting {diff} {unit} more than {name1}, how much is left over for {name1}?"

        answer_cot = f"Let x be the amount of fence {name1} gets and y be the amount {name2} gets. We know that y = x + {diff}, and y + x = {total}.\nSubstituting the first equation into the second equation, we get 2x+{diff}={total}\nSubtracting the {diff} from both sides, we get 2x={total-diff}\nWe divide each side by two, leaving x={amount1}. This means {name1} has {amount1} {unit} of fence left over.\n#### {amount1}"

        return {
            "question": question,
            "answer": str(amount1),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["Sam", "Harry", "Tom", "John", "Mike", "Dave", "Steve", "Bob"]
        units = ["feet", "yards", "meters"]

        name1, name2 = rng.sample(names, 2)
        unit = rng.choice(units)

        # Scale ranges by difficulty while maintaining integer division
        diff = int(rng.randrange(20, int(200 * difficulty), 10))
        total = int(rng.randrange(diff + 20, int(1000 * difficulty), 20))

        # Ensure conditions are met
        while total - diff <= 10 or (total - diff) % 2 != 0:
            diff = int(rng.randrange(20, int(200 * difficulty), 10))
            total = int(rng.randrange(diff + 20, int(1000 * difficulty), 20))

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


def generate_84(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, miles: str, time_cold: int, extra_time: int, multiplier: float, distance: int
    ) -> Dict[str, Any]:
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
            "answer": str(time_difference),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["Ray", "Jim", "Bob", "Tom", "Mike", "John", "Steve"]
        units = ["mile", "kilometer"]

        name = rng.choice(names)
        unit = rng.choice(units)

        time_cold = int(rng.randint(10, int(50 * difficulty)))
        extra_time = int(rng.randint(1, int(10 * difficulty)))
        multiplier = 2.0  # "twice" specified in original
        distance = int(rng.randint(2, int(10 * difficulty)))

        # Check conditions
        while (
            time_cold >= 60
            or extra_time + multiplier * time_cold >= 60
            or distance * (extra_time + multiplier * time_cold) - distance * time_cold <= 0
        ):
            time_cold = int(rng.randint(10, int(50 * difficulty)))
            extra_time = int(rng.randint(1, int(10 * difficulty)))
            distance = int(rng.randint(2, int(10 * difficulty)))

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


def generate_85(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, room_type: str, area: int, length: int, unit1: str, unit2: str
    ) -> Dict[str, Any]:
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
            "answer": str(perimeter),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["William", "James", "John", "Michael", "David", "Robert", "Thomas"]
        room_types = ["living room", "study", "office", "kitchen"]
        units = ["feet", "meters"]

        name = rng.choice(names)
        room_type = rng.choice(room_types)
        unit1 = rng.choice(units)
        unit2 = "yards" if unit1 == "feet" else "meters"

        length = int(rng.randint(5, int(44 * difficulty)))

        # Ensure width is larger than length and area calculation results in integer
        conversion = 3 if unit1 == "feet" and unit2 == "yards" else 1
        width = int(rng.randint(length * conversion + 1, int(100 * difficulty)))
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


def generate_86(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:
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
            "answer": str(trips),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_87(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

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
            "answer": str(int(remaining)),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_88(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(school: str, venue: str, total: int, graduates: int, faculty: int) -> Dict[str, Any]:
        remaining_seats = total - (graduates + faculty)
        tickets_per_graduate = remaining_seats // graduates

        question = f"{school} is holding graduation in their {venue} this year which has space for {total} people. After accounting for {graduates} seats for graduates and {faculty} seats for faculty attending, how many tickets would each graduate receive to give to their friends and family if the tickets are split equally?"

        answer_cot = f"Add graduate and faculty seats together. {graduates} + {faculty} = {graduates+faculty} seats for faculty and graduates\nMinus seats for faculty and graduates from total seats allowed. {total} - {graduates+faculty} = {remaining_seats} remaining seats.\nDivide remaining seats by the number of graduates. {remaining_seats}/{graduates} = {tickets_per_graduate} tickets\n#### {tickets_per_graduate}"

        return {
            "question": question,
            "answer": str(tickets_per_graduate),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_89(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:
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
            "answer": str(remaining),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_90(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        device: str, currency: str, rate1: float, rate2: float, threshold: int, total_mins: int
    ) -> Dict[str, Any]:
        first_period = threshold
        second_period = total_mins - threshold

        cost1 = first_period * rate1
        cost2 = second_period * rate2
        total_cost = int(cost1 + cost2)

        question = f"To make a call from a {device}, you must pay {currency}{rate1} for each minute of your call. After {threshold} minutes, that price drops to {currency}{rate2} per minute. How much would a {total_mins}-minute call cost?"

        answer_cot = f"First {threshold} minutes would be a cost of {threshold} * {rate1} = {currency}{int(cost1)}.\nAfter that, there are {total_mins} - {threshold} = {second_period} minutes of the call left.\nAnd these {second_period} minutes cost {second_period} * {rate2} = {currency}{int(cost2)}.\nSo in total, the {total_mins}-minute call would cost {int(cost1)} + {int(cost2)} = {currency}{total_cost}.\n#### {total_cost}"

        return {
            "question": question,
            "answer": str(total_cost),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        devices = ["payphone", "phone booth", "hotel room phone"]
        currencies = ["$", "£", "€"]

        device = rng.choice(devices)
        currency = rng.choice(currencies)

        # Generate rates ensuring rate2 < rate1
        rate1 = round(rng.uniform(0.2, 0.5 * difficulty), 2)
        rate2 = round(rng.uniform(0.1, rate1), 2)

        threshold = int(rng.randint(10, int(50 * difficulty)))
        total_mins = int(rng.randint(threshold + 10, int(100 * difficulty)))

        # Ensure calculations result in integers
        while not (threshold * rate1).is_integer() or not ((total_mins - threshold) * rate2).is_integer():
            rate1 = round(rng.uniform(0.2, 0.5 * difficulty), 2)
            rate2 = round(rng.uniform(0.1, rate1), 2)

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


def generate_91(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, fruit: str, area: str, field_size: int, density: int, months: int
    ) -> Dict[str, Any]:
        fruits_per_harvest = field_size * density
        harvests_per_year = 12 // months
        total_fruits = fruits_per_harvest * harvests_per_year

        question = f"{name} has {field_size} {area}s of a {fruit} field. There are {density} {fruit}s per {area}. {name} can harvest his {fruit}s every {months} months. How many {fruit}s can {name} harvest within a year?"

        answer_cot = f"{name} has {density} x {field_size}= {fruits_per_harvest} {fruit}s on his field.\n{name} can harvest his {fruit}s 12 ÷ {months} = {harvests_per_year} times per year\nTherefore {name} can harvest {fruits_per_harvest} x {harvests_per_year} = {total_fruits} {fruit}s per year.\n#### {total_fruits}"

        return {
            "question": question,
            "answer": str(total_fruits),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_92(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

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
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_93(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

        kills_arthur = int(n1 * frac1)
        kills_walter = int(kills_arthur * mult1)
        kills_bruce = int(kills_walter * frac2)

        question = f"{name1} slew {n1} {creature} with his mighty {weapon1}, while {name2}, using a {weapon2}, slew {frac1} as many {creature} as {name1}. Using a {weapon3}, {name3} slew {mult1} as many {creature} as {name2}. But {name4}, having forgot his {weapon4} at home, slew {frac2} as many {creature} as {name3} using a {weapon5}. How many {creature} has {name4} slain?"

        answer_cot = f"{name2} slew {frac1} as many {creature} as {name1}, or {n1}*{frac1}={kills_arthur} {creature}.\n{name3} slew {mult1} as many {creature} as {name2}, or {mult1}*{kills_arthur}={kills_walter} {creature}.\n{name4} slew {frac2} as many {creature} as {name3}, or {kills_walter}*{frac2}={kills_bruce} {creature}.\n#### {kills_bruce}"

        return {
            "question": question,
            "answer": str(kills_bruce),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_94(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, num_shares: int, price_per_share: int, increase_pct: int, decrease_pct: int
    ) -> Dict[str, Any]:
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
            "answer": str(int(final_value)),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_95(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name1: str, name2: str, relation: str, food: str, n1: int, n2: int, n3: int, time_unit: str, time_period: str
    ) -> Dict[str, Any]:
        daily_total = n1 + n2 + n3
        total = daily_total * (7 if time_period == "week" else 30)

        question = f"{name1} eats {n1} {food} per {time_unit}, {name2} eats {n2} {food} per {time_unit}, and their {relation} eats {n3} {food} per {time_unit}. How many {food} does this family eat in one {time_period}?"

        answer_cot = f"The number of {food} they eat in one {time_unit} is {n1} + {n2} + {n3} = {daily_total} {food}.\nThe number of {food} they eat in a {time_period} is {daily_total} * {7 if time_period == 'week' else 30} = {total} {food}.\n#### {total}"

        return {
            "question": question,
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_96(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, food: str, animal1: str, animal2: str, n1: int, n2: int, k1: int, k2: int, unit: str
    ) -> Dict[str, Any]:
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
            "answer": str(total),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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


def generate_97(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, mult_run: int, frac_skip: float, skip_speed: int, total_time: int, frac_run: float, frac_walk: float
    ) -> Dict[str, Any]:
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
            "answer": str(total_dist),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
        names = ["Dana", "Emma", "Sarah", "Julia", "Sophie", "Maria"]
        name = rng.choice(names)

        mult_run = rng.randint(2, int(6 * difficulty))
        frac_skip = 0.5  # Keep simple fraction
        skip_speed = rng.randint(2, int(10 * difficulty))
        total_time = rng.randrange(4, int(12 * difficulty), 2)

        # Ensure fractions add to 1
        frac_run = 1 / 3  # Keep simple fraction
        frac_walk = 2 / 3  # Keep simple fraction

        # Validate conditions
        while not (
            skip_speed / frac_skip < 13
            and (total_time * frac_walk * (skip_speed / frac_skip / mult_run)).is_integer()
            and (skip_speed / frac_skip).is_integer()
            and (total_time * frac_run).is_integer()
            and (total_time * frac_walk).is_integer()
        ):
            skip_speed = rng.randint(2, int(10 * difficulty))
            total_time = rng.randrange(4, int(12 * difficulty), 2)

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


def generate_98(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

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
    ) -> Dict[str, Any]:

        total_passenger_weight = num_passengers * weight_passenger
        total_weight = weight_vehicle + weight_item + total_passenger_weight
        force_needed = int((total_weight * force_percent) / 100)

        question = f"{name}'s {vehicle} breaks down. The {vehicle} weighs {weight_vehicle} {unit} and he has {item} in it weighing {weight_item} {unit}. He also has his {num_passengers} young {passenger_type} who weigh {weight_passenger} {unit} each in it. If the force to move the {vehicle} is {force_percent}% of the weight, how much force does he need to push the {vehicle}?"

        answer_cot = f"His {num_passengers} {passenger_type} weigh {weight_passenger}*{num_passengers}={total_passenger_weight} {unit}\nSo the total weight of the {vehicle} and everything is {weight_vehicle}+{weight_item}+{total_passenger_weight}={total_weight} {unit}\nSo he needs to generate {total_weight}*{force_percent/100}={force_needed} {unit}\n#### {force_needed}"

        return {
            "question": question,
            "answer": str(force_needed),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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

        weight_vehicle = int(rng.randrange(1000, int(3000 * difficulty), 50))
        weight_item = int(rng.randrange(100, int(500 * difficulty), 25))
        weight_passenger = int(rng.randrange(50, int(100 * difficulty), 5))
        num_passengers = int(rng.randint(2, int(5 * difficulty)))
        force_percent = int(rng.randint(1, int(6 * difficulty)))

        # Ensure force calculation results in integer
        total_weight = weight_vehicle + weight_item + (num_passengers * weight_passenger)
        while (total_weight * force_percent) % 100 != 0:
            force_percent = int(rng.randint(1, int(6 * difficulty)))

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


def generate_99(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:

    def generate_from_variables(
        name: str, currency: str, initial_amount: float, quantity: int, item: str, store_type: str, unit_price: float
    ) -> Dict[str, Any]:
        total_cost = quantity * unit_price
        remaining = initial_amount - total_cost

        question = f"{name} has {currency}{initial_amount:.2f} and wants to buy {quantity} {item}s from a bin at the {store_type} store. Each {item} costs {currency}{unit_price:.2f}. How much money does {name} have left after paying for the {item}s?"

        answer_cot = f"{name} paid {quantity} * {currency}{unit_price:.2f} = {currency}{total_cost:.2f} for the {item}s.\n{name} has {currency}{initial_amount:.2f} - {currency}{total_cost:.2f} = {currency}{int(remaining)} left.\n#### {int(remaining)}"

        return {
            "question": question,
            "answer": str(int(remaining)),
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

    def generate_example(rng: Random, difficulty: float = 1.0) -> Dict[str, Any]:
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
