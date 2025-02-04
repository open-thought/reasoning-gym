"""
clues.py

These are all the clue types that a puzzle can have. Things like "the tea drinker lives in the
green house" and "the cat owner lives left of the person who likes grilled cheese."

There's a Clue ABC that requires you implement an `as_cnf` method, to convert the clue to an
and-of-ors (probably using things defined in `sat_utils`), and a human-readable __repr__ that
can be used in a puzzle description.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from itertools import product
from typing import Iterable, List, Tuple

from reasoning_gym.logic.contrib.logic_puzzle.literals import Literal
from reasoning_gym.logic.contrib.logic_puzzle.sat_utils import from_dnf, neg


def _capitalize_first(repr_func):
    """
    Decorator for a __repr__ function that capitalizes the first letter without chagning the rest

    (in contrast to str.capitalize(), which capitalizes the first letter and makes the rest lower)
    """

    @wraps(repr_func)
    def wrapper(*args, **kwargs):
        output = repr_func(*args, **kwargs)
        return output[0].upper() + output[1:]

    return wrapper


class Clue(ABC):
    """Base class for the types of clues that we allow."""

    @abstractmethod
    def as_cnf(self) -> Iterable[Tuple[str]]: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    def __lt__(self, other) -> bool:
        return str(self) < str(other)


def comb(value: Literal, house: int) -> str:
    """Format how a value is shown at a given house"""

    return f"{value} {house}"


@dataclass(eq=True, frozen=True)
class found_at(Clue):
    """
    A literal is known to be at a specific house

    Examples:
     - the tea drinker lives in the middle house
     - the fourth house is red
    """

    value: Literal
    house: int

    def as_cnf(self) -> List[Tuple[str]]:
        return [(comb(self.value, self.house),)]

    @_capitalize_first
    def __repr__(self) -> str:
        houses = [None, "first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
        return f"{self.value.value} is in the {houses[self.house]} house."


@dataclass(eq=True, frozen=True)
class not_at(Clue):
    """
    Two values are known *not* to be at the same house

    Examples:
     - the musician does not drink tea
     - the red house does not contain a cat
    """

    value: Literal
    house: int

    def as_cnf(self) -> List[Tuple[str]]:
        return [(neg(comb(self.value, self.house)),)]

    @_capitalize_first
    def __repr__(self) -> str:
        houses = [None, "first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
        return f"{self.value.value} is not in the {houses[self.house]} house."


@dataclass(eq=True, frozen=True)
class same_house(Clue):
    """
    Two values are known to be at the same house

    Examples:
     - the musician drinks tea
     - the red house contains a cat
    """

    value1: Literal
    value2: Literal
    houses: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3, 4, 5))

    def as_cnf(self) -> List[Tuple[str]]:
        return from_dnf((comb(self.value1, i), comb(self.value2, i)) for i in self.houses)

    @_capitalize_first
    def __repr__(self) -> str:
        return f"{self.value1.value} is {self.value2.value}."


@dataclass(eq=True, frozen=True)
class consecutive(Clue):
    """
    The first value is directly to the left of the second value

    Examples:
     - the green house is directly to the left of the white house
       (green in 1, white in 2 OR green in 2, white in 3 OR etc.)
     - the house with the kittens is directly to the right of the tea drinker's home
       (kittens in 2, tea in 1 OR kittens in 3, tea in 2 OR etc.)
    """

    value1: Literal
    value2: Literal
    houses: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3, 4, 5))

    def as_cnf(self) -> List[Tuple[str]]:
        return from_dnf((comb(self.value1, i), comb(self.value2, j)) for i, j in zip(self.houses, self.houses[1:]))

    @_capitalize_first
    def __repr__(self) -> str:
        return f"{self.value1.value} is directly left of {self.value2.value}."


@dataclass(eq=True, frozen=True)
class beside(Clue):
    """
    The two values occur side-by-side (either left or right)

    Examples:
     - the coffee drinker is (left or right) of the tea drinker
     - the cat owner is (left or right) of the green house
    """

    value1: Literal
    value2: Literal
    houses: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3, 4, 5))

    def as_cnf(self) -> List[Tuple[str]]:
        return from_dnf(
            [(comb(self.value1, i), comb(self.value2, j)) for i, j in zip(self.houses, self.houses[1:])]
            + [(comb(self.value2, i), comb(self.value1, j)) for i, j in zip(self.houses, self.houses[1:])]
        )

    @_capitalize_first
    def __repr__(self) -> str:
        return f"{self.value1.value} and {self.value2.value} are next to each other."


@dataclass(eq=True, frozen=True)
class left_of(Clue):
    """
    The first value is somewhere to the left of the second value

    Examples:
     - the tea drinker is in house 1 and the musician in 2, 3, 4, or 5;
       OR the tea drinker in 2, and musician in 3, 4, or 5;
       OR the tea drinker in 3, musician in 4, 5; OR tea 4, musician 5.
    """

    value1: Literal
    value2: Literal
    houses: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3, 4, 5))

    def as_cnf(self) -> List[Tuple[str]]:
        return from_dnf(
            (comb(self.value1, i), comb(self.value2, j)) for i, j in product(self.houses, self.houses) if i < j
        )

    @_capitalize_first
    def __repr__(self) -> str:
        return f"{self.value1.value} is somewhere to the left of {self.value2.value}."


@dataclass(eq=True, frozen=True)
class right_of(Clue):
    """
    The first value is somewhere to the right of the second value.

    Examples:
     - the coffee drinker is in house 5 and the artist in 1, 2, 3, 4;
       OR the coffee drinker in 4, and artist in 1, 2, or 3;
       OR the coffee drinker in 3, artist in 1, 2; OR coffee 2, artist 1.
    """

    value1: Literal
    value2: Literal
    houses: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3, 4, 5))

    def as_cnf(self) -> List[Tuple[str]]:
        return sat_utils.from_dnf(
            (comb(self.value1, i), comb(self.value2, j)) for i, j in product(self.houses, self.houses) if i > j
        )

    @_capitalize_first
    def __repr__(self) -> str:
        return f"{self.value1.value} is somewhere to the right of {self.value2.value}."


@dataclass(eq=True, frozen=True)
class one_between(Clue):
    """
    The values are separated by one house

    Examples (if 5 houses):
     - the cat is in house 1 and tea drinker in house 3; OR cat 2, tea 4;
       OR cat 4 house 5
     - the green house is #1 and the musician in house 3; or green house 2, musician 4;
       OR green house 3, musician 5.
    """

    value1: Literal
    value2: Literal
    houses: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3, 4, 5))

    def as_cnf(self) -> List[Tuple[str]]:
        return from_dnf(
            [(comb(self.value1, i), comb(self.value2, j)) for i, j in zip(self.houses, self.houses[2:])]
            + [(comb(self.value2, i), comb(self.value1, j)) for i, j in zip(self.houses, self.houses[2:])]
        )

    def __repr__(self) -> str:
        return f"There is one house between {self.value1.value} and {self.value2.value}."


@dataclass(eq=True, frozen=True)
class two_between(Clue):
    """
    The values are separated by two houses

    Examples (if 5 houses):
     - the cat is in house 1 and artist in house 4; or cat 2, artist 5
     - the dog is in house 1 and red house is #4; or dog 2, red house 5
    """

    value1: Literal
    value2: Literal
    houses: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3, 4, 5))

    def as_cnf(self) -> List[Tuple[str]]:
        return from_dnf(
            [(comb(self.value1, i), comb(self.value2, j)) for i, j in zip(self.houses, self.houses[3:])]
            + [(comb(self.value2, i), comb(self.value1, j)) for i, j in zip(self.houses, self.houses[3:])]
        )

    def __repr__(self) -> str:
        return f"There are two houses between {self.value1.value} and {self.value2.value}."
