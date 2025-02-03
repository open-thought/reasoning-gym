"""
literals.py

This is a collection of "puzzle elements"---categories, literals, whatever you want to call
them---which are used as the building blocks of zebra puzzles. Examples include people's
favorite colors, preferred drinks, pets, etc.

Each class must provide (but we have no way of enforcing this) a description of each
puzzle element. These get used to make human-readable clues. The classes must also provide
a custom __repr__ method that gets used in the puzzle description.

Included is a base Literal class from which all literals should inherit. To extend these,
just import Literal and implement a class like the ones here.

"""

from enum import Enum


class Literal(Enum):
    """
    Common parent class for all puzzle elements (colors, occupations, pets, etc.).

    We can't make this an ABC because ABC and Enum have different metaclasses, and that'd be
    super confusing. But override the description method!
    """

    @classmethod
    def description(cls) -> str:
        return "".join(cls.__members__)  # type:ignore


class Color(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Each person has a favorite color:"

    yellow = "the person who loves yellow"
    red = "the person whose favorite color is red"
    white = "the person who loves white"
    green = "the person whose favorite color is green"
    blue = "the person who loves blue"
    purple = "the person who loves purple"
    brown = "the person who loves brown"






class Nationality(Literal):
    @classmethod
    def description(cls) -> str:
        return f"The people are of nationalities:"

    dane = "the Dane"
    brit = "the British person"
    swede = "the Swedish person"
    norwegian = "the Norwegian"
    german = "the German"
    chinese = "the Chinese"
    mexican = "the Mexican"


class Animal(Literal):
    @classmethod
    def description(cls) -> str:
        return f"The people keep different animals:"

    horse = "the person who keeps horses"
    cat = "the cat lover"
    bird = "the bird keeper"
    fish = "the fish enthusiast"
    dog = "the dog owner"
    rabbit = "the rabbit owner"
    pig = "the pig owner"


class Drink(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Each person has a favorite drink:"

    water = "the one who only drinks water"
    tea = "the tea drinker"
    milk = "the person who likes milk"
    coffee = "the coffee drinker"
    root_beer = "the root beer lover"
    boba_tea = "the boba tea drinker"
    wine = "the wine drinker"


class Cigar(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Everyone has a different favorite cigar:"

    pall_mall = "the person partial to Pall Mall"
    prince = "the Prince smoker"
    blue_master = "the person who smokes Blue Master"
    dunhill = "the Dunhill smoker"
    blends = "the person who smokes many different blends"
    yellow_monster = "the person who smokes Yellow Monster"
    red_eye = "the person who smokes Red Eye"


class Mother(Literal):
    @classmethod
    def description(cls) -> str: 
        return f"The mothers' names are:" 

    aniya = "The person whose mother's name is Aniya"
    holly = "The person whose mother's name is Holly"
    janelle = "The person whose mother's name is Janelle"
    kailyn = "The person whose mother's name is Kailyn"
    penny = "The person whose mother's name is Penny"


class Children(Literal):
    @classmethod
    def description(cls) -> str:
        return (
            f"Each mother is accompanied by their child: Bella, Fred, Meredith, Samantha, Timothy."
        )

    bella = "the person's child is named Bella"
    fred = "the person's child is named Fred"
    meredith = "the person's child is named Meredith"
    samantha = "the person's child is named Samantha"
    timothy = "the person who is the mother of Timothy"
    alice = "the person's child is named Alice"
    billy = "the person who is the mother of Billy"


class Flower(Literal):
    @classmethod
    def description(cls) -> str:
        return f"They all have a different favorite flower:"

    carnations = "the person who loves a carnations arrangement"
    daffodils = "the person who loves a bouquet of daffodils"
    lilies = "the person who loves the boquet of lilies"
    roses = "the person who loves the rose bouquet"
    tulips = "the person who loves the vase of tulips"
    iris = "the person who loves the boquet of iris"
    orchid = "the person who loves the boquet of orchid"


class Food(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Everyone has something different for lunch:"

    grilled_cheese = "the person who loves eating grilled cheese"
    pizza = "the person who is a pizza lover"
    spaghetti = "the person who loves the spaghetti eater"
    stew = "the person who loves the stew"
    stir_fry = "the person who loves stir fry"
    soup = "the person who loves the soup"
    sushi = "the person who loves the sushi"


class Kiro(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Each house has a different type of Kiro:"

    kaya = "the Kaya Kiro"
    sugar_sketch = "the Sugar Sketch Kiro"
    silosaur = "the Kiro disguised as a Silosaur"
    skyrant = "the Kiro in a Skyrant costume "
    traptor_costume = "the Kiro in a Traptor costume"
    terasaur_costume = "the Terasaur Costume Kiro"
    skeleko = "the Skeleko Kiro"
    zodiac_dragon = "the Zodiac Dragon Kiro"
    gem_dragon = "the Gem Dragon Kiro"
    plushie = "the Plushie Kiro"
    gloray = "the Gloray Kiro"
    rabbit = "the Rabbit Kiro"
    holiday_sweets = "the Holiday Sweets Kiro"
    baby = "the Baby Kiro"
    zaeris = "the Zaeris Kiro"


class Smoothie(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Everyone has a favorite smoothie:"

    cherry = "the person who likes Cherry smoothies"
    desert = "the Desert smoothie lover"
    watermelon = "the Watermelon smoothie lover"
    dragonfruit = "the Dragonfruit smoothie lover"
    lime = "the person who drinks Lime smoothies"
    blueberry = "the person who drinks Blueberry smoothies"
    lemon = "the Lemon smoothie lover"
    dusk = "the person whose favorite smoothie is Dusk"
    dawn = "the person who likes Dawn smoothies"
    spring = "the person who likes Spring smoothies"
    seafoam = "the person who likes Seafoam smoothies"
    phantom_spring = "the person who likes Phantom Spring smoothies"
    abyss = "the person whose favorite smoothie is Abyss"
    butterscotch = "the Butterscotch smoothie drinker"
    lilac = "the Lilac smoothie drinker"
    sakura = "the person whose favorite smoothie is Sakura"
    life = "the Life smoothie drinker"
    darkness = "the Darkness smoothie drinker"
    earth = "the person who likes Earth smoothies"


class Bottlecap(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Everyone keeps a certain type of Bottlecap:"

    red = "the person who has red bottlecaps"
    yellow = "the person who likes YBC"
    green = "the GBC keeper"
    blue = "the blue bottlecap hoarder"
    silver = "the SBC winner"


class RecolorMedal(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Everyone has a recolor or medal:"

    top_level = "the top level person"
    second_ed = "the 2nd edition person"
    ghost = "the ghost recolor"
    pink = "the pink person"
    gold = "the person with a heart of gold"


class NPC(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Each is an NPC on the site:"

    jim = "Dirt Digger Jim"
    amelia = "Amelia"
    chip = "Fishin' Chip"
    riley = "Ringmaster Riley"
    crowley = "Crowley"
    silver = "Silver the Kua"
    jagger = "Jagger"


class FavoriteGame(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Everyone has a favorite game:"

    dirt_digger = "the person who likes Dirt Digger"
    guess_the_number = "the one who plays Guess the Number"
    fishing_fever = "the Fishing Fever lover"
    sock_summoning = "the person who plays Sock Summoning"
    wonder_wheel = "the person who spins the Wonder Wheel"
    fetch = "the person playing Fetch"
    quality_assurance = "the person who plays Quality Assurance"
    stop_and_swap = "the one who often plays Stop and Swap"
    uchi_fusion = "the one who plays Uchi Fusion"
    freedom_forest = "the one in Freedom Forest"


class Tribe(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Everyone has an Altazan tribe:"

    quake = "the one in the Quake tribe"
    cursed = "the Cursed tribe member"
    forest = "the Forest tribe member"
    volcano = "the person in the Volcano tribe"
    storm = "the person in the Storm tribe"


class Kaya(Literal):
    @classmethod
    def description(cls) -> str:
        return f"They are five different types of Kaya:"

    joy = "the Kaya of Joy"
    life = "the Kaya of Life"
    harmony = "the Kaya of Harmony"
    wisdom = "the Kaya of Wisdom"
    love = "the Kaya of Love"


class TraptorPrimary(Literal):
    @classmethod
    def description(cls) -> str:
        return f"They have different primary colors:"

    majestic = "the Majestic Traptor"
    grand = "the Grand Traptor"
    stunning = "the Stunning Traptor"
    marvellous = "the Marvellous Traptor"
    heroic = "the Heroic Traptor"


class TraptorSecondary(Literal):
    @classmethod
    def description(cls) -> str:
        return f"They have different secondary colors:"

    sky = "the Sky Traptor"
    forest = "the Forest Traptor"
    night = "the Night Traptor"
    sun = "the Sun Traptor"
    sand = "the Sand Traptor"


class TraptorTertiary(Literal):
    @classmethod
    def description(cls) -> str:
        return f"They have different tertiary colors:"

    soarer = "the Soarer Traptor"
    diver = "the Diver Traptor"
    screecher = "the Screecher Traptor"
    hunter = "the Hunter Traptor"
    nurturer = "the Nurturer Traptor"


class Egg(Literal):
    @classmethod
    def description(cls) -> str:
        return f"They are each giving out a type of egg:"

    golden = "the one giving out Golden Eggs"
    trollden = "the one who keeps Trollden Eggs"
    topaz = "the one with Topaz Eggs"
    crystal = "the one giving out Crystal Eggs"
    traptor = "the one who has Traptor Eggs"
    marinodon = "the one giving out Marinodon Eggs"


class Dinomon(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Each is a different species of Dinomon:"

    terasaur = "the Terasaur"
    carnodon = "the Carnodon"
    silosaur = "the Silosaur"
    marinodon = "the Marinodon"
    traptor = "the Traptor"


# added by yuchenl  

class Name(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Each person has a unique name:"

    arnold = "Arnold"
    eric = "Eric"
    peter = "Peter"
    alice = "Alice"
    bob = "Bob"
    carol = "Carol"
    david = "David"
    emily = "Emily"

class Age(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Each person has a unique age:"

    age_7 = "the person who is 7 years old"
    age_8 = "the person who is 8"
    age_9 = "the person who is 9"
    age_10 = "the person who is 10"
    age_11 = "the person who is 11"
    age_12 = "the person who is 12"
    age_13 = "the person who is 13"
    age_14 = "the person who is 14"

class Birthday(Literal):
    @classmethod
    def description(cls) -> str:
        return f"Each person has a unique birthday month:"

    april = "the person whose birthday is in April"
    sept = "the person whose birthday is in September"
    jan = "the person whose birthday is in January"
    feb = "the person whose birthday is in February"
    mar = "the person whose birthday is in March"
    may = "the person whose birthday is in May"
    june = "the person whose birthday is in June"
    july = "the person whose birthday is in July"


# ChatGPT generated 

"""
generate more classes like this according to the common properties that you would describe a person or a house, such as the occupation, age, height, hair color, and so on

"""
 
 

class Occupation(Literal):
    @classmethod
    def description(cls) -> str:
        return "Each person has an occupation: doctor, engineer, teacher, artist, lawyer, nurse, or accountant."

    doctor = "the person who is a doctor"
    engineer = "the person who is an engineer"
    teacher = "the person who is a teacher"
    artist = "the person who is an artist"
    lawyer = "the person who is a lawyer"
    nurse = "the person who is a nurse"
    accountant = "the person who is an accountant"

# class AgeGroup(Literal):
#     @classmethod
#     def description(cls) -> str:
#         return "People are grouped by age: infant, child, teenager, young adult, adult, or senior."

#     infant = "the person who is an infant"
#     child = "the person who is a child"
#     teenager = "the person who is a teenager"
#     young_adult = "the person who is a young adult"
#     adult = "the person who is an adult"
#     senior = "the person who is a senior"

class Height(Literal):
    @classmethod
    def description(cls) -> str:
        return "People have different heights: very short, short, average, tall, very tall, or super tall."

    very_short = "the person who is very short"
    short = "the person who is short"
    average = "the person who has an average height"
    tall = "the person who is tall"
    very_tall = "the person who is very tall"
    super_tall = "the person who is super tall"

class HairColor(Literal):
    @classmethod
    def description(cls) -> str:
        return "People have different hair colors: black, brown, blonde, red, gray, auburn, or white."

    black = "the person who has black hair"
    brown = "the person who has brown hair"
    blonde = "the person who has blonde hair"
    red = "the person who has red hair"
    gray = "the person who has gray hair"
    auburn = "the person who has auburn hair"
    white = "the person who has white hair"

class CarModel(Literal):
    @classmethod
    def description(cls) -> str:
        return "People own different car models: Tesla Model 3, Ford F-150, Toyota Camry, Honda Civic, BMW 3 Series, Chevrolet Silverado, or Audi A4."

    tesla_model_3 = "the person who owns a Tesla Model 3"
    ford_f150 = "the person who owns a Ford F-150"
    toyota_camry = "the person who owns a Toyota Camry"
    honda_civic = "the person who owns a Honda Civic"
    bmw_3_series = "the person who owns a BMW 3 Series"
    chevrolet_silverado = "the person who owns a Chevrolet Silverado"
    audi_a4 = "the person who owns an Audi A4"

class PhoneModel(Literal):
    @classmethod
    def description(cls) -> str:
        return "People use different phone models: iPhone 13, Samsung Galaxy S21, Google Pixel 6, OnePlus 9, Huawei P50, Xiaomi Mi 11, or Sony Xperia 5."

    iphone_13 = "the person who uses an iPhone 13"
    samsung_galaxy_s21 = "the person who uses a Samsung Galaxy S21"
    google_pixel_6 = "the person who uses a Google Pixel 6"
    oneplus_9 = "the person who uses a OnePlus 9"
    huawei_p50 = "the person who uses a Huawei P50"
    xiaomi_mi_11 = "the person who uses a Xiaomi Mi 11"
    sony_xperia_5 = "the person who uses a Sony Xperia 5"


class FavoriteSport(Literal):
    @classmethod
    def description(cls) -> str:
        return "People have different favorite sports: soccer, basketball, tennis, swimming, baseball, volleyball, or golf."
    soccer = "the person who loves soccer"
    basketball = "the person who loves basketball"
    tennis = "the person who loves tennis"
    swimming = "the person who loves swimming"
    baseball = "the person who loves baseball"
    volleyball = "the person who loves volleyball"
    golf = "the person who loves golf"

class MusicGenre(Literal):
    @classmethod
    def description(cls) -> str:
        return "People have different favorite music genres: rock, pop, classical, jazz, hip-hop, country, or electronic."
    rock = "the person who loves rock music"
    pop = "the person who loves pop music"
    classical = "the person who loves classical music"
    jazz = "the person who loves jazz music"
    hip_hop = "the person who loves hip-hop music"
    country = "the person who loves country music"
    electronic = "the person who loves electronic music"

class BookGenre(Literal):
    @classmethod
    def description(cls) -> str:
        return "People have different favorite book genres: mystery, science fiction, romance, fantasy, biography, historical fiction, or non-fiction."
    mystery = "the person who loves mystery books"
    science_fiction = "the person who loves science fiction books"
    romance = "the person who loves romance books"
    fantasy = "the person who loves fantasy books"
    biography = "the person who loves biography books"
    historical_fiction = "the person who loves historical fiction books"
    non_fiction = "the person who loves non-fiction books"

