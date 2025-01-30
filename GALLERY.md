# Reasoning Gym Dataset Gallery
This gallery shows examples from all available datasets using their default configurations.

## Available Datasets
- [base_conversion](#base-conversion)
- [basic_arithmetic](#basic-arithmetic)
- [caesar_cipher](#caesar-cipher)
- [chain_sum](#chain-sum)
- [color_cube_rotation](#color-cube-rotation)
- [countdown](#countdown)
- [family_relationships](#family-relationships)
- [figlet_font](#figlet-font)
- [fraction_simplification](#fraction-simplification)
- [game_of_life](#game-of-life)
- [gcd](#gcd)
- [lcm](#lcm)
- [leg_counting](#leg-counting)
- [letter_counting](#letter-counting)
- [letter_jumble](#letter-jumble)
- [maze](#maze)
- [mini_sudoku](#mini-sudoku)
- [number_filtering](#number-filtering)
- [number_sequence](#number-sequence)
- [number_sorting](#number-sorting)
- [polynomial_equations](#polynomial-equations)
- [prime_factorization](#prime-factorization)
- [propositional_logic](#propositional-logic)
- [quantum_lock](#quantum-lock)
- [rubiks_cube](#rubiks-cube)
- [sentence_reordering](#sentence-reordering)
- [simple_equations](#simple-equations)
- [spell_backward](#spell-backward)
- [sudoku](#sudoku)
- [syllogism](#syllogism)
- [word_sequence_reversal](#word-sequence-reversal)
- [word_sorting](#word-sorting)

## Dataset Examples
### base_conversion
Generates base conversion tasks

Default configuration:
```python
min_base = 2
max_base = 16
min_value = 0
max_value = 1000
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Convert the base-15 number 15 to binary
Answer: 10101
Metadata: {'decimal_value': 21, 'source_base': 15, 'target_base': 2, 'source_repr': '15', 'target_repr': '10101'}

Example 2:
Question: Convert the base-15 number de to base-6
Answer: de
Metadata: {'decimal_value': 222, 'source_base': 15, 'target_base': 6, 'source_repr': 'de', 'target_repr': 'de'}

Example 3:
Question: Convert the base-10 number 4e to binary
Answer: 1001110
Metadata: {'decimal_value': 78, 'source_base': 10, 'target_base': 2, 'source_repr': '4e', 'target_repr': '1001110'}

```

### basic_arithmetic
Dataset that generates basic arithmetic tasks with configurable complexity

Default configuration:
```python
min_terms = 2
max_terms = 6
min_digits = 1
max_digits = 4
operators = ('+', '-', '*', '/')
allow_parentheses = True
allow_negation = True
seed = None
size = 500
format_style = simple
whitespace = single
```

Example tasks:
```
Example 1:
Question: 19 + 61 * -43 / 1 + 89 - 98 =
Answer: -2613
Metadata: {'num_terms': 6, 'num_digits': 2, 'expression': '19 + 61 * -43 / 1 + 89 - 98'}

Example 2:
Question: ( 9240 + -702 ) =
Answer: 8538
Metadata: {'num_terms': 2, 'num_digits': 4, 'expression': '( 9240 + -702 )'}

Example 3:
Question: -68 * 12 - 6 / 2 + -60 =
Answer: -879
Metadata: {'num_terms': 5, 'num_digits': 2, 'expression': '-68 * 12 - 6 / 2 + -60'}

```

### caesar_cipher
Generates Caesar cipher encryption/decryption tasks

Default configuration:
```python
delimiter = .
min_words = 3
max_words = 20
min_rotation = 1
max_rotation = 25
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Decrypt this Caesar cipher text: UVYAO MVY AOL VM IBA AOL ZVBAO MVY AOL SHAPUZ
Answer: NORTH FOR THE OF BUT THE SOUTH FOR THE LATINS
Metadata: {'rotation': 7, 'cipher_text': 'UVYAO MVY AOL VM IBA AOL ZVBAO MVY AOL SHAPUZ', 'clear_text': 'NORTH FOR THE OF BUT THE SOUTH FOR THE LATINS'}

Example 2:
Question: Decrypt this Caesar cipher text: ER MRHITIRHIRX KSZIVRQIRX
Answer: AN INDEPENDENT GOVERNMENT
Metadata: {'rotation': 4, 'cipher_text': 'ER MRHITIRHIRX KSZIVRQIRX', 'clear_text': 'AN INDEPENDENT GOVERNMENT'}

Example 3:
Question: Decrypt this Caesar cipher text: IYE WKI ECO DRSC OLYYU PYB XOKBVI KXI ZEBZYCO CEMR KC MBOKDSYX YP NOBSFKDSFO ZOBPYBWKXMOC KXN BOCOKBMR
Answer: YOU MAY USE THIS EBOOK FOR NEARLY ANY PURPOSE SUCH AS CREATION OF DERIVATIVE PERFORMANCES AND RESEARCH
Metadata: {'rotation': 10, 'cipher_text': 'IYE WKI ECO DRSC OLYYU PYB XOKBVI KXI ZEBZYCO CEMR KC MBOKDSYX YP NOBSFKDSFO ZOBPYBWKXMOC KXN BOCOKBMR', 'clear_text': 'YOU MAY USE THIS EBOOK FOR NEARLY ANY PURPOSE SUCH AS CREATION OF DERIVATIVE PERFORMANCES AND RESEARCH'}

```

### chain_sum
Generates simple arithmetic tasks using only + and - operators

Default configuration:
```python
min_terms = 2
max_terms = 6
min_digits = 1
max_digits = 4
allow_negation = False
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: 3 - 6 + 4 =
Answer: 1
Metadata: {'num_terms': 3, 'num_digits': 1, 'expression': '3 - 6 + 4'}

Example 2:
Question: 6516 - 9002 - 5380 - 2663 =
Answer: -10529
Metadata: {'num_terms': 4, 'num_digits': 4, 'expression': '6516 - 9002 - 5380 - 2663'}

Example 3:
Question: 3352 + 3153 - 3475 + 1726 - 8711 - 7863 =
Answer: -11818
Metadata: {'num_terms': 6, 'num_digits': 4, 'expression': '3352 + 3153 - 3475 + 1726 - 8711 - 7863'}

```

### color_cube_rotation
Generates color cube rotation reasoning tasks

Default configuration:
```python
min_rotations = 1
max_rotations = 3
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: A cube has:
- a red top side
- a brown right side
- a cyan front side
- a gray left side
- a silver back side
- a purple bottom side

The cube is rotated so that the side which was before at the front is now at the top.

Now the cube is rotated to place its right side at the top.

What is now the color of the top side of the cube?
Answer: brown
Metadata: {'initial_state': {'top': 'red', 'right': 'brown', 'front': 'cyan', 'left': 'gray', 'back': 'silver', 'bottom': 'purple'}, 'rotations': ['front', 'right'], 'target_side': 'top', 'num_rotations': 2}

Example 2:
Question: A cube has:
- a yellow top side
- a cyan right side
- a white front side
- a blue left side
- a red back side
- a pink bottom side

The cube is rotated so that the side which was before at the left is now at the top.

Then the cube is rotated to bring the front side to the top.

Next, the front side is rotated to become the top face.

What is now the color of the front side of the cube?
Answer: red
Metadata: {'initial_state': {'top': 'yellow', 'right': 'cyan', 'front': 'white', 'left': 'blue', 'back': 'red', 'bottom': 'pink'}, 'rotations': ['left', 'front', 'front'], 'target_side': 'front', 'num_rotations': 3}

Example 3:
Question: A cube has:
- a indigo top side
- a violet right side
- a silver front side
- a pink left side
- a magenta back side
- a cyan bottom side

The cube is rotated so that the side which was before at the front is now at the top.

What is now the color of the top side of the cube?
Answer: silver
Metadata: {'initial_state': {'top': 'indigo', 'right': 'violet', 'front': 'silver', 'left': 'pink', 'back': 'magenta', 'bottom': 'cyan'}, 'rotations': ['front'], 'target_side': 'top', 'num_rotations': 1}

```

### countdown
Generates Countdown Number Game tasks

Default configuration:
```python
min_numbers = 4
max_numbers = 6
min_value = 1
max_value = 100
min_target = 100
max_target = 999
operators = ('+', '-', '*', '/')
shuffle = True
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Calculate 421 using the numbers 10, 30, 26, 59.
Each number may be used at most once.
Answer: 30*(26 - 10) - 59
Metadata: {'numbers': [10, 30, 26, 59], 'target': 421, 'expression': '30*(26 - 10) - 59'}

Example 2:
Question: Calculate 229 using the numbers 55, 80, 34, 60.
Each number may be used at most once.
Answer: 80 + 34 + 60 + 55
Metadata: {'numbers': [55, 80, 34, 60], 'target': 229, 'expression': '80 + 34 + 60 + 55'}

Example 3:
Question: Calculate 840 using the numbers 41, 18, 32, 45, 84.
Each number may be used at most once.
Answer: 84*(41 - 45 + 32 - 18)
Metadata: {'numbers': [41, 18, 32, 45, 84], 'target': 840, 'expression': '84*(41 - 45 + 32 - 18)'}

```

### family_relationships
Generates family relationship reasoning tasks

Default configuration:
```python
min_family_size = 4
max_family_size = 8
male_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles', 'Peter', 'Daniel', 'Matthew', 'Christopher', 'Andrew', 'George', 'Edward', 'Benjamin', 'Henry', 'Samuel', 'Alexander', 'Oliver', 'Jack', 'Harry', 'Jacob', 'Noah', 'Ethan', 'Lucas', 'Mason', 'Logan', 'Sebastian', 'Theodore', 'Owen', 'Liam', 'Aiden', 'Kai', 'Jayden', 'Zion', 'Phoenix', 'Atlas', 'Axel', 'Ryder', 'Finn']
female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Emma', 'Lisa', 'Anna', 'Margaret', 'Victoria', 'Charlotte', 'Sophia', 'Isabella', 'Olivia', 'Ava', 'Mia', 'Emily', 'Abigail', 'Amelia', 'Eleanor', 'Grace', 'Alice', 'Lucy', 'Chloe', 'Sophie', 'Lily', 'Hannah', 'Zoe', 'Luna', 'Nova', 'Aria', 'Willow', 'Aurora', 'Sage', 'River', 'Winter', 'Sky', 'Rain']
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Jack is married to Elizabeth. They have a child called Oliver. Oliver is married to Abigail. They have a child called Logan. Alexander is married to Mia. They have a child called Abigail.

What relation is Mia to Abigail?
Answer: mother
Metadata: {'person1': 'Mia', 'person2': 'Abigail', 'relationship': 'mother', 'family_size': 7}

Example 2:
Question: James is married to Sarah. They have a child called Atlas. Atlas is married to Sophie. They have children called Jennifer and Aria.

What is Aria to Jennifer?
Answer: sister
Metadata: {'person1': 'Aria', 'person2': 'Jennifer', 'relationship': 'sister', 'family_size': 6}

Example 3:
Question: Lucas is married to Willow. They have a child called Samuel. Samuel is married to Zoe. They have a child called William. Henry is married to Emma. They have a child called Zoe.

What is Lucas to Willow?
Answer: husband
Metadata: {'person1': 'Lucas', 'person2': 'Willow', 'relationship': 'husband', 'family_size': 7}

```

### figlet_font
Generates FigletFont tasks

Default configuration:
```python
static_word = None
static_font = None
space_letters = True
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Please read the following figlet font:

  ()     _     _       _    _ __     ()  ,
  /\    ' )   /       | )  ' )  )    /`-'|
 /  )    / / /    ,---|/    /  /    /   /
/__/__  (_(_/      \_/ \_  /  (_   /__-<_



Answer: SWING
Metadata: {'font': 'slscript', 'space_letters': True}

Example 2:
Question: What word does this say?

     dBBBP     dBBBBBb        dBBBP     dBP dBP    dBBBP
                    BB
   dBP          dBP BB      dBP       dBBBBBP    dBBP
  dBP          dBP  BB     dBP       dBP dBP    dBP
 dBBBBP       dBBBBBBB    dBBBBP    dBP dBP    dBBBBP


Answer: CACHE
Metadata: {'font': 'trek', 'space_letters': True}

Example 3:
Question: Please read the following figlet font:

.---. .---. .-. .-..-. .-..-.
 \ \  | |-' | | | .` |  >  /
`---' `-'   `-' `-'`-'  `-'


Answer: SPINY
Metadata: {'font': 'linux', 'space_letters': True}

```

### fraction_simplification
Generates fraction simplification tasks

Default configuration:
```python
min_value = 1
max_value = 1000
min_factor = 1
max_factor = 100
styles = ('plain', 'latex_inline', 'latex_frac', 'latex_dfrac')
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Simplify the fraction $1380/6180$ to its lowest terms
Answer: $23/103$
Metadata: {'numerator': 1380, 'denominator': 6180, 'simplified_numerator': 23, 'simplified_denominator': 103, 'reduction_factor': 60, 'style': 'latex_inline'}

Example 2:
Question: Simplify the fraction 15552/49984 to its lowest terms
Answer: 243/781
Metadata: {'numerator': 15552, 'denominator': 49984, 'simplified_numerator': 243, 'simplified_denominator': 781, 'reduction_factor': 64, 'style': 'plain'}

Example 3:
Question: Simplify the fraction $56100/80500$ to its lowest terms
Answer: $561/805$
Metadata: {'numerator': 56100, 'denominator': 80500, 'simplified_numerator': 561, 'simplified_denominator': 805, 'reduction_factor': 100, 'style': 'latex_inline'}

```

### game_of_life
Generates Game of Life games with configurable parameters

Default configuration:
```python
grid_size_x = 20
grid_size_y = 20
filled_cells = 100
simulation_steps = 1
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: What will this Game of Life board look like after 1 steps of simulation?

[[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
 [0 1 1 1 0 1 1 0 0 0 1 0 0 0 0 0 0 0 1 0]
 [0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1]
 [0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 1 0 1]
 [0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 0 0 1]
 [0 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 0 1 1 0]
 [0 0 1 0 1 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0]
 [0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0]
 [1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1]
 [0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0]
 [0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 1 0]
 [0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 1]
 [0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 1 0 1 0]
 [0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0]
 [1 1 0 0 1 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0]]
Answer: [[1 0 0 1 1 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0]
 [0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1]
 [0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0]
 [0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1 1 0]
 [0 0 1 0 0 0 0 1 1 0 1 1 0 1 0 0 0 1 1 0]
 [0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0]
 [0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0]
 [0 1 0 0 1 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0]
 [0 1 1 0 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0]
 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0]
 [0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 1]
 [0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 1 1 1 0 0 1 1 0 0 1 1 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0]]
Metadata: {'grid_size_x': 20, 'grid_size_y': 20, 'filled_cells': 100, 'simulation_steps': 1}

Example 2:
Question: What will this Game of Life board look like after 1 steps of simulation?

[[0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0]
 [0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0]
 [1 0 0 0 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 1 0 0 1 0 0 1 1 0 0 0 0]
 [0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 1]
 [0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 1]
 [1 0 0 0 1 0 0 0 0 0 0 1 0 0 1 1 1 0 0 0]
 [0 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1]
 [0 0 1 0 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 1]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0]
 [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1]
 [1 0 1 0 0 0 0 1 0 0 0 0 0 1 1 1 0 1 0 0]
 [0 1 0 0 1 0 1 1 0 0 1 1 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 0 0]
 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
 [0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0]]
Answer: [[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0]
 [0 0 0 1 1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1]
 [0 0 0 0 1 0 0 0 0 0 1 1 0 1 1 0 0 0 0 0]
 [0 0 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 0 0 1]
 [1 0 0 0 0 0 1 1 0 1 1 0 0 0 1 1 0 0 1 1]
 [0 1 0 0 0 0 0 0 1 1 1 0 0 0 1 1 1 0 1 1]
 [1 0 1 1 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0]
 [0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [0 1 1 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 1]
 [1 1 0 0 0 0 1 1 1 0 1 0 0 0 1 1 0 0 0 0]
 [0 1 0 0 0 0 1 1 0 1 1 1 1 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 0]]
Metadata: {'grid_size_x': 20, 'grid_size_y': 20, 'filled_cells': 100, 'simulation_steps': 1}

Example 3:
Question: What will this Game of Life board look like after 1 steps of simulation?

[[1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1]
 [0 0 0 1 0 1 0 1 1 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 1 0 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 1 0 0 0 0 1 0 1 1 1 0 0 1]
 [1 1 0 0 0 1 1 0 0 0 0 1 1 1 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0]
 [0 0 1 1 0 0 1 0 0 1 0 1 0 0 1 0 0 1 0 0]
 [0 0 1 1 0 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0]
 [0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1]
 [0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0]
 [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 1 1 0 0 1 1 0 0 0 1 1 0 0]
 [0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 1]
 [0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0]
 [1 0 0 1 0 0 0 0 0 0 1 1 1 0 0 1 0 0 0 0]]
Answer: [[1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1]
 [0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0]
 [0 0 1 1 1 0 0 1 0 0 0 1 1 0 1 0 0 0 0 0]
 [1 1 1 0 0 0 0 1 0 0 1 0 0 0 1 1 1 0 0 0]
 [1 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0]
 [0 1 1 0 0 1 1 0 0 0 0 1 0 1 0 1 1 0 0 0]
 [0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 1 1 0 0 1 1 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 1 0]
 [0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0]
 [0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0]
 [0 0 0 1 1 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 1 1 1 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0]]
Metadata: {'grid_size_x': 20, 'grid_size_y': 20, 'filled_cells': 100, 'simulation_steps': 1}

```

### gcd
Generates Greatest Common Divisor (GCD) tasks

Default configuration:
```python
min_numbers = 2
max_numbers = 2
min_value = 1
max_value = 1000
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 226, 512
Answer: 2
Metadata: {'numbers': [226, 512], 'result': 2}

Example 2:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 999, 495
Answer: 9
Metadata: {'numbers': [999, 495], 'result': 9}

Example 3:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 999, 719
Answer: 1
Metadata: {'numbers': [999, 719], 'result': 1}

```

### lcm
Generates Least Common Multiple (LCM) tasks

Default configuration:
```python
min_numbers = 2
max_numbers = 2
min_value = 1
max_value = 100
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Find the Least Common Multiple (LCM) of these numbers: 30, 69
Answer: 690
Metadata: {'numbers': [30, 69], 'result': 690}

Example 2:
Question: Find the Least Common Multiple (LCM) of these numbers: 57, 99
Answer: 1881
Metadata: {'numbers': [57, 99], 'result': 1881}

Example 3:
Question: Find the Least Common Multiple (LCM) of these numbers: 3, 24
Answer: 24
Metadata: {'numbers': [3, 24], 'result': 24}

```

### leg_counting
Generates leg counting arithmetic tasks

Default configuration:
```python
min_animals = 2
max_animals = 5
max_instances = 3
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: How many legs are there in total if you have 1 starfish, 3 crabs, 3 chickens, 3 cows, 1 woodlouse?
Answer: 67
Metadata: {'animals': {'starfish': 1, 'crab': 3, 'chicken': 3, 'cow': 3, 'woodlouse': 1}, 'total_legs': 67}

Example 2:
Question: How many legs are there in total if you have 2 sheeps, 1 butterfly, 1 ant, 3 humans, 2 wasps?
Answer: 38
Metadata: {'animals': {'sheep': 2, 'butterfly': 1, 'ant': 1, 'human': 3, 'wasp': 2}, 'total_legs': 38}

Example 3:
Question: How many legs are there in total if you have 3 chickens, 3 cockroachs, 3 woodlouses, 2 elephants, 2 sea slugs?
Answer: 74
Metadata: {'animals': {'chicken': 3, 'cockroach': 3, 'woodlouse': 3, 'elephant': 2, 'sea slug': 2}, 'total_legs': 74}

```

### letter_counting
Generates letter counting tasks from text spans

Default configuration:
```python
min_words = 5
max_words = 15
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: How many times does the letter "r" appear in the text: "You decline All is over then murmured the British agent sadly The"?
Answer: 4
Metadata: {'span_length': 12, 'target_letter': 'r', 'span': ['You', 'decline', 'All', 'is', 'over', 'then', 'murmured', 'the', 'British', 'agent', 'sadly', 'The']}

Example 2:
Question: How many times does the letter "l" appear in the text: "coffined and laid in a tomb Time went on September 25th 2889"?
Answer: 1
Metadata: {'span_length': 12, 'target_letter': 'l', 'span': ['coffined', 'and', 'laid', 'in', 'a', 'tomb', 'Time', 'went', 'on', 'September', '25th', '2889']}

Example 3:
Question: How many times does the letter "i" appear in the text: "to the works took more time than he had anticipated It was"?
Answer: 4
Metadata: {'span_length': 12, 'target_letter': 'i', 'span': ['to', 'the', 'works', 'took', 'more', 'time', 'than', 'he', 'had', 'anticipated', 'It', 'was']}

```

### letter_jumble
Generates word letter jumbling tasks

Default configuration:
```python
min_word_len = 1
max_word_len = 64
min_words = 3
max_words = 20
min_corruption_level = 0.1
max_corruption_level = 0.9
consecutive_words = True
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Unscramble these words: moon abotu faec hA trehe s somethnig ni htat driec eht owt nem ta ocne dnA dndeei
Answer: moon about face Ah there s something in that cried the two men at once And indeed
Metadata: {'num_words': 17, 'corruption_level': 0.16056171448414203, 'scrambled_words': ['moon', 'abotu', 'faec', 'hA', 'trehe', 's', 'somethnig', 'ni', 'htat', 'driec', 'eht', 'owt', 'nem', 'ta', 'ocne', 'dnA', 'dndeei'], 'original_words': ['moon', 'about', 'face', 'Ah', 'there', 's', 'something', 'in', 'that', 'cried', 'the', 'two', 'men', 'at', 'once', 'And', 'indeed']}

Example 2:
Question: Unscramble these words: lla het aosssen eth msea I psrooep ot od toshmeign etrtbe itlsl amrnsTrfo toni aeht a tiooprn fo het
Answer: all the seasons the same I propose to do something better still Transform into heat a portion of the
Metadata: {'num_words': 19, 'corruption_level': 0.8984516776838924, 'scrambled_words': ['lla', 'het', 'aosssen', 'eth', 'msea', 'I', 'psrooep', 'ot', 'od', 'toshmeign', 'etrtbe', 'itlsl', 'amrnsTrfo', 'toni', 'aeht', 'a', 'tiooprn', 'fo', 'het'], 'original_words': ['all', 'the', 'seasons', 'the', 'same', 'I', 'propose', 'to', 'do', 'something', 'better', 'still', 'Transform', 'into', 'heat', 'a', 'portion', 'of', 'the']}

Example 3:
Question: Unscramble these words: od ubt si ti fo yna sue Waht ew need si csoudl ont iarn oG dais eh addressing
Answer: do but is it of any use What we need is clouds not rain Go said he addressing
Metadata: {'num_words': 18, 'corruption_level': 0.21786426698317396, 'scrambled_words': ['od', 'ubt', 'si', 'ti', 'fo', 'yna', 'sue', 'Waht', 'ew', 'need', 'si', 'csoudl', 'ont', 'iarn', 'oG', 'dais', 'eh', 'addressing'], 'original_words': ['do', 'but', 'is', 'it', 'of', 'any', 'use', 'What', 'we', 'need', 'is', 'clouds', 'not', 'rain', 'Go', 'said', 'he', 'addressing']}

```

### maze
Generates mazes with guaranteed shortest path distance from start to goal
    within [min_dist, max_dist].

Default configuration:
```python
min_dist = 5
max_dist = 10
min_grid_size = 5
max_grid_size = 10
seed = None
size = 50
```

Example tasks:
```
Example 1:
Question: Navigate from 'F' (start) to 'S' (goal):

```DDDDDDD
D]D]]DD
DD]DD]D
DDS]]]D
D]]D]]D
D]]]]FD
DDDDDDD```
Legend: 'D' = Wall, ']' = Passage

What is the minimum number of steps to reach the goal?
Answer: 5
Metadata: {'grid_size': 7, 'grid': ['DDDDDDD', 'D]D]]DD', 'DD]DD]D', 'DDS]]]D', 'D]]D]]D', 'D]]]]FD', 'DDDDDDD'], 'shortest_path_length': 5, 'start': 'F', 'goal': 'S', 'wall': 'D', 'path': ']'}

Example 2:
Question: Navigate from 'V' (start) to 'S' (goal):

```77777777
77SUU777
7U7UUUU7
77UUU777
7UU7UUU7
77U7UUU7
7UUU7UV7
77777777```
Legend: '7' = Wall, 'U' = Passage

What is the minimum number of steps to reach the goal?
Answer: 9
Metadata: {'grid_size': 8, 'grid': ['77777777', '77SUU777', '7U7UUUU7', '77UUU777', '7UU7UUU7', '77U7UUU7', '7UUU7UV7', '77777777'], 'shortest_path_length': 9, 'start': 'V', 'goal': 'S', 'wall': '7', 'path': 'U'}

Example 3:
Question: Navigate from 'z' (start) to '4' (goal):

```$$$$$$$
$~~~~~$
$$~$~~$
$~$~$4$
$$~~~~$
$~z~~~$
$$$$$$$```
Legend: '$' = Wall, '~' = Passage

What is the minimum number of steps to reach the goal?
Answer: 5
Metadata: {'grid_size': 7, 'grid': ['$$$$$$$', '$~~~~~$', '$$~$~~$', '$~$~$4$', '$$~~~~$', '$~z~~~$', '$$$$$$$'], 'shortest_path_length': 5, 'start': 'z', 'goal': '4', 'wall': '$', 'path': '~'}

```

### mini_sudoku
Generates 4x4 sudoku puzzles with configurable difficulty

Default configuration:
```python
min_empty = 8
max_empty = 12
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Solve this 4x4 Mini Sudoku puzzle:
1 _ _ _
_ 4 _ _
_ _ _ 3
_ _ 1 4
Answer: 1 3 4 2
2 4 3 1
4 1 2 3
3 2 1 4
Metadata: {'puzzle': [[1, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 3], [0, 0, 1, 4]], 'solution': [[1, 3, 4, 2], [2, 4, 3, 1], [4, 1, 2, 3], [3, 2, 1, 4]], 'num_empty': 11}

Example 2:
Question: Solve this 4x4 Mini Sudoku puzzle:
_ _ _ 2
2 _ _ 4
_ 4 _ _
_ 2 4 _
Answer: 4 3 1 2
2 1 3 4
1 4 2 3
3 2 4 1
Metadata: {'puzzle': [[0, 0, 0, 2], [2, 0, 0, 4], [0, 4, 0, 0], [0, 2, 4, 0]], 'solution': [[4, 3, 1, 2], [2, 1, 3, 4], [1, 4, 2, 3], [3, 2, 4, 1]], 'num_empty': 10}

Example 3:
Question: Solve this 4x4 Mini Sudoku puzzle:
4 2 _ _
3 _ 2 4
_ _ _ _
_ 4 3 2
Answer: 4 2 1 3
3 1 2 4
2 3 4 1
1 4 3 2
Metadata: {'puzzle': [[4, 2, 0, 0], [3, 0, 2, 4], [0, 0, 0, 0], [0, 4, 3, 2]], 'solution': [[4, 2, 1, 3], [3, 1, 2, 4], [2, 3, 4, 1], [1, 4, 3, 2]], 'num_empty': 8}

```

### number_filtering
Generates number filtering tasks

Default configuration:
```python
min_numbers = 3
max_numbers = 10
min_decimals = 0
max_decimals = 4
min_value = -100.0
max_value = 100.0
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Remove all numbers smaller than -78.527 in this list: ['-14.14', '10.92', '-56.57', '-56', '-84.8', '20']
Answer: ['-14.14', '10.92', '-56.57', '-56', '20']
Metadata: {'original_numbers': ['-14.14', '10.92', '-56.57', '-56', '-84.8', '20'], 'filter_value': '-78.527', 'operation': 'remove_smaller', 'result': ['-14.14', '10.92', '-56.57', '-56', '20']}

Example 2:
Question: Remove all numbers larger than 19 in this list: ['20', '66', '-22.729', '-21.62', '-6.2198', '4', '34.0', '-43.9360', '98.011', '-1.2024']
Answer: ['-22.729', '-21.62', '-6.2198', '4', '-43.9360', '-1.2024']
Metadata: {'original_numbers': ['20', '66', '-22.729', '-21.62', '-6.2198', '4', '34.0', '-43.9360', '98.011', '-1.2024'], 'filter_value': '19', 'operation': 'remove_larger', 'result': ['-22.729', '-21.62', '-6.2198', '4', '-43.9360', '-1.2024']}

Example 3:
Question: Keep all numbers smaller than 2.319 in this list: ['99', '-21', '-77.530', '7', '-11', '87.2816', '94.319', '-36', '-25.7766', '30.013']
Answer: ['-21', '-77.530', '-11', '-36', '-25.7766']
Metadata: {'original_numbers': ['99', '-21', '-77.530', '7', '-11', '87.2816', '94.319', '-36', '-25.7766', '30.013'], 'filter_value': '2.319', 'operation': 'keep_smaller', 'result': ['-21', '-77.530', '-11', '-36', '-25.7766']}

```

### number_sequence
Generates number sequence completion tasks with dynamic pattern generation

Default configuration:
```python
min_terms = 4
max_terms = 8
min_value = -100
max_value = 100
max_complexity = 3
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: 9, 4, 2, 1, 0, 0, 0, ?
Answer: 0
Metadata: {'rule': 'halve', 'complexity': 2, 'sequence': [9, 4, 2, 1, 0, 0, 0, 0]}

Example 2:
Question: -2, 1, 7, 19, 43, 91, 187, 379, ?
Answer: 763
Metadata: {'rule': 'double then add 5', 'complexity': 1, 'sequence': [-2, 1, 7, 19, 43, 91, 187, 379, 763]}

Example 3:
Question: 1, 0, 0, 0, 0, 0, 0, ?
Answer: 0
Metadata: {'rule': 'halve then multiply by 8', 'complexity': 1, 'sequence': [1, 0, 0, 0, 0, 0, 0, 0]}

```

### number_sorting
Generates number sorting tasks

Default configuration:
```python
min_numbers = 3
max_numbers = 10
min_decimals = 0
max_decimals = 2
min_value = -100.0
max_value = 100.0
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Sort these numbers in ascending order: -6.78, -92.30, 91.23, -77.49, 95.03, 74.19, 70.26, -67.10
Answer: ['-92.30', '-77.49', '-67.10', '-6.78', '70.26', '74.19', '91.23', '95.03']
Metadata: {'original_numbers': ['-6.78', '-92.30', '91.23', '-77.49', '95.03', '74.19', '70.26', '-67.10'], 'direction': 'ascending', 'sorted_numbers': ['-92.30', '-77.49', '-67.10', '-6.78', '70.26', '74.19', '91.23', '95.03']}

Example 2:
Question: Sort these numbers in descending order: -10.32, 68.71, -89.59, 57.02, 12.29, -75.18, 49.79, -62.58, -58.82
Answer: ['68.71', '57.02', '49.79', '12.29', '-10.32', '-58.82', '-62.58', '-75.18', '-89.59']
Metadata: {'original_numbers': ['-10.32', '68.71', '-89.59', '57.02', '12.29', '-75.18', '49.79', '-62.58', '-58.82'], 'direction': 'descending', 'sorted_numbers': ['68.71', '57.02', '49.79', '12.29', '-10.32', '-58.82', '-62.58', '-75.18', '-89.59']}

Example 3:
Question: Sort these numbers in descending order: 10.13, 72.60, 72.13, 14.65, 1.16, -26.82, 55.17, 37.38, 76.73, -82.92
Answer: ['76.73', '72.60', '72.13', '55.17', '37.38', '14.65', '10.13', '1.16', '-26.82', '-82.92']
Metadata: {'original_numbers': ['10.13', '72.60', '72.13', '14.65', '1.16', '-26.82', '55.17', '37.38', '76.73', '-82.92'], 'direction': 'descending', 'sorted_numbers': ['76.73', '72.60', '72.13', '55.17', '37.38', '14.65', '10.13', '1.16', '-26.82', '-82.92']}

```

### polynomial_equations
Generates random polynomial equations of degree in [min_degree, max_degree].
    - The polynomial is formed by summing random terms of the form: coeff * x^exponent.
    - Then we solve "polynomial_expr = 0" using Sympy.
    - The solution may be real or complex; we filter real solutions by default for simplicity.

Default configuration:
```python
min_terms = 2
max_terms = 4
min_value = 1
max_value = 100
min_degree = 1
max_degree = 3
operators = ('+', '-')
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Determine the real value(s) of a tha satisfies: -35*a**2 = 0
Answer: [0.0]
Metadata: {'polynomial_expr': '-35*a**2', 'variable': 'a', 'degree': 2, 'real_solutions': [0.0]}

Example 2:
Question: Solve for real l: 27*l**2 + 175*l - 1 = 0
Answer: [-6.487190738158517, 0.005709256677035911]
Metadata: {'polynomial_expr': '27*l**2 + 175*l - 1', 'variable': 'l', 'degree': 2, 'real_solutions': [-6.487190738158517, 0.005709256677035911]}

Example 3:
Question: Find the real value(s) of t in the equation: 94 - 9*t**2 = 0
Answer: [-3.2317865716108862, 3.2317865716108862]
Metadata: {'polynomial_expr': '94 - 9*t**2', 'variable': 't', 'degree': 2, 'real_solutions': [-3.2317865716108862, 3.2317865716108862]}

```

### prime_factorization
Generates prime factorization tasks

Default configuration:
```python
min_value = 2
max_value = 1000
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Find the prime factorization of 973. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 7 × 139
Metadata: {'number': 973, 'factors': [7, 139]}

Example 2:
Question: Find the prime factorization of 153. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 3 × 3 × 17
Metadata: {'number': 153, 'factors': [3, 3, 17]}

Example 3:
Question: Find the prime factorization of 390. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 2 × 3 × 5 × 13
Metadata: {'number': 390, 'factors': [2, 3, 5, 13]}

```

### propositional_logic
Generates propositional logic reasoning tasks

Default configuration:
```python
min_vars = 2
max_vars = 4
min_statements = 2
max_statements = 4
max_complexity = 3
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Given:
1. (Q → P)
2. (P → P)
3. ((P ∨ Q) ↔ (P ↔ Q))
4. (Q ∨ P)
What can we conclude?
Answer: (P ∧ P)
Metadata: {'premises': ['(Q → P)', '(P → P)', '((P ∨ Q) ↔ (P ↔ Q))', '(Q ∨ P)'], 'variables': ['P', 'Q'], 'complexity': 3}

Example 2:
Question: Given:
1. P
2. ¬(P ∧ P)
3. Q
What can we conclude?
Answer: (P ∧ P)
Metadata: {'premises': ['P', '¬(P ∧ P)', 'Q'], 'variables': ['P', 'Q', 'R'], 'complexity': 3}

Example 3:
Question: Given:
1. ¬(R → P)
2. ¬P
What can we conclude?
Answer: (Q ↔ Q)
Metadata: {'premises': ['¬(R → P)', '¬P'], 'variables': ['P', 'Q', 'R'], 'complexity': 3}

```

### quantum_lock
Generates QuantumLock tasks

Default configuration:
```python
difficulty = 10
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value.

Start: 0 (red)
Target: 38
Buttons:
A: Multiply 2 (when any)
B: Add 2 (when red)
C: Multiply 3 (when any)
Answer: B → A → C → C → B
Metadata: {'difficulty': 10, 'solution_path': ['B', 'A', 'C', 'C', 'B'], 'target_value': 38, 'buttons': [{'name': 'A', 'type': 'multiply', 'value': 2, 'active_state': 'any'}, {'name': 'B', 'type': 'add', 'value': 2, 'active_state': 'red'}, {'name': 'C', 'type': 'multiply', 'value': 3, 'active_state': 'any'}], 'initial_state': 'red', 'initial_value': 0}

Example 2:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value.

Start: 0 (red)
Target: 42
Buttons:
A: Multiply 3 (when any)
B: Add 2 (when any)
C: Add 3 (when any)
Answer: B → B → A → B → A
Metadata: {'difficulty': 10, 'solution_path': ['B', 'B', 'A', 'B', 'A'], 'target_value': 42, 'buttons': [{'name': 'A', 'type': 'multiply', 'value': 3, 'active_state': 'any'}, {'name': 'B', 'type': 'add', 'value': 2, 'active_state': 'any'}, {'name': 'C', 'type': 'add', 'value': 3, 'active_state': 'any'}], 'initial_state': 'red', 'initial_value': 0}

Example 3:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value.

Start: 0 (red)
Target: 35
Buttons:
A: Multiply 3 (when red)
B: Add 2 (when green)
C: Subtract 3 (when any)
Answer: A → B → A → C → A → B → A → B
Metadata: {'difficulty': 10, 'solution_path': ['A', 'B', 'A', 'C', 'A', 'B', 'A', 'B'], 'target_value': 35, 'buttons': [{'name': 'A', 'type': 'multiply', 'value': 3, 'active_state': 'red'}, {'name': 'B', 'type': 'add', 'value': 2, 'active_state': 'green'}, {'name': 'C', 'type': 'subtract', 'value': 3, 'active_state': 'any'}], 'initial_state': 'red', 'initial_value': 0}

```

### rubiks_cube
Generates RubiksCube tasks

Default configuration:
```python
scramble_steps = 3
cube_size = 3
remove_ansi = True
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: You are given a 3x3x3 Rubik's cube. It looks like this:

          Y  Y  Y
          Y  Y  Y
          Y  Y  Y
 G  G  G  O  O  O  B  B  B  R  R  R
 R  R  R  G  G  G  O  O  O  B  B  B
 R  R  R  G  G  G  O  O  O  B  B  B
          W  W  W
          W  W  W
          W  W  W


Please provide a solution to solve this cube using Singmaster notation.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': "U L L'", 'example_correct_answer': "U'"}

Example 2:
Question: You see a size 3 Rubik's cube. It is arranged this:

          Y  Y  O
          Y  Y  O
          Y  Y  B
 R  R  R  G  G  Y  O  G  G  W  B  B
 R  R  Y  O  G  G  W  O  O  B  B  B
 R  R  Y  O  G  G  W  O  O  B  B  B
          G  R  R
          W  W  W
          W  W  W


Please provide a solution to solve this cube.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': "U F' U'", 'example_correct_answer': "U F U'"}

Example 3:
Question: You see a size 3 Rubik's cube. It is arranged this:

          R  R  R
          B  Y  Y
          O  O  O
 G  R  Y  G  G  G  W  O  B  W  W  W
 W  R  Y  G  G  G  W  O  Y  B  B  B
 W  R  B  Y  Y  Y  G  O  Y  B  B  B
          R  R  R
          G  W  W
          O  O  O


Please provide a solution to solve this cube.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': "L B' F'", 'example_correct_answer': "B L' F U F U' F' U F R U R' U' F' R U R' U R U U R' U' R U R' U R U U R' U' L U' R' U L' U' R U L U' R' U L' U' D' R D R' D' R D R' D' R D R' D' R D U R' D' R D R' D' R D U'"}

```

### sentence_reordering
Generates sentence reordering tasks from text spans

Default configuration:
```python
min_words_in_sentence = 3
max_words_in_sentence = 20
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Restore the correct order of words in the following sentence: thing first that Mr. The
Answer: The first thing that Mr.
Metadata: {'word_count': 5}

Example 2:
Question: Restore the correct order of words in the following sentence: shall The to called be the attention of government the matter. Chinese
Answer: The attention of the the Chinese government shall be called to matter.
Metadata: {'word_count': 12}

Example 3:
Question: Restore the correct order of words in the following sentence: wonderful we are the accumulators. indebted instruments those new for Jackson To
Answer: To Jackson we are indebted for those wonderful instruments the new accumulators.
Metadata: {'word_count': 12}

```

### simple_equations
Generates simple equations with one variable to solve

Default configuration:
```python
min_terms = 2
max_terms = 4
min_value = 1
max_value = 100
operators = ('+', '-', '*')
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Solve for j: 69 - 47*j = -4020
Answer: 87
Metadata: {'equation': '69 - 47*j = -4020', 'variable': 'j'}

Example 2:
Question: Solve for o: 210000*o + 98 = 840098
Answer: 4
Metadata: {'equation': '210000*o + 98 = 840098', 'variable': 'o'}

Example 3:
Question: Find the value of a in the equation: 6930*a = 297990
Answer: 43
Metadata: {'equation': '6930*a = 297990', 'variable': 'a'}

```

### spell_backward
Generates tasks to spell words backward

Default configuration:
```python
min_word_len = 3
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Spell this word backward (example: sun -> nus): only
Answer: ylno
Metadata: {'word': 'only', 'word_len': 4}

Example 2:
Question: Spell this word backward (example: sun -> nus): from
Answer: morf
Metadata: {'word': 'from', 'word_len': 4}

Example 3:
Question: Spell this word backward (example: sun -> nus): anxiously
Answer: ylsuoixna
Metadata: {'word': 'anxiously', 'word_len': 9}

```

### sudoku
Generates sudoku puzzles with configurable difficulty

Default configuration:
```python
min_empty = 30
max_empty = 50
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Solve this Sudoku puzzle:
_ 8 _ 2 _ _ _ _ 3
_ _ 4 _ 7 _ _ 8 9
2 5 6 3 _ _ _ 4 7
_ _ 8 _ 6 _ 9 5 _
9 _ 2 7 _ 5 _ _ _
3 6 _ _ 2 9 8 _ _
_ 4 3 _ 5 2 7 _ _
_ _ 1 _ _ _ 4 2 8
6 2 _ 8 4 1 3 9 5
Answer: 7 8 9 2 1 4 5 6 3
1 3 4 5 7 6 2 8 9
2 5 6 3 9 8 1 4 7
4 7 8 1 6 3 9 5 2
9 1 2 7 8 5 6 3 4
3 6 5 4 2 9 8 7 1
8 4 3 9 5 2 7 1 6
5 9 1 6 3 7 4 2 8
6 2 7 8 4 1 3 9 5
Metadata: {'puzzle': [[0, 8, 0, 2, 0, 0, 0, 0, 3], [0, 0, 4, 0, 7, 0, 0, 8, 9], [2, 5, 6, 3, 0, 0, 0, 4, 7], [0, 0, 8, 0, 6, 0, 9, 5, 0], [9, 0, 2, 7, 0, 5, 0, 0, 0], [3, 6, 0, 0, 2, 9, 8, 0, 0], [0, 4, 3, 0, 5, 2, 7, 0, 0], [0, 0, 1, 0, 0, 0, 4, 2, 8], [6, 2, 0, 8, 4, 1, 3, 9, 5]], 'solution': [[7, 8, 9, 2, 1, 4, 5, 6, 3], [1, 3, 4, 5, 7, 6, 2, 8, 9], [2, 5, 6, 3, 9, 8, 1, 4, 7], [4, 7, 8, 1, 6, 3, 9, 5, 2], [9, 1, 2, 7, 8, 5, 6, 3, 4], [3, 6, 5, 4, 2, 9, 8, 7, 1], [8, 4, 3, 9, 5, 2, 7, 1, 6], [5, 9, 1, 6, 3, 7, 4, 2, 8], [6, 2, 7, 8, 4, 1, 3, 9, 5]], 'num_empty': 38}

Example 2:
Question: Solve this Sudoku puzzle:
5 _ _ _ 3 4 _ 6 _
_ _ 3 _ _ _ _ _ _
_ _ 8 5 9 _ _ _ 2
_ 5 7 6 4 _ _ 8 _
_ 4 6 _ _ _ _ 5 3
_ 3 _ _ _ 5 _ _ _
6 8 1 _ _ 9 _ _ _
_ 9 5 _ 2 _ _ 4 _
_ 2 _ _ 8 6 1 9 5
Answer: 5 7 2 1 3 4 8 6 9
9 1 3 2 6 8 5 7 4
4 6 8 5 9 7 3 1 2
2 5 7 6 4 3 9 8 1
8 4 6 9 1 2 7 5 3
1 3 9 8 7 5 4 2 6
6 8 1 4 5 9 2 3 7
3 9 5 7 2 1 6 4 8
7 2 4 3 8 6 1 9 5
Metadata: {'puzzle': [[5, 0, 0, 0, 3, 4, 0, 6, 0], [0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 8, 5, 9, 0, 0, 0, 2], [0, 5, 7, 6, 4, 0, 0, 8, 0], [0, 4, 6, 0, 0, 0, 0, 5, 3], [0, 3, 0, 0, 0, 5, 0, 0, 0], [6, 8, 1, 0, 0, 9, 0, 0, 0], [0, 9, 5, 0, 2, 0, 0, 4, 0], [0, 2, 0, 0, 8, 6, 1, 9, 5]], 'solution': [[5, 7, 2, 1, 3, 4, 8, 6, 9], [9, 1, 3, 2, 6, 8, 5, 7, 4], [4, 6, 8, 5, 9, 7, 3, 1, 2], [2, 5, 7, 6, 4, 3, 9, 8, 1], [8, 4, 6, 9, 1, 2, 7, 5, 3], [1, 3, 9, 8, 7, 5, 4, 2, 6], [6, 8, 1, 4, 5, 9, 2, 3, 7], [3, 9, 5, 7, 2, 1, 6, 4, 8], [7, 2, 4, 3, 8, 6, 1, 9, 5]], 'num_empty': 47}

Example 3:
Question: Solve this Sudoku puzzle:
9 8 6 _ _ _ _ _ 3
4 _ _ _ _ _ _ 6 _
_ _ 3 6 7 _ _ _ 8
_ _ 9 _ _ 3 6 _ _
_ _ _ _ _ _ 7 4 2
_ _ _ 4 _ _ _ _ _
_ _ 2 5 _ _ _ 1 _
_ 3 1 _ 4 6 8 9 7
7 9 _ 8 _ _ _ _ 6
Answer: 9 8 6 1 2 4 5 7 3
4 2 7 3 8 5 1 6 9
1 5 3 6 7 9 4 2 8
2 4 9 7 1 3 6 8 5
3 1 5 9 6 8 7 4 2
6 7 8 4 5 2 9 3 1
8 6 2 5 9 7 3 1 4
5 3 1 2 4 6 8 9 7
7 9 4 8 3 1 2 5 6
Metadata: {'puzzle': [[9, 8, 6, 0, 0, 0, 0, 0, 3], [4, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 3, 6, 7, 0, 0, 0, 8], [0, 0, 9, 0, 0, 3, 6, 0, 0], [0, 0, 0, 0, 0, 0, 7, 4, 2], [0, 0, 0, 4, 0, 0, 0, 0, 0], [0, 0, 2, 5, 0, 0, 0, 1, 0], [0, 3, 1, 0, 4, 6, 8, 9, 7], [7, 9, 0, 8, 0, 0, 0, 0, 6]], 'solution': [[9, 8, 6, 1, 2, 4, 5, 7, 3], [4, 2, 7, 3, 8, 5, 1, 6, 9], [1, 5, 3, 6, 7, 9, 4, 2, 8], [2, 4, 9, 7, 1, 3, 6, 8, 5], [3, 1, 5, 9, 6, 8, 7, 4, 2], [6, 7, 8, 4, 5, 2, 9, 3, 1], [8, 6, 2, 5, 9, 7, 3, 1, 4], [5, 3, 1, 2, 4, 6, 8, 9, 7], [7, 9, 4, 8, 3, 1, 2, 5, 6]], 'num_empty': 50}

```

### syllogism
Generates syllogism reasoning tasks

Default configuration:
```python
terms = None
allow_all = True
allow_no = True
allow_some = True
allow_some_not = True
include_invalid = True
invalid_ratio = 0.3
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Consider these statements:
1. Some humans are reptiles
2. Some reptiles are insects

Does it logically follow that:
Some ... are not humans are insects?
(Answer Yes or No)
Answer: No
Metadata: {'premise1': 'Some humans are reptiles', 'premise2': 'Some reptiles are insects', 'conclusion': 'Some ... are not humans are insects', 'is_valid': False}

Example 2:
Question: Consider these statements:
1. All mortals are teachers
2. Some teachers are ants

Does it logically follow that:
Some ... are not mortals are ants?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'All mortals are teachers', 'premise2': 'Some teachers are ants', 'conclusion': 'Some ... are not mortals are ants', 'is_valid': True}

Example 3:
Question: Consider these statements:
1. No mortals are whales
2. No whales are bees

Does it logically follow that:
No mortals are bees?
(Answer Yes or No)
Answer: No
Metadata: {'premise1': 'No mortals are whales', 'premise2': 'No whales are bees', 'conclusion': 'No mortals are bees', 'is_valid': False}

```

### word_sequence_reversal
Generates word sequence reversal tasks from text spans

Default configuration:
```python
min_words = 3
max_words = 8
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Reverse this list of words: Africa, harmless, moral
Answer: moral, harmless, Africa
Metadata: {'num_words': 3, 'words': ['Africa', 'harmless', 'moral']}

Example 2:
Question: Reverse this list of words: efforts, well, set, these, back, Her, for
Answer: for, Her, back, these, set, well, efforts
Metadata: {'num_words': 7, 'words': ['efforts', 'well', 'set', 'these', 'back', 'Her', 'for']}

Example 3:
Question: Reverse this list of words: fellow, compliance, few, which, in, famous, Not
Answer: Not, famous, in, which, few, compliance, fellow
Metadata: {'num_words': 7, 'words': ['fellow', 'compliance', 'few', 'which', 'in', 'famous', 'Not']}

```

### word_sorting
Generates word sorting tasks

Default configuration:
```python
min_words = 3
max_words = 10
min_word_length = 3
max_word_length = 12
transformation = original
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Sort these words in descending order (using ASCII/Unicode ordering) and return them as a comma-separated list:
prepare, provide, speak, surplus, after, unlink, change, 000
Answer: unlink, surplus, speak, provide, prepare, change, after, 000
Metadata: {'original_words': ['prepare', 'provide', 'speak', 'surplus', 'after', 'unlink', 'change', '000'], 'transformed_words': ['prepare', 'provide', 'speak', 'surplus', 'after', 'unlink', 'change', '000'], 'direction': 'descending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['unlink', 'surplus', 'speak', 'provide', 'prepare', 'change', 'after', '000']}

Example 2:
Question: Sort these words in descending order (using ASCII/Unicode ordering) and return them as a comma-separated list:
501, differences, Thus, cupola, longer, remaining, mummy, Paris, DISTRIBUTE
Answer: remaining, mummy, longer, differences, cupola, Thus, Paris, DISTRIBUTE, 501
Metadata: {'original_words': ['501', 'differences', 'Thus', 'cupola', 'longer', 'remaining', 'mummy', 'Paris', 'DISTRIBUTE'], 'transformed_words': ['501', 'differences', 'Thus', 'cupola', 'longer', 'remaining', 'mummy', 'Paris', 'DISTRIBUTE'], 'direction': 'descending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['remaining', 'mummy', 'longer', 'differences', 'cupola', 'Thus', 'Paris', 'DISTRIBUTE', '501']}

Example 3:
Question: Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list:
discontinue, access, office, luminous, distributing
Answer: access, discontinue, distributing, luminous, office
Metadata: {'original_words': ['discontinue', 'access', 'office', 'luminous', 'distributing'], 'transformed_words': ['discontinue', 'access', 'office', 'luminous', 'distributing'], 'direction': 'ascending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['access', 'discontinue', 'distributing', 'luminous', 'office']}

```
