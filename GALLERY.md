# Reasoning Gym Dataset Gallery
This gallery shows examples from all available datasets using their default configurations.

## Available Datasets
- [ab](#ab)
- [advanced_geometry](#advanced_geometry)
- [aiw](#aiw)
- [arc_1d](#arc_1d)
- [arc_agi](#arc_agi)
- [base_conversion](#base_conversion)
- [basic_arithmetic](#basic_arithmetic)
- [bf](#bf)
- [binary_alternation](#binary_alternation)
- [binary_matrix](#binary_matrix)
- [bitwise_arithmetic](#bitwise_arithmetic)
- [caesar_cipher](#caesar_cipher)
- [calendar_arithmetic](#calendar_arithmetic)
- [chain_sum](#chain_sum)
- [circuit_logic](#circuit_logic)
- [color_cube_rotation](#color_cube_rotation)
- [complex_arithmetic](#complex_arithmetic)
- [count_bits](#count_bits)
- [count_primes](#count_primes)
- [countdown](#countdown)
- [course_schedule](#course_schedule)
- [cryptarithm](#cryptarithm)
- [decimal_arithmetic](#decimal_arithmetic)
- [decimal_chain_sum](#decimal_chain_sum)
- [dice](#dice)
- [emoji_mystery](#emoji_mystery)
- [family_relationships](#family_relationships)
- [figlet_font](#figlet_font)
- [fraction_simplification](#fraction_simplification)
- [futoshiki](#futoshiki)
- [game_of_life](#game_of_life)
- [gcd](#gcd)
- [graph_color](#graph_color)
- [group_anagrams](#group_anagrams)
- [gsm_symbolic](#gsm_symbolic)
- [intermediate_integration](#intermediate_integration)
- [isomorphic_strings](#isomorphic_strings)
- [jugs](#jugs)
- [knight_swap](#knight_swap)
- [largest_island](#largest_island)
- [lcm](#lcm)
- [leg_counting](#leg_counting)
- [letter_counting](#letter_counting)
- [letter_jumble](#letter_jumble)
- [list_functions](#list_functions)
- [manipulate_matrix](#manipulate_matrix)
- [maze](#maze)
- [mini_sudoku](#mini_sudoku)
- [n_queens](#n_queens)
- [needle_haystack](#needle_haystack)
- [number_filtering](#number_filtering)
- [number_format](#number_format)
- [number_sequence](#number_sequence)
- [number_sorting](#number_sorting)
- [palindrome](#palindrome)
- [palindrome_partitioning](#palindrome_partitioning)
- [polynomial_equations](#polynomial_equations)
- [polynomial_multiplication](#polynomial_multiplication)
- [pool_matrix](#pool_matrix)
- [power_function](#power_function)
- [prime_factorization](#prime_factorization)
- [products](#products)
- [propositional_logic](#propositional_logic)
- [quantum_lock](#quantum_lock)
- [ransom_note](#ransom_note)
- [rearc](#rearc)
- [rectangle_count](#rectangle_count)
- [rotate_matrix](#rotate_matrix)
- [rotten_oranges](#rotten_oranges)
- [rubiks_cube](#rubiks_cube)
- [rush_hour](#rush_hour)
- [self_reference](#self_reference)
- [sentence_reordering](#sentence_reordering)
- [shortest_path](#shortest_path)
- [simple_equations](#simple_equations)
- [simple_geometry](#simple_geometry)
- [simple_integration](#simple_integration)
- [sokoban](#sokoban)
- [spell_backward](#spell_backward)
- [spiral_matrix](#spiral_matrix)
- [string_insertion](#string_insertion)
- [string_manipulation](#string_manipulation)
- [string_splitting](#string_splitting)
- [string_synthesis](#string_synthesis)
- [sudoku](#sudoku)
- [syllogism](#syllogism)
- [time_intervals](#time_intervals)
- [tower_of_hanoi](#tower_of_hanoi)
- [tsumego](#tsumego)
- [word_ladder](#word_ladder)
- [word_sequence_reversal](#word_sequence_reversal)
- [word_sorting](#word_sorting)
- [zebra_puzzles](#zebra_puzzles)

## Dataset Examples
### ab
Generates A::B tasks, as described by @VictorTaelin [here](https://x.com/VictorTaelin/status/1776096481704804789)

Default configuration:
```python
seed = 42
size = 500
length = 10
```

Example tasks:
````
Example 1:
Question: A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.

An A::B program is a sequence of tokens. Example:

    B# A# #B #A B#

To *compute* a program, we must rewrite neighbor tokens, using the rules:

    A# #A ... becomes ... nothing
    A# #B ... becomes ... #B A#
    B# #A ... becomes ... #A B#
    B# #B ... becomes ... nothing

In other words, whenever two neighbor tokens have their '#' facing each-other,
they must be rewritten according to the corresponding rule. For example, the
first example shown here is computed as:

    B# A# #B #A B# =
    B# #B A# #A B# =
    A# #A B# =
    B#

The steps were:
1. We replaced `A# #B` by `#B A#`.
2. We replaced `B# #B` by nothing.
3. We replaced `A# #A` by nothing.
The final result was just `B#`.

Now, consider the following program:

A# A# #A B# B# B# A# A# #B A#

Return the final state of the program.

Answer: A# B# B# A# A# A#

Example 2:
Question: A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.

An A::B program is a sequence of tokens. Example:

    B# A# #B #A B#

To *compute* a program, we must rewrite neighbor tokens, using the rules:

    A# #A ... becomes ... nothing
    A# #B ... becomes ... #B A#
    B# #A ... becomes ... #A B#
    B# #B ... becomes ... nothing

In other words, whenever two neighbor tokens have their '#' facing each-other,
they must be rewritten according to the corresponding rule. For example, the
first example shown here is computed as:

    B# A# #B #A B# =
    B# #B A# #A B# =
    A# #A B# =
    B#

The steps were:
1. We replaced `A# #B` by `#B A#`.
2. We replaced `B# #B` by nothing.
3. We replaced `A# #A` by nothing.
The final result was just `B#`.

Now, consider the following program:

A# #A B# #B #A A# #B #B A# #B

Return the final state of the program.

Answer: #A #B #B #B A# A#

Example 3:
Question: A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.

An A::B program is a sequence of tokens. Example:

    B# A# #B #A B#

To *compute* a program, we must rewrite neighbor tokens, using the rules:

    A# #A ... becomes ... nothing
    A# #B ... becomes ... #B A#
    B# #A ... becomes ... #A B#
    B# #B ... becomes ... nothing

In other words, whenever two neighbor tokens have their '#' facing each-other,
they must be rewritten according to the corresponding rule. For example, the
first example shown here is computed as:

    B# A# #B #A B# =
    B# #B A# #A B# =
    A# #A B# =
    B#

The steps were:
1. We replaced `A# #B` by `#B A#`.
2. We replaced `B# #B` by nothing.
3. We replaced `A# #A` by nothing.
The final result was just `B#`.

Now, consider the following program:

#B A# B# #B B# #A A# B# A# A#

Return the final state of the program.

Answer: #B B# A# B# A# A#

````

### advanced_geometry
A dataset for advanced geometry tasks using coordinate geometry.

Default configuration:
```python
min_coord = -10
max_coord = 10
size = 50
seed = 42
task_types = ['orthocenter', 'incircle_radius', 'angle_measure']
```

Example tasks:
````
Example 1:
Question: In triangle ABC with coordinates A=(-7, -10), B=(-2, -3), and C=(-3, -6), find the measure (in degrees) of angle ABC. For all geometry problems:
1. Give coordinates in the form (x, y)
2. Round decimal answers to 3 decimal places
3. Use the degree symbol ° for angles
4. Return only th angle, coordinates, or radius as your answer.
Answer: 17.10°
Metadata: {'A': (-7, -10), 'B': (-2, -3), 'C': (-3, -6), 'angle_ABC_degrees': 17.10272896905237, 'task_type': 'angle_measure'}

Example 2:
Question: For triangle with vertices A=(-1, -6), B=(4, 1), and C=(-7, 4), determine the orthocenter (intersection of altitudes). For all geometry problems:
1. Give coordinates in the form (x, y)
2. Round decimal answers to 3 decimal places
3. Use the degree symbol ° for angles
4. Return only th angle, coordinates, or radius as your answer.
Answer: (0.304, -1.217)
Metadata: {'A': (-1, -6), 'B': (4, 1), 'C': (-7, 4), 'ortho': (7/23, -28/23), 'orthocenter_exact': ('7/23', '-28/23'), 'orthocenter_approx': (0.30434782608695654, -1.2173913043478262), 'task_type': 'orthocenter'}

Example 3:
Question: Find the incircle radius of triangle ABC whose vertices are A=(6, 7), B=(-7, -5), and C=(2, -3). For all geometry problems:
1. Give coordinates in the form (x, y)
2. Round decimal answers to 3 decimal places
3. Use the degree symbol ° for angles
4. Return only th angle, coordinates, or radius as your answer.
Answer: 2.176
Metadata: {'A': (6, 7), 'B': (-7, -5), 'C': (2, -3), 'incircle_radius_exact': 'sqrt(-sqrt(29) + sqrt(85)/2 + sqrt(313)/2)*sqrt(-sqrt(313)/2 + sqrt(85)/2 + sqrt(29))*sqrt(-sqrt(85)/2 + sqrt(29) + sqrt(313)/2)/sqrt(sqrt(85)/2 + sqrt(29) + sqrt(313)/2)', 'incircle_radius_approx': 2.176123777286009, 'task_type': 'incircle_radius'}

````

### aiw
A procedural dataset inspired by the "Alice in Wonderland" paper.

    The dataset is inspired by the following paper:
       @inproceedings{nezhurina2024alice,
       title={Alice in Wonderland: Simple Tasks Reveal Severe Generalization and
              Basic Reasoning Deficits in State-Of-the-Art Large Language Models},
       author={Marianna Nezhurina and Lucia Cipolina-Kun and Mehdi Cherti and
              Jenia Jitsev},
       booktitle={NeurIPS 2024 Workshop on Scientific Methods for Understanding
                  Deep Learning},
       year={2024},
       url={https://openreview.net/forum?id=Mkl7dzjYiW}
       }

Default configuration:
```python
male_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles', 'Bob']
female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Margaret', 'Alice']
task_types = [<TaskType.SIBLINGS: 'siblings'>, <TaskType.FRIENDS: 'friends'>, <TaskType.COLLEAGUES: 'colleagues'>]
seed = 42
size = 10
max_entities = 6
```

Example tasks:
````
Example 1:
Question: Patricia has 6 male colleagues and she also has 3 female colleagues. These are all colleagues that Patricia has. All these mentioned persons around Patricia are colleagues of each other. James has 2 male colleagues and 2 female colleagues in total. All these mentioned persons around James are colleagues of each other. The people in the circle around James do not have other colleagues aside - with the only exception of Matilda. She is colleague of James and she is also colleague of Patricia, being part of Patricia's circle. How many female colleagues does Matilda have?
Answer: 4
Metadata: {'task_type': 'colleagues'}

Example 2:
Question: Elizabeth has 4 brothers and she also has 3 sisters. How many sisters does Elizabeth's brother have?
Answer: 4
Metadata: {'task_type': 'siblings'}

Example 3:
Question: Sarah has 6 male friends and she also has 1 female friends. They all are friends with each other and have no other friends aside. How many female friends does Thomas, a male friend of Sarah, have?
Answer: 2
Metadata: {'task_type': 'friends'}

````

### arc_1d
Generates ARC 1D tasks by randomly selecting from available task generators

    This dataset is a procedural variant of the 1D-ARC dataset which is described in the paper:
    `LLMs and the Abstraction and Reasoning Corpus:  Successes, Failures, and the Importance
    of Object-based Representations` (https://arxiv.org/abs/2305.18354)

    Ilya Sheprut (optozorax) created rust generators for most of the ARC 1d tasks. For
    reasoning-gym rust tasks were machine-converted to python via Sonnet.

    Ilya's original rust code can be found here: https://github.com/optozorax/arc_1d/

Default configuration:
```python
min_size = 10
max_size = 30
num_train = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:
Input:  0 0 0 2 9 2 3 4 4 0
Output: 2 9 2 3 4 4 0 0 0 0

Example 2:
Input:  0 0 0 0 4 4 2 1 1 0
Output: 0 4 4 2 1 1 0 0 0 0

Example 3:
Input:  0 0 0 7 9 4 9 1 0 0
Output: 7 9 4 9 1 0 0 0 0 0

Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. Your final answer must be placed in <output></output> tags and should be just be the text output grid itself.

Input:
0 0 0 0 0 1 5 0 0 0
Answer: 0 0 1 5 0 0 0 0 0 0
Metadata: {'task_name': 'move_3pix_colorful_left', 'size': 10, 'train_examples': [{'input': [0, 0, 0, 2, 9, 2, 3, 4, 4, 0], 'output': [2, 9, 2, 3, 4, 4, 0, 0, 0, 0]}, {'input': [0, 0, 0, 0, 4, 4, 2, 1, 1, 0], 'output': [0, 4, 4, 2, 1, 1, 0, 0, 0, 0]}, {'input': [0, 0, 0, 7, 9, 4, 9, 1, 0, 0], 'output': [7, 9, 4, 9, 1, 0, 0, 0, 0, 0]}], 'test_example': {'input': [0, 0, 0, 0, 0, 1, 5, 0, 0, 0], 'output': [0, 0, 1, 5, 0, 0, 0, 0, 0, 0]}}

Example 2:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:
Input:  0 0 0 0 0 0 0 6 2 8 8 1 0 0 0 0 0 0 0
Output: 0 0 0 0 0 0 0 0 6 2 8 8 1 0 0 0 0 0 0

Example 2:
Input:  0 6 9 7 7 3 1 2 2 7 3 2 3 9 8 3 7 9 0
Output: 0 0 6 9 7 7 3 1 2 2 7 3 2 3 9 8 3 7 9

Example 3:
Input:  0 0 0 0 0 0 0 0 0 3 7 2 1 1 3 1 3 5 0
Output: 0 0 0 0 0 0 0 0 0 0 3 7 2 1 1 3 1 3 5

Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. Your final answer must be placed in <output></output> tags and should be just be the text output grid itself.

Input:
0 9 2 1 2 8 6 6 9 8 0 0 0 0 0 0 0 0 0
Answer: 0 0 9 2 1 2 8 6 6 9 8 0 0 0 0 0 0 0 0
Metadata: {'task_name': 'move_1pix_colorful_right', 'size': 19, 'train_examples': [{'input': [0, 0, 0, 0, 0, 0, 0, 6, 2, 8, 8, 1, 0, 0, 0, 0, 0, 0, 0], 'output': [0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 8, 8, 1, 0, 0, 0, 0, 0, 0]}, {'input': [0, 6, 9, 7, 7, 3, 1, 2, 2, 7, 3, 2, 3, 9, 8, 3, 7, 9, 0], 'output': [0, 0, 6, 9, 7, 7, 3, 1, 2, 2, 7, 3, 2, 3, 9, 8, 3, 7, 9]}, {'input': [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 2, 1, 1, 3, 1, 3, 5, 0], 'output': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 2, 1, 1, 3, 1, 3, 5]}], 'test_example': {'input': [0, 9, 2, 1, 2, 8, 6, 6, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'output': [0, 0, 9, 2, 1, 2, 8, 6, 6, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0]}}

Example 3:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:
Input:  0 0 0 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 0 0 0
Output: 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0

Example 2:
Input:  0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0
Output: 0 0 0 0 0 0 0 3 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0

Example 3:
Input:  5 5 5 5 5 5 5 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Output: 5 0 0 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. Your final answer must be placed in <output></output> tags and should be just be the text output grid itself.

Input:
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
Answer: 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0
Metadata: {'task_name': 'two_points_and_fill_inv', 'size': 26, 'train_examples': [{'input': [0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0], 'output': [0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0]}, {'input': [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'output': [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'input': [5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'output': [5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}], 'test_example': {'input': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0], 'output': [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]}}

````

### arc_agi
Default configuration:
```python
use_train = True
use_eval = True
board_format_opts = BoardFormattingOptions(alphabet=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], col_delimiter=' ', row_delimiter='\n', array_brackets=False)
rotations = ['90', '180', '270']
mirrors = ['horizontal', 'vertical', 'diagonal', 'counterdiagonal']
use_color_permutation = True
shuffle_example_order = True
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
7 7 7 7 7 6 3 6 7 7 7 6 6 7
7 7 7 7 7 6 6 6 7 7 7 6 6 7
6 6 6 6 7 6 6 6 7 7 7 6 6 7
6 3 6 6 7 7 7 7 7 7 7 7 7 7
6 6 6 6 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 6 6 3 6 7
7 7 7 7 7 7 7 7 7 6 3 6 6 7
7 7 7 6 6 6 6 7 7 6 6 6 3 7
7 7 7 6 6 3 6 7 7 7 7 7 7 7
7 7 7 6 3 6 6 7 7 7 7 7 7 7
7 7 7 6 6 6 6 7 7 7 6 3 6 6
7 7 7 7 7 7 7 7 7 7 6 6 6 3
7 7 7 7 7 7 7 7 7 7 6 3 3 6
7 7 7 7 7 7 7 7 7 7 6 6 6 6
Output:
7 7 7 7 7 6 3 6 7 7 7 6 6 7
7 7 7 7 7 6 6 6 7 7 7 6 6 7
6 6 6 6 7 6 6 6 7 7 7 6 6 7
6 3 6 6 7 7 7 7 7 7 7 7 7 7
6 6 6 6 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7

Example 2:

Input:
7 7 7 7 7 6 6 6 6 7 7 3 6 7 7
6 6 6 6 7 3 6 6 3 7 7 6 3 7 7
6 3 6 6 7 6 6 6 6 7 7 7 7 7 7
6 6 6 6 7 6 6 3 6 7 7 6 6 6 6
7 7 7 7 7 7 7 7 7 7 7 6 3 6 6
7 7 7 7 7 7 7 7 7 7 7 6 6 6 6
7 7 6 6 3 6 6 7 7 7 7 7 7 7 7
7 7 6 6 6 3 6 7 7 7 7 7 7 7 7
7 7 6 3 6 6 6 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 6 6 3 7 7 7
7 7 6 6 6 6 7 7 7 6 3 6 7 7 7
7 7 6 6 6 6 7 7 7 6 6 6 7 7 7
7 7 6 6 6 6 7 7 7 3 6 3 7 7 7
Output:
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
6 6 6 6 7 7 7 7 7 7 7 7 7 7 7
6 3 6 6 7 7 7 7 7 7 7 7 7 7 7
6 6 6 6 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 7 7 7 6 3 6 6
7 7 7 7 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 6 6 6 6 7 7 7 7 7 7 7 7 7
7 7 6 6 6 6 7 7 7 7 7 7 7 7 7
7 7 6 6 6 6 7 7 7 7 7 7 7 7 7

Example 3:

Input:
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 6 3 6 7 7 7 7 7 7 7 7 7 7
7 6 6 3 7 6 6 6 7 7 6 3 7 7
7 7 7 7 7 6 3 6 7 7 6 6 7 7
7 7 7 7 7 6 6 3 7 7 7 7 7 7
7 7 7 7 7 3 6 6 7 7 7 6 6 6
7 7 7 7 7 7 7 7 7 7 7 6 3 6
7 6 6 3 7 7 7 7 7 7 7 6 6 6
7 3 6 6 7 7 7 7 7 7 7 7 7 7
7 6 6 6 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 3 6 6 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
Output:
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 6 3 7 7
7 7 7 7 7 7 7 7 7 7 6 6 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 6 6 6
7 7 7 7 7 7 7 7 7 7 7 6 3 6
7 7 7 7 7 7 7 7 7 7 7 6 6 6
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 3 6 6 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
7 7 7 7 7 7 7 7 6 3 6 6
6 6 6 7 7 7 7 7 6 6 6 6
3 6 6 7 7 7 7 7 6 3 6 3
6 6 6 7 3 6 6 7 7 7 7 7
7 7 7 7 6 6 6 7 7 7 7 7
7 7 7 7 6 6 3 7 7 7 7 7
7 7 7 7 6 6 6 7 6 6 6 6
7 7 7 7 7 7 7 7 6 6 3 6
7 6 6 6 6 6 6 7 6 6 6 6
7 6 6 6 6 3 6 7 6 6 6 6
7 6 3 6 6 6 6 7 7 7 7 7
7 6 6 6 6 6 6 7 6 6 6 7
7 7 7 7 7 7 7 7 6 6 6 7

Answer: 7 7 7 7 7 7 7 7 7 7 7 7
6 6 6 7 7 7 7 7 7 7 7 7
3 6 6 7 7 7 7 7 7 7 7 7
6 6 6 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 6 6 3 6
7 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 6 6 6 7
7 7 7 7 7 7 7 7 6 6 6 7
Metadata: {'input': ((7, 7, 7, 7, 7, 7, 7, 7, 6, 3, 6, 6), (6, 6, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6), (3, 6, 6, 7, 7, 7, 7, 7, 6, 3, 6, 3), (6, 6, 6, 7, 3, 6, 6, 7, 7, 7, 7, 7), (7, 7, 7, 7, 6, 6, 6, 7, 7, 7, 7, 7), (7, 7, 7, 7, 6, 6, 3, 7, 7, 7, 7, 7), (7, 7, 7, 7, 6, 6, 6, 7, 6, 6, 6, 6), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 3, 6), (7, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6), (7, 6, 6, 6, 6, 3, 6, 7, 6, 6, 6, 6), (7, 6, 3, 6, 6, 6, 6, 7, 7, 7, 7, 7), (7, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 7), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7)), 'output': ((7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7), (3, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7), (6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 3, 6), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7)), 'task_id': 'a934301b'}

Example 2:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
2 8 8 8 8 8 8 8 8 9
2 8 8 0 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 0 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 0 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
Output:
2 8 8 8 8 8 8 8 8 9
2 8 8 2 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 9 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 9 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9

Example 2:

Input:
6 6 6 6 6 6 6 6 6 6
8 8 8 8 8 8 8 8 8 8
8 8 0 8 8 8 8 8 0 8
8 8 8 8 8 8 0 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 0 8 8 8 8
8 0 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
1 1 1 1 1 1 1 1 1 1
Output:
6 6 6 6 6 6 6 6 6 6
8 8 8 8 8 8 8 8 8 8
8 8 6 8 8 8 8 8 6 8
8 8 8 8 8 8 6 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 1 8 8 8 8
8 1 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
1 1 1 1 1 1 1 1 1 1

Example 3:

Input:
5 5 5 5 5 5 5 5 5 5
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 0 8 8 8 8
8 8 0 8 8 8 8 8 0 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 0 8 8 8 8 0 8
8 8 8 8 8 8 0 8 8 8
8 8 8 8 8 8 8 8 8 8
7 7 7 7 7 7 7 7 7 7
Output:
5 5 5 5 5 5 5 5 5 5
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 5 8 8 8 8
8 8 5 8 8 8 8 8 5 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 7 8 8 8 8 7 8
8 8 8 8 8 8 7 8 8 8
8 8 8 8 8 8 8 8 8 8
7 7 7 7 7 7 7 7 7 7


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
6 8 8 8 8 8 8 8 0 4
6 0 8 8 0 8 8 8 8 4
6 8 8 8 8 8 8 8 8 4
6 8 8 8 8 8 0 8 8 4
6 8 8 0 8 8 8 8 8 4
6 8 8 8 8 8 0 8 8 4
6 8 8 8 8 8 8 8 8 4
6 8 8 8 8 0 8 8 8 4
6 8 8 0 8 8 8 0 8 4
6 8 8 8 8 8 8 8 8 4

Answer: 6 8 8 8 8 8 8 8 4 4
6 6 8 8 6 8 8 8 8 4
6 8 8 8 8 8 8 8 8 4
6 8 8 8 8 8 4 8 8 4
6 8 8 6 8 8 8 8 8 4
6 8 8 8 8 8 4 8 8 4
6 8 8 8 8 8 8 8 8 4
6 8 8 8 8 4 8 8 8 4
6 8 8 6 8 8 8 4 8 4
6 8 8 8 8 8 8 8 8 4
Metadata: {'input': ((6, 8, 8, 8, 8, 8, 8, 8, 0, 4), (6, 0, 8, 8, 0, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 0, 8, 8, 4), (6, 8, 8, 0, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 0, 8, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 0, 8, 8, 8, 4), (6, 8, 8, 0, 8, 8, 8, 0, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4)), 'output': ((6, 8, 8, 8, 8, 8, 8, 8, 4, 4), (6, 6, 8, 8, 6, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 4, 8, 8, 4), (6, 8, 8, 6, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 4, 8, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 4, 8, 8, 8, 4), (6, 8, 8, 6, 8, 8, 8, 4, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4)), 'task_id': '2204b7a8'}

Example 3:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 8 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 8 5
5 5 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 5 5 5 5 5
5 5 8 8 8 8 5 5 5 5 5 8 8 8 8 5 5 5 5 5
2 5 8 8 8 8 5 5 5 5 5 8 8 8 8 5 5 5 5 2
5 5 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 8 8 8 8 8 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 5 5
Output:
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 8 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 8 5
5 5 8 8 8 8 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 8 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 8 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 2 2 2 2 5 5 5 5 5
5 5 2 2 2 2 5 2 5 5 5 2 2 2 2 5 5 5 5 5
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
5 5 2 2 2 2 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 2 2 2 2 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 5 5

Example 2:

Input:
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 5 5 5 5 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 5 5 5 5 5 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 8 8 8 8 8 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 8 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 8 8 8 8 8 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 8 8 8 8 8 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
Output:
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 5 5 2 5 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 5 5 5 2 5 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 2 2 2 2 2 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 2 2 2 2 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 2 2 2 2 2 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 2 2 2 2 2 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5

Example 3:

Input:
5 8 8 8 8 8 5 2 5 5 5 5 5 5
5 8 8 8 8 8 5 5 5 5 5 8 8 8
5 5 5 5 5 5 5 5 5 5 5 8 8 8
5 5 5 5 8 8 8 8 8 8 5 8 8 8
5 5 5 5 8 8 8 8 8 8 5 8 8 8
5 5 5 5 8 8 8 8 8 8 5 8 8 8
8 8 5 5 8 8 8 8 8 8 5 5 5 5
8 8 5 5 8 8 8 8 8 8 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 5 5 8 8 8 5 5 5 5
2 5 8 8 8 5 5 8 8 8 5 5 5 2
5 5 8 8 8 5 5 5 5 5 5 5 5 5
5 5 8 8 8 5 5 2 5 5 5 5 5 5
Output:
5 8 8 8 8 8 5 2 5 5 5 5 5 5
5 8 8 8 8 8 5 2 5 5 5 8 8 8
5 5 5 5 5 5 5 2 5 5 5 8 8 8
5 5 5 5 2 2 2 2 2 2 5 8 8 8
5 5 5 5 2 2 2 2 2 2 5 8 8 8
5 5 5 5 2 2 2 2 2 2 5 8 8 8
8 8 5 5 2 2 2 2 2 2 5 5 5 5
8 8 5 5 2 2 2 2 2 2 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5
5 5 2 2 2 5 5 2 2 2 5 5 5 5
2 2 2 2 2 2 2 2 2 2 2 2 2 2
5 5 2 2 2 5 5 2 5 5 5 5 5 5
5 5 2 2 2 5 5 2 5 5 5 5 5 5


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 8 8 5 5 5 5 5 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 8 8 5 5 8 8 8 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 5 5 5 5 8 8 8 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 5 5 5 5 8 8 8 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 8 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 8 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
2 8 8 8 8 8 5 5 5 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 2
5 8 8 8 8 8 5 5 5 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 5
5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 8 8 5
5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 8 8 5
5 5 5 5 8 8 8 5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 5 5 5
2 5 5 5 8 8 8 5 5 5 5 5 5 8 8 8 8 8 5 8 8 8 8 5 2
5 5 5 5 8 8 8 5 5 8 8 8 5 8 8 8 8 8 5 8 8 8 8 5 5
5 5 5 5 5 5 5 5 5 8 8 8 5 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 5 5 8 8 8 5 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 2 5 5 5 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5

Answer: 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 8 8 5 5 5 5 5 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 8 8 5 5 8 8 8 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 5 5 5 5 8 8 8 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 5 5 5 5 8 8 8 5
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 2 2 2 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 2 2 2 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
5 2 2 2 2 2 5 5 5 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 5 5 5 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 5
5 5 5 5 5 2 5 5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 8 8 5
5 5 5 5 5 2 5 5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 8 8 5
5 5 5 5 2 2 2 5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 5 5 5
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
5 5 5 5 2 2 2 5 5 2 2 2 5 2 2 2 2 2 5 2 2 2 2 5 5
5 5 5 5 5 2 5 5 5 2 2 2 5 5 5 5 5 5 5 2 2 2 2 5 5
5 5 5 5 5 2 5 5 5 2 2 2 5 5 5 5 5 5 5 2 2 2 2 5 5
5 5 5 5 5 2 5 5 5 2 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5
Metadata: {'input': ((5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 8, 8, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (2, 8, 8, 8, 8, 8, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2), (5, 8, 8, 8, 8, 8, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5), (2, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 5, 2), (5, 5, 5, 5, 8, 8, 8, 5, 5, 8, 8, 8, 5, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)), 'output': ((5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 8, 8, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), (5, 2, 2, 2, 2, 2, 5, 5, 5, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5), (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), (5, 5, 5, 5, 2, 2, 2, 5, 5, 2, 2, 2, 5, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)), 'task_id': '0d87d2a6'}

````

### base_conversion
Generates base conversion tasks

Default configuration:
```python
min_base = 2
max_base = 16
min_value = 0
max_value = 1000
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to convert a number between two different bases.

If the target base is > 10, use lowercase letters a-z for digits above 9.

Example:
- Input: Convert the base-9 number 440 to base-5
- Output: 2420
- Explanation
    - First, we convert the base-9 number 440 to base-10: 4 * 9**2 + 4 * 9**1 + 0 * 9**0 = 324 + 36 + 0 = 360
    - Next, we convert the base-10 number 360 to base-5:
        - 360 // 5 = 72 remainder 0
        - 72 // 5 = 14 remainder 2
        - 14 // 5 = 2 remainder 4
        - 2 // 5 = 0 remainder 2
    - Reading the remainders in reverse order gives us the base-5 number 2 4 2 0
    - Hence, the final answer is 2420

Now, convert the base-3 number 220020 to binary

Answer: 1010001110
Metadata: {'decimal_value': 654, 'source_base': 3, 'target_base': 2, 'source_repr': '220020', 'target_repr': '1010001110'}

Example 2:
Question: Your task is to convert a number between two different bases.

If the target base is > 10, use lowercase letters a-z for digits above 9.

Example:
- Input: Convert the base-9 number 440 to base-5
- Output: 2420
- Explanation
    - First, we convert the base-9 number 440 to base-10: 4 * 9**2 + 4 * 9**1 + 0 * 9**0 = 324 + 36 + 0 = 360
    - Next, we convert the base-10 number 360 to base-5:
        - 360 // 5 = 72 remainder 0
        - 72 // 5 = 14 remainder 2
        - 14 // 5 = 2 remainder 4
        - 2 // 5 = 0 remainder 2
    - Reading the remainders in reverse order gives us the base-5 number 2 4 2 0
    - Hence, the final answer is 2420

Now, convert the base-6 number 103 to base-13

Answer: 30
Metadata: {'decimal_value': 39, 'source_base': 6, 'target_base': 13, 'source_repr': '103', 'target_repr': '30'}

Example 3:
Question: Your task is to convert a number between two different bases.

If the target base is > 10, use lowercase letters a-z for digits above 9.

Example:
- Input: Convert the base-9 number 440 to base-5
- Output: 2420
- Explanation
    - First, we convert the base-9 number 440 to base-10: 4 * 9**2 + 4 * 9**1 + 0 * 9**0 = 324 + 36 + 0 = 360
    - Next, we convert the base-10 number 360 to base-5:
        - 360 // 5 = 72 remainder 0
        - 72 // 5 = 14 remainder 2
        - 14 // 5 = 2 remainder 4
        - 2 // 5 = 0 remainder 2
    - Reading the remainders in reverse order gives us the base-5 number 2 4 2 0
    - Hence, the final answer is 2420

Now, convert the base-10 number 418 to base-13

Answer: 262
Metadata: {'decimal_value': 418, 'source_base': 10, 'target_base': 13, 'source_repr': '418', 'target_repr': '262'}

````

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
seed = 42
size = 500
format_style = simple
whitespace = single
```

Example tasks:
````
Example 1:
Question: Calculate -5 * -6.
Answer: 30
Metadata: {'num_terms': 2, 'num_digits': 1, 'expression': '-5 * -6'}

Example 2:
Question: Calculate 965 / 5.
Answer: 193
Metadata: {'num_terms': 2, 'num_digits': 3, 'expression': '965 / 5'}

Example 3:
Question: Calculate 0 + -2 + -4 * 0 * 3.
Answer: -2
Metadata: {'num_terms': 5, 'num_digits': 1, 'expression': '0 + -2 + -4 * 0 * 3'}

````

### bf
Generates BF tasks

Default configuration:
```python
seed = 42
size = 500
difficulty = 1
```

Example tasks:
````
Example 1:
Question: This is a BF (Brainf*ck) computer program. What is the output?

>[-]>[-]<>++++++++++[<+++++++++++>-]<+.-.+++++.--------------.+++++++++++++++.<

Respond only with the exact output of the program.
Answer: onset
Metadata: {'bfit_code': '\nint main() {\n    print("onset");\n}\n', 'bf_program': '>[-]>[-]<>++++++++++[<+++++++++++>-]<+.-.+++++.--------------.+++++++++++++++.<'}

Example 2:
Question: Consider the following BF (Brainf*ck) code. What would it output?

>[-]>[-]<>++++++++[<++++++++++++++>-]<.-----------.+++++++++++++.---------------.+++++.<

Provide only the exact output of the code.
Answer: perch
Metadata: {'bfit_code': '\nint main() {\n    print("perch");\n}\n', 'bf_program': '>[-]>[-]<>++++++++[<++++++++++++++>-]<.-----------.+++++++++++++.---------------.+++++.<'}

Example 3:
Question: This is a BF (Brainf*ck) computer program. What is the output?

>[-]>[-]<>+++++++++[<+++++++++++++>-]<.-------.----------.+.+++++++++++++.<

Respond only with the exact output of the program.
Answer: under
Metadata: {'bfit_code': '\nint main() {\n    print("under");\n}\n', 'bf_program': '>[-]>[-]<>+++++++++[<+++++++++++++>-]<.-------.----------.+.+++++++++++++.<'}

````

### binary_alternation
Generates Binary Alternation exercises with configurable difficulty

Default configuration:
```python
min_n = 10
max_n = 30
p_solvable = 0.8
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.

The string is called alternating if no two adjacent characters are equal. For example, the strings "010" and "1010" are alternating, while the string "0100" is not.

Any two characters may be swapped, even if they are not adjacent.

Example:
- Input: Determine the minimum number of swaps to make the following binary string alternating: 111000
- Output: 1

Now, determine the minimum number of swaps to make the following binary string alternating: 0010101011

Answer: 1
Metadata: {'string': '0010101011', 'solution': 1, 'solvable': True}

Example 2:
Question: Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.

The string is called alternating if no two adjacent characters are equal. For example, the strings "010" and "1010" are alternating, while the string "0100" is not.

Any two characters may be swapped, even if they are not adjacent.

Example:
- Input: Determine the minimum number of swaps to make the following binary string alternating: 111000
- Output: 1

Now, determine the minimum number of swaps to make the following binary string alternating: 00011111001010

Answer: 3
Metadata: {'string': '00011111001010', 'solution': 3, 'solvable': True}

Example 3:
Question: Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.

The string is called alternating if no two adjacent characters are equal. For example, the strings "010" and "1010" are alternating, while the string "0100" is not.

Any two characters may be swapped, even if they are not adjacent.

Example:
- Input: Determine the minimum number of swaps to make the following binary string alternating: 111000
- Output: 1

Now, determine the minimum number of swaps to make the following binary string alternating: 100000100111110000000111111

Answer: 7
Metadata: {'string': '100000100111110000000111111', 'solution': 7, 'solvable': True}

````

### binary_matrix
Generates Binary Matrix exercises with configurable difficulty

Default configuration:
```python
min_n = 3
max_n = 10
p_zero = 0.25
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.

Example:
- Input: Find the distance to the nearest 0 for each cell in the matrix below:
0 0 0
0 1 0
1 1 1
- Output:
0 0 0
0 1 0
1 2 1
- Explanation
    - Each cell with a 0 has a distance of 0 to itself.
    - The cell at (1, 1) has a distance of 1 to the nearest 0 (any of the three 0's at (1, 0), (0, 1), (1, 2)).
    - The cell at (2, 0) has a distance of 1 to the nearest 0 (the 0 at (1, 0)).
    - The cell at (2, 1) has a distance of 2 to the nearest 0 (any of the two 0's at (1, 0), (1, 2))
    - The cell at (2, 2) has a distance of 1 to the nearest 0 (the 0 at (1, 2)).
    - Hence, the final answer is the matrix is the output shown above, where each cell contains the distance to the nearest 0, in the same format as the input matrix.

Find the distance to the nearest 0 for each cell in the matrix below:
1 1 0 1
0 0 0 0
1 1 1 0
1 0 1 0

Answer: 1 1 0 1
0 0 0 0
1 1 1 0
1 0 1 0
Metadata: {'matrix': [[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0]], 'solution': [[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0]]}

Example 2:
Question: Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.

Example:
- Input: Find the distance to the nearest 0 for each cell in the matrix below:
0 0 0
0 1 0
1 1 1
- Output:
0 0 0
0 1 0
1 2 1
- Explanation
    - Each cell with a 0 has a distance of 0 to itself.
    - The cell at (1, 1) has a distance of 1 to the nearest 0 (any of the three 0's at (1, 0), (0, 1), (1, 2)).
    - The cell at (2, 0) has a distance of 1 to the nearest 0 (the 0 at (1, 0)).
    - The cell at (2, 1) has a distance of 2 to the nearest 0 (any of the two 0's at (1, 0), (1, 2))
    - The cell at (2, 2) has a distance of 1 to the nearest 0 (the 0 at (1, 2)).
    - Hence, the final answer is the matrix is the output shown above, where each cell contains the distance to the nearest 0, in the same format as the input matrix.

Find the distance to the nearest 0 for each cell in the matrix below:
1 0 1
1 1 1
1 0 1

Answer: 1 0 1
2 1 2
1 0 1
Metadata: {'matrix': [[1, 0, 1], [1, 1, 1], [1, 0, 1]], 'solution': [[1, 0, 1], [2, 1, 2], [1, 0, 1]]}

Example 3:
Question: Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.

Example:
- Input: Find the distance to the nearest 0 for each cell in the matrix below:
0 0 0
0 1 0
1 1 1
- Output:
0 0 0
0 1 0
1 2 1
- Explanation
    - Each cell with a 0 has a distance of 0 to itself.
    - The cell at (1, 1) has a distance of 1 to the nearest 0 (any of the three 0's at (1, 0), (0, 1), (1, 2)).
    - The cell at (2, 0) has a distance of 1 to the nearest 0 (the 0 at (1, 0)).
    - The cell at (2, 1) has a distance of 2 to the nearest 0 (any of the two 0's at (1, 0), (1, 2))
    - The cell at (2, 2) has a distance of 1 to the nearest 0 (the 0 at (1, 2)).
    - Hence, the final answer is the matrix is the output shown above, where each cell contains the distance to the nearest 0, in the same format as the input matrix.

Find the distance to the nearest 0 for each cell in the matrix below:
0 1 1 1 1 1 1 0 1
1 1 0 1 0 1 0 1 1
1 0 1 1 0 1 0 1 0
1 1 1 1 1 1 1 0 1
1 1 1 1 0 1 1 0 1
1 1 1 1 1 1 1 1 1
0 1 1 1 1 0 1 1 0
1 1 1 1 1 1 1 1 1
0 0 1 1 1 1 1 1 1

Answer: 0 1 1 2 1 2 1 0 1
1 1 0 1 0 1 0 1 1
1 0 1 1 0 1 0 1 0
2 1 2 2 1 2 1 0 1
2 2 2 1 0 1 1 0 1
1 2 3 2 1 1 2 1 1
0 1 2 2 1 0 1 1 0
1 1 2 3 2 1 2 2 1
0 0 1 2 3 2 3 3 2
Metadata: {'matrix': [[0, 1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1]], 'solution': [[0, 1, 1, 2, 1, 2, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 1, 0], [2, 1, 2, 2, 1, 2, 1, 0, 1], [2, 2, 2, 1, 0, 1, 1, 0, 1], [1, 2, 3, 2, 1, 1, 2, 1, 1], [0, 1, 2, 2, 1, 0, 1, 1, 0], [1, 1, 2, 3, 2, 1, 2, 2, 1], [0, 0, 1, 2, 3, 2, 3, 3, 2]]}

````

### bitwise_arithmetic
Dataset that generates tasks testing understanding of bitwise arithmetic operations.

    Generates expressions combining:
    - Standard arithmetic operators (+, -, *)
    - Bitwise shift operators (<<, >>)
    - Multi-byte hexadecimal numbers (e.g. 0x100 to 0xFFFF)

    The difficulty parameter controls expression complexity:
    - Level 1: Simple expressions like (0x123 + 0x456)
    - Level 2: Nested expressions with shifts like ((0x123 + 0x456) << 1)
    - Level 3+: Deeper nesting like ((0x123 + 0x456) << (0x789 >> 1))

    Each task provides:
    - A question asking to evaluate an expression
    - The correct answer in hexadecimal format
    - Metadata including the raw expression

    The dataset verifies answers by evaluating them as Python expressions,
    supporting both integer and hexadecimal string formats.

Default configuration:
```python
difficulty = 2
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Please solve this problem. Assume there is arbitrary bit depth and that there are signed integers. Reply only with the final hexidecimal value.
((0x3a24 - 0x24b8) + (0x1741 >> 0x3))
Answer: 0x1854
Metadata: {'problem': '((0x3a24 - 0x24b8) + (0x1741 >> 0x3))'}

Example 2:
Question: Please solve this problem. Assume there is arbitrary bit depth and that there are signed integers. Reply only with the final hexidecimal value.
((0xacf1 * 0xb3cc) - (0x9a4b << 0x0))
Answer: 0x7975b8c1
Metadata: {'problem': '((0xacf1 * 0xb3cc) - (0x9a4b << 0x0))'}

Example 3:
Question: Please solve this problem. Assume there is arbitrary bit depth and that there are signed integers. Reply only with the final hexidecimal value.
((0x2e39 + 0x622b) >> 0x0)
Answer: 0x9064
Metadata: {'problem': '((0x2e39 + 0x622b) >> 0x0)'}

````

### caesar_cipher
Generates Caesar cipher encryption/decryption tasks

Default configuration:
```python
delimiter = .
min_words = 3
max_words = 20
min_rotation = 1
max_rotation = 25
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Decrypt this Caesar cipher text: JNJUBUF ZPVS BTTPDJBUF XIPN J XBT DPNQMJNFOUJOH B NPNFOU BHP. Provide only the decrypted text as your final answer.
Answer: IMITATE YOUR ASSOCIATE WHOM I WAS COMPLIMENTING A MOMENT AGO
Metadata: {'rotation': 1, 'cipher_text': 'JNJUBUF ZPVS BTTPDJBUF XIPN J XBT DPNQMJNFOUJOH B NPNFOU BHP', 'clear_text': 'IMITATE YOUR ASSOCIATE WHOM I WAS COMPLIMENTING A MOMENT AGO'}

Example 2:
Question: Decrypt this Caesar cipher text: PBSDJ XKZYVOYX CWSDR LYEQRD SD PYB K WOBO KXN YBSQSXKDON DOVOZRYXSM TYEBXKVSCW. Provide only the decrypted text as your final answer.
Answer: FRITZ NAPOLEON SMITH BOUGHT IT FOR A MERE AND ORIGINATED TELEPHONIC JOURNALISM
Metadata: {'rotation': 10, 'cipher_text': 'PBSDJ XKZYVOYX CWSDR LYEQRD SD PYB K WOBO KXN YBSQSXKDON DOVOZRYXSM TYEBXKVSCW', 'clear_text': 'FRITZ NAPOLEON SMITH BOUGHT IT FOR A MERE AND ORIGINATED TELEPHONIC JOURNALISM'}

Example 3:
Question: Decrypt this Caesar cipher text: ZW PFLI JKFDRTY ZJ FLK FW ZK DLJK SV DVEUVU. Provide only the decrypted text as your final answer.
Answer: IF YOUR STOMACH IS OUT OF IT MUST BE MENDED
Metadata: {'rotation': 17, 'cipher_text': 'ZW PFLI JKFDRTY ZJ FLK FW ZK DLJK SV DVEUVU', 'clear_text': 'IF YOUR STOMACH IS OUT OF IT MUST BE MENDED'}

````

### calendar_arithmetic
Default configuration:
```python
year = 2022
tasks = ['weekday_offset', 'weekday_of_date', 'weekday_of_date_from_first_day', 'recurring_event_day', 'count_days', 'count_business_days', 'is_leap_year']
offset_upper_bound = 100
leap_year_range = 200
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Between Sunday, February 27, 2022 and Wednesday, March 2, 2022 (counting both dates), what's the total count of business days (Monday through Friday)? Give the count numerically.
Answer: 3
Metadata: {'task': 'count_business_days', 'start_date': '2022-02-27', 'end_date': '2022-03-02'}

Example 2:
Question: Starting from Monday, May 23, 2022, which weekday was it 98 days before? Write out the full weekday name.
Answer: Monday
Metadata: {'task': 'weekday_offset', 'start_date': '2022-05-23', 'offset_days': -98, 'target_date': '2022-02-14'}

Example 3:
Question: If a meeting is scheduled on the last Saturday of September 2022, on which day of the month does it occur? Respond with just the number. Answer with -1 if the ordinal does not exist in the month.
Answer: 24
Metadata: {'task': 'recurring_event_day', 'year': 2022, 'month': 9, 'ordinal': 'last', 'weekday': 'Saturday'}

````

### chain_sum
Generates simple arithmetic tasks using only + and - operators

Default configuration:
```python
min_terms = 2
max_terms = 6
min_digits = 1
max_digits = 4
allow_negation = False
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: State the final answer to the following arithmetic problem: 4 + 3 =
Answer: 7
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 1}, 'expression': '4 + 3'}

Example 2:
Question: State the final answer to the following arithmetic problem: 812 + 880 =
Answer: 1692
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 3}, 'expression': '812 + 880'}

Example 3:
Question: State the final answer to the following arithmetic problem: 2 + 6 + 3 + 4 + 0 =
Answer: 15
Metadata: {'difficulty': {'num_terms': 5, 'num_digits': 1}, 'expression': '2 + 6 + 3 + 4 + 0'}

````

### circuit_logic
Generates random digital logic circuits (in ASCII) together with:
      - a random Boolean expression,
      - random input assignments,
      - the final evaluated output.

    Each item in the dataset is a dict with:
       {
           "question": <str>,
           "answer": <str>,
           "metadata": {
               "diagram": <ASCII circuit diagram>,
               "expression": <str>,
               "term_strings": <list of term_strings>,
               "assignments": <dict of input->0/1>,
               "final_gate": <str>,
               "inputs": <list of input names>,
           }
       }

Default configuration:
```python
num_terms = 5
min_inputs = 2
max_inputs = 4
neg_prob = 0.3
allow_reuse = True
size = 100
seed = 42
```

Example tasks:
````
Example 1:
Question: Below is a randomly generated logic circuit.

A: ─────────────────┐
B: ───────────────┐ │
C: ─────────────┐ │ │
D: ───────────┐ │ │ │
E: ─────────┐ │ │ │ │
F: ───────┐ │ │ │ │ │
G: ─────┐ │ │ │ │ │ │
H: ───┐ │ │ │ │ │ │ │
I: ─┐ │ │ │ │ │ │ │ │
    │ │ │ │ │ │ │ │ ├─>o─│&&
    │ │ │ │ │ │ │ │ ├────│&&───┐
    │ │ │ │ │ │ │ ├───>o─│&&   │
    │ │ │ │ │ │ │ │ ├─>o─│&&   │
    │ │ │ │ │ │ │ │ │          │
    │ │ │ │ │ │ ├────────│&&   │
    │ │ │ │ │ ├──────────│&&──┐│
    │ │ │ │ │ └──────────│&&  ││
    │ │ │ │ ├─────────>o─│&&  ││
    │ │ │ │ │   │ │ │         │└───│++
    │ │ │ │ │   ├─────>o─│&&  └────│++
    │ │ │ └──────────────│&&───────│++─── OUT
    │ │ │   │   │ │ └────│&&  ┌────│++
    │ │ │   │   └────────│&&  │┌───│++
    │ │ │   │     │           ││
    │ │ └────────────────│⊕⊕  ││
    │ │     ├─────────>o─│⊕⊕──┘│
    │ ├──────────────────│⊕⊕   │
    │ │     │     │            │
    │ │     │     └───>o─│↑↑   │
    │ │     └────────────│↑↑───┘
    └────────────────────│↑↑
      └──────────────────│↑↑


Legend for gates:
&&: AND
↑↑: NAND
⊕⊕: XOR
>o: Negate
++: OR

Given the following input assignments:
  A = 1
  B = 0
  C = 1
  D = 1
  E = 0
  F = 1
  G = 0
  H = 0
  I = 0

What is the final output?
Answer: 1
Metadata: {'expression': "(A'&A&B'&A')+(C&D&D&E')+(C'&F&A&C)+(G⊕E'⊕H)+(B'↑E↑I↑H)", 'assignments': {'A': 1, 'B': 0, 'C': 1, 'D': 1, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 0}, 'term_strings': ["A'&A&B'&A'", "C&D&D&E'", "C'&F&A&C", "G⊕E'⊕H", "B'↑E↑I↑H"], 'final_gate': 'OR', 'inputs': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']}

Example 2:
Question: Below is a randomly generated logic circuit.

A: ─────────────────────┐
B: ───────────────────┐ │
C: ─────────────────┐ │ │
D: ───────────────┐ │ │ │
E: ─────────────┐ │ │ │ │
F: ───────────┐ │ │ │ │ │
G: ─────────┐ │ │ │ │ │ │
H: ───────┐ │ │ │ │ │ │ │
I: ─────┐ │ │ │ │ │ │ │ │
J: ───┐ │ │ │ │ │ │ │ │ │
K: ─┐ │ │ │ │ │ │ │ │ │ │
    │ │ │ │ │ │ │ │ │ │ ├────│↑↑
    │ │ │ │ │ │ │ │ │ ├──────│↑↑───┐
    │ │ │ │ │ │ │ │ └─────>o─│↑↑   │
    │ │ │ │ │ │ │ └──────────│↑↑   │
    │ │ │ │ │ │ │     │ │          │
    │ │ │ │ │ │ └────────────│⊕⊕   │
    │ │ │ │ │ └──────────────│⊕⊕──┐│
    │ │ │ │ └────────────────│⊕⊕  ││
    │ │ │ │           │ │         │└───│++
    │ │ │ ├──────────────────│&&─┐└────│++
    │ │ │ └───────────────>o─│&& └─────│++─── OUT
    │ │ │             │ │         ┌────│++
    │ │ └─────────────────>o─│↑↑  │┌───│++
    │ │               ├───>o─│↑↑──┘│
    │ │               │ └────│↑↑   │
    │ │               └──────│↑↑   │
    │ │                            │
    │ ├──────────────────────│&&   │
    │ └──────────────────────│&&───┘
    ├────────────────────────│&&
    └────────────────────────│&&


Legend for gates:
&&: AND
↑↑: NAND
⊕⊕: XOR
>o: Negate
++: OR

Given the following input assignments:
  A = 0
  B = 0
  C = 0
  D = 1
  E = 0
  F = 1
  G = 1
  H = 0
  I = 0
  J = 0
  K = 1

What is the final output?
Answer: 1
Metadata: {'expression': "(A↑B↑C'↑D)+(E⊕F⊕G)+(H&H')+(I'↑B'↑A↑B)+(J&J&K&K)", 'assignments': {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 0, 'F': 1, 'G': 1, 'H': 0, 'I': 0, 'J': 0, 'K': 1}, 'term_strings': ["A↑B↑C'↑D", 'E⊕F⊕G', "H&H'", "I'↑B'↑A↑B", 'J&J&K&K'], 'final_gate': 'OR', 'inputs': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']}

Example 3:
Question: Below is a randomly generated logic circuit.

A: ───────────┐
B: ─────────┐ │
C: ───────┐ │ │
D: ─────┐ │ │ │
E: ───┐ │ │ │ │
F: ─┐ │ │ │ │ │
    │ │ │ │ │ ├────│⊕⊕
    │ │ │ │ │ ├─>o─│⊕⊕───┐
    │ │ │ │ │ ├────│⊕⊕   │
    │ │ │ │ │ ├────│⊕⊕   │
    │ │ │ │ │ │          │
    │ │ │ │ │ ├────│↑↑   │
    │ │ │ │ │ ├────│↑↑──┐│
    │ │ │ │ │ ├────│↑↑  ││
    │ │ │ │ │ ├─>o─│↑↑  ││
    │ │ │ │ │ │         │└───│&&
    │ │ │ │ │ ├─>o─│⊕⊕  └────│&&
    │ │ │ │ │ ├─>o─│⊕⊕───────│&&─── OUT
    │ │ │ │ │ ├────│⊕⊕  ┌────│&&
    │ │ │ │ │ │         │┌───│&&
    │ │ │ │ └──────│&&  ││
    │ │ │ ├────────│&&──┘│
    │ │ └──────────│&&   │
    │ ├────────────│&&   │
    │ │   │   │          │
    │ └─────────>o─│↑↑   │
    │     └────────│↑↑───┘
    └──────────────│↑↑
              └─>o─│↑↑


Legend for gates:
&&: AND
↑↑: NAND
⊕⊕: XOR
>o: Negate
&&: AND

Given the following input assignments:
  A = 0
  B = 1
  C = 1
  D = 0
  E = 1
  F = 0

What is the final output?
Answer: 0
Metadata: {'expression': "(A⊕A'⊕A⊕A)&(A↑A↑A↑A')&(A'⊕A'⊕A)&(B&C&D&E)&(E'↑C↑F↑A')", 'assignments': {'A': 0, 'B': 1, 'C': 1, 'D': 0, 'E': 1, 'F': 0}, 'term_strings': ["A⊕A'⊕A⊕A", "A↑A↑A↑A'", "A'⊕A'⊕A", 'B&C&D&E', "E'↑C↑F↑A'"], 'final_gate': 'AND', 'inputs': ['A', 'B', 'C', 'D', 'E', 'F']}

````

### color_cube_rotation
Generates color cube rotation reasoning tasks

Default configuration:
```python
min_rotations = 1
max_rotations = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: A cube has:
- a pink top side
- a gray right side
- a orange front side
- a purple left side
- a indigo back side
- a cyan bottom side

The cube is rotated so that the side which was before at the bottom is now at the top.

What is now the color of the back side of the cube?
Provide only the color as your final answer.
Answer: orange
Metadata: {'initial_state': {'top': 'pink', 'right': 'gray', 'front': 'orange', 'left': 'purple', 'back': 'indigo', 'bottom': 'cyan'}, 'rotations': ['bottom'], 'target_side': 'back', 'num_rotations': 1}

Example 2:
Question: A cube has:
- a gray top side
- a brown right side
- a silver front side
- a red left side
- a purple back side
- a yellow bottom side

The cube is rotated so that the side which was before at the left is now at the top.

Next, the bottom side is rotated to become the top face.

After that the cube is turned to make the bottom face the top.

What is now the color of the left side of the cube?
Provide only the color as your final answer.
Answer: yellow
Metadata: {'initial_state': {'top': 'gray', 'right': 'brown', 'front': 'silver', 'left': 'red', 'back': 'purple', 'bottom': 'yellow'}, 'rotations': ['left', 'bottom', 'bottom'], 'target_side': 'left', 'num_rotations': 3}

Example 3:
Question: A cube has:
- a orange top side
- a cyan right side
- a violet front side
- a pink left side
- a gray back side
- a gold bottom side

The cube is rotated so that the side which was before at the left is now at the top.

Now the cube is rotated to place its back side at the top.

Now the cube is rotated to place its bottom side at the top.

What is now the color of the left side of the cube?
Provide only the color as your final answer.
Answer: gold
Metadata: {'initial_state': {'top': 'orange', 'right': 'cyan', 'front': 'violet', 'left': 'pink', 'back': 'gray', 'bottom': 'gold'}, 'rotations': ['left', 'back', 'bottom'], 'target_side': 'left', 'num_rotations': 3}

````

### complex_arithmetic
Generates complex number arithmetic problems.

Default configuration:
```python
min_real = -10
max_real = 10
min_imag = -10
max_imag = 10
operations = ('+', '-', '*', '/')
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Add the complex numbers: (-10.0 - 2.0i) + (-3.0 - 3.0i)
Answer: -13.0 - 5.0i
Metadata: {'num1': (-10.0, -2.0), 'num2': (-3.0, -3.0), 'operation': '+', 'result': (-13, -5)}

Example 2:
Question: Add the complex numbers: (-1.0 - 6.0i) + (4.0 + 1.0i)
Answer: 3.0 - 5.0i
Metadata: {'num1': (-1.0, -6.0), 'num2': (4.0, 1.0), 'operation': '+', 'result': (3, -5)}

Example 3:
Question: Divide the complex numbers: (-7.0 - 79.0i) ÷ (-7.0 - 5.0i)
Answer: 6.0 + 7.0i
Metadata: {'num1': (-7.0, -79.0), 'num2': (-7.0, -5.0), 'operation': '/', 'result': (6, 7)}

````

### count_bits
Generates Count Bits exercises with configurable difficulty

Default configuration:
```python
max_n = 2147483647
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: How many 1 bits are there in the binary representation of the number 1373158607?
Answer: 18
Metadata: {'number': 1373158607, 'solution': 18, 'binary': '1010001110110001011110011001111'}

Example 2:
Question: How many 1 bits are there in the binary representation of the number 82789451?
Answer: 14
Metadata: {'number': 82789451, 'solution': 14, 'binary': '100111011110100010001001011'}

Example 3:
Question: How many 1 bits are there in the binary representation of the number 877324117?
Answer: 16
Metadata: {'number': 877324117, 'solution': 16, 'binary': '110100010010101110011101010101'}

````

### count_primes
Generates Count Primes exercises with configurable difficulty

Default configuration:
```python
max_n = 10000
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Count how many prime numbers there are between 1825 and 2029 (inclusive) ?
Answer: 27
Metadata: {'start': 1825, 'end': 2029, 'primes': [False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True], 'solution': 27}

Example 2:
Question: Count how many prime numbers there are between 632 and 5319 (inclusive) ?
Answer: 589
Metadata: {'start': 632, 'end': 5319, 'primes': [False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False], 'solution': 589}

Example 3:
Question: Count how many prime numbers there are between 6694 and 8824 (inclusive) ?
Answer: 236
Metadata: {'start': 6694, 'end': 8824, 'primes': [False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False], 'solution': 236}

````

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
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Calculate 139 using the numbers 36, 29, 95, 32, 4, 15.
Each number may be used at most once.
Final answer format instructions:
1. Provide your solution as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.
5. Do not include any other text or formatting.

Answer: 15 - 4 + 95 + 36 - 32 + 29
Metadata: {'numbers': [36, 29, 95, 32, 4, 15], 'target': 139, 'expression': '15 - 4 + 95 + 36 - 32 + 29'}

Example 2:
Question: Using the numbers 74, 48, 56, 66, create an expression that equals 132.
You can only use each number once.
Final answer format instructions:
1. Provide your solution as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.
5. Do not include any other text or formatting.

Answer: 66 - 56 + 74 + 48
Metadata: {'numbers': [74, 48, 56, 66], 'target': 132, 'expression': '66 - 56 + 74 + 48'}

Example 3:
Question: Using the numbers 5, 41, 38, 81, 14, create an expression that equals 450.
You can only use each number once.
Final answer format instructions:
1. Provide your solution as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.
5. Do not include any other text or formatting.

Answer: 41*14 - 81 - 38 - 5
Metadata: {'numbers': [5, 41, 38, 81, 14], 'target': 450, 'expression': '41*14 - 81 - 38 - 5'}

````

### course_schedule
Generates Course Schedule exercises with configurable difficulty

Default configuration:
```python
num_courses = 5
max_num_prerequisites = 2
p_solvable = 0.5
min_cycle_length = 3
max_cycle_length = 5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: There are a total of 5 courses you have to take, labeled from 0 to 4.

You are given the following list of prerequisites, where prerequisites[i] = (a_i, b_i) indicates that you must first take course b_i if you want to take course a_i:
[(2, 1), (4, 2), (4, 3), (2, 3)]

Return True if you can finish all courses considering the prerequisites, or False otherwise.

Answer: True
Metadata: {'courses': [3, 1, 2, 4, 0], 'prerequisites': [(2, 1), (4, 2), (4, 3), (2, 3)], 'solution': True, 'solvable': True}

Example 2:
Question: There are a total of 5 courses you have to take, labeled from 0 to 4.

You are given the following list of prerequisites, where prerequisites[i] = (a_i, b_i) indicates that you must first take course b_i if you want to take course a_i:
[(3, 0), (2, 4), (2, 3), (4, 1), (3, 1), (0, 1), (0, 2), (1, 3)]

Return True if you can finish all courses considering the prerequisites, or False otherwise.

Answer: False
Metadata: {'courses': [1, 4, 3, 2, 0], 'prerequisites': [(3, 0), (2, 4), (2, 3), (4, 1), (3, 1), (0, 1), (0, 2), (1, 3)], 'solution': False, 'solvable': False}

Example 3:
Question: There are a total of 5 courses you have to take, labeled from 0 to 4.

You are given the following list of prerequisites, where prerequisites[i] = (a_i, b_i) indicates that you must first take course b_i if you want to take course a_i:
[]

Return True if you can finish all courses considering the prerequisites, or False otherwise.

Answer: True
Metadata: {'courses': [2, 1, 4, 0, 3], 'prerequisites': [], 'solution': True, 'solvable': True}

````

### cryptarithm
Generates cryptarithm puzzles by:
      1) Randomly choosing integers for each "addend" (with no leading zero if not allowed),
      2) Summing them,
      3) Mapping distinct digits (0..9) to letters (A..Z),
      4) Formatting the puzzle text.

    This approach guarantees sum correctness and avoids repeated failures.

Default configuration:
```python
min_words = 2
max_words = 3
allow_leading_zero = False
include_example = True
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve this cryptarithm:

    FOM
+ IKPLO
-------
  IKIZL

Each letter stands for a unique digit (0-9). No leading letter can be zero.
Provide a comma separated mapping from letters to digits that satisfies the equation in your final answer. Output format: "A=1,B=2,C=3" (without quotes)

Here's an example:
- Input:
  BASE
+ BALL
------
 GAMES

- Output: B=7, A=4, S=8, E=3, L=5, M=9, G=1
- Explanation:
    * BASE + BALL = GAMES, two 4-digit numbers sum to 5 digits, so G = 1.
    * Units: E + L = S (no carry).
    * Tens: S + L = E + 10 (carry 1). Substitute S = E + L to get E + 2L = E + 10, so L = 5.
    * Since S = E + 5 and S is one digit, E < 5.
    * Hundreds: 2A + 1 = M (with carry).
    * Thousands: 2B = A + 10 (carry makes G = 1). So A = 2B - 10.
    * Try B = 7: Then A = 4 and M = 2(4) + 1 = 9.
    * With E < 5, try E = 3: Then S = 8.
    * Solution: B = 7, A = 4, S = 8, E = 3, L = 5, M = 9, G = 1
    * Verify: BASE (7483) + BALL (7455) = GAMES (14938).

Answer: F=3,I=4,K=2,L=9,M=1,O=8,P=0,Z=7
Metadata: {'letters': ['L', 'O', 'K', 'I', 'P', 'Z', 'M', 'F'], 'word_values': [381, 42098], 'sum_number': 42479, 'words_letters': ['FOM', 'IKPLO'], 'result_letters': 'IKIZL', 'digit_to_letter': {'9': 'L', '8': 'O', '2': 'K', '4': 'I', '0': 'P', '7': 'Z', '1': 'M', '3': 'F'}, 'letter_to_digit': {'L': 9, 'O': 8, 'K': 2, 'I': 4, 'P': 0, 'Z': 7, 'M': 1, 'F': 3}}

Example 2:
Question: Solve this cryptarithm:

   HHPD
+ JIOKP
-------
  JHEDH

Each letter stands for a unique digit (0-9). No leading letter can be zero.
Provide a comma separated mapping from letters to digits that satisfies the equation in your final answer. Output format: "A=1,B=2,C=3" (without quotes)

Here's an example:
- Input:
  BASE
+ BALL
------
 GAMES

- Output: B=7, A=4, S=8, E=3, L=5, M=9, G=1
- Explanation:
    * BASE + BALL = GAMES, two 4-digit numbers sum to 5 digits, so G = 1.
    * Units: E + L = S (no carry).
    * Tens: S + L = E + 10 (carry 1). Substitute S = E + L to get E + 2L = E + 10, so L = 5.
    * Since S = E + 5 and S is one digit, E < 5.
    * Hundreds: 2A + 1 = M (with carry).
    * Thousands: 2B = A + 10 (carry makes G = 1). So A = 2B - 10.
    * Try B = 7: Then A = 4 and M = 2(4) + 1 = 9.
    * With E < 5, try E = 3: Then S = 8.
    * Solution: B = 7, A = 4, S = 8, E = 3, L = 5, M = 9, G = 1
    * Verify: BASE (7483) + BALL (7455) = GAMES (14938).

Answer: D=8,E=9,H=3,I=0,J=7,K=2,O=6,P=5
Metadata: {'letters': ['O', 'K', 'H', 'P', 'I', 'D', 'E', 'J'], 'word_values': [3358, 70625], 'sum_number': 73983, 'words_letters': ['HHPD', 'JIOKP'], 'result_letters': 'JHEDH', 'digit_to_letter': {'6': 'O', '2': 'K', '3': 'H', '5': 'P', '0': 'I', '8': 'D', '9': 'E', '7': 'J'}, 'letter_to_digit': {'O': 6, 'K': 2, 'H': 3, 'P': 5, 'I': 0, 'D': 8, 'E': 9, 'J': 7}}

Example 3:
Question: Solve this cryptarithm:

   RZRHA
   PPXZZ
+  ZHGZA
--------
  XXNXHZ

Each letter stands for a unique digit (0-9). No leading letter can be zero.
Provide a comma separated mapping from letters to digits that satisfies the equation in your final answer. Output format: "A=1,B=2,C=3" (without quotes)

Here's an example:
- Input:
  BASE
+ BALL
------
 GAMES

- Output: B=7, A=4, S=8, E=3, L=5, M=9, G=1
- Explanation:
    * BASE + BALL = GAMES, two 4-digit numbers sum to 5 digits, so G = 1.
    * Units: E + L = S (no carry).
    * Tens: S + L = E + 10 (carry 1). Substitute S = E + L to get E + 2L = E + 10, so L = 5.
    * Since S = E + 5 and S is one digit, E < 5.
    * Hundreds: 2A + 1 = M (with carry).
    * Thousands: 2B = A + 10 (carry makes G = 1). So A = 2B - 10.
    * Try B = 7: Then A = 4 and M = 2(4) + 1 = 9.
    * With E < 5, try E = 3: Then S = 8.
    * Solution: B = 7, A = 4, S = 8, E = 3, L = 5, M = 9, G = 1
    * Verify: BASE (7483) + BALL (7455) = GAMES (14938).

Answer: A=0,G=7,H=9,N=8,P=3,R=2,X=1,Z=5
Metadata: {'letters': ['Z', 'H', 'N', 'G', 'X', 'A', 'R', 'P'], 'word_values': [25290, 33155, 59750], 'sum_number': 118195, 'words_letters': ['RZRHA', 'PPXZZ', 'ZHGZA'], 'result_letters': 'XXNXHZ', 'digit_to_letter': {'5': 'Z', '9': 'H', '8': 'N', '7': 'G', '1': 'X', '0': 'A', '2': 'R', '3': 'P'}, 'letter_to_digit': {'Z': 5, 'H': 9, 'N': 8, 'G': 7, 'X': 1, 'A': 0, 'R': 2, 'P': 3}}

````

### decimal_arithmetic
Dataset that generates basic arithmetic tasks using Decimal arithmetic and proper operator precedence.

Default configuration:
```python
min_num_decimal_places = 6
max_num_decimal_places = 6
precision = 28
terms = 6
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Please solve this problem to a maximum of 28 significant digits, rounding up from the half. Only reply with the final value.
(0.419611*3.744855)-(9.149733+0.533225)+3.668137-9.416130 = ?
Answer: -13.859568648595

Example 2:
Question: Please solve this problem to a maximum of 28 significant digits, rounding up from the half. Only reply with the final value.
(4.799697-6.205510+(8.359621+9.674082*6.609140)-1.800269) = ?
Answer: 69.09090130948

Example 3:
Question: Please solve this problem to a maximum of 28 significant digits, rounding up from the half. Only reply with the final value.
((8.724497+6.368109)-0.488171-9.541022+(2.628828*9.915288)) = ?
Answer: 31.128999722464

````

### decimal_chain_sum
Generates simple decimal arithmetic tasks using only + and - operators

Default configuration:
```python
min_terms = 2
max_terms = 6
min_digits = 1
max_digits = 4
min_decimal_places = 1
max_decimal_places = 4
allow_negation = False
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: State the final answer to the following arithmetic problem: 4.23 + 3.96 =
Answer: 8.19
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 1}, 'expression': '4.23 + 3.96'}

Example 2:
Question: State the final answer to the following arithmetic problem: 812.57 - 880.2577 =
Answer: -67.6877
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 3}, 'expression': '812.57 - 880.2577'}

Example 3:
Question: State the final answer to the following arithmetic problem: 2.75 - 6.5 - 3.7 + 4.7 - 0.98 =
Answer: -3.73
Metadata: {'difficulty': {'num_terms': 5, 'num_digits': 1}, 'expression': '2.75 - 6.5 - 3.7 + 4.7 - 0.98'}

````

### dice
Generates Dice-based puzzles with configurable parameters

Default configuration:
```python
num_dice = 4
max_dice_size = 20
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: I have these dice: 1d20, 1d10, 1d5, 1d2. What are the odds of rolling 18 or higher? (Assume that all dice are rolled at once, and that '1d6' represents one roll of a 6-sided dice.) Please respond with a reduced fraction representing the probability [ex., 1/60].
Answer: 13/20

Example 2:
Question: I have these dice: 1d20, 1d11, 1d6, 1d3. What are the odds of rolling 23 or higher? (Assume that all dice are rolled at once, and that '1d6' represents one roll of a 6-sided dice.) Please respond with a reduced fraction representing the probability [ex., 1/60].
Answer: 19/40

Example 3:
Question: I have these dice: 1d20, 1d19, 1d18, 1d15. What are the odds of rolling 48 or higher? (Assume that all dice are rolled at once, and that '1d6' represents one roll of a 6-sided dice.) Please respond with a reduced fraction representing the probability [ex., 1/60].
Answer: 9677/51300

````

### emoji_mystery
Default configuration:
```python
size = 1000
seed = 42
min_words_in_sentence = 3
max_words_in_sentence = 35
```

Example tasks:
````
Example 1:
Question: The following emoji is encoded with a sentence.
Decode the following sentence from the emoji: 😽󠄶󠅢󠅙󠅤󠅪󠄐󠄾󠅑󠅠󠅟󠅜󠅕󠅟󠅞󠄐󠅃󠅝󠅙󠅤󠅘󠄐󠅑󠅧󠅟󠅛󠅕󠄐󠅙󠅞󠄐󠅦󠅕󠅢󠅩󠄐󠅒󠅑󠅔󠄐󠅘󠅥󠅝󠅟󠅢󠄞
Here is a hint: 
```python
def variance_selector_to_byte(variation_selector):
    variation_selector_codepoint = ord(variation_selector)
    if 0xFE00 <= variation_selector_codepoint <= 0xFE0F:
        return variation_selector_codepoint - 0xFE00
    elif 0xE0100 <= variation_selector_codepoint <= 0xE01EF:
        return variation_selector_codepoint - 0xE0100 + 16
    else:
        return None
def decode(encoded_sentence):
    decoded_bytes = []
    variation_selectors_part = encoded_sentence[1:]
    for char in variation_selectors_part:
        byte_val = variance_selector_to_byte(char)
        if byte_val is not None:
            decoded_bytes.append(byte_val)
    return bytes(decoded_bytes).decode('utf-8')
```

Return the secret sentence as your final answer.
Answer: Fritz Napoleon Smith awoke in very bad humor.
Metadata: {'emoji': '😽'}

Example 2:
Question: The following emoji is encoded with a sentence.
Decode the following sentence from the emoji: 😆󠄱󠅞󠅔󠄐󠅙󠅞󠅔󠅕󠅕󠅔󠄜󠄐󠅣󠅟󠄐󠅓󠅟󠅞󠅖󠅙󠅔󠅕󠅞󠅤󠄐󠅧󠅑󠅣󠄐󠅤󠅘󠅕󠅙󠅢󠄐󠅑󠅙󠅢󠄜󠄐󠅤󠅘󠅕󠅩󠄐󠅣󠅕󠅕󠅝󠅕󠅔󠄐󠅤󠅟󠄐󠅘󠅑󠅦󠅕󠄐󠅞󠅟󠄐󠅔󠅟󠅥󠅒󠅤󠄐󠅑󠅣󠄐󠅤󠅟󠄐󠅤󠅘󠅕󠄐󠅠󠅟󠅣󠅣󠅙󠅒󠅙󠅜󠅙󠅤󠅩󠄐󠅟󠅖󠄐󠅣󠅥󠅓󠅓󠅕󠅣󠅣󠄐󠅙󠅞󠄐󠅣󠅥󠅓󠅘󠄐󠅑󠅞󠄐󠅥󠅞󠅔󠅕󠅢󠅤󠅑󠅛󠅙󠅞󠅗󠄞
Here is a hint: 
```python
def variance_selector_to_byte(variation_selector):
    variation_selector_codepoint = ord(variation_selector)
    if 0xFE00 <= variation_selector_codepoint <= 0xFE0F:
        return variation_selector_codepoint - 0xFE00
    elif 0xE0100 <= variation_selector_codepoint <= 0xE01EF:
        return variation_selector_codepoint - 0xE0100 + 16
    else:
        return None
def decode(encoded_sentence):
    decoded_bytes = []
    variation_selectors_part = encoded_sentence[1:]
    for char in variation_selectors_part:
        byte_val = variance_selector_to_byte(char)
        if byte_val is not None:
            decoded_bytes.append(byte_val)
    return bytes(decoded_bytes).decode('utf-8')
```

Return the secret sentence as your final answer.
Answer: And indeed, so confident was their air, they seemed to have no doubt as to the possibility of success in such an undertaking.
Metadata: {'emoji': '😆'}

Example 3:
Question: The following emoji is encoded with a sentence.
Decode the following sentence from the emoji: 😱󠄹󠅞󠄐󠅖󠅑󠅓󠅤󠄜󠄐󠅟󠅞󠅕󠄐󠅟󠅖󠄐󠅤󠅘󠅕󠅝󠄐󠅧󠅑󠅞󠅤󠅕󠅔󠄐󠅤󠅟󠄐󠅢󠅕󠅦󠅙󠅦󠅕󠄐󠅠󠅑󠅙󠅞󠅤󠅙󠅞󠅗󠄜󠄐󠅑󠅞󠄐󠅑󠅢󠅤󠄐󠅖󠅑󠅜󠅜󠅕󠅞󠄐󠅙󠅞󠅤󠅟󠄐󠅔󠅕󠅣󠅥󠅕󠅤󠅥󠅔󠅕󠄐󠅟󠅧󠅙󠅞󠅗󠄐󠅤󠅟󠄐󠅤󠅘󠅕󠄐󠅠󠅢󠅟󠅗󠅢󠅕󠅣󠅣󠄐󠅝󠅑󠅔󠅕󠄐󠅙󠅞󠄐󠅓󠅟󠅜󠅟󠅢󠄝󠅠󠅘󠅟󠅤󠅟󠅗󠅢󠅑󠅠󠅘󠅩󠄞
Here is a hint: 
```python
def variance_selector_to_byte(variation_selector):
    variation_selector_codepoint = ord(variation_selector)
    if 0xFE00 <= variation_selector_codepoint <= 0xFE0F:
        return variation_selector_codepoint - 0xFE00
    elif 0xE0100 <= variation_selector_codepoint <= 0xE01EF:
        return variation_selector_codepoint - 0xE0100 + 16
    else:
        return None
def decode(encoded_sentence):
    decoded_bytes = []
    variation_selectors_part = encoded_sentence[1:]
    for char in variation_selectors_part:
        byte_val = variance_selector_to_byte(char)
        if byte_val is not None:
            decoded_bytes.append(byte_val)
    return bytes(decoded_bytes).decode('utf-8')
```

Return the secret sentence as your final answer.
Answer: In fact, one of them wanted to revive painting, an art fallen into desuetude owing to the progress made in color-photography.
Metadata: {'emoji': '😱'}

````

### family_relationships
Generates family relationship reasoning tasks

Default configuration:
```python
min_family_size = 4
max_family_size = 8
male_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles', 'Peter', 'Daniel', 'Matthew', 'Christopher', 'Andrew', 'George', 'Edward', 'Benjamin', 'Henry', 'Samuel', 'Alexander', 'Oliver', 'Jack', 'Harry', 'Jacob', 'Noah', 'Ethan', 'Lucas', 'Mason', 'Logan', 'Sebastian', 'Theodore', 'Owen', 'Liam', 'Aiden', 'Kai', 'Jayden', 'Zion', 'Phoenix', 'Atlas', 'Axel', 'Ryder', 'Finn']
female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Emma', 'Lisa', 'Anna', 'Margaret', 'Victoria', 'Charlotte', 'Sophia', 'Isabella', 'Olivia', 'Ava', 'Mia', 'Emily', 'Abigail', 'Amelia', 'Eleanor', 'Grace', 'Alice', 'Lucy', 'Chloe', 'Sophie', 'Lily', 'Hannah', 'Zoe', 'Luna', 'Nova', 'Aria', 'Willow', 'Aurora', 'Sage', 'River', 'Winter', 'Sky', 'Rain']
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: John is married to Isabella. They have a child called Edward. Edward is married to Victoria.

What is Isabella to Edward? Respond only with the word that describes their relationship.
Answer: mother
Metadata: {'person1': 'Isabella', 'person2': 'Edward', 'relationship': 'mother', 'family_size': 4}

Example 2:
Question: Henry is married to Karen. They have a child called Sebastian. Sebastian is married to Eleanor.

What relation is Henry to Karen? Answer with a single word.
Answer: husband
Metadata: {'person1': 'Henry', 'person2': 'Karen', 'relationship': 'husband', 'family_size': 4}

Example 3:
Question: Liam is married to Nova. They have a child called Noah. Noah is married to Charlotte. They have a child called Patricia. Joseph is married to Lisa. They have a child called Charlotte.

What is Liam to Noah? Respond only with the word that describes their relationship.
Answer: father
Metadata: {'person1': 'Liam', 'person2': 'Noah', 'relationship': 'father', 'family_size': 7}

````

### figlet_font
Generates FigletFont tasks

Default configuration:
```python
static_word = None
static_font = None
space_letters = True
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Please read the following figlet font:

  sSSSs        d s  b        sss.      d sss        sss sssss 
 S     S       S  S S      d           S                S     
S       S      S   SS      Y           S                S     
S       S      S    S        ss.       S sSSs           S     
S       S      S    S           b      S                S     
 S     S       S    S           P      S                S     
  "sss"        P    P      ` ss'       P sSSss          P     
                                                              

Answer: ONSET
Metadata: {'font': 'amc_tubes', 'space_letters': True}

Example 2:
Question: What word does this say?

######   ######   ######     ####   ##    ## 
 ##  ##   ##  ##   ##  ##   ##  ##   ##  ##  
 ##  ##   ##       ##  ##  ##   ##   ##  ##  
 #####    ####     #####   ##        ######  
 ##       ##       ## ##   ##   ##   ##  ##  
 ##       ##  ##   ## ##    ##  ##   ##  ##  
####     ######   ### ###    ####   ##    ## 
                                             

Answer: PERCH
Metadata: {'font': 'demo_2__', 'space_letters': True}

Example 3:
Question: What word does this say?

                                              
                                              
                                              
### ###   ### ###   #####    ######   #####   
 ## ##     ##  #     ## ##    ##  #    ## ##  
 ## ##     ### #     ## ##    ####     ## ##  
 ## ##     #####     ## ##    ##       ####   
 ## ##     ## ##     ## ##    ## ##    ## ##  
  ###     ### ##    #####    ######   #### ## 
                                              
                                              

Answer: UNDER
Metadata: {'font': 'xcourb', 'space_letters': True}

````

### fraction_simplification
Generates fraction simplification tasks

Default configuration:
```python
min_value = 1
max_value = 1000
min_factor = 1
max_factor = 100
styles = ('plain', 'latex_inline', 'latex_frac', 'latex_dfrac')
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Simplify the fraction $\frac{92}{524}$ to its lowest terms. Give only the simplified fraction as your final answer.
Answer: $\frac{23}{131}$
Metadata: {'numerator': 92, 'denominator': 524, 'simplified_numerator': 23, 'simplified_denominator': 131, 'reduction_factor': 4, 'style': 'latex_frac'}

Example 2:
Question: Simplify the fraction $3600/26370$ to its lowest terms. Give only the simplified fraction as your final answer.
Answer: $40/293$
Metadata: {'numerator': 3600, 'denominator': 26370, 'simplified_numerator': 40, 'simplified_denominator': 293, 'reduction_factor': 90, 'style': 'latex_inline'}

Example 3:
Question: Simplify the fraction 29330/37310 to its lowest terms. Give only the simplified fraction as your final answer.
Answer: 419/533
Metadata: {'numerator': 29330, 'denominator': 37310, 'simplified_numerator': 419, 'simplified_denominator': 533, 'reduction_factor': 70, 'style': 'plain'}

````

### futoshiki
Generates Futoshiki puzzles with configurable board size and difficulty

Default configuration:
```python
board_size = 4
difficulty = 1
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve the following 4x4 Futoshiki puzzle:

_ > _   _   _
             
4   _   _   _
             
_   1   3   _
             
1 < _   _   _

Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.
Use < and > for horizontal constraints. Use ∧ and ∨ for vertical constraints.
Remember, in Futoshiki each row and column must contain each number from 1 to 4 exactly once.
Answer: 3 > 2   4   1
             
4   3   1   2
             
2   1   3   4
             
1 < 4   2   3
Metadata: {'puzzle': [[0, 0, 0, 0], [4, 0, 0, 0], [0, 1, 3, 0], [1, 0, 0, 0]], 'constraints': {((0, 0), (0, 1)): '>', ((3, 0), (3, 1)): '<'}, 'solution': [[3, 2, 4, 1], [4, 3, 1, 2], [2, 1, 3, 4], [1, 4, 2, 3]], 'board_size': 4, 'difficulty': 1}

Example 2:
Question: Solve the following 4x4 Futoshiki puzzle:

_   _   _   _
            ∧
_   _   _   _
             
_   _   3   4
    ∧       ∨
_   2   _ < _

Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.
Use < and > for horizontal constraints. Use ∧ and ∨ for vertical constraints.
Remember, in Futoshiki each row and column must contain each number from 1 to 4 exactly once.
Answer: 3   4   2   1
            ∧
1   3   4   2
             
2   1   3   4
    ∧       ∨
4   2   1 < 3
Metadata: {'puzzle': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3, 4], [0, 2, 0, 0]], 'constraints': {((0, 3), (1, 3)): '<', ((2, 1), (3, 1)): '<', ((2, 3), (3, 3)): '>', ((3, 2), (3, 3)): '<'}, 'solution': [[3, 4, 2, 1], [1, 3, 4, 2], [2, 1, 3, 4], [4, 2, 1, 3]], 'board_size': 4, 'difficulty': 1}

Example 3:
Question: Solve the following 4x4 Futoshiki puzzle:

_   _   _   _
             
_   4   _   2
             
_   _   _   _
            ∧
1   _   4   _

Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.
Use < and > for horizontal constraints. Use ∧ and ∨ for vertical constraints.
Remember, in Futoshiki each row and column must contain each number from 1 to 4 exactly once.
Answer: 2   1   3   4
             
3   4   1   2
             
4   3   2   1
            ∧
1   2   4   3
Metadata: {'puzzle': [[0, 0, 0, 0], [0, 4, 0, 2], [0, 0, 0, 0], [1, 0, 4, 0]], 'constraints': {((2, 3), (3, 3)): '<'}, 'solution': [[2, 1, 3, 4], [3, 4, 1, 2], [4, 3, 2, 1], [1, 2, 4, 3]], 'board_size': 4, 'difficulty': 1}

````

### game_of_life
Generates Game of Life games with configurable parameters

Default configuration:
```python
grid_size_x = 10
grid_size_y = 10
filled_cells = 100
simulation_steps = 1
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: What will this Game of Life board look like after 1 steps of simulation? Reply as array of arrays representing rows in the grid from top to bottom in JSON format. (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])

[[0,1,0,1,1,0,0,0,1,0],
 [1,0,0,1,0,1,1,1,1,1],
 [0,0,1,1,1,1,1,1,1,0],
 [1,1,1,1,0,0,0,0,1,1],
 [1,1,1,1,0,0,1,0,1,1],
 [1,1,0,1,1,0,1,1,0,1],
 [1,0,0,1,1,0,0,0,0,1],
 [1,1,1,0,0,1,1,0,1,1],
 [1,1,1,1,1,0,0,1,0,1],
 [0,1,1,1,0,1,1,0,1,0]].
Answer: [[0,1,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,0,0,0],[0,0,0,0,0,0,1,1,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,0,1,0]]
Metadata: {'grid_size_x': 10, 'grid_size_y': 10, 'filled_cells': 100, 'simulation_steps': 1}

Example 2:
Question: What will this Game of Life board look like after 1 steps of simulation? Reply as array of arrays representing rows in the grid from top to bottom in JSON format. (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])

[[1,1,1,1,1,1,0,1,1,1],
 [0,0,1,1,1,1,1,1,1,1],
 [0,1,0,0,0,0,0,1,1,1],
 [1,0,0,1,1,1,1,0,0,1],
 [0,1,0,1,1,0,0,1,1,0],
 [1,1,1,1,0,1,1,0,1,1],
 [0,1,1,0,1,1,1,0,0,1],
 [0,0,1,0,1,1,0,0,1,1],
 [0,1,1,0,1,0,1,0,1,1],
 [1,1,1,0,1,1,1,0,0,1]].
Answer: [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,1,0,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
Metadata: {'grid_size_x': 10, 'grid_size_y': 10, 'filled_cells': 100, 'simulation_steps': 1}

Example 3:
Question: What will this Game of Life board look like after 1 steps of simulation? Reply as array of arrays representing rows in the grid from top to bottom in JSON format. (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])

[[0,1,0,1,1,1,1,0,0,1],
 [0,1,0,0,1,1,1,0,1,1],
 [0,1,1,1,1,0,1,0,1,0],
 [1,0,0,1,1,0,1,1,1,1],
 [1,1,1,0,0,1,1,0,1,1],
 [0,1,0,0,1,1,0,1,0,1],
 [0,1,1,0,0,0,1,0,1,1],
 [0,1,1,0,1,1,1,1,0,1],
 [1,1,1,1,1,1,0,1,1,0],
 [1,1,1,0,0,1,1,0,1,0]].
Answer: [[0,0,0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,1,1],[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0]]
Metadata: {'grid_size_x': 10, 'grid_size_y': 10, 'filled_cells': 100, 'simulation_steps': 1}

````

### gcd
Generates Greatest Common Divisor (GCD) tasks

Default configuration:
```python
min_numbers = 2
max_numbers = 2
min_value = 1
max_value = 1000
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 26, 760. Give only the GCD as your final answer.
Answer: 2
Metadata: {'numbers': [26, 760], 'result': 2}

Example 2:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 688, 716. Give only the GCD as your final answer.
Answer: 4
Metadata: {'numbers': [688, 716], 'result': 4}

Example 3:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 297, 30. Give only the GCD as your final answer.
Answer: 3
Metadata: {'numbers': [297, 30], 'result': 3}

````

### graph_color
Generates graph coloring problems with configurable parameters

Default configuration:
```python
num_colors = 4
num_vertices = 10
edge_probability = 0.4
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Please provide a coloring for this graph such that every vertex is not connected to a vertex of the same color. The graph has these properties:

Vertices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Edges: [(0, 2), (0, 3), (0, 4), (0, 8), (1, 2), (1, 3), (1, 5), (1, 6), (1, 9), (2, 5), (2, 8), (2, 9), (3, 5), (3, 6), (3, 7), (4, 9), (6, 9), (7, 8), (7, 9), (8, 9)]
Possible colors: [1, 2, 3, 4]

Return your solution as a JSON map of vertices to colors. (For example: {0: 1, 1: 2, 2: 3})

Answer: None
Metadata: {'possible_answer': {0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 1, 8: 3, 9: 4}, 'puzzle': {'vertices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'edges': [(0, 2), (0, 3), (0, 4), (0, 8), (1, 2), (1, 3), (1, 5), (1, 6), (1, 9), (2, 5), (2, 8), (2, 9), (3, 5), (3, 6), (3, 7), (4, 9), (6, 9), (7, 8), (7, 9), (8, 9)], 'num_colors': 4, 'color_options': [1, 2, 3, 4]}}

Example 2:
Question: Please provide a coloring for this graph such that every vertex is not connected to a vertex of the same color. The graph has these properties:

Vertices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Edges: [(0, 1), (0, 3), (0, 9), (1, 3), (1, 8), (2, 4), (2, 5), (3, 6), (3, 7), (3, 8), (4, 6), (4, 9), (6, 7), (7, 9)]
Possible colors: [1, 2, 3, 4]

Return your solution as a JSON map of vertices to colors. (For example: {0: 1, 1: 2, 2: 3})

Answer: None
Metadata: {'possible_answer': {0: 1, 1: 2, 2: 1, 3: 3, 4: 2, 5: 2, 6: 1, 7: 2, 8: 1, 9: 3}, 'puzzle': {'vertices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'edges': [(0, 1), (0, 3), (0, 9), (1, 3), (1, 8), (2, 4), (2, 5), (3, 6), (3, 7), (3, 8), (4, 6), (4, 9), (6, 7), (7, 9)], 'num_colors': 4, 'color_options': [1, 2, 3, 4]}}

Example 3:
Question: Please provide a coloring for this graph such that every vertex is not connected to a vertex of the same color. The graph has these properties:

Vertices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Edges: [(0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 5), (1, 8), (1, 9), (2, 5), (2, 6), (2, 7), (2, 9), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (6, 9)]
Possible colors: [1, 2, 3, 4]

Return your solution as a JSON map of vertices to colors. (For example: {0: 1, 1: 2, 2: 3})

Answer: None
Metadata: {'possible_answer': {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 2}, 'puzzle': {'vertices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'edges': [(0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 5), (1, 8), (1, 9), (2, 5), (2, 6), (2, 7), (2, 9), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (6, 9)], 'num_colors': 4, 'color_options': [1, 2, 3, 4]}}

````

### group_anagrams
Generates Group Anagrams exercises with configurable difficulty

Default configuration:
```python
anagram_groups = 10
max_words_per_group = 5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: An anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

Your job is to group the anagrams together. You can return the answer in any order.

Example:
Input: ["eat", "tea", "tan", "ate", "nat", "bat"]
Output: [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
Explanation:
    - There is no string in the input that can be rearranged to form "bat".
    - The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.

Group the following list of words into anagrams:
["tinglers", "argonon", "ditas", "palinodist", "merocyte", "conterminal", "canny", "nancy", "outasight", "autosight", "oversauciness", "applauders", "suprapedal"]

Answer: [["applauders", "suprapedal"], ["argonon"], ["autosight", "outasight"], ["canny", "nancy"], ["conterminal"], ["ditas"], ["merocyte"], ["oversauciness"], ["palinodist"], ["tinglers"]]
Metadata: {'words': ['tinglers', 'argonon', 'ditas', 'palinodist', 'merocyte', 'conterminal', 'canny', 'nancy', 'outasight', 'autosight', 'oversauciness', 'applauders', 'suprapedal'], 'solution': [['applauders', 'suprapedal'], ['argonon'], ['autosight', 'outasight'], ['canny', 'nancy'], ['conterminal'], ['ditas'], ['merocyte'], ['oversauciness'], ['palinodist'], ['tinglers']]}

Example 2:
Question: An anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

Your job is to group the anagrams together. You can return the answer in any order.

Example:
Input: ["eat", "tea", "tan", "ate", "nat", "bat"]
Output: [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
Explanation:
    - There is no string in the input that can be rearranged to form "bat".
    - The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.

Group the following list of words into anagrams:
["regear", "escrod", "coders", "decors", "credos", "scored", "semitaur", "muriates", "peripterous", "zanies", "expatiater", "wooled", "meningomyelocele", "myelomeningocele", "vainest", "natives", "naivest", "preludes", "repulsed"]

Answer: [["coders", "credos", "decors", "escrod", "scored"], ["expatiater"], ["meningomyelocele", "myelomeningocele"], ["muriates", "semitaur"], ["naivest", "natives", "vainest"], ["peripterous"], ["preludes", "repulsed"], ["regear"], ["wooled"], ["zanies"]]
Metadata: {'words': ['regear', 'escrod', 'coders', 'decors', 'credos', 'scored', 'semitaur', 'muriates', 'peripterous', 'zanies', 'expatiater', 'wooled', 'meningomyelocele', 'myelomeningocele', 'vainest', 'natives', 'naivest', 'preludes', 'repulsed'], 'solution': [['coders', 'credos', 'decors', 'escrod', 'scored'], ['expatiater'], ['meningomyelocele', 'myelomeningocele'], ['muriates', 'semitaur'], ['naivest', 'natives', 'vainest'], ['peripterous'], ['preludes', 'repulsed'], ['regear'], ['wooled'], ['zanies']]}

Example 3:
Question: An anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

Your job is to group the anagrams together. You can return the answer in any order.

Example:
Input: ["eat", "tea", "tan", "ate", "nat", "bat"]
Output: [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
Explanation:
    - There is no string in the input that can be rearranged to form "bat".
    - The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.

Group the following list of words into anagrams:
["eagerest", "granitite", "helium", "nizam", "nazim", "striplings", "slipstring", "rearrest", "arrester", "bf", "tadpolism", "canun", "cunan", "isotonic"]

Answer: [["arrester", "rearrest"], ["bf"], ["canun", "cunan"], ["eagerest"], ["granitite"], ["helium"], ["isotonic"], ["nazim", "nizam"], ["slipstring", "striplings"], ["tadpolism"]]
Metadata: {'words': ['eagerest', 'granitite', 'helium', 'nizam', 'nazim', 'striplings', 'slipstring', 'rearrest', 'arrester', 'bf', 'tadpolism', 'canun', 'cunan', 'isotonic'], 'solution': [['arrester', 'rearrest'], ['bf'], ['canun', 'cunan'], ['eagerest'], ['granitite'], ['helium'], ['isotonic'], ['nazim', 'nizam'], ['slipstring', 'striplings'], ['tadpolism']]}

````

### gsm_symbolic
Default configuration:
```python
difficulty = 1.0
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: There are 12 students playing basketball and twice that number playing volleyball. There are 17 boys and 17 girls playing table tennis. If each student only participates in one group, how many students are there in total? Give only the result as your final answer.
Answer: 70
Metadata: {'difficulty': 1.0, 'answer_value': 70, 'answer_cot': 'There are 12 x 2 = 24 students playing volleyball.\nThere are 17 + 17 = 34 students playing table tennis.\nIn total there are 12 + 24 + 34 = 70 students.\n#### 70', 'variables': {'tennis_players': 12, 'volleyball_players': 24, 'soccer_boys': 17, 'soccer_girls': 17, 'total_soccer': 34, 'total_students': 70, 'sports': ['basketball', 'volleyball', 'table tennis']}}

Example 2:
Question: In Ms. Johnson's class of 100 students, 80% of the class are volleyball players. Out of the remaining class, 65% of the students are choir members or part of robotics club members. These 3 groups of students will need to leave early today to travel to an away performance. How many students are leaving early? Give only the result as your final answer.
Answer: 93
Metadata: {'difficulty': 1.0, 'answer_value': 93, 'answer_cot': "80% of the 100 student class are volleyball players so that's 0.8*100 = 80 students\nThere are 100 students and 80 are volleyball players so that leaves 100-80 = 20 students\n65% of the remaining 20 students are part of robotics club members or choir members so that's 0.65*20 = 13 students\n80 students are volleyball players and 13 are part of robotics club members/choir members so 80+13 = 93 students will be leaving early\n#### 93", 'variables': {'teacher': 'Ms. Johnson', 'total_students': 100, 'percent_group1': 80, 'percent_group23': 65, 'group1': 'volleyball players', 'group2': 'choir members', 'group3': 'robotics club members', 'event': 'performance', 'group1_count': 80, 'group23_count': 13}}

Example 3:
Question: Olivia is trying to decide whether to do her business accounting herself or hire an accountant. If she does it herself, she'll be able to do 7 fewer hours of consulting work, losing €57/hour in missed income. The accountant charges €57. How much more money will she have if she hires the accountant? Give only the result as your final answer.
Answer: 342
Metadata: {'difficulty': 1.0, 'answer_value': 342, 'answer_cot': "First find the total lost revenue if Olivia does her business accounting herself: €57/hour * 7 hours = €399\nThen subtract the accountant's charge to find how much money Olivia saves: €399 - €57 = €342\n#### 342", 'variables': {'name': 'Olivia', 'task': 'her business accounting', 'profession': 'accountant', 'hours': 7, 'work_type': 'consulting', 'hourly_rate': 57, 'fee': 57, 'currency': '€', 'lost_income': 399}}

````

### intermediate_integration
Generates intermediate integration problem - either
    by substitution or by parts

Default configuration:
```python
problem_types = ('substitution', 'by_parts')
substitution_types = ('linear', 'trigonometric', 'exponential', 'radical')
by_parts_types = ('polynomial_exp_trig', 'log_inverse_trig', 'cyclic', 'repeated_parts')
seed = 42
size = 500
linear_lower_bound = 1
linear_upper_bound = 10
min_linear_degree = 2
max_linear_degree = 4
outer_constant_min = 1
outer_constant_max = 3
min_poly_degree = 1
max_poly_degree = 3
symbols = ('x', 'X')
operators = ('+', '-')
```

Example tasks:
````
Example 1:
Question: Find the indefinite integral: ∫ -3*exp(3*x + 9) dx
In addition, when doing calculation, use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps. For example Use [-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C] instead of [-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C].

Answer: -exp(3*x + 9) + C
Metadata: {'integrand': '-3*exp(3*x + 9)', 'problem_type': 'substitution', 'variable': 'x', 'type': 'exponential', 'expected_answer_expression': -exp(3*x + 9)}

Example 2:
Question: Evaluate the indefinite integral: ∫ -6*sin(2*X + 10)*cos(2*X + 10)**4 dx
In addition, when doing calculation, use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps. For example Use [-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C] instead of [-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C].

Answer: 3*cos(2*X + 10)**5/5 + C
Metadata: {'integrand': '-6*sin(2*X + 10)*cos(2*X + 10)**4', 'problem_type': 'substitution', 'variable': 'X', 'type': 'trigonometric', 'expected_answer_expression': 3*cos(2*X + 10)**5/5}

Example 3:
Question: Find the indefinite integral: ∫ 2*asin(x) dx
In addition, when doing calculation, use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps. For example Use [-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C] instead of [-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C].

Answer: 2*Integral(asin(x), x) + C
Metadata: {'integrand': '2*asin(x)', 'problem_type': 'by_parts', 'variable': 'x', 'type': 'log_inverse_trig', 'expected_answer_expression': 2*Integral(asin(x), x)}

````

### isomorphic_strings
Generates Isomorphic Strings exercises with configurable difficulty

Default configuration:
```python
max_string_length = 10
p_solvable = 0.5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Two strings are isomorphic if the characters in one string can be replaced to get the second string.

All occurrences of a character must be replaced with another character while preserving the order of characters.

No two characters may map to the same character, but a character may map to itself.

Example 1:
Input: egg add
Output: True
Explanation: The strings s and t can be made identical by:
    - Mapping 'e' to 'a'.
    - Mapping 'g' to 'd'.

Example 2:
Input: foo bar
Output: False
Explanation:
    - The strings cannot be made identical as 'o' needs to be mapped to both 'a' and 'r'.

Return True if the following two strings are isomorphic, or False otherwise:
cc bw

Answer: False
Metadata: {'words': ['cc', 'bw'], 'solution': False, 'solvable': False}

Example 2:
Question: Two strings are isomorphic if the characters in one string can be replaced to get the second string.

All occurrences of a character must be replaced with another character while preserving the order of characters.

No two characters may map to the same character, but a character may map to itself.

Example 1:
Input: egg add
Output: True
Explanation: The strings s and t can be made identical by:
    - Mapping 'e' to 'a'.
    - Mapping 'g' to 'd'.

Example 2:
Input: foo bar
Output: False
Explanation:
    - The strings cannot be made identical as 'o' needs to be mapped to both 'a' and 'r'.

Return True if the following two strings are isomorphic, or False otherwise:
nai oik

Answer: True
Metadata: {'words': ['nai', 'oik'], 'solution': True, 'solvable': True}

Example 3:
Question: Two strings are isomorphic if the characters in one string can be replaced to get the second string.

All occurrences of a character must be replaced with another character while preserving the order of characters.

No two characters may map to the same character, but a character may map to itself.

Example 1:
Input: egg add
Output: True
Explanation: The strings s and t can be made identical by:
    - Mapping 'e' to 'a'.
    - Mapping 'g' to 'd'.

Example 2:
Input: foo bar
Output: False
Explanation:
    - The strings cannot be made identical as 'o' needs to be mapped to both 'a' and 'r'.

Return True if the following two strings are isomorphic, or False otherwise:
hogtytyof kgqwfwfgh

Answer: True
Metadata: {'words': ['hogtytyof', 'kgqwfwfgh'], 'solution': True, 'solvable': True}

````

### jugs
Generates water jug puzzles inspired by [this scene from _Die Hard 3_](https://www.youtube.com/watch?v=6cAbgAaEOVE), with configurable parameters

Default configuration:
```python
num_jugs = 3
difficulty = 10
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: You are a police officer. A maniac has planted a bomb next to a public fountain.

To defuse the bomb, you must solve a puzzle. The puzzle is solved when you fill any of the available jugs with the target amount of water.

You have three move types: 'fill', 'empty' and 'pour'.

To fill Jug A, you 'fill A'.
To empty Jug B, you 'empty B'.
To pour the contents of Jug A into Jug B, you 'pour A->B'.
All jugs are empty to begin with.

The empty jugs hold this many litres of water: A:13, B:13, C:4
And your target is: 10 litres.

How do you defuse the bomb?

Reply as a JSON-parsable list of moves which result in any of the jugs being filled with the target amount.

Answer: ["fill A", "pour A->C", "fill B", "empty C", "pour A->C", "empty C", "pour A->C", "empty C", "pour A->C", "pour B->C"]
Metadata: {'puzzle': {'jug_capacities': [13, 13, 4], 'target': 10, 'min_moves': 10}}

Example 2:
Question: You are a police officer. A maniac has planted a bomb next to a public fountain.

To defuse the bomb, you must solve a puzzle. The puzzle is solved when you fill any of the available jugs with the target amount of water.

You have three move types: 'fill', 'empty' and 'pour'.

To fill Jug A, you 'fill A'.
To empty Jug B, you 'empty B'.
To pour the contents of Jug A into Jug B, you 'pour A->B'.
All jugs are empty to begin with.

The empty jugs hold this many litres of water: A:7, B:10, C:10
And your target is: 5 litres.

How do you defuse the bomb?

Reply as a JSON-parsable list of moves which result in any of the jugs being filled with the target amount.

Answer: ["fill A", "pour A->B", "fill A", "pour A->B", "pour A->C", "fill A", "pour A->C", "empty B", "pour A->B", "fill A", "pour A->B", "fill A", "pour A->B"]
Metadata: {'puzzle': {'jug_capacities': [7, 10, 10], 'target': 5, 'min_moves': 13}}

Example 3:
Question: You are a police officer. A maniac has planted a bomb next to a public fountain.

To defuse the bomb, you must solve a puzzle. The puzzle is solved when you fill any of the available jugs with the target amount of water.

You have three move types: 'fill', 'empty' and 'pour'.

To fill Jug A, you 'fill A'.
To empty Jug B, you 'empty B'.
To pour the contents of Jug A into Jug B, you 'pour A->B'.
All jugs are empty to begin with.

The empty jugs hold this many litres of water: A:7, B:10, C:7
And your target is: 2 litres.

How do you defuse the bomb?

Reply as a JSON-parsable list of moves which result in any of the jugs being filled with the target amount.

Answer: ["fill B", "pour B->A", "empty A", "pour B->A", "fill B", "pour B->A", "empty A", "pour B->A", "fill B", "pour B->A", "pour B->C"]
Metadata: {'puzzle': {'jug_capacities': [7, 10, 7], 'target': 2, 'min_moves': 11}}

````

### knight_swap
Generates Knight Swap puzzles with configurable parameters.

Default configuration:
```python
min_nodes = 6
max_nodes = 9
min_pieces = 2
max_pieces = 2
min_steps = 4
max_steps = 20
max_attempts = 100
seed = 42
size = 5
impossible_ratio = 0.2
```

Example tasks:
````
Example 1:
Question: Knight Swap Challenge:

```
    A   B   C   D
   ----------------
3 |   | . |   | . |
   ----------------
2 | B | w |   |   |
   ----------------
1 |   |   | B | w |
   ----------------
```

Legend:
- 'w' = White Knight
- 'B' = Black Knight
- Empty squares are marked with '.'

Objective:
Swap the positions of all white knights with all black knights through valid moves.

Rules:
1. Knights move in L-shape (2 squares + 1 square perpendicular)
2. Knights can only move to empty squares
3. w moves first, then players alternate
4. All knights must reach their target positions (white ↔ black)

Question:
Is it possible to swap all knights' positions? If yes, list the moves.

Answer Format:
- For impossible puzzles: "No"
- For possible puzzles: List moves as ["color,from,to", ...]
  Example: ["w,A1,B3"] means white knight moves A1→B3

Answer: No
Metadata: {'board': {'C1': ['A2', 'B3', 'D3'], 'A2': ['C1'], 'B3': ['C1'], 'D1': ['B2'], 'B2': ['D1', 'D3'], 'D3': ['B2', 'C1']}, 'pieces': {'C1': 'B', 'A2': 'B', 'B3': None, 'D1': 'w', 'B2': 'w', 'D3': None}, 'start_turn': 'w', 'solution': None, 'is_possible': False, 'num_steps': 0, 'board_states': None}

Example 2:
Question: Knight Swap Challenge:

```
    A   B   C   D
   ----------------
3 |   | w | . |   |
   ----------------
2 | w |   |   | B |
   ----------------
1 |   |   | . | B |
   ----------------
```

Legend:
- 'w' = White Knight
- 'B' = Black Knight
- Empty squares are marked with '.'

Objective:
Swap the positions of all white knights with all black knights through valid moves.

Rules:
1. Knights move in L-shape (2 squares + 1 square perpendicular)
2. Knights can only move to empty squares
3. w moves first, then players alternate
4. All knights must reach their target positions (white ↔ black)

Question:
Is it possible to swap all knights' positions? If yes, list the moves.

Answer Format:
- For impossible puzzles: "No"
- For possible puzzles: List moves as ["color,from,to", ...]
  Example: ["w,A1,B3"] means white knight moves A1→B3

Answer: No
Metadata: {'board': {'B3': ['C1'], 'D1': ['C3'], 'C3': ['A2', 'D1'], 'C1': ['A2', 'B3'], 'D2': [], 'A2': ['C1', 'C3']}, 'pieces': {'B3': 'w', 'D1': 'B', 'C3': None, 'C1': None, 'D2': 'B', 'A2': 'w'}, 'start_turn': 'w', 'solution': None, 'is_possible': False, 'num_steps': 0, 'board_states': None}

Example 3:
Question: Knight Swap Challenge:

```
    A   B   C
   ------------
3 | . |   | B |
   ------------
2 | w |   | . |
   ------------
1 |   | w | B |
   ------------
```

Legend:
- 'w' = White Knight
- 'B' = Black Knight
- Empty squares are marked with '.'

Objective:
Swap the positions of all white knights with all black knights through valid moves.

Rules:
1. Knights move in L-shape (2 squares + 1 square perpendicular)
2. Knights can only move to empty squares
3. w moves first, then players alternate
4. All knights must reach their target positions (white ↔ black)

Question:
Is it possible to swap all knights' positions? If yes, list the moves.

Answer Format:
- For impossible puzzles: "No"
- For possible puzzles: List moves as ["color,from,to", ...]
  Example: ["w,A1,B3"] means white knight moves A1→B3

Answer: No
Metadata: {'board': {'B1': ['A3'], 'A3': ['B1', 'C2'], 'A2': ['C1', 'C3'], 'C3': ['A2'], 'C1': ['A2'], 'C2': ['A3']}, 'pieces': {'B1': 'w', 'A3': None, 'A2': 'w', 'C3': 'B', 'C1': 'B', 'C2': None}, 'start_turn': 'w', 'solution': None, 'is_possible': False, 'num_steps': 0, 'board_states': None}

````

### largest_island
Generates Largest Island exercises with configurable difficulty

Default configuration:
```python
rows = 10
cols = 10
max_num_islands = 5
max_island_size = 10
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: You are given the following 10 x 10 binary matrix grid:
0 0 0 1 0 0 0 0 0 0
1 1 0 1 0 0 0 0 0 1
0 1 0 1 1 0 0 0 0 1
0 1 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 0 0 0 1 0
1 1 0 1 1 0 0 0 1 1
1 1 1 1 1 0 0 0 0 0

An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical).
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

Answer: 10
Metadata: {'grid': [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0, 0, 1], [0, 1, 0, 1, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 1, 1, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], 'solution': 10}

Example 2:
Question: You are given the following 10 x 10 binary matrix grid:
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0

An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical).
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

Answer: 0
Metadata: {'grid': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'solution': 0}

Example 3:
Question: You are given the following 10 x 10 binary matrix grid:
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 0 0 0 0 0

An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical).
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

Answer: 3
Metadata: {'grid': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'solution': 3}

````

### lcm
Generates Least Common Multiple (LCM) tasks

Default configuration:
```python
min_numbers = 2
max_numbers = 2
min_value = 1
max_value = 100
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the Least Common Multiple (LCM) of these numbers: 95, 14
Answer: 1330
Metadata: {'numbers': [95, 14], 'result': 1330}

Example 2:
Question: Find the Least Common Multiple (LCM) of these numbers: 60, 48
Answer: 240
Metadata: {'numbers': [60, 48], 'result': 240}

Example 3:
Question: Find the Least Common Multiple (LCM) of these numbers: 38, 4
Answer: 76
Metadata: {'numbers': [38, 4], 'result': 76}

````

### leg_counting
Generates leg counting arithmetic tasks

Default configuration:
```python
min_animals = 3
max_animals = 10
max_instances = 15
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to count how many legs there are in total when given a list of animals.

Example:
- Input: How many legs are there in total if you have 1 duck, 2 deers, 1 spider, 3 cows?
- Output: 30
- Explanation:
    - Ducks have 2 legs each, so 1 duck has 2 legs.
    - Deers have 4 legs each, so 2 deers have 8 legs.
    - Spiders have 8 legs each, so 1 spider has 8 legs.
    - Cows have 4 legs each, so 3 cows have 12 legs.
    - Therefore, the total number of legs is 2 + 8 + 8 + 12 = 30

Now, how many legs are there in total if you have 3 sea slugs, 12 deers, 2 giraffes, 11 elephants?

Answer: 100
Metadata: {'difficulty': {'num_animals': 4}, 'animals': {'sea slug': 3, 'deer': 12, 'giraffe': 2, 'elephant': 11}, 'total_legs': 100}

Example 2:
Question: Your task is to count how many legs there are in total when given a list of animals.

Example:
- Input: How many legs are there in total if you have 1 duck, 2 deers, 1 spider, 3 cows?
- Output: 30
- Explanation:
    - Ducks have 2 legs each, so 1 duck has 2 legs.
    - Deers have 4 legs each, so 2 deers have 8 legs.
    - Spiders have 8 legs each, so 1 spider has 8 legs.
    - Cows have 4 legs each, so 3 cows have 12 legs.
    - Therefore, the total number of legs is 2 + 8 + 8 + 12 = 30

Now, how many legs are there in total if you have 6 sheeps, 11 dogs, 12 praying mantiss?

Answer: 140
Metadata: {'difficulty': {'num_animals': 3}, 'animals': {'sheep': 6, 'dog': 11, 'praying mantis': 12}, 'total_legs': 140}

Example 3:
Question: Your task is to count how many legs there are in total when given a list of animals.

Example:
- Input: How many legs are there in total if you have 1 duck, 2 deers, 1 spider, 3 cows?
- Output: 30
- Explanation:
    - Ducks have 2 legs each, so 1 duck has 2 legs.
    - Deers have 4 legs each, so 2 deers have 8 legs.
    - Spiders have 8 legs each, so 1 spider has 8 legs.
    - Cows have 4 legs each, so 3 cows have 12 legs.
    - Therefore, the total number of legs is 2 + 8 + 8 + 12 = 30

Now, how many legs are there in total if you have 2 crabs, 10 lobsters, 1 human, 2 cows, 3 bees, 13 elephants, 9 dogs, 12 snakes, 5 shrimps?

Answer: 286
Metadata: {'difficulty': {'num_animals': 9}, 'animals': {'crab': 2, 'lobster': 10, 'human': 1, 'cow': 2, 'bee': 3, 'elephant': 13, 'dog': 9, 'snake': 12, 'shrimp': 5}, 'total_legs': 286}

````

### letter_counting
Generates letter counting tasks from text spans

Default configuration:
```python
min_words = 5
max_words = 15
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: How many times does the letter "a" appear in the text: "bed and enters his mechanical dresser Two minutes later the machine deposited him all dressed"?
Answer: 6
Metadata: {'span_length': 15, 'target_letter': 'a', 'span': ['bed', 'and', 'enters', 'his', 'mechanical', 'dresser', 'Two', 'minutes', 'later', 'the', 'machine', 'deposited', 'him', 'all', 'dressed']}

Example 2:
Question: How many times does the letter "w" appear in the text: "it into a watering place"?
Answer: 1
Metadata: {'span_length': 5, 'target_letter': 'w', 'span': ['it', 'into', 'a', 'watering', 'place']}

Example 3:
Question: How many times does the letter "t" appear in the text: "readable form accessible by the widest array of equipment including outdated"?
Answer: 5
Metadata: {'span_length': 11, 'target_letter': 't', 'span': ['readable', 'form', 'accessible', 'by', 'the', 'widest', 'array', 'of', 'equipment', 'including', 'outdated']}

````

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
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to unsramble words in a sentence.

For each word in a sentence, the letter may have been randomly shuffled. Your task is to unscramble the words.

The order of the words in the sentence is preserved. Moreover, the style of the sentence is preserved (i.e. punctuation, capitalization, new lines, etc.).

Example:
- Input: Unscramble these words: raendgmeins yWh nya hilcd anc od hatt
- Output: meanderings Why any child can do that
- Explanation
    - We unscramble each of the words independently.
    - raendgmeins -> meanderings
    - yWh -> Why
    - nya -> any
    - hilcd -> child
    - anc -> can
    - od -> do
    - hatt -> that
    - The final answer is: meanderings Why any child can do that
    - Notice that the order of the words is preserved, no new words / symbols (e.g. new lines) are added.

Now, unscramble these words: ew hsall eb ebla ot puodrce

Answer: we shall be able to produce
Metadata: {'num_words': 6, 'corruption_level': 0.12000860417813355, 'scrambled_words': ['ew', 'hsall', 'eb', 'ebla', 'ot', 'puodrce'], 'original_words': ['we', 'shall', 'be', 'able', 'to', 'produce']}

Example 2:
Question: Your task is to unsramble words in a sentence.

For each word in a sentence, the letter may have been randomly shuffled. Your task is to unscramble the words.

The order of the words in the sentence is preserved. Moreover, the style of the sentence is preserved (i.e. punctuation, capitalization, new lines, etc.).

Example:
- Input: Unscramble these words: raendgmeins yWh nya hilcd anc od hatt
- Output: meanderings Why any child can do that
- Explanation
    - We unscramble each of the words independently.
    - raendgmeins -> meanderings
    - yWh -> Why
    - nya -> any
    - hilcd -> child
    - anc -> can
    - od -> do
    - hatt -> that
    - The final answer is: meanderings Why any child can do that
    - Notice that the order of the words is preserved, no new words / symbols (e.g. new lines) are added.

Now, unscramble these words: ni oiurnalmsj Well Cahs

Answer: in journalism Well Cash
Metadata: {'num_words': 4, 'corruption_level': 0.3288673442377109, 'scrambled_words': ['ni', 'oiurnalmsj', 'Well', 'Cahs'], 'original_words': ['in', 'journalism', 'Well', 'Cash']}

Example 3:
Question: Your task is to unsramble words in a sentence.

For each word in a sentence, the letter may have been randomly shuffled. Your task is to unscramble the words.

The order of the words in the sentence is preserved. Moreover, the style of the sentence is preserved (i.e. punctuation, capitalization, new lines, etc.).

Example:
- Input: Unscramble these words: raendgmeins yWh nya hilcd anc od hatt
- Output: meanderings Why any child can do that
- Explanation
    - We unscramble each of the words independently.
    - raendgmeins -> meanderings
    - yWh -> Why
    - nya -> any
    - hilcd -> child
    - anc -> can
    - od -> do
    - hatt -> that
    - The final answer is: meanderings Why any child can do that
    - Notice that the order of the words is preserved, no new words / symbols (e.g. new lines) are added.

Now, unscramble these words: dear rchAdbali keep no nSice yrstyedae atnhks ot oyu rheet si a gain fo sucrbbisesr rM

Answer: dear Archibald keep on Since yesterday thanks to you there is a gain of subscribers Mr
Metadata: {'num_words': 16, 'corruption_level': 0.516016391169858, 'scrambled_words': ['dear', 'rchAdbali', 'keep', 'no', 'nSice', 'yrstyedae', 'atnhks', 'ot', 'oyu', 'rheet', 'si', 'a', 'gain', 'fo', 'sucrbbisesr', 'rM'], 'original_words': ['dear', 'Archibald', 'keep', 'on', 'Since', 'yesterday', 'thanks', 'to', 'you', 'there', 'is', 'a', 'gain', 'of', 'subscribers', 'Mr']}

````

### list_functions
Default configuration:
```python
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: You are an expert at inductive reasoning. Generate an output corresponding to the given input.
The output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements
Examples:
Input 1: [4, 95, 36, 32]
Output 1: [4, 32, 36, 95]
Input 2: [18, 95, 14, 87, 95, 70]
Output 2: [14, 18, 70, 87, 95, 95]
Input 3: [76, 55, 5, 4]
Output 3: [4, 5, 55, 76]
Input 4: [28, 30, 65, 78]
Output 4: [28, 30, 65, 78]


Input: [72, 26, 92]
Output:

Answer: [26, 72, 92]

Example 2:
Question: You are an expert at inductive reasoning. Generate an output corresponding to the given input.
The output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements
Examples:
Input 1: [37, 90, 98]
Output 1: [37, 90, 98]
Input 2: [60, 48, 86, 90, 13]
Output 2: [60, 48, 86, 90, 13]
Input 3: [77, 64, 78, 3, 66, 56, 74, 48, 80, 71]
Output 3: [77, 64, 78, 3, 66, 56, 74, 48, 80, 71]
Input 4: [51, 23, 8, 14, 16, 49, 20, 13, 21]
Output 4: [51, 23, 8, 14, 16, 49, 20, 13, 21]


Input: [17, 99, 50, 77, 65, 35, 74, 24, 49, 9]
Output:

Answer: [17, 99, 50, 77, 65, 35, 74, 24, 49, 9]

Example 3:
Question: You are an expert at inductive reasoning. Generate an output corresponding to the given input.
The output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements
Examples:
Input 1: [4, 29, 49, 15, 90, 23, 38, 5, 67, 5, 70]
Output 1: [2]
Input 2: [37, 66, 21, 15, 44, 46, 80, 10]
Output 2: [0]
Input 3: [13, 45, 5, 5, 5, 50, 5]
Output 3: [4]
Input 4: [88, 6, 87]
Output 4: [0]


Input: [59, 5, 81, 5, 20, 5, 61, 76, 48, 70, 5, 30]
Output:

Answer: [4]

````

### manipulate_matrix
Generates Manipulate Matrix exercises with configurable difficulty

Default configuration:
```python
min_rows = 1
min_cols = 1
max_rows = 10
max_cols = 10
max_transforms = 5
p_rotate = 0.2
p_hmirror = 0.2
p_vmirror = 0.2
p_dmirror = 0.2
p_cmirror = 0.2
p_map = 0.2
p_crop = 0.2
p_remove_every_nth_row = 0.2
p_remove_every_nth_col = 0.2
p_zero_divisible = 0.2
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: For the following matrix:
4
3

Perform the following series of operations in order:
- Identity transformation, i.e. no change


Answer: 4
3
Metadata: {'matrix': [[4], [3]], 'solution': [[4], [3]], 'operations': []}

Example 2:
Question: For the following matrix:
2 7 5 1 7

Perform the following series of operations in order:
- Identity transformation, i.e. no change


Answer: 2 7 5 1 7
Metadata: {'matrix': [[2, 7, 5, 1, 7]], 'solution': [[2, 7, 5, 1, 7]], 'operations': []}

Example 3:
Question: For the following matrix:
8 1 2 6 3 4 0 3 1
9 0 1 2 8 4 6 9 6
5 5 1 5 4 9 2 1 8
1 9 1 4 5 1 4 0 5
6 1 7 7 3 3 2 4 3
0 0 6 0 5 5 7 7 9
8 2 3 7 7 5 9 0 4

Perform the following series of operations in order:
- Identity transformation, i.e. no change


Answer: 8 1 2 6 3 4 0 3 1
9 0 1 2 8 4 6 9 6
5 5 1 5 4 9 2 1 8
1 9 1 4 5 1 4 0 5
6 1 7 7 3 3 2 4 3
0 0 6 0 5 5 7 7 9
8 2 3 7 7 5 9 0 4
Metadata: {'matrix': [[8, 1, 2, 6, 3, 4, 0, 3, 1], [9, 0, 1, 2, 8, 4, 6, 9, 6], [5, 5, 1, 5, 4, 9, 2, 1, 8], [1, 9, 1, 4, 5, 1, 4, 0, 5], [6, 1, 7, 7, 3, 3, 2, 4, 3], [0, 0, 6, 0, 5, 5, 7, 7, 9], [8, 2, 3, 7, 7, 5, 9, 0, 4]], 'solution': [[8, 1, 2, 6, 3, 4, 0, 3, 1], [9, 0, 1, 2, 8, 4, 6, 9, 6], [5, 5, 1, 5, 4, 9, 2, 1, 8], [1, 9, 1, 4, 5, 1, 4, 0, 5], [6, 1, 7, 7, 3, 3, 2, 4, 3], [0, 0, 6, 0, 5, 5, 7, 7, 9], [8, 2, 3, 7, 7, 5, 9, 0, 4]], 'operations': []}

````

### maze
Generates mazes with guaranteed shortest path distance from start to goal
    within [min_dist, max_dist].

Default configuration:
```python
min_dist = 5
max_dist = 10
min_grid_size = 5
max_grid_size = 10
seed = 42
size = 50
```

Example tasks:
````
Example 1:
Question: Navigate from '3' (start) to 'z' (goal):

```
>>>>>>>>>
>eeee>e>>
>ee>>>>>>
>eeeeee>>
>e>ee>>e>
>>ez>3e>>
>eee>e>e>
>eeeee>e>
>>>>>>>>>
```
Legend: '>' = Wall, 'e' = Passage

What is the minimum number of steps to reach the goal?
Give only the number of steps as your final answer, no other text or formatting.
Answer: 6
Metadata: {'grid_size': 9, 'grid': ['>>>>>>>>>', '>eeee>e>>', '>ee>>>>>>', '>eeeeee>>', '>e>ee>>e>', '>>ez>3e>>', '>eee>e>e>', '>eeeee>e>', '>>>>>>>>>'], 'shortest_path_length': 6, 'start': '3', 'goal': 'z', 'wall': '>', 'path': 'e'}

Example 2:
Question: Navigate from '`' (start) to 'i' (goal):

```
4444444
4AAAAi4
4A4A4A4
4A4AA44
44AAAA4
44A`444
4444444
```
Legend: '4' = Wall, 'A' = Passage

What is the minimum number of steps to reach the goal?
Give only the number of steps as your final answer, no other text or formatting.
Answer: 6
Metadata: {'grid_size': 7, 'grid': ['4444444', '4AAAAi4', '4A4A4A4', '4A4AA44', '44AAAA4', '44A`444', '4444444'], 'shortest_path_length': 6, 'start': '`', 'goal': 'i', 'wall': '4', 'path': 'A'}

Example 3:
Question: Navigate from '(' (start) to '`' (goal):

```
QQQQQQQ
QQ%%%%Q
QQ`%Q%Q
Q%%Q%%Q
Q%%%Q%Q
Q%QQ%(Q
QQQQQQQ
```
Legend: 'Q' = Wall, '%' = Passage

What is the minimum number of steps to reach the goal?
Give only the number of steps as your final answer, no other text or formatting.
Answer: 8
Metadata: {'grid_size': 7, 'grid': ['QQQQQQQ', 'QQ%%%%Q', 'QQ`%Q%Q', 'Q%%Q%%Q', 'Q%%%Q%Q', 'Q%QQ%(Q', 'QQQQQQQ'], 'shortest_path_length': 8, 'start': '(', 'goal': '`', 'wall': 'Q', 'path': '%'}

````

### mini_sudoku
Generates 4x4 sudoku puzzles with configurable difficulty

Default configuration:
```python
min_empty = 8
max_empty = 12
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: In 4x4 Mini Sudoku:
- Each row must contain each number from 1-4 exactly once
- Each column must contain each number 1-4 exactly once
- Each 2x2 subgrid must contain each number 1-4 exactly once
Solve this 4x4 Mini Sudoku puzzle:
4 _ _ _
_ 3 _ _
_ 1 3 _
_ _ _ _
Format your response as the puzzle above, with spaces separating each number within a row, and newlines separating rows.

Answer: 4 2 1 3
1 3 4 2
2 1 3 4
3 4 2 1
Metadata: {'puzzle': [[4, 0, 0, 0], [0, 3, 0, 0], [0, 1, 3, 0], [0, 0, 0, 0]], 'solution': [[4, 2, 1, 3], [1, 3, 4, 2], [2, 1, 3, 4], [3, 4, 2, 1]], 'num_empty': 12}

Example 2:
Question: In 4x4 Mini Sudoku:
- Each row must contain each number from 1-4 exactly once
- Each column must contain each number 1-4 exactly once
- Each 2x2 subgrid must contain each number 1-4 exactly once
Solve this 4x4 Mini Sudoku puzzle:
3 _ _ _
_ _ 4 _
4 2 _ _
_ _ _ 4
Format your response as the puzzle above, with spaces separating each number within a row, and newlines separating rows.

Answer: 3 4 1 2
2 1 4 3
4 2 3 1
1 3 2 4
Metadata: {'puzzle': [[3, 0, 0, 0], [0, 0, 4, 0], [4, 2, 0, 0], [0, 0, 0, 4]], 'solution': [[3, 4, 1, 2], [2, 1, 4, 3], [4, 2, 3, 1], [1, 3, 2, 4]], 'num_empty': 11}

Example 3:
Question: In 4x4 Mini Sudoku:
- Each row must contain each number from 1-4 exactly once
- Each column must contain each number 1-4 exactly once
- Each 2x2 subgrid must contain each number 1-4 exactly once
Solve this 4x4 Mini Sudoku puzzle:
_ _ _ _
1 3 4 _
3 _ 2 4
4 _ _ 1
Format your response as the puzzle above, with spaces separating each number within a row, and newlines separating rows.

Answer: 2 4 1 3
1 3 4 2
3 1 2 4
4 2 3 1
Metadata: {'puzzle': [[0, 0, 0, 0], [1, 3, 4, 0], [3, 0, 2, 4], [4, 0, 0, 1]], 'solution': [[2, 4, 1, 3], [1, 3, 4, 2], [3, 1, 2, 4], [4, 2, 3, 1]], 'num_empty': 8}

````

### n_queens
Generates N Queens puzzles with configurable difficulty

Default configuration:
```python
n = 8
min_remove = 1
max_remove = 7
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your job is to complete an n x n chess board with n Queens in total, such that no two attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

You can place a queen by replacing an underscore (_) with a Q.

Example:
- Input: Given the below board of size 4 x 4 your job is to place 2 queen(s) on the board such that no two queens attack each other.
_ Q _ _
_ _ _ _
_ _ _ _
_ _ Q _
- Output:
_ Q _ _
_ _ _ Q
Q _ _ _
_ _ Q _
- Explanation
    - None of the queens attack each other vertically, horizontally, or diagonally.
    - The added queens are marked with Q at the positions (1, 3) and (2, 0).

Given the below board of size 8 x 8 your job is to place 1 queen(s) on the board such that no two queens attack each other.
_ _ _ _ _ _ Q _
_ Q _ _ _ _ _ _
_ _ _ Q _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ Q
_ _ _ _ Q _ _ _
_ _ Q _ _ _ _ _
_ _ _ _ _ Q _ _

Answer: _ _ _ _ _ _ Q _
_ Q _ _ _ _ _ _
_ _ _ Q _ _ _ _
Q _ _ _ _ _ _ _
_ _ _ _ _ _ _ Q
_ _ _ _ Q _ _ _
_ _ Q _ _ _ _ _
_ _ _ _ _ Q _ _
Metadata: {'puzzle': [['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']], 'solutions': [[['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']]], 'num_removed': 1, 'valid_answers': ['_ _ _ _ _ _ Q _\n_ Q _ _ _ _ _ _\n_ _ _ Q _ _ _ _\nQ _ _ _ _ _ _ _\n_ _ _ _ _ _ _ Q\n_ _ _ _ Q _ _ _\n_ _ Q _ _ _ _ _\n_ _ _ _ _ Q _ _']}

Example 2:
Question: Your job is to complete an n x n chess board with n Queens in total, such that no two attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

You can place a queen by replacing an underscore (_) with a Q.

Example:
- Input: Given the below board of size 4 x 4 your job is to place 2 queen(s) on the board such that no two queens attack each other.
_ Q _ _
_ _ _ _
_ _ _ _
_ _ Q _
- Output:
_ Q _ _
_ _ _ Q
Q _ _ _
_ _ Q _
- Explanation
    - None of the queens attack each other vertically, horizontally, or diagonally.
    - The added queens are marked with Q at the positions (1, 3) and (2, 0).

Given the below board of size 8 x 8 your job is to place 3 queen(s) on the board such that no two queens attack each other.
_ Q _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ Q _ _
_ _ _ _ _ _ _ Q
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ Q _
_ _ _ _ Q _ _ _

Answer: _ Q _ _ _ _ _ _
_ _ _ Q _ _ _ _
_ _ _ _ _ Q _ _
_ _ _ _ _ _ _ Q
_ _ Q _ _ _ _ _
Q _ _ _ _ _ _ _
_ _ _ _ _ _ Q _
_ _ _ _ Q _ _ _
Metadata: {'puzzle': [['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', '_', '_', '_', 'Q', '_', '_', '_']], 'solutions': [[['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', '_', '_', '_', 'Q', '_', '_', '_']]], 'num_removed': 3, 'valid_answers': ['_ Q _ _ _ _ _ _\n_ _ _ Q _ _ _ _\n_ _ _ _ _ Q _ _\n_ _ _ _ _ _ _ Q\n_ _ Q _ _ _ _ _\nQ _ _ _ _ _ _ _\n_ _ _ _ _ _ Q _\n_ _ _ _ Q _ _ _']}

Example 3:
Question: Your job is to complete an n x n chess board with n Queens in total, such that no two attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

You can place a queen by replacing an underscore (_) with a Q.

Example:
- Input: Given the below board of size 4 x 4 your job is to place 2 queen(s) on the board such that no two queens attack each other.
_ Q _ _
_ _ _ _
_ _ _ _
_ _ Q _
- Output:
_ Q _ _
_ _ _ Q
Q _ _ _
_ _ Q _
- Explanation
    - None of the queens attack each other vertically, horizontally, or diagonally.
    - The added queens are marked with Q at the positions (1, 3) and (2, 0).

Given the below board of size 8 x 8 your job is to place 5 queen(s) on the board such that no two queens attack each other.
_ _ _ _ _ _ _ _
_ Q _ _ _ _ _ _
_ _ _ _ _ _ _ _
Q _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ Q _ _

Answer: _ _ _ _ Q _ _ _
_ Q _ _ _ _ _ _
_ _ _ _ _ _ _ Q
Q _ _ _ _ _ _ _
_ _ _ Q _ _ _ _
_ _ _ _ _ _ Q _
_ _ Q _ _ _ _ _
_ _ _ _ _ Q _ _
Metadata: {'puzzle': [['_', '_', '_', '_', '_', '_', '_', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']], 'solutions': [[['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']], [['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']], [['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']]], 'num_removed': 5, 'valid_answers': ['_ _ _ _ Q _ _ _\n_ Q _ _ _ _ _ _\n_ _ _ _ _ _ _ Q\nQ _ _ _ _ _ _ _\n_ _ _ Q _ _ _ _\n_ _ _ _ _ _ Q _\n_ _ Q _ _ _ _ _\n_ _ _ _ _ Q _ _', '_ _ _ _ _ _ Q _\n_ Q _ _ _ _ _ _\n_ _ _ Q _ _ _ _\nQ _ _ _ _ _ _ _\n_ _ _ _ _ _ _ Q\n_ _ _ _ Q _ _ _\n_ _ Q _ _ _ _ _\n_ _ _ _ _ Q _ _', '_ _ _ _ _ _ _ Q\n_ Q _ _ _ _ _ _\n_ _ _ Q _ _ _ _\nQ _ _ _ _ _ _ _\n_ _ _ _ _ _ Q _\n_ _ _ _ Q _ _ _\n_ _ Q _ _ _ _ _\n_ _ _ _ _ Q _ _']}

````

### needle_haystack
Generates "Needle in a Haystack tasks

Default configuration:
```python
num_statements = 50
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Caolain is neutral toward music. Alexx desires writing novels. Jake bears boxing. Harold gripes about dusting the furniture. Frederick disdains ironing the curtains. Cooper enjoys astronomy hobby. Caiden-Paul applauds all-terrain vehicles. Shayne delights in politics. Bradyn accepts artificial intelligence. Tyrnan supports climbing. Michal yearns for acting. Alvin deifies penguins. Allen relishes sailing. Brooke overlooks archery. Flynn prizes cleaning the patio. Grady can’t bear brewing beer. Rio ridicules acting. Wen is committed to emptying the dishwasher. Alfy execrates weeding the garden. Sweyn deifies bats. Emlyn laments bats. Shayan is passionate about snowboarding. Mehmet idolizes bird photography. Francis pines octopuses. Nikash worships ice skating. Tymom fancies motorcycles. Jaosha rejects balance. Abdur celebrates anime. Darryn bemoans logic. Michee revels in cleaning the ceiling fan. Khaleel worships trains. Jamie rails against the color amber. Daragh exults in astronomy. Finlay scoffs at minibikes. Kenyon desires collecting postcards. Caiden worships cocktails. Brodie reviles writing novels. Linton extols virtual reality. Bryson covets playing volleyball. Kyan begrudges listening to jazz. Kieran-Scott disapproves of collecting postcards. Willum esteems indie films. Isaa is addicted to ballet dancing. Arafat finds pleasure in triathlon. Oluwafemi disapproves of astronomy hobby. Seamas is keen on diving. Cian blasts playing the banjo. Liam-Stephen loathes the color sapphire. Bilal shrugs off playing the accordion. Sol is crazy about hip-hop dancing. 
Who execrates weeding the garden? Reply only with a name.
Answer: Alfy
Metadata: {'question': 'Who execrates weeding the garden? Reply only with a name.'}

Example 2:
Question: Angus disdains composting. Jazz adores trail running. Craig eschews ballet dancing. Orrin resents wolves. Leigh adores playing ping pong. Bryn spurns washing the dishes. Nyah dotes foxes. Vuyolwethu finds fulfillment in DJing. Rhoridh rails against baking cakes. Yaseen idolizes goats. Ajayraj lusts after visiting theme parks. Rooke damns building model airplanes. Morton approves of bird photography. Tiarnan curses trucks. Lennon endorses deer. Zidane resents turtles. Shergo stomachs curry. Muhammad rejoices in hip-hop dancing. Machlan bears curiosity. Diarmaid fancies ice skating. Asrar is apathetic about peacocks. Callan celebrates listening to jazz. Chukwuemeka glories in cycling. Levon is crazy about cleaning the microwave. Danniel rails against innovation. Bryden regrets luxury sedans. Daumantas enjoys solving crossword puzzles. Rokas finds pleasure in indie films. Reuben blasts cupcakes. Cobain derides listening to classical music. Loukas is keen on resilience. Vincenzo glorifies watering the garden. Riyaj is partial to scooters. Jagat shrugs off playing the harp. Thorben tolerates the color ruby. Dominick is committed to religion. Lex despises parrots. Ayden extols ultimate frisbee. Arlo is fond of listening to jazz. Tjay favors the color plum. Averon yearns surfing. Dylan-Patrick is nuts about dancing. Avi prefers space shuttles. Dedeniseoluwa celebrates playing the banjo. Johnathan finds fulfillment in beatboxing. Jakey is partial to optimism. Berkay approves of rhinos. Ryden is keen on playing water polo. Zhi is crazy about fishing. Caie disdains hip-hop dancing. 
Who extols ultimate frisbee? Reply only with a name.
Answer: Ayden
Metadata: {'question': 'Who extols ultimate frisbee? Reply only with a name.'}

Example 3:
Question: Marlin pines the color teal. Rufus mocks geocaching. Sharland yearns for the color yellow. Cejay yearns exploring caves. Diarmuid reveres limousines. Lincon exults resilience. Gareth ridicules playing board games. Jerome gripes about off-road vehicles. Aliyaan loves courage. Gabriel worships trucks. Cejay craves triathlon. Taylor-Jay detests off-road vehicles. Abu adores determination. Caedyn spurns pie. Darien is indifferent to resilience. Ronnie scorns all-terrain vehicles. Josan tolerates playing saxophone. Liam scorns playing cricket. Tyson longs for scorpions. Marc-Anthony ignores making coffee. Kayne bears trail running. Kurtis blasts creativity. Beau appreciates racing cars. Kerr laments the color khaki. Jayden-Paul relishes mopping the floor. Zak appreciates metaphysics. Darroch detests beauty. Carlo regrets building model cars. Rogan stomachs listening to folk music. Baley execrates omelettes. Tyler-Jay despises washing the dishes. Bruno fancies popcorn. Jacky puts up with zoology. Kajetan mocks cleaning the oven. Calley desires the color fuchsia. Zishan supports optimism. Jeronimo can’t bear vacuuming the floor. Amolpreet mocks roller skating. Kierin regrets metaphysics. Loudon approves of ducks. Brydon despises camels. Prinay eschews roller skating. Precious reveres coffee. Edison damns playing cricket. Eason yearns ants. Codey lusts after the color ruby. Ian revels in virtual reality. Hashim respects the color blue. Armaan derides performing magic. Arafat revels in canoeing. 
Who damns playing cricket? Reply only with a name.
Answer: Edison
Metadata: {'question': 'Who damns playing cricket? Reply only with a name.'}

````

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
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Keep all numbers larger than -90 in this list: ['-95.00', '-51.0', '47.2942', '-82.612']
Return the new list in the same format.
Answer: ['-51.0', '47.2942', '-82.612']
Metadata: {'original_numbers': ['-95.00', '-51.0', '47.2942', '-82.612'], 'filter_value': '-90', 'operation': 'keep_larger', 'result': ['-51.0', '47.2942', '-82.612']}

Example 2:
Question: Remove all numbers larger than 18.236 in this list: ['-42.8', '91.88', '34']
Return the new list in the same format.
Answer: ['-42.8']
Metadata: {'original_numbers': ['-42.8', '91.88', '34'], 'filter_value': '18.236', 'operation': 'remove_larger', 'result': ['-42.8']}

Example 3:
Question: Keep all numbers larger than 19.8962 in this list: ['4', '-64.7', '-42.1', '-77', '-79.9640', '37.76', '38.702', '18.20', '-28.34']
Return the new list in the same format.
Answer: ['37.76', '38.702']
Metadata: {'original_numbers': ['4', '-64.7', '-42.1', '-77', '-79.9640', '37.76', '38.702', '18.20', '-28.34'], 'filter_value': '19.8962', 'operation': 'keep_larger', 'result': ['37.76', '38.702']}

````

### number_format
Generates Count Bits exercises with configurable difficulty

Default configuration:
```python
max_num_candidates = 5
min_n = 1000
max_n = 1000000000
max_delta = 1000
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your task is to pick the largest/smallest number out of several options.

Example
- Input: Pick the largest number of the following candidates: 857575.23 8.975554e+05 887,555.62
- Output: 8.975554e+05
- Explanation:
    - Sorting the numbers written in various notations we get: 857575.23 < 887,555.62 < 8.975554e+05
    - Therefore, the largest number is 8.975554e+05

Now, pick the largest number of the following candidates: 25011730.212000 25011280.271000

Answer: 25011730.212
Metadata: {'candidates': [25011730.212, 25011280.271], 'solution': 25011730.212, 'formatted_candidates': ['25011730.212000', '25011280.271000'], 'size': 'largest'}

Example 2:
Question: Your task is to pick the largest/smallest number out of several options.

Example
- Input: Pick the largest number of the following candidates: 857575.23 8.975554e+05 887,555.62
- Output: 8.975554e+05
- Explanation:
    - Sorting the numbers written in various notations we get: 857575.23 < 887,555.62 < 8.975554e+05
    - Therefore, the largest number is 8.975554e+05

Now, pick the largest number of the following candidates: 286,084,894.213 286,085,419.581

Answer: 286085419.581
Metadata: {'candidates': [286084894.213, 286085419.581], 'solution': 286085419.581, 'formatted_candidates': ['286,084,894.213', '286,085,419.581'], 'size': 'largest'}

Example 3:
Question: Your task is to pick the largest/smallest number out of several options.

Example
- Input: Pick the largest number of the following candidates: 857575.23 8.975554e+05 887,555.62
- Output: 8.975554e+05
- Explanation:
    - Sorting the numbers written in various notations we get: 857575.23 < 887,555.62 < 8.975554e+05
    - Therefore, the largest number is 8.975554e+05

Now, pick the largest number of the following candidates: 520020968.942000 520021372.170000 5.200202022530000e+08 520020728.080000 520020548.078000

Answer: 520021372.16999996
Metadata: {'candidates': [520020968.942, 520021372.16999996, 520020202.25299996, 520020728.08, 520020548.07799995], 'solution': 520021372.16999996, 'formatted_candidates': ['520020968.942000', '520021372.170000', '5.200202022530000e+08', '520020728.080000', '520020548.078000'], 'size': 'largest'}

````

### number_sequence
Generates number sequence completion tasks with dynamic pattern generation

Default configuration:
```python
min_terms = 4
max_terms = 8
min_value = -100
max_value = 100
max_complexity = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: 3, 6, 12, 24, 48, 96, 192, 384, ?
Answer: 768
Metadata: {'rule': 'double', 'complexity': 3, 'sequence': [3, 6, 12, 24, 48, 96, 192, 384, 768]}

Example 2:
Question: 8, 14, 20, 26, 32, 38, 44, ?
Answer: 50
Metadata: {'rule': 'add 6', 'complexity': 1, 'sequence': [8, 14, 20, 26, 32, 38, 44, 50]}

Example 3:
Question: 8, 4, 2, 1, 0, 0, 0, ?
Answer: 0
Metadata: {'rule': 'halve', 'complexity': 2, 'sequence': [8, 4, 2, 1, 0, 0, 0, 0]}

````

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
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Sort these numbers in ascending order: 48, -51, -72, -80
Please follow the instruction below:
## 1. Let all your answers be a list of numbers. Instead of reporting your answer as -69, -13, 1, 7, 11, 43, 59, 61, use ['-69', '-13', '1', '7', '11', '43', '59', '61'] instead
## 2. Convert all numbers in the square brackets as strings. For example, ['-69', '-13', '1', '7', '11', '43', '59', '61']

Answer: ['-80', '-72', '-51', '48']
Metadata: {'original_numbers': ['48', '-51', '-72', '-80'], 'direction': 'ascending', 'sorted_numbers': ['-80', '-72', '-51', '48']}

Example 2:
Question: Sort these numbers in ascending order: 39.2, -71.2, -7.5
Please follow the instruction below:
## 1. Let all your answers be a list of numbers. Instead of reporting your answer as -69, -13, 1, 7, 11, 43, 59, 61, use ['-69', '-13', '1', '7', '11', '43', '59', '61'] instead
## 2. Convert all numbers in the square brackets as strings. For example, ['-69', '-13', '1', '7', '11', '43', '59', '61']

Answer: ['-71.2', '-7.5', '39.2']
Metadata: {'original_numbers': ['39.2', '-71.2', '-7.5'], 'direction': 'ascending', 'sorted_numbers': ['-71.2', '-7.5', '39.2']}

Example 3:
Question: Sort these numbers in descending order: 8.39, 72.41, -64.67, -54.97, -94.18, -76.67, -98.24, -68.66, 2.74
Please follow the instruction below:
## 1. Let all your answers be a list of numbers. Instead of reporting your answer as -69, -13, 1, 7, 11, 43, 59, 61, use ['-69', '-13', '1', '7', '11', '43', '59', '61'] instead
## 2. Convert all numbers in the square brackets as strings. For example, ['-69', '-13', '1', '7', '11', '43', '59', '61']

Answer: ['72.41', '8.39', '2.74', '-54.97', '-64.67', '-68.66', '-76.67', '-94.18', '-98.24']
Metadata: {'original_numbers': ['8.39', '72.41', '-64.67', '-54.97', '-94.18', '-76.67', '-98.24', '-68.66', '2.74'], 'direction': 'descending', 'sorted_numbers': ['72.41', '8.39', '2.74', '-54.97', '-64.67', '-68.66', '-76.67', '-94.18', '-98.24']}

````

### palindrome
Generates a set of letters that can be assembled into a palindrome.

Default configuration:
```python
min_length = 3
max_length = 10
seed = 42
size = 50
```

Example tasks:
````
Example 1:
Question: Your task is, given a list of letters, to form a valid palindrome.

A palindrome is a phrase that reads the same forwards and backwards.

If there are multiple possible answers, only respond with one of them. You must use all the letters provided.

Example:
- Input: Form a valid palindrome using the following letters: a, a, b
- Output: aba
- Explanation:
    - The phrase aba reads the same forwards and backwards.
    - The output answer is a valid palindrome using all the letters provided.
    - The answer is a string, rather than a list of characters.

Now, form a valid palindrome using the following letters: h, a, h, a

Answer: ahha
Metadata: {'letters': ['h', 'a', 'h', 'a'], 'generated_palindrome': 'ahha'}

Example 2:
Question: Your task is, given a list of letters, to form a valid palindrome.

A palindrome is a phrase that reads the same forwards and backwards.

If there are multiple possible answers, only respond with one of them. You must use all the letters provided.

Example:
- Input: Form a valid palindrome using the following letters: a, a, b
- Output: aba
- Explanation:
    - The phrase aba reads the same forwards and backwards.
    - The output answer is a valid palindrome using all the letters provided.
    - The answer is a string, rather than a list of characters.

Now, form a valid palindrome using the following letters: h, y, h

Answer: hyh
Metadata: {'letters': ['h', 'y', 'h'], 'generated_palindrome': 'hyh'}

Example 3:
Question: Your task is, given a list of letters, to form a valid palindrome.

A palindrome is a phrase that reads the same forwards and backwards.

If there are multiple possible answers, only respond with one of them. You must use all the letters provided.

Example:
- Input: Form a valid palindrome using the following letters: a, a, b
- Output: aba
- Explanation:
    - The phrase aba reads the same forwards and backwards.
    - The output answer is a valid palindrome using all the letters provided.
    - The answer is a string, rather than a list of characters.

Now, form a valid palindrome using the following letters: n, j, n, j, d, j, s, s, d

Answer: nsdjjjdsn
Metadata: {'letters': ['n', 'j', 'n', 'j', 'd', 'j', 's', 's', 'd'], 'generated_palindrome': 'nsdjjjdsn'}

````

### palindrome_partitioning
Generates Palindrome Partitioning exercises with configurable difficulty

Default configuration:
```python
min_string_len = 5
max_string_len = 15
max_substring_palindome_len = 5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a string, partition it such that every substring is a palindrome.

A palindrome is a word that reads the same backward as forward.

You may return all possible palindrome partitioning in any order.

Example:
- Input: Partition the following string into palindromes: aab
- Output: [["a","a","b"],["aa","b"]]
- Explanation:
    - One way to partition the string is "a" | "a" | "b", where each substring is a palindrome.
    - Another way to partition the string is "aa" | "b", where again each substring is a palindrome.
    - Therefore, the final result is a list of the two palindrome partitions.

Partition the following string into palindromes: agegvckakcgnnrw

Answer: [["a", "g", "e", "g", "v", "c", "k", "a", "k", "c", "g", "n", "n", "r", "w"], ["a", "g", "e", "g", "v", "c", "k", "a", "k", "c", "g", "nn", "r", "w"], ["a", "g", "e", "g", "v", "c", "kak", "c", "g", "n", "n", "r", "w"], ["a", "g", "e", "g", "v", "c", "kak", "c", "g", "nn", "r", "w"], ["a", "g", "e", "g", "v", "ckakc", "g", "n", "n", "r", "w"], ["a", "g", "e", "g", "v", "ckakc", "g", "nn", "r", "w"], ["a", "geg", "v", "c", "k", "a", "k", "c", "g", "n", "n", "r", "w"], ["a", "geg", "v", "c", "k", "a", "k", "c", "g", "nn", "r", "w"], ["a", "geg", "v", "c", "kak", "c", "g", "n", "n", "r", "w"], ["a", "geg", "v", "c", "kak", "c", "g", "nn", "r", "w"], ["a", "geg", "v", "ckakc", "g", "n", "n", "r", "w"], ["a", "geg", "v", "ckakc", "g", "nn", "r", "w"]]
Metadata: {'string': 'agegvckakcgnnrw', 'solution': [['a', 'g', 'e', 'g', 'v', 'c', 'k', 'a', 'k', 'c', 'g', 'n', 'n', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'c', 'k', 'a', 'k', 'c', 'g', 'nn', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'c', 'kak', 'c', 'g', 'n', 'n', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'c', 'kak', 'c', 'g', 'nn', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'ckakc', 'g', 'n', 'n', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'ckakc', 'g', 'nn', 'r', 'w'], ['a', 'geg', 'v', 'c', 'k', 'a', 'k', 'c', 'g', 'n', 'n', 'r', 'w'], ['a', 'geg', 'v', 'c', 'k', 'a', 'k', 'c', 'g', 'nn', 'r', 'w'], ['a', 'geg', 'v', 'c', 'kak', 'c', 'g', 'n', 'n', 'r', 'w'], ['a', 'geg', 'v', 'c', 'kak', 'c', 'g', 'nn', 'r', 'w'], ['a', 'geg', 'v', 'ckakc', 'g', 'n', 'n', 'r', 'w'], ['a', 'geg', 'v', 'ckakc', 'g', 'nn', 'r', 'w']]}

Example 2:
Question: Given a string, partition it such that every substring is a palindrome.

A palindrome is a word that reads the same backward as forward.

You may return all possible palindrome partitioning in any order.

Example:
- Input: Partition the following string into palindromes: aab
- Output: [["a","a","b"],["aa","b"]]
- Explanation:
    - One way to partition the string is "a" | "a" | "b", where each substring is a palindrome.
    - Another way to partition the string is "aa" | "b", where again each substring is a palindrome.
    - Therefore, the final result is a list of the two palindrome partitions.

Partition the following string into palindromes: sesjj

Answer: [["s", "e", "s", "j", "j"], ["s", "e", "s", "jj"], ["ses", "j", "j"], ["ses", "jj"]]
Metadata: {'string': 'sesjj', 'solution': [['s', 'e', 's', 'j', 'j'], ['s', 'e', 's', 'jj'], ['ses', 'j', 'j'], ['ses', 'jj']]}

Example 3:
Question: Given a string, partition it such that every substring is a palindrome.

A palindrome is a word that reads the same backward as forward.

You may return all possible palindrome partitioning in any order.

Example:
- Input: Partition the following string into palindromes: aab
- Output: [["a","a","b"],["aa","b"]]
- Explanation:
    - One way to partition the string is "a" | "a" | "b", where each substring is a palindrome.
    - Another way to partition the string is "aa" | "b", where again each substring is a palindrome.
    - Therefore, the final result is a list of the two palindrome partitions.

Partition the following string into palindromes: owfwofaafsd

Answer: [["o", "w", "f", "w", "o", "f", "a", "a", "f", "s", "d"], ["o", "w", "f", "w", "o", "f", "aa", "f", "s", "d"], ["o", "w", "f", "w", "o", "faaf", "s", "d"], ["o", "wfw", "o", "f", "a", "a", "f", "s", "d"], ["o", "wfw", "o", "f", "aa", "f", "s", "d"], ["o", "wfw", "o", "faaf", "s", "d"], ["owfwo", "f", "a", "a", "f", "s", "d"], ["owfwo", "f", "aa", "f", "s", "d"], ["owfwo", "faaf", "s", "d"]]
Metadata: {'string': 'owfwofaafsd', 'solution': [['o', 'w', 'f', 'w', 'o', 'f', 'a', 'a', 'f', 's', 'd'], ['o', 'w', 'f', 'w', 'o', 'f', 'aa', 'f', 's', 'd'], ['o', 'w', 'f', 'w', 'o', 'faaf', 's', 'd'], ['o', 'wfw', 'o', 'f', 'a', 'a', 'f', 's', 'd'], ['o', 'wfw', 'o', 'f', 'aa', 'f', 's', 'd'], ['o', 'wfw', 'o', 'faaf', 's', 'd'], ['owfwo', 'f', 'a', 'a', 'f', 's', 'd'], ['owfwo', 'f', 'aa', 'f', 's', 'd'], ['owfwo', 'faaf', 's', 'd']]}

````

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
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the real value(s) of w in the equation: -127*w = 0
In solving the equations, please abide by the following instruction:
## 1. All answers should be comma-separated. For example "-0.3773, 0.4005" etc.
## 2. In cases where your answer is b = 2 + sqrt(4560) / 172 and b = 2 - sqrt(4560) / 172. Since b can be 2 numbers, resolve your answer like this instead, "-0.3773, 0.4005".
## 3. If there are no real values of i that satisfy the equation, report your answer as empty string, "".
## 4. If there are 2 answers, resolve the answers as comma-separated floats of 2 numbers, if 3 answers, make it comma-separated floats of 3 numbers.
## 5. Resolve all numbers as floats in the string of comma-separated numbers. Round the floats higher than 4 decimal place(d.p) down to 4 d.p.

Answer: 0.0
Metadata: {'polynomial_expr': '-127*w', 'variable': 'w', 'degree': 1, 'real_solutions': [0.0]}

Example 2:
Question: Determine the real value(s) of b that satisfies: 86*b**2 - 2*b - 13 = 0
In solving the equations, please abide by the following instruction:
## 1. All answers should be comma-separated. For example "-0.3773, 0.4005" etc.
## 2. In cases where your answer is b = 2 + sqrt(4560) / 172 and b = 2 - sqrt(4560) / 172. Since b can be 2 numbers, resolve your answer like this instead, "-0.3773, 0.4005".
## 3. If there are no real values of i that satisfy the equation, report your answer as empty string, "".
## 4. If there are 2 answers, resolve the answers as comma-separated floats of 2 numbers, if 3 answers, make it comma-separated floats of 3 numbers.
## 5. Resolve all numbers as floats in the string of comma-separated numbers. Round the floats higher than 4 decimal place(d.p) down to 4 d.p.

Answer: -0.3773, 0.4006
Metadata: {'polynomial_expr': '86*b**2 - 2*b - 13', 'variable': 'b', 'degree': 2, 'real_solutions': [-0.3773, 0.4006]}

Example 3:
Question: Determine the real value(s) of p that satisfies: 71*p**3 - 2*p - 29 = 0
In solving the equations, please abide by the following instruction:
## 1. All answers should be comma-separated. For example "-0.3773, 0.4005" etc.
## 2. In cases where your answer is b = 2 + sqrt(4560) / 172 and b = 2 - sqrt(4560) / 172. Since b can be 2 numbers, resolve your answer like this instead, "-0.3773, 0.4005".
## 3. If there are no real values of i that satisfy the equation, report your answer as empty string, "".
## 4. If there are 2 answers, resolve the answers as comma-separated floats of 2 numbers, if 3 answers, make it comma-separated floats of 3 numbers.
## 5. Resolve all numbers as floats in the string of comma-separated numbers. Round the floats higher than 4 decimal place(d.p) down to 4 d.p.

Answer: 0.7546
Metadata: {'polynomial_expr': '71*p**3 - 2*p - 29', 'variable': 'p', 'degree': 3, 'real_solutions': [0.7546]}

````

### polynomial_multiplication
Generates [min_polynomials, max_polynomials] random polynomials of degree in [min_degree, max_degree].
    - The polynomial is formed by summing random terms of the form: coeff * x^exponent.
    - Then we find "F = P_0 * ... * P_1" using Sympy.

Default configuration:
```python
min_terms = 2
max_terms = 4
min_value = 1
max_value = 100
min_degree = 0
max_degree = 3
min_polynomials = 2
max_polynomials = 3
variables = ('x', 'y', 'z')
allow_cross_variable_product = False
allow_multivariate_polynomials = False
operators = ('+', '-')
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Calculate the following: (-95*z**3 + 18*z)*(-12*z**2 + 78*z - 104)
In addition, When doing calculation, Use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps and even in reporting answers.

Answer: 1140*z**5 - 7410*z**4 + 9664*z**3 + 1404*z**2 - 1872*z
Metadata: {'polynomial_expr': '(-95*z**3 + 18*z)*(-12*z**2 + 78*z - 104)', 'result': '1140*z**5 - 7410*z**4 + 9664*z**3 + 1404*z**2 - 1872*z', 'variables': [z]}

Example 2:
Question: Simplify this expression: (-49*x**3 + 77*x + 8)*(8*x**3 - 163*x**2 - 49)*(16*x**3 + 74*x + 98)
In addition, When doing calculation, Use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps and even in reporting answers.

Answer: -6272*x**9 + 127792*x**8 - 19152*x**7 + 391246*x**6 + 807446*x**5 - 746364*x**4 - 1091196*x**3 - 406994*x**2 - 398762*x - 38416
Metadata: {'polynomial_expr': '(-49*x**3 + 77*x + 8)*(8*x**3 - 163*x**2 - 49)*(16*x**3 + 74*x + 98)', 'result': '-6272*x**9 + 127792*x**8 - 19152*x**7 + 391246*x**6 + 807446*x**5 - 746364*x**4 - 1091196*x**3 - 406994*x**2 - 398762*x - 38416', 'variables': [x]}

Example 3:
Question: Calculate the following: (29*y**2 - 49*y)*(21*y**3 + 49)
In addition, When doing calculation, Use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps and even in reporting answers.

Answer: 609*y**5 - 1029*y**4 + 1421*y**2 - 2401*y
Metadata: {'polynomial_expr': '(29*y**2 - 49*y)*(21*y**3 + 49)', 'result': '609*y**5 - 1029*y**4 + 1421*y**2 - 2401*y', 'variables': [y]}

````

### pool_matrix
Generates Pool Matrix exercises with configurable difficulty

Default configuration:
```python
min_rows = 2
min_cols = 2
max_rows = 10
max_cols = 10
max_pool_size = 3
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your job is to perform max/average pooling on the given matrix.
The stride is equal to the kernel size, meaning there is no overlap between the pooling regions.

Example 1:
- Input: Perform max pooling on the following matrix with a kernel size of 2:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
- Output:
6 8
14 16

Example 2:
- Input: Perform average pooling on the following matrix with a kernel size of 2:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
- Output:
3.5 5.5
11.5 13.5

Perform max pooling on the following matrix with a kernel size of 3:
6 3
7 4
6 9

Answer: 9
Metadata: {'matrix': [[6, 3], [7, 4], [6, 9]], 'pool_type': 'max', 'pool_size': 3, 'solution': [[9]]}

Example 2:
Question: Your job is to perform max/average pooling on the given matrix.
The stride is equal to the kernel size, meaning there is no overlap between the pooling regions.

Example 1:
- Input: Perform max pooling on the following matrix with a kernel size of 2:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
- Output:
6 8
14 16

Example 2:
- Input: Perform average pooling on the following matrix with a kernel size of 2:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
- Output:
3.5 5.5
11.5 13.5

Perform average pooling on the following matrix with a kernel size of 3:
4 0 1 5 0 3
1 2 7 0 3 2

Answer: 2.5 2.17
Metadata: {'matrix': [[4, 0, 1, 5, 0, 3], [1, 2, 7, 0, 3, 2]], 'pool_type': 'average', 'pool_size': 3, 'solution': [[2.5, 2.1666666666666665]]}

Example 3:
Question: Your job is to perform max/average pooling on the given matrix.
The stride is equal to the kernel size, meaning there is no overlap between the pooling regions.

Example 1:
- Input: Perform max pooling on the following matrix with a kernel size of 2:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
- Output:
6 8
14 16

Example 2:
- Input: Perform average pooling on the following matrix with a kernel size of 2:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
- Output:
3.5 5.5
11.5 13.5

Perform average pooling on the following matrix with a kernel size of 3:
4 3 1 3 0 4 3 8 7 7
6 9 3 7 3 3 6 5 4 5
9 1 8 7 4 5 3 0 4 9
2 8 8 6 2 0 3 4 8 3
2 2 1 2 2 9 8 1 8 9
4 2 4 6 7 5 5 6 2 5
1 8 9 1 8 0 9 3 5 9
5 0 8 0 4 2 9 7 6 6

Answer: 4.89 4.0 4.44 7.0
3.67 4.33 5.0 5.67
5.17 2.5 6.5 7.5
Metadata: {'matrix': [[4, 3, 1, 3, 0, 4, 3, 8, 7, 7], [6, 9, 3, 7, 3, 3, 6, 5, 4, 5], [9, 1, 8, 7, 4, 5, 3, 0, 4, 9], [2, 8, 8, 6, 2, 0, 3, 4, 8, 3], [2, 2, 1, 2, 2, 9, 8, 1, 8, 9], [4, 2, 4, 6, 7, 5, 5, 6, 2, 5], [1, 8, 9, 1, 8, 0, 9, 3, 5, 9], [5, 0, 8, 0, 4, 2, 9, 7, 6, 6]], 'pool_type': 'average', 'pool_size': 3, 'solution': [[4.888888888888889, 4.0, 4.444444444444445, 7.0], [3.6666666666666665, 4.333333333333333, 5.0, 5.666666666666667], [5.166666666666667, 2.5, 6.5, 7.5]]}

````

### power_function
Generates Power Function exercises with configurable difficulty

Default configuration:
```python
min_base = -1000.0
max_base = 1000.0
min_exponent = -8
max_exponent = 8
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your task is to compute an exponentiation of a number.

Example:
- Input: Compute 2^3
- Output: 8
- Explanation:
    - 2^3 = 2 * 2 * 2 = 8
    - Therefore, the final answer is 8

Example:
- Input: Compute 412.5^3
- Output: 70189453.125
- Explanation:
    - 412.5^3 = 412.5 * 412.5 * 412.5 = 70189453.125
    - Therefore, the final answer is 70189453.125

Compute 278.8536^-8

Answer: 2.7352054627088526e-20
Metadata: {'base': 278.8536, 'exponent': -8, 'solution': 2.7352054627088526e-20}

Example 2:
Question: Your task is to compute an exponentiation of a number.

Example:
- Input: Compute 2^3
- Output: 8
- Explanation:
    - 2^3 = 2 * 2 * 2 = 8
    - Therefore, the final answer is 8

Example:
- Input: Compute 412.5^3
- Output: 70189453.125
- Explanation:
    - 412.5^3 = 412.5 * 412.5 * 412.5 = 70189453.125
    - Therefore, the final answer is 70189453.125

Compute -922.8963^-4

Answer: 1.3784416297559e-12
Metadata: {'base': -922.8963, 'exponent': -4, 'solution': 1.3784416297559e-12}

Example 3:
Question: Your task is to compute an exponentiation of a number.

Example:
- Input: Compute 2^3
- Output: 8
- Explanation:
    - 2^3 = 2 * 2 * 2 = 8
    - Therefore, the final answer is 8

Example:
- Input: Compute 412.5^3
- Output: 70189453.125
- Explanation:
    - 412.5^3 = 412.5 * 412.5 * 412.5 = 70189453.125
    - Therefore, the final answer is 70189453.125

Compute -182.9282^-5

Answer: -4.881987860097121e-12
Metadata: {'base': -182.9282, 'exponent': -5, 'solution': -4.881987860097121e-12}

````

### prime_factorization
Generates prime factorization tasks

Default configuration:
```python
min_value = 2
max_value = 1000
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the prime factorization of 656. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 2 × 2 × 2 × 2 × 41
Metadata: {'number': 656, 'factors': [2, 2, 2, 2, 41]}

Example 2:
Question: Find the prime factorization of 41. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 41
Metadata: {'number': 41, 'factors': [41]}

Example 3:
Question: Find the prime factorization of 420. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 2 × 2 × 3 × 5 × 7
Metadata: {'number': 420, 'factors': [2, 2, 3, 5, 7]}

````

### products
Generates multiplication tasks with configurable number of terms

Default configuration:
```python
min_terms = 2
max_terms = 2
min_digits = 1
max_digits = 5
allow_negation = False
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve the following multiplication: 4 * 3. Give only the result as your final answer.
Answer: 12
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 1}, 'expression': '4 * 3'}

Example 2:
Question: Solve the following multiplication: 812 * 880. Give only the result as your final answer.
Answer: 714560
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 3}, 'expression': '812 * 880'}

Example 3:
Question: Solve the following multiplication: 81037 * 25290. Give only the result as your final answer.
Answer: 2049425730
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 5}, 'expression': '81037 * 25290'}

````

### propositional_logic
Generates propositional logic reasoning tasks

Default configuration:
```python
min_vars = 2
max_vars = 4
min_statements = 2
max_statements = 4
max_complexity = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: The following question is a propositional logic reasoning question.
In the question we provide a list of premises
The task is to infer a correct conclusion from the premise.
FORMAT INSTRUCTIONS:
Return the conclusion logic statement, as your final answer.
Use the following notation to denote symbols
OR = ∨
AND = ∧
IMPLIES = →
IFF = ↔
NOT = ¬
Here is the question:Given:
1. R
.2. Q
.What can we conclude from the above statements?
Answer: None
Metadata: {'premises': ['R', 'Q'], 'variables': ['P', 'Q', 'R', 'S'], 'complexity': 3, 'example_answer': '(P ∨ Q)'}

Example 2:
Question: The following question is a propositional logic reasoning question.
In the question we provide a list of premises
The task is to infer a correct conclusion from the premise.
FORMAT INSTRUCTIONS:
Return the conclusion logic statement, as your final answer.
Use the following notation to denote symbols
OR = ∨
AND = ∧
IMPLIES = →
IFF = ↔
NOT = ¬
Here is the question:Given:
1. ((Q → P) ∨ (Q → P))
.2. ((Q ↔ Q) → (P → P))
.3. P
.What can we conclude from the above statements?
Answer: None
Metadata: {'premises': ['((Q → P) ∨ (Q → P))', '((Q ↔ Q) → (P → P))', 'P'], 'variables': ['P', 'Q'], 'complexity': 3, 'example_answer': '(Q ∨ P)'}

Example 3:
Question: The following question is a propositional logic reasoning question.
In the question we provide a list of premises
The task is to infer a correct conclusion from the premise.
FORMAT INSTRUCTIONS:
Return the conclusion logic statement, as your final answer.
Use the following notation to denote symbols
OR = ∨
AND = ∧
IMPLIES = →
IFF = ↔
NOT = ¬
Here is the question:Given:
1. ((Q ∨ P) ∧ ¬P)
.2. P
.3. ((P ∧ R) ∧ ¬R)
.4. ((Q ↔ R) → ¬Q)
.What can we conclude from the above statements?
Answer: None
Metadata: {'premises': ['((Q ∨ P) ∧ ¬P)', 'P', '((P ∧ R) ∧ ¬R)', '((Q ↔ R) → ¬Q)'], 'variables': ['P', 'Q', 'R'], 'complexity': 3, 'example_answer': '(Q ∧ Q)'}

````

### quantum_lock
Generates QuantumLock tasks

Default configuration:
```python
difficulty = 10
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value. Your answer should be a sequence of buttons separated by '→', for example: A → B → C

Start: 0 (red)
Target: 46
Buttons:
A: Add 3 (when any)
B: Add 2 (when any)
C: Multiply 2 (when any)
Answer: A → B → C → C → A → C
Metadata: {'difficulty': 10, 'solution_path': ['A', 'B', 'C', 'C', 'A', 'C'], 'target_value': 46, 'buttons': [{'name': 'A', 'type': 'add', 'value': 3, 'active_state': 'any'}, {'name': 'B', 'type': 'add', 'value': 2, 'active_state': 'any'}, {'name': 'C', 'type': 'multiply', 'value': 2, 'active_state': 'any'}], 'initial_state': 'red', 'initial_value': 0}

Example 2:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value. Your answer should be a sequence of buttons separated by '→', for example: A → B → C

Start: 0 (red)
Target: 30
Buttons:
A: Add 2 (when green)
B: Subtract 3 (when red)
C: Multiply 2 (when red)
Answer: C → A → C → A → C → A → C → A
Metadata: {'difficulty': 10, 'solution_path': ['C', 'A', 'C', 'A', 'C', 'A', 'C', 'A'], 'target_value': 30, 'buttons': [{'name': 'A', 'type': 'add', 'value': 2, 'active_state': 'green'}, {'name': 'B', 'type': 'subtract', 'value': 3, 'active_state': 'red'}, {'name': 'C', 'type': 'multiply', 'value': 2, 'active_state': 'red'}], 'initial_state': 'red', 'initial_value': 0}

Example 3:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value. Your answer should be a sequence of buttons separated by '→', for example: A → B → C

Start: 0 (red)
Target: 45
Buttons:
A: Subtract 2 (when any)
B: Add 3 (when any)
C: Add 2 (when any)
Answer: B → B → B → B → B → B → B → B → B → B → B → B → B → B → B
Metadata: {'difficulty': 10, 'solution_path': ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'], 'target_value': 45, 'buttons': [{'name': 'A', 'type': 'subtract', 'value': 2, 'active_state': 'any'}, {'name': 'B', 'type': 'add', 'value': 3, 'active_state': 'any'}, {'name': 'C', 'type': 'add', 'value': 2, 'active_state': 'any'}], 'initial_state': 'red', 'initial_value': 0}

````

### ransom_note
Generates Ransom Note exercises with configurable difficulty

Default configuration:
```python
max_note_length = 10
max_magazine_length = 30
p_solvable = 0.5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and False otherwise.

Each letter in the magazine string can only be used once in your ransom note.

Ransom note: c
Magazine: kjjfnerbv

Answer: False
Metadata: {'ransom_note': 'c', 'magazine': 'kjjfnerbv', 'solution': False, 'solvable': False}

Example 2:
Question: Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and False otherwise.

Each letter in the magazine string can only be used once in your ransom note.

Ransom note: pan
Magazine: pipmrxluyrkumtnaynmqosywf

Answer: True
Metadata: {'ransom_note': 'pan', 'magazine': 'pipmrxluyrkumtnaynmqosywf', 'solution': True, 'solvable': True}

Example 3:
Question: Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and False otherwise.

Each letter in the magazine string can only be used once in your ransom note.

Ransom note: yuothygge
Magazine: gpfslbehhhhagoutvejfoytuuyy

Answer: True
Metadata: {'ransom_note': 'yuothygge', 'magazine': 'gpfslbehhhhagoutvejfoytuuyy', 'solution': True, 'solvable': True}

````

### rearc
Default configuration:
```python
min_examples = 3
max_examples = 5
diff_lb = 0
diff_ub = 0.2
board_format_opts = BoardFormattingOptions(alphabet=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], col_delimiter=' ', row_delimiter='\n', array_brackets=False)
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 9
Output:
9 9 9 9
1 1 1 1
9 9 9 9
1 1 1 1
1 9 9 9
1 9 1 1
1 9 1 9

Example 2:

Input:
4 8 8 8 8 8 8
8 8 8 8 8 8 8
8 8 8 8 8 8 8
8 8 8 8 8 8 8
8 8 8 8 8 8 8
Output:
4 8 4 8 4 8 4
8 8 4 8 4 8 4
4 4 4 8 4 8 4
8 8 8 8 4 8 4
4 4 4 4 4 8 4

Example 3:

Input:
2 2 2 2
2 2 2 2
2 2 2 2
2 2 2 2
2 2 2 2
2 2 2 2
2 2 2 2
5 2 2 2
Output:
2 2 2 2
5 5 5 5
2 2 2 2
5 5 5 5
2 2 2 2
5 5 5 2
2 2 5 2
5 2 5 2


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
3 3 3 3 3 3 3 9
3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3

Answer: 3 9 3 9 3 9 3 9
3 9 3 9 3 9 3 3
3 9 3 9 3 9 9 9
3 9 3 9 3 3 3 3
3 9 3 9 9 9 9 9
Metadata: {'input': ((3, 3, 3, 3, 3, 3, 3, 9), (3, 3, 3, 3, 3, 3, 3, 3), (3, 3, 3, 3, 3, 3, 3, 3), (3, 3, 3, 3, 3, 3, 3, 3), (3, 3, 3, 3, 3, 3, 3, 3)), 'output': ((3, 9, 3, 9, 3, 9, 3, 9), (3, 9, 3, 9, 3, 9, 3, 3), (3, 9, 3, 9, 3, 9, 9, 9), (3, 9, 3, 9, 3, 3, 3, 3), (3, 9, 3, 9, 9, 9, 9, 9)), 'task_id': 'd22278a0', 'difficulty': {'rng': 0.07173948707162241, 'pso': 0.12314814814814816}}

Example 2:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
6 6 6 6 6 6 6 6
6 6 6 6 6 6 6 6
6 6 9 6 6 6 9 6
6 6 6 9 6 9 6 6
6 6 6 6 9 6 6 6
6 6 6 9 6 9 6 6
6 6 9 6 6 6 9 6
6 6 6 6 6 6 6 6
6 6 6 6 6 6 6 6
6 6 6 6 6 6 6 6
6 6 6 6 6 6 6 6
Output:
6 6 6 6 6 6 6 6
6 6 6 6 6 6 6 6
6 6 9 6 6 6 9 6
6 6 6 9 6 9 6 6
6 6 6 6 9 6 6 6
6 6 6 9 6 9 6 6
6 6 9 6 6 6 9 6
6 6 6 6 6 6 6 6
6 6 6 6 6 6 6 6
6 6 6 6 6 6 6 6
6 6 6 6 6 6 6 6

Example 2:

Input:
5 5 5 5 5 5 5 5 5 5
5 5 8 5 8 5 8 5 5 5
5 5 5 5 5 5 5 5 5 5
5 5 8 5 2 5 8 5 5 5
5 5 5 5 5 5 5 5 5 5
5 5 8 5 8 5 8 5 5 5
5 5 5 5 5 5 5 5 5 5
Output:
5 5 5 5 5 5 5 5 5 5
5 5 8 5 8 5 8 5 5 5
5 5 5 5 5 5 5 5 5 5
5 5 8 5 2 5 8 5 5 5
5 5 5 5 5 5 5 5 5 5
5 5 8 5 8 5 8 5 5 5
5 5 5 5 5 5 5 5 5 5

Example 3:

Input:
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 2 1 2 1 1 1
1 1 1 1 2 1 1 1 1
1 1 1 2 1 2 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
Output:
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 2 1 2 1 1 1
1 1 1 1 2 1 1 1 1
1 1 1 2 1 2 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1

Example 4:

Input:
7 7 7 7 7 7 7 7 7 7
7 7 7 1 7 1 7 1 7 7
7 7 7 7 7 7 7 7 7 7
7 7 7 1 7 1 7 1 7 7
7 7 7 7 7 7 7 7 7 7
7 7 7 1 7 1 7 1 7 7
7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7
Output:
7 7 7 7 7 7 7 7 7 7
7 7 7 1 7 1 7 1 7 7
7 7 7 7 7 7 7 7 7 7
7 7 7 1 7 1 7 1 7 7
7 7 7 7 7 7 7 7 7 7
7 7 7 1 7 1 7 1 7 7
7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7

Example 5:

Input:
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 6 3 3 3 6 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 6 3 3 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 6 3 3 3 6 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3 3 3 3
Output:
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 6 3 3 3 6 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 6 3 3 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 6 3 3 3 6 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3 3 3 3


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 8 7 7 7
7 7 7 7 7 7 8 7 8 7 7
7 7 7 7 7 8 7 8 7 8 7
7 7 7 7 7 7 8 7 8 7 7
7 7 7 7 7 7 7 8 7 7 7
7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7

Answer: 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 8 7 7 7
7 7 7 7 7 7 8 7 8 7 7
7 7 7 7 7 8 7 8 7 8 7
7 7 7 7 7 7 8 7 8 7 7
7 7 7 7 7 7 7 8 7 7 7
7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7
Metadata: {'input': ((7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 8, 7, 7, 7), (7, 7, 7, 7, 7, 7, 8, 7, 8, 7, 7), (7, 7, 7, 7, 7, 8, 7, 8, 7, 8, 7), (7, 7, 7, 7, 7, 7, 8, 7, 8, 7, 7), (7, 7, 7, 7, 7, 7, 7, 8, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)), 'output': ((7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 8, 7, 7, 7), (7, 7, 7, 7, 7, 7, 8, 7, 8, 7, 7), (7, 7, 7, 7, 7, 8, 7, 8, 7, 8, 7), (7, 7, 7, 7, 7, 7, 8, 7, 8, 7, 7), (7, 7, 7, 7, 7, 7, 7, 8, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)), 'task_id': '11852cab', 'difficulty': {'rng': 0.09651305327452808, 'pso': 0.15228956228956228}}

Example 3:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
9 9
9 9
Output:
9 9
9 9
9 9
9 9

Example 2:

Input:
4 4 4 6
Output:
4 4 4 6
4 4 4 6

Example 3:

Input:
4 1 1
4 4 4
Output:
4 1 1
4 4 4
4 4 4
4 1 1


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
1 1 1 1 1
1 1 1 1 1

Answer: 1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
Metadata: {'input': ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1)), 'output': ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)), 'task_id': '8be77c9e', 'difficulty': {'rng': 0.09322002370336528, 'pso': 0.0638888888888889}}

````

### rectangle_count
Generates ASCII rectangle counting puzzles with configurable parameters

Default configuration:
```python
max_rectangles = 10
width = 80
height = 80
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to count how many rectangles are present in an ASCII grid.

Single rectangles are outlined with a '#', overlapping rectangles (max 2) are shown with '█'.

Example:
- Input: How many rectangles are in the grid below?

              ####
              #  #
              ####










 #########
 #       █##
 #       █ #
 ########█ #
         # #
         ###
- Output: 3
- Explanation:
    - The first rectangle is the 3x4 rectangle in the top right.
    - The other two rectangles are overlapping in the bottom left corner.
    - Therefore, the final answer is 3.

Now, it's your turn. How many rectangles do you see in the grid below?
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                 ##################################################             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 ##################################################             
                                                                                
                                                                                
                                                                                
                                                                                
   ######################################                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   ######################################                                       
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                


Answer: 2
Metadata: {'puzzle': '                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                 ##################################################             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 ##################################################             \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n   ######################################                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   ######################################                                       \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n', 'solution': 2}

Example 2:
Question: Your task is to count how many rectangles are present in an ASCII grid.

Single rectangles are outlined with a '#', overlapping rectangles (max 2) are shown with '█'.

Example:
- Input: How many rectangles are in the grid below?

              ####
              #  #
              ####










 #########
 #       █##
 #       █ #
 ########█ #
         # #
         ###
- Output: 3
- Explanation:
    - The first rectangle is the 3x4 rectangle in the top right.
    - The other two rectangles are overlapping in the bottom left corner.
    - Therefore, the final answer is 3.

Now, it's your turn. How many rectangles do you see in the grid below?
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                    ############                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    ############                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                


Answer: 1
Metadata: {'puzzle': '                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                    ############                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    ############                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n', 'solution': 1}

Example 3:
Question: Your task is to count how many rectangles are present in an ASCII grid.

Single rectangles are outlined with a '#', overlapping rectangles (max 2) are shown with '█'.

Example:
- Input: How many rectangles are in the grid below?

              ####
              #  #
              ####










 #########
 #       █##
 #       █ #
 ########█ #
         # #
         ###
- Output: 3
- Explanation:
    - The first rectangle is the 3x4 rectangle in the top right.
    - The other two rectangles are overlapping in the bottom left corner.
    - Therefore, the final answer is 3.

Now, it's your turn. How many rectangles do you see in the grid below?
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                         #########################              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       ############   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                    #####█#######################██#########█#  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #####█#######################██#########█#  
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #      ##########       ##         #   
                                         #      #        #       ############   
                                         #      #        #       #              
                                         #      ##########       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #########################              
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
            #######################                                             
            #                     #                                             
            #                     #                                             
            #                     #                                             
            #                     #                                             
            #                     #                                             
            #                     #                                             
            #               ######█###                                          
            #               #     #  #                                          
            #               ######█###                                          
            #                     #   ###########################               
            #                     #   #                         #               
            #                     #   #                         #               
            #######################   ###########################               
                                                                                


Answer: 7
Metadata: {'puzzle': '                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                         #########################              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       ############   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                    #####█#######################██#########█#  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #####█#######################██#########█#  \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #      ##########       ##         #   \n                                         #      #        #       ############   \n                                         #      #        #       #              \n                                         #      ##########       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #########################              \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n            #######################                                             \n            #                     #                                             \n            #                     #                                             \n            #                     #                                             \n            #                     #                                             \n            #                     #                                             \n            #                     #                                             \n            #               ######█###                                          \n            #               #     #  #                                          \n            #               ######█###                                          \n            #                     #   ###########################               \n            #                     #   #                         #               \n            #                     #   #                         #               \n            #######################   ###########################               \n                                                                                \n', 'solution': 7}

````

### rotate_matrix
Generates Rotate Matrix exercises with configurable difficulty

Default configuration:
```python
max_n = 10
max_rotations = 4
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a square matrix, your job is to rotate it clockwise.

Example:

Input: Rotate the matrix below by 90 degrees clockwise:
1 2 3
4 5 6
7 8 9

Output:
7 4 1
8 5 2
9 6 3

Rotate the matrix below by 90 degrees clockwise:
3 1
2 0

Answer: 2 3
0 1
Metadata: {'matrix': [[3, 1], [2, 0]], 'num_rotations': 1, 'solution': [[2, 3], [0, 1]]}

Example 2:
Question: Given a square matrix, your job is to rotate it clockwise.

Example:

Input: Rotate the matrix below by 90 degrees clockwise:
1 2 3
4 5 6
7 8 9

Output:
7 4 1
8 5 2
9 6 3

Rotate the matrix below by 180 degrees clockwise:
0

Answer: 0
Metadata: {'matrix': [[0]], 'num_rotations': 2, 'solution': [[0]]}

Example 3:
Question: Given a square matrix, your job is to rotate it clockwise.

Example:

Input: Rotate the matrix below by 90 degrees clockwise:
1 2 3
4 5 6
7 8 9

Output:
7 4 1
8 5 2
9 6 3

Rotate the matrix below by 180 degrees clockwise:
28 17 38 29 8 15 26
35 13 37 39 27 40 20
4 30 23 16 3 5 48
9 25 2 46 47 21 22
31 12 41 43 19 32 10
6 0 36 45 42 1 18
14 24 11 7 44 34 33

Answer: 33 34 44 7 11 24 14
18 1 42 45 36 0 6
10 32 19 43 41 12 31
22 21 47 46 2 25 9
48 5 3 16 23 30 4
20 40 27 39 37 13 35
26 15 8 29 38 17 28
Metadata: {'matrix': [[28, 17, 38, 29, 8, 15, 26], [35, 13, 37, 39, 27, 40, 20], [4, 30, 23, 16, 3, 5, 48], [9, 25, 2, 46, 47, 21, 22], [31, 12, 41, 43, 19, 32, 10], [6, 0, 36, 45, 42, 1, 18], [14, 24, 11, 7, 44, 34, 33]], 'num_rotations': 2, 'solution': [[33, 34, 44, 7, 11, 24, 14], [18, 1, 42, 45, 36, 0, 6], [10, 32, 19, 43, 41, 12, 31], [22, 21, 47, 46, 2, 25, 9], [48, 5, 3, 16, 23, 30, 4], [20, 40, 27, 39, 37, 13, 35], [26, 15, 8, 29, 38, 17, 28]]}

````

### rotten_oranges
Generates Rotten Oranges exercises with configurable difficulty

Default configuration:
```python
min_n = 10
max_n = 30
p_oranges = 0.85
p_rotten = 0.1
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: You are given an n x n grid where each cell can have one of three values:
- 0 representing an empty cell
- 1 representing a fresh orange
- 2 representing a rotten orange

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Your task is determine the minimum number of minutes that must elapse until no cell has a fresh orange.
If this is impossible, return -1.

Example:
- Input: Determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
    2 1 1
    1 1 0
    0 1 1
- Output: 4

Now, determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
1 1 1 1 2 1 1 1 1 0 1 1 1 1 1 2 1 0 1 1 1 1 1 0 0 1 1 1 1 1
1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 2 1 0
1 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 2 1 1 2 1 1
1 1 1 1 1 1 0 1 2 1 1 1 1 0 1 0 1 2 1 1 1 0 2 1 1 1 1 1 2 2
2 1 2 1 2 0 1 1 2 1 1 1 1 1 0 0 1 2 1 1 1 1 1 0 1 1 0 1 1 1
1 1 0 1 0 1 2 1 0 1 1 1 1 1 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1
1 1 1 1 1 0 1 1 1 1 1 2 0 1 0 1 1 1 1 2 1 1 0 1 1 0 1 1 1 1
1 1 1 1 1 1 1 1 1 0 1 1 2 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 1
1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 1 1 0 1 1
1 1 0 0 1 1 1 0 1 1 1 1 1 1 1 2 0 2 1 1 1 0 1 1 0 1 1 1 1 1
1 1 1 1 1 1 2 2 1 1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 2 1 1 2 1 1 1 2 1 1 1 1
1 1 1 0 1 1 1 1 1 1 1 1 2 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1
1 1 2 1 1 1 1 0 1 0 1 1 1 1 1 2 1 1 2 0 2 1 1 1 1 0 2 1 1 1
1 1 1 0 1 1 1 1 1 2 1 1 2 1 1 0 1 1 1 0 0 1 0 1 1 1 1 1 1 1
2 0 1 0 0 1 1 2 1 1 1 1 1 1 2 0 1 1 2 2 1 1 1 1 1 1 1 1 0 1
2 0 0 1 1 1 0 1 1 2 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1
0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 1 1 1 1 1 0 1 1
1 1 1 0 1 2 1 0 2 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 2 1 1 1
1 1 2 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 2 1 1 1 1 1 0 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 2 1 1 1 1 1 1
1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 1 2 1 1 1 1 1 1 2
0 1 1 1 1 1 1 1 1 2 2 1 1 1 1 0 2 0 1 1 0 1 1 1 1 0 1 1 1 2
1 1 1 0 0 1 1 0 1 1 2 1 1 1 0 0 1 2 1 1 1 1 1 1 1 0 1 1 1 0
2 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 0 1 1 1 2 1 2 0 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 1 1 2 1 1 1 1 1 1 1 1
0 1 1 1 1 2 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 0 1 1 2 1 1 0 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0
1 1 1 1 1 2 1 1 1 1 1 1 1 0 1 1 1 0 1 2 1 1 1 1 1 1 1 1 2 0
1 1 1 1 1 1 1 1 1 1 1 1 2 0 0 1 0 1 1 1 1 2 1 1 1 1 1 1 1 1

Answer: 6
Metadata: {'matrix': [[1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0], [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 1, 2, 1, 1], [1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1, 2, 2], [2, 1, 2, 1, 2, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], [1, 1, 2, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 2, 0, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], [2, 0, 1, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], [2, 0, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 1], [1, 1, 1, 0, 1, 2, 1, 0, 2, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1], [1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1, 2], [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 2, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 2], [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 1, 0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0], [2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 2, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]], 'solution': 6}

Example 2:
Question: You are given an n x n grid where each cell can have one of three values:
- 0 representing an empty cell
- 1 representing a fresh orange
- 2 representing a rotten orange

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Your task is determine the minimum number of minutes that must elapse until no cell has a fresh orange.
If this is impossible, return -1.

Example:
- Input: Determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
    2 1 1
    1 1 0
    0 1 1
- Output: 4

Now, determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
1 0 1 1 1 1 0 0 0 2 1
1 1 1 1 1 2 1 1 0 1 2
1 1 1 1 1 0 1 2 0 1 0
1 1 1 1 0 1 1 1 1 1 2
1 1 1 1 1 2 1 1 0 1 1
2 1 1 1 1 1 1 1 2 0 1
1 1 1 1 1 1 1 1 1 1 1
1 0 1 1 2 1 1 1 0 1 1
1 1 1 1 1 1 2 1 1 1 1
0 2 1 1 1 1 0 1 1 1 1
1 0 1 1 1 1 1 1 0 1 1

Answer: -1
Metadata: {'matrix': [[1, 0, 1, 1, 1, 1, 0, 0, 0, 2, 1], [1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 2], [1, 1, 1, 1, 1, 0, 1, 2, 0, 1, 0], [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1], [2, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1], [0, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]], 'solution': -1}

Example 3:
Question: You are given an n x n grid where each cell can have one of three values:
- 0 representing an empty cell
- 1 representing a fresh orange
- 2 representing a rotten orange

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Your task is determine the minimum number of minutes that must elapse until no cell has a fresh orange.
If this is impossible, return -1.

Example:
- Input: Determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
    2 1 1
    1 1 0
    0 1 1
- Output: 4

Now, determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 2 0 1 1 1 1 1
1 0 1 1 1 1 1 1 0 1 1 1 0 1 1 0 2 1 2 1 1 0 0
1 0 0 0 1 1 1 1 0 1 1 1 1 1 1 1 2 1 1 1 1 0 1
0 0 2 0 1 1 1 0 1 1 0 2 1 1 2 2 0 1 1 2 1 0 1
1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 1 1 1
1 1 2 1 1 1 1 2 1 1 1 1 1 1 2 2 1 1 1 1 1 0 1
1 1 1 1 1 2 1 1 2 1 1 1 1 0 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 2 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1
0 1 1 1 0 0 1 1 1 0 1 1 0 2 1 1 2 1 0 1 2 0 1
1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 2 1 1 0 1 1 1 1
1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1
1 1 1 0 0 1 1 0 1 1 1 1 2 1 1 0 1 0 1 1 1 1 1
2 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1
1 1 1 0 0 0 1 2 1 1 1 1 1 2 0 1 1 1 1 1 1 1 0
1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 0 2 1 1 1 1 2
1 1 1 1 1 1 1 1 1 2 0 1 1 0 2 1 1 1 1 1 1 1 1
1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1
2 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 1 2 1
1 1 0 1 2 1 1 1 1 1 2 1 1 2 1 1 1 1 1 1 1 1 0
1 1 0 1 1 1 1 2 1 1 1 2 1 1 1 0 1 2 1 1 1 1 1
1 2 1 1 2 1 0 1 0 1 2 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 0 1 2 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1
1 1 0 1 1 2 1 1 1 1 1 0 1 1 0 1 1 1 0 1 0 0 1

Answer: 13
Metadata: {'matrix': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 2, 1, 2, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1], [0, 0, 2, 0, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 2, 2, 0, 1, 1, 2, 1, 0, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 2, 1, 0, 1, 2, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1], [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 2, 1], [1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1], [1, 2, 1, 1, 2, 1, 0, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1]], 'solution': 13}

````

### rubiks_cube
Generates RubiksCube tasks

Default configuration:
```python
scramble_steps = 3
cube_size = 3
remove_ansi = True
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: You are given a 3x3x3 Rubik's cube. It looks like this:

          G  Y  G                   
          G  Y  G                   
          G  R  G                   
 W  W  W  O  G  O  Y  Y  Y  R  B  R 
 R  R  R  W  G  W  O  O  O  Y  B  Y 
 R  R  R  W  G  W  O  O  O  Y  B  Y 
          B  O  B                   
          B  W  B                   
          B  W  B                   
 

Please provide a solution to solve this cube using Singmaster notation.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': "F L' R", 'example_correct_answer': "L F' U' R D B' D' U R U' R' U B U' B' U' R' U R U B U' B' U R' U R U B U' B' U' B' U B U L U' L' U' B' U B U L U' L' U B' U B U L U' L' F R U R' U' F' U' R U R' U R U U R' F U' B' U F' U' B R' D' R D R' D' R D R' D' R D R' D' R D U R' D' R D R' D' R D U R' D' R D R' D' R D R' D' R D R' D' R D U R' D' R D R' D' R D U"}

Example 2:
Question: You are given a 3x3x3 Rubik's cube. It looks like this:

          Y  Y  R                   
          Y  Y  R                   
          G  G  R                   
 B  B  Y  R  R  B  W  W  W  G  O  O 
 R  R  W  G  G  G  Y  O  O  B  B  Y 
 R  R  W  G  G  G  Y  O  O  B  B  Y 
          O  O  O                   
          B  W  W                   
          B  W  W                   
 

Please provide a solution to solve this cube using Singmaster notation.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': "L' F U'", 'example_correct_answer': "U' D' B D L' U' F D R' D' U' R U' R' F' U U F U F U' F' U' L' U L U F U' F' U L' U L U F U' F' R U' R' U' F' U F R' U R U B U' B' U' U' B' U B U L U' L' F R U R' U' R U R' U' F' U R U R' U R U U R' U' R U R' U R U U R' U' R U' L' U R' U' L U F U' B' U F' U' B R' D' R D R' D' R D U U R' D' R D R' D' R D U R' D' R D R' D' R D U"}

Example 3:
Question: You are given a 3x3x3 Rubik's cube. It looks like this:

          Y  Y  W                   
          Y  Y  W                   
          Y  Y  W                   
 G  G  G  O  O  B  O  O  O  G  R  R 
 R  R  R  G  G  B  O  O  O  G  B  B 
 R  R  R  G  G  R  B  B  B  O  B  B 
          W  W  Y                   
          W  W  Y                   
          W  W  Y                   
 

Please provide a solution to solve this cube using Singmaster notation.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': "U R' R'", 'example_correct_answer': "R R U'"}

````

### rush_hour
Generates Rush Hour puzzle configurations from pre-computed database

Default configuration:
```python
min_moves = 1
max_moves = 50
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Move the red car (AA) to the exit on the right.
Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...'
where the letter is the vehicle and +/- number is spaces to move right/left or down/up.

Board:
.xBBCC
..x.K.
G.AAK.
G.IJDD
H.IJ..
HEEFFF

Answer: None
Metadata: {'board_config': 'oxCCDDooxoMoIoAAMoIoKLFFJoKLooJGGHHH', 'min_moves': 10}

Example 2:
Question: Move the red car (AA) to the exit on the right.
Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...'
where the letter is the vehicle and +/- number is spaces to move right/left or down/up.

Board:
EBBCCC
E....H
F.xAAH
F.G...
..GDDD
......

Answer: None
Metadata: {'board_config': 'FCCDDDFooooIGoxAAIGoHoooooHEEEoooooo', 'min_moves': 6}

Example 3:
Question: Move the red car (AA) to the exit on the right.
Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...'
where the letter is the vehicle and +/- number is spaces to move right/left or down/up.

Board:
GBBIJK
G..IJK
AAHI..
..HCCC
..xDD.
EEEFF.

Answer: None
Metadata: {'board_config': 'HBBJKLHooJKLAAIJooooICCCooxEEoFFFGGo', 'min_moves': 30}

````

### self_reference
Generates self-referential puzzles

Default configuration:
```python
difficulty = 5
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Given the truthfulness of these statements, please tell me the number of possible solutions: 
 - Statement 1: 'At least 1 of these 7 statements are true.'
 - Statement 2: 'At most 3 of these 7 statements are false.'
 - Statement 3: 'Exactly 4 of these 7 statements are true.'
 - Statement 4: 'Exactly 3 of these 7 statements are false.'
 - Statement 5: 'Either Statement 3 or Statement 4 is true, but not both.'
 - Statement 6: 'The number of true statements is a prime number.'
 - Statement 7: 'The number of false statements is a composite number.'

Answer: 4

Example 2:
Question: Given the truthfulness of these statements, please tell me the number of possible solutions: 
 - Statement 1: 'At least 4 of these 7 statements are true.'
 - Statement 2: 'At most 5 of these 7 statements are false.'
 - Statement 3: 'Exactly 7 of these 7 statements are true.'
 - Statement 4: 'Exactly 1 of these 7 statements are false.'
 - Statement 5: 'Either Statement 3 or Statement 4 is true, but not both.'
 - Statement 6: 'The number of true statements is a prime number.'
 - Statement 7: 'The number of false statements is a composite number.'

Answer: 4

Example 3:
Question: Given the truthfulness of these statements, please tell me the number of possible solutions: 
 - Statement 1: 'At least 2 of these 7 statements are true.'
 - Statement 2: 'At most 5 of these 7 statements are false.'
 - Statement 3: 'Exactly 0 of these 7 statements are true.'
 - Statement 4: 'Exactly 3 of these 7 statements are false.'
 - Statement 5: 'Either Statement 3 or Statement 4 is true, but not both.'
 - Statement 6: 'The number of true statements is a prime number.'
 - Statement 7: 'The number of false statements is a composite number.'

Answer: 2

````

### sentence_reordering
Generates sentence reordering tasks from text spans

Default configuration:
```python
min_words_in_sentence = 3
max_words_in_sentence = 20
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Restore the correct order of words in the following sentence: wish could get I sleep. "I some
Answer: "I wish I could get some sleep.
Metadata: {'word_count': 7}

Example 2:
Question: Restore the correct order of words in the following sentence: the high level name. itself its unable it maintain at was of to Unfortunately,
Answer: Unfortunately, it was unable to maintain itself at the high level of its name.
Metadata: {'word_count': 14}

Example 3:
Question: Restore the correct order of words in the following sentence: developed by For the unutilized. energy falls ages went the
Answer: For ages the the energy developed by falls went unutilized.
Metadata: {'word_count': 10}

````

### shortest_path
Generates Shortest Path exercises with configurable difficulty

Default configuration:
```python
min_rows = 5
max_rows = 8
min_cols = 5
max_cols = 8
p_blocked = 0.4
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your task is to find the shortest path from the start to the destination point in a grid.

The grid is represented as a matrix with the following types of cells:
- *: your starting point
- #: your destination point
- O: an open cell
- X: a blocked cell

Therefore, you need to find the shortest path from * to #, moving only through open cells.
If there is no path from * to #, simply write "infeasible" (without quotes).

Example 1:
- Input: Find the length of the shortest path from * to # in the following grid:
    X X X X X
    X * O O X
    X O X O X
    X X X O #
- Output: right right down down right

Example 2:
- Input: Find the length of the shortest path from * to # in the following grid:
    X X X X X
    X * O O X
    X O X O X
    X X X X #
- Output: infeasible

Now, find the length of the shortest path from * to # in the following grid:
O X X X O
O O X X X
O O # O O
* X O O X
O X X O X

Answer: up right right
Metadata: {'matrix': [['O', 'X', 'X', 'X', 'O'], ['O', 'O', 'X', 'X', 'X'], ['O', 'O', '#', 'O', 'O'], ['*', 'X', 'O', 'O', 'X'], ['O', 'X', 'X', 'O', 'X']], 'solution': ['up', 'right', 'right']}

Example 2:
Question: Your task is to find the shortest path from the start to the destination point in a grid.

The grid is represented as a matrix with the following types of cells:
- *: your starting point
- #: your destination point
- O: an open cell
- X: a blocked cell

Therefore, you need to find the shortest path from * to #, moving only through open cells.
If there is no path from * to #, simply write "infeasible" (without quotes).

Example 1:
- Input: Find the length of the shortest path from * to # in the following grid:
    X X X X X
    X * O O X
    X O X O X
    X X X O #
- Output: right right down down right

Example 2:
- Input: Find the length of the shortest path from * to # in the following grid:
    X X X X X
    X * O O X
    X O X O X
    X X X X #
- Output: infeasible

Now, find the length of the shortest path from * to # in the following grid:
# X O O O O O
X O X O O O O
X O O X X O O
O O O O X X X
O O X O O * O

Answer: infeasible
Metadata: {'matrix': [['#', 'X', 'O', 'O', 'O', 'O', 'O'], ['X', 'O', 'X', 'O', 'O', 'O', 'O'], ['X', 'O', 'O', 'X', 'X', 'O', 'O'], ['O', 'O', 'O', 'O', 'X', 'X', 'X'], ['O', 'O', 'X', 'O', 'O', '*', 'O']], 'solution': []}

Example 3:
Question: Your task is to find the shortest path from the start to the destination point in a grid.

The grid is represented as a matrix with the following types of cells:
- *: your starting point
- #: your destination point
- O: an open cell
- X: a blocked cell

Therefore, you need to find the shortest path from * to #, moving only through open cells.
If there is no path from * to #, simply write "infeasible" (without quotes).

Example 1:
- Input: Find the length of the shortest path from * to # in the following grid:
    X X X X X
    X * O O X
    X O X O X
    X X X O #
- Output: right right down down right

Example 2:
- Input: Find the length of the shortest path from * to # in the following grid:
    X X X X X
    X * O O X
    X O X O X
    X X X X #
- Output: infeasible

Now, find the length of the shortest path from * to # in the following grid:
X X X X X
X O O O X
O # X X O
O X X X O
X O O X X
O O X X X
X O O O X
O O O X *

Answer: infeasible
Metadata: {'matrix': [['X', 'X', 'X', 'X', 'X'], ['X', 'O', 'O', 'O', 'X'], ['O', '#', 'X', 'X', 'O'], ['O', 'X', 'X', 'X', 'O'], ['X', 'O', 'O', 'X', 'X'], ['O', 'O', 'X', 'X', 'X'], ['X', 'O', 'O', 'O', 'X'], ['O', 'O', 'O', 'X', '*']], 'solution': []}

````

### simple_equations
Generates simple equations with one variable to solve

Default configuration:
```python
min_terms = 2
max_terms = 4
min_value = 1
max_value = 100
operators = ('+', '-', '*')
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Determine the value of u that satisfies: 32*u + 4 = 580
Answer: 18
Metadata: {'equation': '32*u + 4 = 580', 'variable': 'u'}

Example 2:
Question: Solve for b: 82080*b = 1067040
Answer: 13
Metadata: {'equation': '82080*b = 1067040', 'variable': 'b'}

Example 3:
Question: Determine the value of n that satisfies: 29*n - 5 = 430
Answer: 15
Metadata: {'equation': '29*n - 5 = 430', 'variable': 'n'}

````

### simple_geometry
A dataset for simple polygon angle-finding tasks.
    We randomly choose the number of sides N within [min_sides, max_sides].
    We then generate (N-1) random angles (in degrees), ensuring their sum is
    strictly less than the total sum for an (N)-sided convex polygon (which is 180*(N-2)).
    The question asks for the missing angle; the answer is computed by subtracting the
    sum of known angles from 180*(N-2).

Default configuration:
```python
min_sides = 3
max_sides = 6
min_angle = 10
max_angle = 170
seed = 42
size = 100
```

Example tasks:
````
Example 1:
Question: Given a convex polygon with 3 sides, its first 2 interior angles are: 16.0°, 80.0°. What is the measure of the remaining interior angle (in degrees)?Return only the angle as your answer.Do not give the units in your answer.
Answer: 84
Metadata: {'n_sides': 3, 'known_angles': [16.0, 80.0], 'sum_of_known_angles': 96.0, 'missing_angle_raw': 84.0, 'missing_angle_rounded': 84, 'total_interior_sum': 180}

Example 2:
Question: A convex polygon has 3 sides. The measures of the first 2 interior angles are: 83.0°, 46.0°. Find the measure of the last interior angle.Return only the angle as your answer.Do not give the units in your answer.
Answer: 51
Metadata: {'n_sides': 3, 'known_angles': [83.0, 46.0], 'sum_of_known_angles': 129.0, 'missing_angle_raw': 51.0, 'missing_angle_rounded': 51, 'total_interior_sum': 180}

Example 3:
Question: Given a convex polygon with 6 sides, its first 5 interior angles are: 143.0°, 148.0°, 39.0°, 55.0°, 107.0°. What is the measure of the remaining interior angle (in degrees)?Return only the angle as your answer.Do not give the units in your answer.
Answer: 228
Metadata: {'n_sides': 6, 'known_angles': [143.0, 148.0, 39.0, 55.0, 107.0], 'sum_of_known_angles': 492.0, 'missing_angle_raw': 228.0, 'missing_angle_rounded': 228, 'total_interior_sum': 720}

````

### simple_integration
Generates simple integration problems with one variable

Default configuration:
```python
min_terms = 2
max_terms = 5
min_degree = 1
max_degree = 10
min_bounds = 1
max_bounds = 10
operators = ('+', '-')
symbols = ('x', 'X')
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the indefinite integral: ∫ 70*x**6 + 12*x**2/5 dx
In addition, When doing calculation, Use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps. For example Use [-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C] instead of [-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C].

Answer: 10*x**7 + 4*x**3/5 + C
Metadata: {'integrand': '70*x**6 + 12*x**2/5', 'variable': 'x', 'expected_answer_expression': 10*x**7 + 4*x**3/5}

Example 2:
Question: Find the indefinite integral: ∫ 49*x**6/10 + 48*x**5 - 4*x - 10/9 dx
In addition, When doing calculation, Use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps. For example Use [-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C] instead of [-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C].

Answer: 7*x**7/10 + 8*x**6 - 2*x**2 - 10*x/9 + C
Metadata: {'integrand': '49*x**6/10 + 48*x**5 - 4*x - 10/9', 'variable': 'x', 'expected_answer_expression': 7*x**7/10 + 8*x**6 - 2*x**2 - 10*x/9}

Example 3:
Question: Find the indefinite integral: ∫ -28*X**3 + 8*X dx
In addition, When doing calculation, Use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps. For example Use [-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C] instead of [-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C].

Answer: -7*X**4 + 4*X**2 + C
Metadata: {'integrand': '-28*X**3 + 8*X', 'variable': 'X', 'expected_answer_expression': -7*X**4 + 4*X**2}

````

### sokoban
Generates Sokoban games with configurable parameters

Default configuration:
```python
min_w = 6
min_h = 6
max_w = 10
max_h = 10
min_boxes = 6
max_boxes = 10
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: You are going to solve a 'sokoban' puzzle.

* - The player
% - The player on a goal
@ - A box
X - A goal
$ - A box on a goal
+ - A wall
- - An empty position

Your solution must be a string of characters, ex: LDURRUDL.

Here is your puzzle:
+ + + + + + + + +  
+ + X - @ * @ X +  
+ + + - - @ - + +  
+ + + - - - X $ +  
+ + + + - + + + +  
+ + $ + + + + + +  
+ + + + + + + + +  


Answer: RLDULLRRDLDR
Metadata: {'gamestr': '+ + + + + + + + +  \n+ + X - @ * @ X +  \n+ + + - - @ - + +  \n+ + + - - - X $ +  \n+ + + + - + + + +  \n+ + $ + + + + + +  \n+ + + + + + + + +  \n\n', 'difficulty': {'size': (7, 9), 'num_steps': 12}}

Example 2:
Question: You are going to solve a 'sokoban' puzzle.

* - The player
% - The player on a goal
@ - A box
X - A goal
$ - A box on a goal
+ - A wall
- - An empty position

Your solution must be a string of characters, ex: LDURRUDL.

Here is your puzzle:
+ + + + + +  
+ - * - - +  
+ @ - - @ +  
+ X - @ - +  
+ - - - X +  
+ X - @ X +  
+ - - - - +  
+ + + + + +  


Answer: LDRRDRDDLLURURDULUURDD
Metadata: {'gamestr': '+ + + + + +  \n+ - * - - +  \n+ @ - - @ +  \n+ X - @ - +  \n+ - - - X +  \n+ X - @ X +  \n+ - - - - +  \n+ + + + + +  \n\n', 'difficulty': {'size': (8, 6), 'num_steps': 22}}

Example 3:
Question: You are going to solve a 'sokoban' puzzle.

* - The player
% - The player on a goal
@ - A box
X - A goal
$ - A box on a goal
+ - A wall
- - An empty position

Your solution must be a string of characters, ex: LDURRUDL.

Here is your puzzle:
+ + + + + + + + + + + +  
+ - $ - X + - - - - - +  
+ - @ - - - - - @ - X +  
+ - * - @ - - X - $ - +  
+ - - - - X + - - - - +  
+ + - - - - + $ - @ - +  
+ + + - - - - - - - - +  
+ + + - - - $ - - - - +  
+ + + + - - - - - - - +  
+ + + + + + + + + + + +  


Answer: RRRRURRRLDDRRDLULDRDLLLLULLDRDRUULUUULDLLURRDRU
Metadata: {'gamestr': '+ + + + + + + + + + + +  \n+ - $ - X + - - - - - +  \n+ - @ - - - - - @ - X +  \n+ - * - @ - - X - $ - +  \n+ - - - - X + - - - - +  \n+ + - - - - + $ - @ - +  \n+ + + - - - - - - - - +  \n+ + + - - - $ - - - - +  \n+ + + + - - - - - - - +  \n+ + + + + + + + + + + +  \n\n', 'difficulty': {'size': (10, 12), 'num_steps': 47}}

````

### spell_backward
Generates tasks to spell words backward

Default configuration:
```python
min_word_len = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Spell this word backward (example: sun -> nus): Project
Answer: tcejorP
Metadata: {'word': 'Project', 'word_len': 7}

Example 2:
Question: Spell this word backward (example: sun -> nus): Would
Answer: dluoW
Metadata: {'word': 'Would', 'word_len': 5}

Example 3:
Question: Spell this word backward (example: sun -> nus): One
Answer: enO
Metadata: {'word': 'One', 'word_len': 3}

````

### spiral_matrix
Generates Spiral Matrix exercises with configurable difficulty

Default configuration:
```python
max_n = 10
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a matrix, your job is to generate a list of elements in spiral order, starting from the top-left element.

Example:
- Input: For the matrix below, what is the list of elements in spiral order?
1 2 3
4 5 6
7 8 9
- Output: 1 2 3 6 9 8 7 4 5
- Explanation:
    - We start from the top-left element (1) and move right until we reach the end of the row: 1 2 3
    - Then, we move down until we reach the last column: 1 2 3 6 9
    - Next, we move left until we reach the first column: 1 2 3 6 9 8 7
    - Then, we move up until we reach the second row (i.e. one below the previously traversed row): 1 2 3 6 9 8 7 4
    - Finally, we move right until we reach the second to last column: 1 2 3 6 9 8 7 4 5
    - The output format is a space-separated list of elements in spiral order (as opposed to a python list)

For the matrix below, what is the list of elements in spiral order?
3 1 3
2 4 9
1 0 8

Answer: 3 1 3 9 8 0 1 2 4
Metadata: {'matrix': [[3, 1, 3], [2, 4, 9], [1, 0, 8]], 'solution': [3, 1, 3, 9, 8, 0, 1, 2, 4]}

Example 2:
Question: Given a matrix, your job is to generate a list of elements in spiral order, starting from the top-left element.

Example:
- Input: For the matrix below, what is the list of elements in spiral order?
1 2 3
4 5 6
7 8 9
- Output: 1 2 3 6 9 8 7 4 5
- Explanation:
    - We start from the top-left element (1) and move right until we reach the end of the row: 1 2 3
    - Then, we move down until we reach the last column: 1 2 3 6 9
    - Next, we move left until we reach the first column: 1 2 3 6 9 8 7
    - Then, we move up until we reach the second row (i.e. one below the previously traversed row): 1 2 3 6 9 8 7 4
    - Finally, we move right until we reach the second to last column: 1 2 3 6 9 8 7 4 5
    - The output format is a space-separated list of elements in spiral order (as opposed to a python list)

For the matrix below, what is the list of elements in spiral order?
5 7
2 4

Answer: 5 7 4 2
Metadata: {'matrix': [[5, 7], [2, 4]], 'solution': [5, 7, 4, 2]}

Example 3:
Question: Given a matrix, your job is to generate a list of elements in spiral order, starting from the top-left element.

Example:
- Input: For the matrix below, what is the list of elements in spiral order?
1 2 3
4 5 6
7 8 9
- Output: 1 2 3 6 9 8 7 4 5
- Explanation:
    - We start from the top-left element (1) and move right until we reach the end of the row: 1 2 3
    - Then, we move down until we reach the last column: 1 2 3 6 9
    - Next, we move left until we reach the first column: 1 2 3 6 9 8 7
    - Then, we move up until we reach the second row (i.e. one below the previously traversed row): 1 2 3 6 9 8 7 4
    - Finally, we move right until we reach the second to last column: 1 2 3 6 9 8 7 4 5
    - The output format is a space-separated list of elements in spiral order (as opposed to a python list)

For the matrix below, what is the list of elements in spiral order?
1 9 9 5 2 9 7 3
1 1 5 0 7 0 4 9
3 5 4 7 8 4 3 4
6 5 3 3 2 7 1 9
6 7 7 0 1 4 1 8
8 2 5 9 0 1 4 0
2 1 5 5 6 4 0 3
1 6 6 0 2 8 8 5

Answer: 1 9 9 5 2 9 7 3 9 4 9 8 0 3 5 8 8 2 0 6 6 1 2 8 6 6 3 1 1 5 0 7 0 4 3 1 1 4 0 4 6 5 5 1 2 7 5 5 4 7 8 4 7 4 1 0 9 5 7 3 3 2 1 0
Metadata: {'matrix': [[1, 9, 9, 5, 2, 9, 7, 3], [1, 1, 5, 0, 7, 0, 4, 9], [3, 5, 4, 7, 8, 4, 3, 4], [6, 5, 3, 3, 2, 7, 1, 9], [6, 7, 7, 0, 1, 4, 1, 8], [8, 2, 5, 9, 0, 1, 4, 0], [2, 1, 5, 5, 6, 4, 0, 3], [1, 6, 6, 0, 2, 8, 8, 5]], 'solution': [1, 9, 9, 5, 2, 9, 7, 3, 9, 4, 9, 8, 0, 3, 5, 8, 8, 2, 0, 6, 6, 1, 2, 8, 6, 6, 3, 1, 1, 5, 0, 7, 0, 4, 3, 1, 1, 4, 0, 4, 6, 5, 5, 1, 2, 7, 5, 5, 4, 7, 8, 4, 7, 4, 1, 0, 9, 5, 7, 3, 3, 2, 1, 0]}

````

### string_insertion
Generates String Insertion exercises with configurable difficulty

Default configuration:
```python
min_string_length = 5
max_string_length = 20
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a string consisting of characters A, B, C, D, and E, your job is to insert a character according to the following pattern:
1. If there is a substring ABCD in the string, insert the character A after the substring.
2. If there is a substring BCDE in the string, insert the character B after the substring.
3. If there is a substring CDEA in the string, insert the character C after the substring.
4. If there is a substring DEAB in the string, insert the character D after the substring.
5. If there is a substring EABC in the string, insert the character E after the substring.

Once you have inserted a character, you have to skip over the substring and the inserted character and continue the search from the next character.

Example
- Input: DDABCDEEDEAB
- Output: DDABCDAEEDEABD
- Explanation:
    - Theere are two inserted characters: DDABCD[A]EEDEAB[D] (shown in square brackets)
    - First, we insert A after ABCD.
    - Even though with the newly inserted 'A' we can obtain the substring BCD[A], we can't use it to insert another character.
    - Lastly, we insert D after DEAB.
    - Therefore, the final answer is DDABCDAEEDEABD (represented as a string, instead of a list of characters).

Given the following string, provide the answer after inserting the characters according to the pattern: ACBBBAEA

Answer: ACBBBAEA
Metadata: {'string': 'ACBBBAEA', 'solution': 'ACBBBAEA'}

Example 2:
Question: Given a string consisting of characters A, B, C, D, and E, your job is to insert a character according to the following pattern:
1. If there is a substring ABCD in the string, insert the character A after the substring.
2. If there is a substring BCDE in the string, insert the character B after the substring.
3. If there is a substring CDEA in the string, insert the character C after the substring.
4. If there is a substring DEAB in the string, insert the character D after the substring.
5. If there is a substring EABC in the string, insert the character E after the substring.

Once you have inserted a character, you have to skip over the substring and the inserted character and continue the search from the next character.

Example
- Input: DDABCDEEDEAB
- Output: DDABCDAEEDEABD
- Explanation:
    - Theere are two inserted characters: DDABCD[A]EEDEAB[D] (shown in square brackets)
    - First, we insert A after ABCD.
    - Even though with the newly inserted 'A' we can obtain the substring BCD[A], we can't use it to insert another character.
    - Lastly, we insert D after DEAB.
    - Therefore, the final answer is DDABCDAEEDEABD (represented as a string, instead of a list of characters).

Given the following string, provide the answer after inserting the characters according to the pattern: CBDCAD

Answer: CBDCAD
Metadata: {'string': 'CBDCAD', 'solution': 'CBDCAD'}

Example 3:
Question: Given a string consisting of characters A, B, C, D, and E, your job is to insert a character according to the following pattern:
1. If there is a substring ABCD in the string, insert the character A after the substring.
2. If there is a substring BCDE in the string, insert the character B after the substring.
3. If there is a substring CDEA in the string, insert the character C after the substring.
4. If there is a substring DEAB in the string, insert the character D after the substring.
5. If there is a substring EABC in the string, insert the character E after the substring.

Once you have inserted a character, you have to skip over the substring and the inserted character and continue the search from the next character.

Example
- Input: DDABCDEEDEAB
- Output: DDABCDAEEDEABD
- Explanation:
    - Theere are two inserted characters: DDABCD[A]EEDEAB[D] (shown in square brackets)
    - First, we insert A after ABCD.
    - Even though with the newly inserted 'A' we can obtain the substring BCD[A], we can't use it to insert another character.
    - Lastly, we insert D after DEAB.
    - Therefore, the final answer is DDABCDAEEDEABD (represented as a string, instead of a list of characters).

Given the following string, provide the answer after inserting the characters according to the pattern: EEABDBCABAEAABECDE

Answer: EEABDBCABAEAABECDE
Metadata: {'string': 'EEABDBCABAEAABECDE', 'solution': 'EEABDBCABAEAABECDE'}

````

### string_manipulation
Generates String Insertion exercises with configurable difficulty

Default configuration:
```python
min_string_length = 5
max_string_length = 20
min_num_rules = 3
max_num_rules = 8
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your job is to repeatedly transform a string according to a set of rules until no further transformations can be performed, or a state is repeated.

Evaluate the following rules in order, and apply the first applicable rule to the string:
1. If the string contains an even number of 'b's (and at least one 'b'), append 'ab' at the end.
2. If the string prefix is 'bc', delete the first two characters and append 'aa' to the end.
3. If the string ends with 'ca', remove the last character.
4. If the string suffix is 'ac', replace it with 'cb'.
5. If the string prefix is 'ab', replace it with 'ca'.
6. If the string contains 'ca' (not at the start), remove the first occurrence found after the first character.
7. If the string suffix is 'bb', delete the last two characters.
8. If the string starts with 'ac', replace the first two characters with 'zz'.

Once you have applied a rule, repeat the process with the new string until no further transformations can be performed (i.e. the string doesn't change), or a state is repeated.
If a state is repeated, the process is terminated, and the repeated state is discarded (i.e. is not considered as the final answer) and the state before the repeated state is considered as the final answer.

Example:
- Input:
    - String: abbac
    - Rules:
        1. If the string prefix is 'ab', replace it with 'ca'.
        2. If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.
        3. If the string ends with 'aa', replace it with 'cc'.
- Output: bbbacc
- Explanation:
    - In the first iteration, rule 1 is applied to the string abbac, resulting in cabac
    - In the second interation, rule 1 doesn't apply, but rule 2 is applied to the string cabac, resulting in bbbacc
    - In the third iteration, none of the rules (1, 2, 3) apply, so the process is terminated, and the final answer is bbbacc

Transform the following string according to the above list of rules:
acbaaaca

Answer: zzbaacbab
Metadata: {'string': 'acbaaaca', 'solution': 'zzbaacbab', 'states': ['acbaaaca', 'acbaaac', 'acbaacb', 'acbaacbab', 'zzbaacbab'], 'selected_rules': ["If the string contains an even number of 'b's (and at least one 'b'), append 'ab' at the end.", "If the string prefix is 'bc', delete the first two characters and append 'aa' to the end.", "If the string ends with 'ca', remove the last character.", "If the string suffix is 'ac', replace it with 'cb'.", "If the string prefix is 'ab', replace it with 'ca'.", "If the string contains 'ca' (not at the start), remove the first occurrence found after the first character.", "If the string suffix is 'bb', delete the last two characters.", "If the string starts with 'ac', replace the first two characters with 'zz'."]}

Example 2:
Question: Your job is to repeatedly transform a string according to a set of rules until no further transformations can be performed, or a state is repeated.

Evaluate the following rules in order, and apply the first applicable rule to the string:
1. If the string suffix is 'bb', delete the last two characters.
2. If the string starts with 'bb', remove the second character.
3. If the string ends with 'aa', replace it with 'cc'.
4. If the string prefix is 'ab', replace it with 'ca'.
5. If the string ends with 'ca', remove the last character.
6. If the string contains 'bca', delete the first occurrence entirely.
7. If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.
8. If the string length is greater than 15, remove the middle character.

Once you have applied a rule, repeat the process with the new string until no further transformations can be performed (i.e. the string doesn't change), or a state is repeated.
If a state is repeated, the process is terminated, and the repeated state is discarded (i.e. is not considered as the final answer) and the state before the repeated state is considered as the final answer.

Example:
- Input:
    - String: abbac
    - Rules:
        1. If the string prefix is 'ab', replace it with 'ca'.
        2. If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.
        3. If the string ends with 'aa', replace it with 'cc'.
- Output: bbbacc
- Explanation:
    - In the first iteration, rule 1 is applied to the string abbac, resulting in cabac
    - In the second interation, rule 1 doesn't apply, but rule 2 is applied to the string cabac, resulting in bbbacc
    - In the third iteration, none of the rules (1, 2, 3) apply, so the process is terminated, and the final answer is bbbacc

Transform the following string according to the above list of rules:
bcabbc

Answer: bc
Metadata: {'string': 'bcabbc', 'solution': 'bc', 'states': ['bcabbc', 'bbc', 'bc'], 'selected_rules': ["If the string suffix is 'bb', delete the last two characters.", "If the string starts with 'bb', remove the second character.", "If the string ends with 'aa', replace it with 'cc'.", "If the string prefix is 'ab', replace it with 'ca'.", "If the string ends with 'ca', remove the last character.", "If the string contains 'bca', delete the first occurrence entirely.", "If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.", 'If the string length is greater than 15, remove the middle character.']}

Example 3:
Question: Your job is to repeatedly transform a string according to a set of rules until no further transformations can be performed, or a state is repeated.

Evaluate the following rules in order, and apply the first applicable rule to the string:
1. If the string contains 'acb', replace the first occurrence with its reverse ('bca').
2. If the string length is greater than 15, remove the middle character.
3. If the string starts with 'ac', replace the first two characters with 'zz'.
4. If the string ends with 'ba', replace it with 'ab'.
5. If the string starts with 'cc', remove the first two characters.
6. If the string suffix is 'ac', replace it with 'cb'.
7. If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.
8. If the string prefix is 'cb', replace it with 'aa' and delete the last character.

Once you have applied a rule, repeat the process with the new string until no further transformations can be performed (i.e. the string doesn't change), or a state is repeated.
If a state is repeated, the process is terminated, and the repeated state is discarded (i.e. is not considered as the final answer) and the state before the repeated state is considered as the final answer.

Example:
- Input:
    - String: abbac
    - Rules:
        1. If the string prefix is 'ab', replace it with 'ca'.
        2. If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.
        3. If the string ends with 'aa', replace it with 'cc'.
- Output: bbbacc
- Explanation:
    - In the first iteration, rule 1 is applied to the string abbac, resulting in cabac
    - In the second interation, rule 1 doesn't apply, but rule 2 is applied to the string cabac, resulting in bbbacc
    - In the third iteration, none of the rules (1, 2, 3) apply, so the process is terminated, and the final answer is bbbacc

Transform the following string according to the above list of rules:
cccaababaaacaaaccb

Answer: bbababcaaaccbc
Metadata: {'string': 'cccaababaaacaaaccb', 'solution': 'bbababcaaaccbc', 'states': ['cccaababaaacaaaccb', 'cccaababaacaaaccb', 'cccaababacaaaccb', 'cccaababcaaaccb', 'caababcaaaccb', 'bbababcaaaccbc'], 'selected_rules': ["If the string contains 'acb', replace the first occurrence with its reverse ('bca').", 'If the string length is greater than 15, remove the middle character.', "If the string starts with 'ac', replace the first two characters with 'zz'.", "If the string ends with 'ba', replace it with 'ab'.", "If the string starts with 'cc', remove the first two characters.", "If the string suffix is 'ac', replace it with 'cb'.", "If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.", "If the string prefix is 'cb', replace it with 'aa' and delete the last character."]}

````

### string_splitting
Generates String Splitting exercises with configurable difficulty

Default configuration:
```python
min_initial_machines = 0
max_initial_machines = 5
max_iterations = 1000
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: There is a dismantling engineer who has old machines A, B, and C.
He discovered that he can obtain a batch of new parts X, Y, Z through the following rules:
1. One unit of machine A can be dismanteled into two units of part X and one unit of part Y.
2. Two units of machine B can be dismanteled into one unit of part X.
3. Two units of machine C can be dismanteled into one unit of part Y.
4. One unit of machine B and one unit of machine C can be combined into one unit of machine A.
5. One unit of part X and one unit of part Y can be combined into one unit of part Z.

Given a certain number of initial machines, your job is to continuously cycle through the rules 1-5 above, exausting one rule at a time, until no more rules can be applied, or until a state (counts of each machine and part type) is repeated.
After you make use of a rule, you should update the counts of each machine and part type accordingly, and then restart the process from rule 1.

The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
For example 1 0 1 5 4 3 means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z.

Example:
- Input: You have 2 machines A, 0 machines B, and 1 machine C.
- Output: 0 0 1 2 0 2
- Explanation
    0. Initial state: 2 0 1 0 0 0
    1. We can apply rule 1 and trade 1 machine A for 2 part X and 1 part Y: 1 0 1 2 1 0
    2. Starting over, we can apply rule 1 again: 0 0 1 4 2 0
    3. In the next iteration, we can apply rule 5 and trade 1 part X and 1 part Y for 1 part Z: 0 0 1 3 1 1
    4. In the next iteration, we can apply rule 5 again: 0 0 1 2 0 2
    5. We can't apply any more rules, so the final answer is 0 0 1 2 0 2

Now, you have 5 machine A, 0 machine B, and 0 machine C. Provide the count of each machine and part type after applying the above rules.

Answer: 0 0 0 5 0 5
Metadata: {'states': [[5, 0, 0, 0, 0, 0], [4, 0, 0, 2, 1, 0], [3, 0, 0, 4, 2, 0], [2, 0, 0, 6, 3, 0], [1, 0, 0, 8, 4, 0], [0, 0, 0, 10, 5, 0], [0, 0, 0, 9, 4, 1], [0, 0, 0, 8, 3, 2], [0, 0, 0, 7, 2, 3], [0, 0, 0, 6, 1, 4], [0, 0, 0, 5, 0, 5]], 'solution': [0, 0, 0, 5, 0, 5]}

Example 2:
Question: There is a dismantling engineer who has old machines A, B, and C.
He discovered that he can obtain a batch of new parts X, Y, Z through the following rules:
1. One unit of machine A can be dismanteled into two units of part X and one unit of part Y.
2. Two units of machine B can be dismanteled into one unit of part X.
3. Two units of machine C can be dismanteled into one unit of part Y.
4. One unit of machine B and one unit of machine C can be combined into one unit of machine A.
5. One unit of part X and one unit of part Y can be combined into one unit of part Z.

Given a certain number of initial machines, your job is to continuously cycle through the rules 1-5 above, exausting one rule at a time, until no more rules can be applied, or until a state (counts of each machine and part type) is repeated.
After you make use of a rule, you should update the counts of each machine and part type accordingly, and then restart the process from rule 1.

The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
For example 1 0 1 5 4 3 means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z.

Example:
- Input: You have 2 machines A, 0 machines B, and 1 machine C.
- Output: 0 0 1 2 0 2
- Explanation
    0. Initial state: 2 0 1 0 0 0
    1. We can apply rule 1 and trade 1 machine A for 2 part X and 1 part Y: 1 0 1 2 1 0
    2. Starting over, we can apply rule 1 again: 0 0 1 4 2 0
    3. In the next iteration, we can apply rule 5 and trade 1 part X and 1 part Y for 1 part Z: 0 0 1 3 1 1
    4. In the next iteration, we can apply rule 5 again: 0 0 1 2 0 2
    5. We can't apply any more rules, so the final answer is 0 0 1 2 0 2

Now, you have 0 machine A, 2 machine B, and 5 machine C. Provide the count of each machine and part type after applying the above rules.

Answer: 0 0 1 0 1 1
Metadata: {'states': [[0, 2, 5, 0, 0, 0], [0, 0, 5, 1, 0, 0], [0, 0, 3, 1, 1, 0], [0, 0, 1, 1, 2, 0], [0, 0, 1, 0, 1, 1]], 'solution': [0, 0, 1, 0, 1, 1]}

Example 3:
Question: There is a dismantling engineer who has old machines A, B, and C.
He discovered that he can obtain a batch of new parts X, Y, Z through the following rules:
1. One unit of machine A can be dismanteled into two units of part X and one unit of part Y.
2. Two units of machine B can be dismanteled into one unit of part X.
3. Two units of machine C can be dismanteled into one unit of part Y.
4. One unit of machine B and one unit of machine C can be combined into one unit of machine A.
5. One unit of part X and one unit of part Y can be combined into one unit of part Z.

Given a certain number of initial machines, your job is to continuously cycle through the rules 1-5 above, exausting one rule at a time, until no more rules can be applied, or until a state (counts of each machine and part type) is repeated.
After you make use of a rule, you should update the counts of each machine and part type accordingly, and then restart the process from rule 1.

The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
For example 1 0 1 5 4 3 means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z.

Example:
- Input: You have 2 machines A, 0 machines B, and 1 machine C.
- Output: 0 0 1 2 0 2
- Explanation
    0. Initial state: 2 0 1 0 0 0
    1. We can apply rule 1 and trade 1 machine A for 2 part X and 1 part Y: 1 0 1 2 1 0
    2. Starting over, we can apply rule 1 again: 0 0 1 4 2 0
    3. In the next iteration, we can apply rule 5 and trade 1 part X and 1 part Y for 1 part Z: 0 0 1 3 1 1
    4. In the next iteration, we can apply rule 5 again: 0 0 1 2 0 2
    5. We can't apply any more rules, so the final answer is 0 0 1 2 0 2

Now, you have 3 machine A, 4 machine B, and 4 machine C. Provide the count of each machine and part type after applying the above rules.

Answer: 0 0 0 3 0 5
Metadata: {'states': [[3, 4, 4, 0, 0, 0], [2, 4, 4, 2, 1, 0], [1, 4, 4, 4, 2, 0], [0, 4, 4, 6, 3, 0], [0, 2, 4, 7, 3, 0], [0, 0, 4, 8, 3, 0], [0, 0, 2, 8, 4, 0], [0, 0, 0, 8, 5, 0], [0, 0, 0, 7, 4, 1], [0, 0, 0, 6, 3, 2], [0, 0, 0, 5, 2, 3], [0, 0, 0, 4, 1, 4], [0, 0, 0, 3, 0, 5]], 'solution': [0, 0, 0, 3, 0, 5]}

````

### string_synthesis
Generates String Synthesis exercises with configurable difficulty

Default configuration:
```python
min_initial_blocks = 0
max_initial_blocks = 5
max_iterations = 1000
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: There are nine different blocks [A] [B] [C] {A} {B} {C} (A) (B) (C)
1. One [A], one [B], and one [C] can be combined to form one {A}.
2. One [A] and one [B] can be combined to form one {C}.
3. One [B] and one [C] can be combined to form one {B}.
4. Two [C] can be combined to form one {C}.
5. One {A} and one {C} can be combined to form one (A) and one (B).
6. Two {B} can be combined to form one (C).

Given a certain number of initial blocks, your job is to cycle through the rules 1-6 above, synthesizing new blocks until no more rules can be applied, or until a state (counts of each block type) is repeated.
In the case a state is repeated the answer is the state before the repetition!

The output should be the count of each block type after the rules have been applied in the order they are listed above.
For example 1 0 3 0 2 0 0 0 1 means that you have 1 [A] 0 [B] 3 [C] 0 {A} 2 {B} 0 {C} 0 (A) 0 (B) 1 (C).

Example:
- Input: You have 2 [A], 3 [B], and 3 [C].
- Output: 0 0 0 2 1 0 0 0 0
- Explanation:
    0. Initial state: 2 3 3 0 0 0 0 0 0
    1. We can apply Rule 1 and obtain 1 {A}. New state: 1 2 2 1 0 0 0 0 0
    2. We can apply Rule 1 again and obtain 1 {A}. New state 0 1 1 2 0 0 0 0 0
    3. We can apply Rule 3 and obtain 1 {B}. New state 0 0 0 2 1 0 0 0 0
    4. No more rules can be applied. The answer is 0 0 0 2 1 0 0 0 0

Now, you have 5 [A], 0 [B], and 0 [C] blocks. Provide the count of each block type after applying the above rules.

Answer: 5 0 0 0 0 0 0 0 0
Metadata: {'states': [[5, 0, 0, 0, 0, 0, 0, 0, 0]], 'solution': [5, 0, 0, 0, 0, 0, 0, 0, 0]}

Example 2:
Question: There are nine different blocks [A] [B] [C] {A} {B} {C} (A) (B) (C)
1. One [A], one [B], and one [C] can be combined to form one {A}.
2. One [A] and one [B] can be combined to form one {C}.
3. One [B] and one [C] can be combined to form one {B}.
4. Two [C] can be combined to form one {C}.
5. One {A} and one {C} can be combined to form one (A) and one (B).
6. Two {B} can be combined to form one (C).

Given a certain number of initial blocks, your job is to cycle through the rules 1-6 above, synthesizing new blocks until no more rules can be applied, or until a state (counts of each block type) is repeated.
In the case a state is repeated the answer is the state before the repetition!

The output should be the count of each block type after the rules have been applied in the order they are listed above.
For example 1 0 3 0 2 0 0 0 1 means that you have 1 [A] 0 [B] 3 [C] 0 {A} 2 {B} 0 {C} 0 (A) 0 (B) 1 (C).

Example:
- Input: You have 2 [A], 3 [B], and 3 [C].
- Output: 0 0 0 2 1 0 0 0 0
- Explanation:
    0. Initial state: 2 3 3 0 0 0 0 0 0
    1. We can apply Rule 1 and obtain 1 {A}. New state: 1 2 2 1 0 0 0 0 0
    2. We can apply Rule 1 again and obtain 1 {A}. New state 0 1 1 2 0 0 0 0 0
    3. We can apply Rule 3 and obtain 1 {B}. New state 0 0 0 2 1 0 0 0 0
    4. No more rules can be applied. The answer is 0 0 0 2 1 0 0 0 0

Now, you have 0 [A], 2 [B], and 5 [C] blocks. Provide the count of each block type after applying the above rules.

Answer: 0 0 1 0 0 1 0 0 1
Metadata: {'states': [[0, 2, 5, 0, 0, 0, 0, 0, 0], [0, 1, 4, 0, 1, 0, 0, 0, 0], [0, 0, 3, 0, 2, 0, 0, 0, 0], [0, 0, 1, 0, 2, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1]], 'solution': [0, 0, 1, 0, 0, 1, 0, 0, 1]}

Example 3:
Question: There are nine different blocks [A] [B] [C] {A} {B} {C} (A) (B) (C)
1. One [A], one [B], and one [C] can be combined to form one {A}.
2. One [A] and one [B] can be combined to form one {C}.
3. One [B] and one [C] can be combined to form one {B}.
4. Two [C] can be combined to form one {C}.
5. One {A} and one {C} can be combined to form one (A) and one (B).
6. Two {B} can be combined to form one (C).

Given a certain number of initial blocks, your job is to cycle through the rules 1-6 above, synthesizing new blocks until no more rules can be applied, or until a state (counts of each block type) is repeated.
In the case a state is repeated the answer is the state before the repetition!

The output should be the count of each block type after the rules have been applied in the order they are listed above.
For example 1 0 3 0 2 0 0 0 1 means that you have 1 [A] 0 [B] 3 [C] 0 {A} 2 {B} 0 {C} 0 (A) 0 (B) 1 (C).

Example:
- Input: You have 2 [A], 3 [B], and 3 [C].
- Output: 0 0 0 2 1 0 0 0 0
- Explanation:
    0. Initial state: 2 3 3 0 0 0 0 0 0
    1. We can apply Rule 1 and obtain 1 {A}. New state: 1 2 2 1 0 0 0 0 0
    2. We can apply Rule 1 again and obtain 1 {A}. New state 0 1 1 2 0 0 0 0 0
    3. We can apply Rule 3 and obtain 1 {B}. New state 0 0 0 2 1 0 0 0 0
    4. No more rules can be applied. The answer is 0 0 0 2 1 0 0 0 0

Now, you have 3 [A], 4 [B], and 4 [C] blocks. Provide the count of each block type after applying the above rules.

Answer: 0 0 0 3 1 0 0 0 0
Metadata: {'states': [[3, 4, 4, 0, 0, 0, 0, 0, 0], [2, 3, 3, 1, 0, 0, 0, 0, 0], [1, 2, 2, 2, 0, 0, 0, 0, 0], [0, 1, 1, 3, 0, 0, 0, 0, 0], [0, 0, 0, 3, 1, 0, 0, 0, 0]], 'solution': [0, 0, 0, 3, 1, 0, 0, 0, 0]}

````

### sudoku
Generates sudoku puzzles with configurable difficulty

Default configuration:
```python
min_empty = 30
max_empty = 50
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve this Sudoku puzzle:
4 _ _ _ 9 2 _ 3 _
_ _ 3 4 6 _ _ _ 7
6 1 2 _ _ 7 8 _ _
2 _ _ _ _ _ 7 9 1
8 _ _ 7 1 _ _ 5 6
1 _ _ 5 _ _ _ _ 3
9 _ 4 _ 7 1 _ _ _
_ 8 _ _ _ _ _ _ _
_ _ _ 9 8 _ _ _ 4
Respond with only your answer, formatted as the puzzle, a 9x9 grid with numbers separated by spaces, and rows separated by newlines.
Answer: 4 7 8 1 9 2 6 3 5
5 9 3 4 6 8 1 2 7
6 1 2 3 5 7 8 4 9
2 4 5 8 3 6 7 9 1
8 3 9 7 1 4 2 5 6
1 6 7 5 2 9 4 8 3
9 5 4 2 7 1 3 6 8
3 8 1 6 4 5 9 7 2
7 2 6 9 8 3 5 1 4
Metadata: {'puzzle': [[4, 0, 0, 0, 9, 2, 0, 3, 0], [0, 0, 3, 4, 6, 0, 0, 0, 7], [6, 1, 2, 0, 0, 7, 8, 0, 0], [2, 0, 0, 0, 0, 0, 7, 9, 1], [8, 0, 0, 7, 1, 0, 0, 5, 6], [1, 0, 0, 5, 0, 0, 0, 0, 3], [9, 0, 4, 0, 7, 1, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 9, 8, 0, 0, 0, 4]], 'solution': [[4, 7, 8, 1, 9, 2, 6, 3, 5], [5, 9, 3, 4, 6, 8, 1, 2, 7], [6, 1, 2, 3, 5, 7, 8, 4, 9], [2, 4, 5, 8, 3, 6, 7, 9, 1], [8, 3, 9, 7, 1, 4, 2, 5, 6], [1, 6, 7, 5, 2, 9, 4, 8, 3], [9, 5, 4, 2, 7, 1, 3, 6, 8], [3, 8, 1, 6, 4, 5, 9, 7, 2], [7, 2, 6, 9, 8, 3, 5, 1, 4]], 'num_empty': 48}

Example 2:
Question: Solve this Sudoku puzzle:
_ _ _ 1 3 2 6 4 5
_ 4 _ 8 5 _ _ 9 _
_ _ 1 9 _ 7 _ _ _
_ 8 9 6 _ _ 7 5 4
_ 3 _ 4 _ 1 9 8 _
4 6 _ 5 9 _ 2 3 1
5 _ 4 7 1 9 3 _ _
9 7 6 _ _ 4 5 1 _
8 _ 3 _ _ _ 4 7 _
Respond with only your answer, formatted as the puzzle, a 9x9 grid with numbers separated by spaces, and rows separated by newlines.
Answer: 7 9 8 1 3 2 6 4 5
3 4 2 8 5 6 1 9 7
6 5 1 9 4 7 8 2 3
1 8 9 6 2 3 7 5 4
2 3 5 4 7 1 9 8 6
4 6 7 5 9 8 2 3 1
5 2 4 7 1 9 3 6 8
9 7 6 3 8 4 5 1 2
8 1 3 2 6 5 4 7 9
Metadata: {'puzzle': [[0, 0, 0, 1, 3, 2, 6, 4, 5], [0, 4, 0, 8, 5, 0, 0, 9, 0], [0, 0, 1, 9, 0, 7, 0, 0, 0], [0, 8, 9, 6, 0, 0, 7, 5, 4], [0, 3, 0, 4, 0, 1, 9, 8, 0], [4, 6, 0, 5, 9, 0, 2, 3, 1], [5, 0, 4, 7, 1, 9, 3, 0, 0], [9, 7, 6, 0, 0, 4, 5, 1, 0], [8, 0, 3, 0, 0, 0, 4, 7, 0]], 'solution': [[7, 9, 8, 1, 3, 2, 6, 4, 5], [3, 4, 2, 8, 5, 6, 1, 9, 7], [6, 5, 1, 9, 4, 7, 8, 2, 3], [1, 8, 9, 6, 2, 3, 7, 5, 4], [2, 3, 5, 4, 7, 1, 9, 8, 6], [4, 6, 7, 5, 9, 8, 2, 3, 1], [5, 2, 4, 7, 1, 9, 3, 6, 8], [9, 7, 6, 3, 8, 4, 5, 1, 2], [8, 1, 3, 2, 6, 5, 4, 7, 9]], 'num_empty': 34}

Example 3:
Question: Solve this Sudoku puzzle:
_ _ 1 9 2 _ _ _ 3
3 _ _ 1 7 5 8 2 6
_ _ _ 4 3 6 1 _ _
1 _ 5 7 _ _ 9 3 _
_ 4 _ _ 5 9 7 1 8
7 _ 9 _ 1 _ 6 4 5
_ _ 3 5 9 _ 2 8 4
_ _ 2 6 8 _ _ 9 1
_ 5 _ 2 4 1 3 _ _
Respond with only your answer, formatted as the puzzle, a 9x9 grid with numbers separated by spaces, and rows separated by newlines.
Answer: 5 6 1 9 2 8 4 7 3
3 9 4 1 7 5 8 2 6
8 2 7 4 3 6 1 5 9
1 8 5 7 6 4 9 3 2
2 4 6 3 5 9 7 1 8
7 3 9 8 1 2 6 4 5
6 1 3 5 9 7 2 8 4
4 7 2 6 8 3 5 9 1
9 5 8 2 4 1 3 6 7
Metadata: {'puzzle': [[0, 0, 1, 9, 2, 0, 0, 0, 3], [3, 0, 0, 1, 7, 5, 8, 2, 6], [0, 0, 0, 4, 3, 6, 1, 0, 0], [1, 0, 5, 7, 0, 0, 9, 3, 0], [0, 4, 0, 0, 5, 9, 7, 1, 8], [7, 0, 9, 0, 1, 0, 6, 4, 5], [0, 0, 3, 5, 9, 0, 2, 8, 4], [0, 0, 2, 6, 8, 0, 0, 9, 1], [0, 5, 0, 2, 4, 1, 3, 0, 0]], 'solution': [[5, 6, 1, 9, 2, 8, 4, 7, 3], [3, 9, 4, 1, 7, 5, 8, 2, 6], [8, 2, 7, 4, 3, 6, 1, 5, 9], [1, 8, 5, 7, 6, 4, 9, 3, 2], [2, 4, 6, 3, 5, 9, 7, 1, 8], [7, 3, 9, 8, 1, 2, 6, 4, 5], [6, 1, 3, 5, 9, 7, 2, 8, 4], [4, 7, 2, 6, 8, 3, 5, 9, 1], [9, 5, 8, 2, 4, 1, 3, 6, 7]], 'num_empty': 33}

````

### syllogism
Generates syllogism reasoning tasks

Default configuration:
```python
allow_all = True
allow_no = True
allow_some = True
allow_some_not = True
invalid_ratio = 0.3
inversion_probability = 0.3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Consider these statements:
1. No students are humans
2. All humans are chefs

Does it logically follow that:
Some chefs are humans?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'No students are humans', 'premise2': 'All humans are chefs', 'selected_premise': 2, 'conclusion': 'Some chefs are humans', 'is_valid': True, 'type': 'inversion'}

Example 2:
Question: Consider these statements:
1. All children are animals
2. Some animals are not doctors

Does it logically follow that:
Some children are not doctors?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'All children are animals', 'premise2': 'Some animals are not doctors', 'conclusion': 'Some children are not doctors', 'is_valid': True, 'type': 'syllogism'}

Example 3:
Question: Consider these statements:
1. Some butterflies are not tigers
2. No tigers are whales

Does it logically follow that:
Some butterflies are whales?
(Answer Yes or No)
Answer: No
Metadata: {'premise1': 'Some butterflies are not tigers', 'premise2': 'No tigers are whales', 'conclusion': 'Some butterflies are whales', 'is_valid': False, 'type': 'syllogism'}

````

### time_intervals
Generates time interval calculation tasks with various formats and complexities

Default configuration:
```python
min_time = 00:00:00
max_time = 23:59:59.999999
max_time_difference_seconds = 86400
min_date = 1900-01-01
max_date = 3000-01-01
max_date_difference_days = 100
task_types = ['time', 'time_seconds', 'time_ms', 'date', 'datetime', 'datetime_tz']
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: A system backup started at 2964-06-17 08:15:14 and completed at 2964-07-04 11:59:09. What was the total backup duration? Answer in D days, HH:MM.
Answer: 17 days, 03:43
Metadata: {'task_type': 'datetime_tz', 'start_time': datetime.datetime(2964, 6, 17, 8, 15, 14), 'end_time': datetime.datetime(2964, 7, 4, 11, 59, 9), 'format': '%Y-%m-%d %H:%M:%S', 'expected_format': 'D days, HH:MM'}

Example 2:
Question: A video call started at 09:44 and ended at 12:22. How long was the call? Answer in HH:MM.
Answer: 02:38
Metadata: {'task_type': 'time', 'start_time': datetime.datetime(2025, 2, 23, 9, 44), 'end_time': datetime.datetime(2025, 2, 23, 12, 22), 'format': '%H:%M', 'expected_format': 'HH:MM'}

Example 3:
Question: Calculate the time difference between Sat Dec 22 2677 and Thu Mar 21 2678. Express the result in D days.
Answer: 89 days
Metadata: {'task_type': 'date', 'start_time': datetime.datetime(2677, 12, 22, 0, 0), 'end_time': datetime.datetime(2678, 3, 21, 0, 0), 'format': '%a %b %d %Y', 'expected_format': 'D days'}

````

### tower_of_hanoi
Generates Tower of Hanoi problems with solutions.
    Supports variable number of pegs using the optimized Frame-Stewart algorithm with Peg State Tracking.

Default configuration:
```python
min_disks = 3
max_disks = 7
min_pegs = 3
max_pegs = 4
size = 500
seed = 42
visualize = False
```

Example tasks:
````
Example 1:
Question: Solve the Tower of Hanoi problem with 3 disks and 3 pegs.
Move all disks from Peg 3 to Peg 2 following the rules:
- Only one disk can be moved at a time.
- A larger disk cannot be placed on top of a smaller disk.
- All disks must be on a peg at all times.
Example:
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2

Provide the sequence of moves.
Formatting guidelines:
Each instruction should be placed on a single line.
Each line should be formatted as 'Move disk X from Peg Y to Peg Z'
Do not include any other text or formatting.

Answer: ['Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 3 from Peg 3 to Peg 2', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2']
Metadata: {'num_disks': 3, 'num_pegs': 3, 'start_peg': 3, 'target_peg': 2, 'auxiliary_pegs': [1], 'solution_length': 7}

Example 2:
Question: Solve the Tower of Hanoi problem with 3 disks and 4 pegs.
Move all disks from Peg 2 to Peg 4 following the rules:
- Only one disk can be moved at a time.
- A larger disk cannot be placed on top of a smaller disk.
- All disks must be on a peg at all times.
Example:
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2

Provide the sequence of moves.
Formatting guidelines:
Each instruction should be placed on a single line.
Each line should be formatted as 'Move disk X from Peg Y to Peg Z'
Do not include any other text or formatting.

Answer: ['Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 3 from Peg 2 to Peg 4', 'Move disk 2 from Peg 3 to Peg 4', 'Move disk 1 from Peg 1 to Peg 4']
Metadata: {'num_disks': 3, 'num_pegs': 4, 'start_peg': 2, 'target_peg': 4, 'auxiliary_pegs': [1, 3], 'solution_length': 5}

Example 3:
Question: Solve the Tower of Hanoi problem with 6 disks and 3 pegs.
Move all disks from Peg 1 to Peg 2 following the rules:
- Only one disk can be moved at a time.
- A larger disk cannot be placed on top of a smaller disk.
- All disks must be on a peg at all times.
Example:
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2

Provide the sequence of moves.
Formatting guidelines:
Each instruction should be placed on a single line.
Each line should be formatted as 'Move disk X from Peg Y to Peg Z'
Do not include any other text or formatting.

Answer: ['Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 3 from Peg 1 to Peg 3', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 4 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 3 from Peg 3 to Peg 2', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 5 from Peg 1 to Peg 3', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 3 from Peg 2 to Peg 1', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 4 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 3 from Peg 1 to Peg 3', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 6 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 3 from Peg 3 to Peg 2', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 4 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 3 from Peg 2 to Peg 1', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 5 from Peg 3 to Peg 2', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 3 from Peg 1 to Peg 3', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 4 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 3 from Peg 3 to Peg 2', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2']
Metadata: {'num_disks': 6, 'num_pegs': 3, 'start_peg': 1, 'target_peg': 2, 'auxiliary_pegs': [3], 'solution_length': 63}

````

### tsumego
Generates Tsumego problems with configurable parameters

Default configuration:
```python
min_board_size = 9
max_board_size = 13
max_stones = 15
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: I have a Go problem for you. Black moves next - can you capture some of the white stones?

   A B C D E F G H I
 9 X . . . X . . . .
 8 . . . . . . . . .
 7 . O . O . . X . .
 6 . . . X . . . . O
 5 O . X O X . . . .
 4 . X O O . O . . .
 3 . . X O X . . . .
 2 . . . X . . . . .
 1 . O . O . . X . .

X - Black
O - White

Specify your move in coordinates (e.g. 'C4' for column C, row 4)
Answer: E4
Metadata: {'difficulty': {'board_size': 9}, 'board': [['X', '.', '.', '.', 'X', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', 'O', '.', 'O', '.', '.', 'X', '.', '.'], ['.', '.', '.', 'X', '.', '.', '.', '.', 'O'], ['O', '.', 'X', 'O', 'X', '.', '.', '.', '.'], ['.', 'X', 'O', 'O', '.', 'O', '.', '.', '.'], ['.', '.', 'X', 'O', 'X', '.', '.', '.', '.'], ['.', '.', '.', 'X', '.', '.', '.', '.', '.'], ['.', 'O', '.', 'O', '.', '.', 'X', '.', '.']], 'solution': 'E4'}

Example 2:
Question: Here's a Go challenge. Playing as Black, how can you capture as many white stones as possible?

   A B C D E F G H I
 9 . . O . . . . . .
 8 . X O . . . . . .
 7 X . X . . . . . .
 6 O O O X . . . . .
 5 X O O . . . . . .
 4 . X . . . . . . O
 3 . X . . . . X . .
 2 O . O . . . . . .
 1 . . . . O . . . .

X - Black
O - White

Specify your move in coordinates (e.g. 'C4' for column C, row 4)
Answer: B7
Metadata: {'difficulty': {'board_size': 9}, 'board': [['.', '.', 'O', '.', '.', '.', '.', '.', '.'], ['.', 'X', 'O', '.', '.', '.', '.', '.', '.'], ['X', '.', 'X', '.', '.', '.', '.', '.', '.'], ['O', 'O', 'O', 'X', '.', '.', '.', '.', '.'], ['X', 'O', 'O', '.', '.', '.', '.', '.', '.'], ['.', 'X', '.', '.', '.', '.', '.', '.', 'O'], ['.', 'X', '.', '.', '.', '.', 'X', '.', '.'], ['O', '.', 'O', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', 'O', '.', '.', '.', '.']], 'solution': 'B7'}

Example 3:
Question: Tsumego time. Black to play and capture some stones.
Find the key move.

   A B C D E F G H I J K L
12 . . . . . . . . . . . .
11 . . X . . . . . . . . .
10 . . . . . . . . . . . .
 9 . . . . . . . . . . . .
 8 X . . . . X . . . X . .
 7 . X . . . . . . . . . .
 6 . O X X . . . . . . . O
 5 . X O O X . . . . . . .
 4 . O O . . . . . O . . O
 3 X . X . . . . . . . . .
 2 . . . . . . . . . . . .
 1 . . . . . . . . . . X .

X - Black
O - White

Specify your move in coordinates (e.g. 'C4' for column C, row 4)
Answer: D4
Metadata: {'difficulty': {'board_size': 12}, 'board': [['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', 'X', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['X', '.', '.', '.', '.', 'X', '.', '.', '.', 'X', '.', '.'], ['.', 'X', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', 'O', 'X', 'X', '.', '.', '.', '.', '.', '.', '.', 'O'], ['.', 'X', 'O', 'O', 'X', '.', '.', '.', '.', '.', '.', '.'], ['.', 'O', 'O', '.', '.', '.', '.', '.', 'O', '.', '.', 'O'], ['X', '.', 'X', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'X', '.']], 'solution': 'D4'}

````

### word_ladder
Generates word ladder transformation tasks

Default configuration:
```python
min_word_length = 4
max_word_length = 4
min_chain_length = -1
max_chain_length = -1
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Transform the word ladder 'HAND' to 'GLEE' by changing one letter at a time.
Provide your answer as a comma-separated sequence of uppercase letters without spaces.
Each step must be a valid English word.
Answer: HAND,HARD,HERD,HEED,FEED,FLED,FLEE,GLEE
Metadata: {'start_word': 'HAND', 'end_word': 'GLEE', 'word_length': 4, 'chain_length': 8}

Example 2:
Question: Transform the word ladder 'JAZZ' to 'DORM' by changing one letter at a time.
Provide your answer as a comma-separated sequence of uppercase letters without spaces.
Each step must be a valid English word.
Answer: JAZZ,JIZZ,FIZZ,FUZZ,FUZE,FAZE,FARE,FORE,FORM,DORM
Metadata: {'start_word': 'JAZZ', 'end_word': 'DORM', 'word_length': 4, 'chain_length': 10}

Example 3:
Question: Transform the word ladder 'SNOG' to 'SUQS' by changing one letter at a time.
Provide your answer as a comma-separated sequence of uppercase letters without spaces.
Each step must be a valid English word.
Answer: SNOG,SNOW,SHOW,SHEW,SHES,SUES,SUQS
Metadata: {'start_word': 'SNOG', 'end_word': 'SUQS', 'word_length': 4, 'chain_length': 7}

````

### word_sequence_reversal
Generates word sequence reversal tasks from text spans

Default configuration:
```python
min_words = 3
max_words = 8
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Reverse this list of words: bed, if, problem, but, Well, an, transmission, nutritive
Answer: nutritive, transmission, an, Well, but, problem, if, bed
Metadata: {'num_words': 8, 'words': ['bed', 'if', 'problem', 'but', 'Well', 'an', 'transmission', 'nutritive']}

Example 2:
Question: Reverse this list of words: it, pleasure, Gutenberg
Answer: Gutenberg, pleasure, it
Metadata: {'num_words': 3, 'words': ['it', 'pleasure', 'Gutenberg']}

Example 3:
Question: Reverse this list of words: readable, to, he, that, to, possession
Answer: possession, to, that, he, to, readable
Metadata: {'num_words': 6, 'words': ['readable', 'to', 'he', 'that', 'to', 'possession']}

````

### word_sorting
Generates word sorting tasks

Default configuration:
```python
min_words = 3
max_words = 10
min_word_length = 3
max_word_length = 12
transformation = original
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.

Example:
- Input: Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list: freely, idea, indemnify, last, END, solving
- Output: END, freely, idea, indemnify, last, solving
- Explanation:
    - Uppercase letters come before lowercase letters, hence why "END" comes first.
    - "freely" comes before "idea" because "f" comes before "i".
    - "idea" comes before "indemnify" because even though they both start with "i", "d" comes before "n".
    - "indemnify" comes before "last" because "i" comes before "l".
    - "last" comes before "solving" because "l" comes before "s".
    - Finally, the output is provided as a comma separated list of the sorted words.

Now, sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list: DIRECT, given, exclaims, dreaming

Answer: DIRECT, dreaming, exclaims, given
Metadata: {'original_words': ['DIRECT', 'given', 'exclaims', 'dreaming'], 'transformed_words': ['DIRECT', 'given', 'exclaims', 'dreaming'], 'direction': 'ascending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['DIRECT', 'dreaming', 'exclaims', 'given']}

Example 2:
Question: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.

Example:
- Input: Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list: freely, idea, indemnify, last, END, solving
- Output: END, freely, idea, indemnify, last, solving
- Explanation:
    - Uppercase letters come before lowercase letters, hence why "END" comes first.
    - "freely" comes before "idea" because "f" comes before "i".
    - "idea" comes before "indemnify" because even though they both start with "i", "d" comes before "n".
    - "indemnify" comes before "last" because "i" comes before "l".
    - "last" comes before "solving" because "l" comes before "s".
    - Finally, the output is provided as a comma separated list of the sorted words.

Now, sort these words in descending order (using ASCII/Unicode ordering) and return them as a comma-separated list: heat, begun, sometimes

Answer: sometimes, heat, begun
Metadata: {'original_words': ['heat', 'begun', 'sometimes'], 'transformed_words': ['heat', 'begun', 'sometimes'], 'direction': 'descending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['sometimes', 'heat', 'begun']}

Example 3:
Question: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.

Example:
- Input: Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list: freely, idea, indemnify, last, END, solving
- Output: END, freely, idea, indemnify, last, solving
- Explanation:
    - Uppercase letters come before lowercase letters, hence why "END" comes first.
    - "freely" comes before "idea" because "f" comes before "i".
    - "idea" comes before "indemnify" because even though they both start with "i", "d" comes before "n".
    - "indemnify" comes before "last" because "i" comes before "l".
    - "last" comes before "solving" because "l" comes before "s".
    - Finally, the output is provided as a comma separated list of the sorted words.

Now, sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list: violates, yes, already, completing, pages, duty, his, EXPRESS, duly

Answer: EXPRESS, already, completing, duly, duty, his, pages, violates, yes
Metadata: {'original_words': ['violates', 'yes', 'already', 'completing', 'pages', 'duty', 'his', 'EXPRESS', 'duly'], 'transformed_words': ['violates', 'yes', 'already', 'completing', 'pages', 'duty', 'his', 'EXPRESS', 'duly'], 'direction': 'ascending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['EXPRESS', 'already', 'completing', 'duly', 'duty', 'his', 'pages', 'violates', 'yes']}

````

### zebra_puzzles
Generates [Zebra Puzzles](https://en.wikipedia.org/wiki/Zebra_Puzzle) with configurable parameters

Default configuration:
```python
num_people = 4
num_characteristics = 4
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: This is a logic puzzle. There are 4 houses (numbered 1 on the left, 4 on the right), from the perspective of someone standing across the street from them. Each has a different person in them. They have different characteristics:
 - Each person has a unique name: carol, arnold, alice, bob
 - People use different phone models: huawei p50, samsung galaxy s21, oneplus 9, google pixel 6
 - Each person has a favorite drink: milk, boba tea, coffee, water
 - The people keep different animals: bird, cat, fish, dog

1. Alice is the cat lover.
2. The person who likes milk is in the third house.
3. The person who uses a Huawei P50 is Bob.
4. The one who only drinks water is the bird keeper.
5. The cat lover is in the second house.
6. The boba tea drinker is the dog owner.
7. The person who uses a Google Pixel 6 is directly left of Carol.
8. The one who only drinks water is Carol.
9. Carol is the person who uses a OnePlus 9.

What is Name of the person who lives in House 1?? Provide only the name of the person as your final answer.
Answer: bob
Metadata: {'num_people': 4, 'num_characteristics': 4}

Example 2:
Question: This is a logic puzzle. There are 4 houses (numbered 1 on the left, 4 on the right), from the perspective of someone standing across the street from them. Each has a different person in them. They have different characteristics:
 - Each person has a unique name: alice, bob, arnold, carol
 - Each mother is accompanied by their child: alice, bella, billy, timothy
 - The people are of nationalities: brit, german, chinese, dane
 - Everyone has something different for lunch: soup, stir fry, grilled cheese, pizza

1. The British person is Arnold.
2. The person's child is named Alice is directly left of the person who loves the soup.
3. The person who loves stir fry is the person's child is named Bella.
4. The Chinese is Carol.
5. The German is the person's child is named Bella.
6. The person's child is named Bella is Bob.
7. The person who loves the soup is in the second house.
8. The person who loves the soup is the British person.
9. The person's child is named Alice is Carol.
10. The British person is directly left of the German.
11. The person who is the mother of Billy is the person who is a pizza lover.

What is Name of the person who lives in House 1?? Provide only the name of the person as your final answer.
Answer: carol
Metadata: {'num_people': 4, 'num_characteristics': 4}

Example 3:
Question: This is a logic puzzle. There are 4 houses (numbered 1 on the left, 4 on the right), from the perspective of someone standing across the street from them. Each has a different person in them. They have different characteristics:
 - Each person has a unique name: alice, arnold, bob, carol
 - Everyone has a different favorite cigar: pall mall, dunhill, blue master, prince
 - Everyone has something different for lunch: stir fry, grilled cheese, soup, pizza
 - Each person has a favorite color: blue, purple, brown, white

1. The person who loves white is the person who loves stir fry.
2. The person who loves brown is directly left of the Prince smoker.
3. The person who is a pizza lover and Arnold are next to each other.
4. The person partial to Pall Mall is the person who loves white.
5. Alice is the person who loves the soup.
6. The person partial to Pall Mall is directly left of the person who loves the soup.
7. The person who smokes Blue Master is directly left of the Dunhill smoker.
8. The Dunhill smoker is Bob.
9. The person who loves the soup is the person who loves blue.

What is Name of the person who lives in House 1?? Provide only the name of the person as your final answer.
Answer: carol
Metadata: {'num_people': 4, 'num_characteristics': 4}

````


