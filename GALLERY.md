# Reasoning Gym Dataset Gallery
This gallery shows examples from all available datasets using their default configurations.

## Available Datasets
- [advanced_geometry](#advanced_geometry)
- [aiw](#aiw)
- [arc_1d](#arc_1d)
- [base_conversion](#base_conversion)
- [basic_arithmetic](#basic_arithmetic)
- [bf](#bf)
- [caesar_cipher](#caesar_cipher)
- [calendar_arithmetic](#calendar_arithmetic)
- [chain_sum](#chain_sum)
- [color_cube_rotation](#color_cube_rotation)
- [countdown](#countdown)
- [course_schedule](#course_schedule)
- [family_relationships](#family_relationships)
- [figlet_font](#figlet_font)
- [fraction_simplification](#fraction_simplification)
- [game_of_life](#game_of_life)
- [gcd](#gcd)
- [gsm_symbolic](#gsm_symbolic)
- [intermediate_integration](#intermediate_integration)
- [largest_island](#largest_island)
- [lcm](#lcm)
- [leg_counting](#leg_counting)
- [letter_counting](#letter_counting)
- [letter_jumble](#letter_jumble)
- [maze](#maze)
- [mini_sudoku](#mini_sudoku)
- [n_queens](#n_queens)
- [number_filtering](#number_filtering)
- [number_sequence](#number_sequence)
- [number_sorting](#number_sorting)
- [palindrome](#palindrome)
- [polynomial_equations](#polynomial_equations)
- [prime_factorization](#prime_factorization)
- [propositional_logic](#propositional_logic)
- [quantum_lock](#quantum_lock)
- [rubiks_cube](#rubiks_cube)
- [sentence_reordering](#sentence_reordering)
- [simple_equations](#simple_equations)
- [simple_geometry](#simple_geometry)
- [simple_integration](#simple_integration)
- [spell_backward](#spell_backward)
- [sudoku](#sudoku)
- [syllogism](#syllogism)
- [time_intervals](#time_intervals)
- [tower_of_hanoi](#tower_of_hanoi)
- [word_ladder](#word_ladder)
- [group_anagrams](#group_anagrams)
- [word_sequence_reversal](#word_sequence_reversal)
- [word_sorting](#word_sorting)
- [zebra_puzzles](#zebra_puzzles)

## Dataset Examples
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
Question: In triangle ABC with coordinates A=(-7, -10), B=(-2, -3), and C=(-3, -6), find the measure (in degrees) of angle ABC.
Answer: 17.10°
Metadata: {'A': (-7, -10), 'B': (-2, -3), 'C': (-3, -6), 'angle_ABC_degrees': 17.10272896905237}

Example 2:
Question: For triangle with vertices A=(-1, -6), B=(4, 1), and C=(-7, 4), determine the orthocenter (intersection of altitudes).
Answer: (0.304, -1.217)
Metadata: {'A': (-1, -6), 'B': (4, 1), 'C': (-7, 4), 'orthocenter_exact': ('7/23', '-28/23'), 'orthocenter_approx': (0.30434782608695654, -1.2173913043478262)}

Example 3:
Question: Find the incircle radius of triangle ABC whose vertices are A=(6, 7), B=(-7, -5), and C=(2, -3).
Answer: 2.176
Metadata: {'A': (6, 7), 'B': (-7, -5), 'C': (2, -3), 'incircle_radius_exact': 'sqrt(-sqrt(29) + sqrt(85)/2 + sqrt(313)/2)*sqrt(-sqrt(313)/2 + sqrt(85)/2 + sqrt(29))*sqrt(-sqrt(85)/2 + sqrt(29) + sqrt(313)/2)/sqrt(sqrt(85)/2 + sqrt(29) + sqrt(313)/2)', 'incircle_radius_approx': 2.176123777286009}

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
Question: Convert the base-3 number 220020 to binary
Answer: 1010001110
Metadata: {'decimal_value': 654, 'source_base': 3, 'target_base': 2, 'source_repr': '220020', 'target_repr': '1010001110'}

Example 2:
Question: Convert the base-6 number 103 to base-13 (use lowercase letters a-z for digits above 9)
Answer: 30
Metadata: {'decimal_value': 39, 'source_base': 6, 'target_base': 13, 'source_repr': '103', 'target_repr': '30'}

Example 3:
Question: Convert the base-10 number 418 to base-13 (use lowercase letters a-z for digits above 9)
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
Question: -5 * -6 =
Answer: 30
Metadata: {'num_terms': 2, 'num_digits': 1, 'expression': '-5 * -6'}

Example 2:
Question: 965 / 5 =
Answer: 193
Metadata: {'num_terms': 2, 'num_digits': 3, 'expression': '965 / 5'}

Example 3:
Question: 0 + -2 + -4 * 0 * 3 =
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
Answer: onset
Metadata: {'bfit_code': '\nint main() {\n    print("onset");\n}\n', 'bf_program': '>[-]>[-]<>++++++++++[<+++++++++++>-]<+.-.+++++.--------------.+++++++++++++++.<'}

Example 2:
Question: This is a BF (Brainf*ck) computer program. What is the output? 

>[-]>[-]<>++++++++[<++++++++++++++>-]<.-----------.+++++++++++++.---------------.+++++.<
Answer: perch
Metadata: {'bfit_code': '\nint main() {\n    print("perch");\n}\n', 'bf_program': '>[-]>[-]<>++++++++[<++++++++++++++>-]<.-----------.+++++++++++++.---------------.+++++.<'}

Example 3:
Question: This is a BF (Brainf*ck) computer program. What is the output? 

>[-]>[-]<>+++++++++[<+++++++++++++>-]<.-------.----------.+.+++++++++++++.<
Answer: under
Metadata: {'bfit_code': '\nint main() {\n    print("under");\n}\n', 'bf_program': '>[-]>[-]<>+++++++++[<+++++++++++++>-]<.-------.----------.+.+++++++++++++.<'}

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
Question: Decrypt this Caesar cipher text: JNJUBUF ZPVS BTTPDJBUF XIPN J XBT DPNQMJNFOUJOH B NPNFOU BHP
Answer: IMITATE YOUR ASSOCIATE WHOM I WAS COMPLIMENTING A MOMENT AGO
Metadata: {'rotation': 1, 'cipher_text': 'JNJUBUF ZPVS BTTPDJBUF XIPN J XBT DPNQMJNFOUJOH B NPNFOU BHP', 'clear_text': 'IMITATE YOUR ASSOCIATE WHOM I WAS COMPLIMENTING A MOMENT AGO'}

Example 2:
Question: Decrypt this Caesar cipher text: PBSDJ XKZYVOYX CWSDR LYEQRD SD PYB K WOBO KXN YBSQSXKDON DOVOZRYXSM TYEBXKVSCW
Answer: FRITZ NAPOLEON SMITH BOUGHT IT FOR A MERE AND ORIGINATED TELEPHONIC JOURNALISM
Metadata: {'rotation': 10, 'cipher_text': 'PBSDJ XKZYVOYX CWSDR LYEQRD SD PYB K WOBO KXN YBSQSXKDON DOVOZRYXSM TYEBXKVSCW', 'clear_text': 'FRITZ NAPOLEON SMITH BOUGHT IT FOR A MERE AND ORIGINATED TELEPHONIC JOURNALISM'}

Example 3:
Question: Decrypt this Caesar cipher text: ZW PFLI JKFDRTY ZJ FLK FW ZK DLJK SV DVEUVU
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
Question: 4 + 3 =
Answer: 7
Metadata: {'num_terms': 2, 'num_digits': 1, 'expression': '4 + 3'}

Example 2:
Question: 812 + 880 =
Answer: 1692
Metadata: {'num_terms': 2, 'num_digits': 3, 'expression': '812 + 880'}

Example 3:
Question: 2 + 6 + 3 + 4 + 0 =
Answer: 15
Metadata: {'num_terms': 5, 'num_digits': 1, 'expression': '2 + 6 + 3 + 4 + 0'}

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
Answer: gold
Metadata: {'initial_state': {'top': 'orange', 'right': 'cyan', 'front': 'violet', 'left': 'pink', 'back': 'gray', 'bottom': 'gold'}, 'rotations': ['left', 'back', 'bottom'], 'target_side': 'left', 'num_rotations': 3}

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
Answer: 15 - 4 + 95 + 36 - 32 + 29
Metadata: {'numbers': [36, 29, 95, 32, 4, 15], 'target': 139, 'expression': '15 - 4 + 95 + 36 - 32 + 29'}

Example 2:
Question: Using the numbers 74, 48, 56, 66, create an expression that equals 132.
You can only use each number once.
Answer: 66 - 56 + 74 + 48
Metadata: {'numbers': [74, 48, 56, 66], 'target': 132, 'expression': '66 - 56 + 74 + 48'}

Example 3:
Question: Using the numbers 5, 41, 38, 81, 14, create an expression that equals 450.
You can only use each number once.
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

What is Isabella to Edward?
Answer: mother
Metadata: {'person1': 'Isabella', 'person2': 'Edward', 'relationship': 'mother', 'family_size': 4}

Example 2:
Question: Henry is married to Karen. They have a child called Sebastian. Sebastian is married to Eleanor.

What relation is Henry to Karen?
Answer: husband
Metadata: {'person1': 'Henry', 'person2': 'Karen', 'relationship': 'husband', 'family_size': 4}

Example 3:
Question: Liam is married to Nova. They have a child called Noah. Noah is married to Charlotte. They have a child called Patricia. Joseph is married to Lisa. They have a child called Charlotte.

What is Liam to Noah?
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
Question: Simplify the fraction $\frac{92}{524}$ to its lowest terms
Answer: $\frac{23}{131}$
Metadata: {'numerator': 92, 'denominator': 524, 'simplified_numerator': 23, 'simplified_denominator': 131, 'reduction_factor': 4, 'style': 'latex_frac'}

Example 2:
Question: Simplify the fraction $3600/26370$ to its lowest terms
Answer: $40/293$
Metadata: {'numerator': 3600, 'denominator': 26370, 'simplified_numerator': 40, 'simplified_denominator': 293, 'reduction_factor': 90, 'style': 'latex_inline'}

Example 3:
Question: Simplify the fraction 29330/37310 to its lowest terms
Answer: 419/533
Metadata: {'numerator': 29330, 'denominator': 37310, 'simplified_numerator': 419, 'simplified_denominator': 533, 'reduction_factor': 70, 'style': 'plain'}

````

### game_of_life
Generates Game of Life games with configurable parameters

Default configuration:
```python
grid_size_x = 20
grid_size_y = 20
filled_cells = 100
simulation_steps = 1
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: What will this Game of Life board look like after 1 steps of simulation?

[[0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0]
 [0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 1 0]
 [1 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 1 0 0]
 [0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 1 0 0]
 [0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 0]
 [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0]
 [1 1 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0]
 [0 0 1 0 0 0 0 1 0 0 0 0 1 1 0 0 1 0 0 1]
 [1 1 0 1 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 1 1]
 [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1]
 [0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1]
 [0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 1 1 0]
 [1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0]
 [1 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1]
 [0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0]
 [0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0]]
Answer: [[0 0 1 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 1 0]
 [0 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 1 1 0]
 [0 0 0 0 0 1 1 1 0 0 1 1 0 1 0 0 1 1 0 0]
 [0 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 1 1 1 0]
 [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1]
 [1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0]
 [0 0 0 0 0 0 1 1 1 0 0 0 1 1 0 0 0 0 0 1]
 [1 1 1 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0]
 [0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 1]
 [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1]
 [1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1]
 [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1]
 [1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0]
 [1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1]
 [0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
Metadata: {'grid_size_x': 20, 'grid_size_y': 20, 'filled_cells': 100, 'simulation_steps': 1}

Example 2:
Question: What will this Game of Life board look like after 1 steps of simulation?

[[1 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 1 0 0 0]
 [0 0 1 1 1 1 0 0 0 1 0 0 0 0 0 1 0 0 1 0]
 [0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1 0 0 0]
 [0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1 1 1]
 [0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]
 [0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0]
 [1 1 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0]
 [0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0]
 [0 1 0 0 1 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0]
 [0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1]
 [0 0 1 1 1 1 0 0 1 0 0 1 1 0 0 0 0 0 0 1]
 [0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1]
 [0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 1]
 [0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1]
 [0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0]
 [0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1]
 [0 1 0 0 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0]
 [0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0]]
Answer: [[0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 1]
 [0 0 1 1 0 1 1 0 0 1 1 0 0 0 0 1 0 1 0 0]
 [0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 1 0 0 1]
 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1]
 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1]
 [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
 [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0]
 [0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [1 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0]
 [1 0 1 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1]
 [1 0 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 1 0 0]
 [1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1]
 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]]
Metadata: {'grid_size_x': 20, 'grid_size_y': 20, 'filled_cells': 100, 'simulation_steps': 1}

Example 3:
Question: What will this Game of Life board look like after 1 steps of simulation?

[[0 0 1 1 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 1]
 [0 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 0 0 0 1]
 [0 0 0 1 0 0 0 0 1 1 1 0 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 1 0]
 [0 0 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 1 0 1]
 [0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0]
 [1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1]
 [0 0 0 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0]
 [0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]
 [0 0 1 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0]
 [0 0 1 1 1 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0]
 [0 0 1 1 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0]
 [1 0 0 1 1 0 1 0 0 1 0 0 0 0 0 1 1 0 0 0]
 [0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 1 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0]]
Answer: [[1 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1]
 [0 0 1 1 0 0 0 1 0 0 0 1 1 0 0 0 0 0 1 1]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 1 1 0 0 1 1 0 0 0 0 0 0 0]
 [0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0]
 [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1]
 [0 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1]
 [0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
 [0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0]
 [0 0 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 1 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 1 0 1 0 0 0 1 0 1 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 1 0]]
Metadata: {'grid_size_x': 20, 'grid_size_y': 20, 'filled_cells': 100, 'simulation_steps': 1}

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
Question: Find the Greatest Common Divisor (GCD) of these numbers: 26, 760
Answer: 2
Metadata: {'numbers': [26, 760], 'result': 2}

Example 2:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 688, 716
Answer: 4
Metadata: {'numbers': [688, 716], 'result': 4}

Example 3:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 297, 30
Answer: 3
Metadata: {'numbers': [297, 30], 'result': 3}

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
Question: There are 12 students playing basketball and twice that number playing volleyball. There are 17 boys and 17 girls playing table tennis. If each student only participates in one group, how many students are there in total?
Answer: 70
Metadata: {'difficulty': 1.0, 'answer_value': 70, 'answer_cot': 'There are 12 x 2 = 24 students playing volleyball.\nThere are 17 + 17 = 34 students playing table tennis.\nIn total there are 12 + 24 + 34 = 70 students.\n#### 70', 'variables': {'tennis_players': 12, 'volleyball_players': 24, 'soccer_boys': 17, 'soccer_girls': 17, 'total_soccer': 34, 'total_students': 70, 'sports': ['basketball', 'volleyball', 'table tennis']}}

Example 2:
Question: In Ms. Johnson's class of 100 students, 80% of the class are volleyball players. Out of the remaining class, 65% of the students are choir members or part of robotics club members. These 3 groups of students will need to leave early today to travel to an away performance. How many students are leaving early?
Answer: 93
Metadata: {'difficulty': 1.0, 'answer_value': 93, 'answer_cot': "80% of the 100 student class are volleyball players so that's 0.8*100 = 80 students\nThere are 100 students and 80 are volleyball players so that leaves 100-80 = 20 students\n65% of the remaining 20 students are part of robotics club members or choir members so that's 0.65*20 = 13 students\n80 students are volleyball players and 13 are part of robotics club members/choir members so 80+13 = 93 students will be leaving early\n#### 93", 'variables': {'teacher': 'Ms. Johnson', 'total_students': 100, 'percent_group1': 80, 'percent_group23': 65, 'group1': 'volleyball players', 'group2': 'choir members', 'group3': 'robotics club members', 'event': 'performance', 'group1_count': 80, 'group23_count': 13}}

Example 3:
Question: Olivia is trying to decide whether to do her business accounting herself or hire an accountant. If she does it herself, she'll be able to do 7 fewer hours of consulting work, losing €57/hour in missed income. The accountant charges €57. How much more money will she have if she hires the accountant?
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
Answer: -exp(3*x + 9) + C
Metadata: {'integrand': '-3*exp(3*x + 9)', 'problem_type': 'substitution', 'variable': 'x', 'type': 'exponential', 'expected_answer_expression': -exp(3*x + 9)}

Example 2:
Question: Evaluate the indefinite integral: ∫ -6*sin(2*X + 10)*cos(2*X + 10)**4 dx
Answer: 3*cos(2*X + 10)**5/5 + C
Metadata: {'integrand': '-6*sin(2*X + 10)*cos(2*X + 10)**4', 'problem_type': 'substitution', 'variable': 'X', 'type': 'trigonometric', 'expected_answer_expression': 3*cos(2*X + 10)**5/5}

Example 3:
Question: Find the indefinite integral: ∫ 2*asin(x) dx
Answer: 2*Integral(asin(x), x) + C
Metadata: {'integrand': '2*asin(x)', 'problem_type': 'by_parts', 'variable': 'x', 'type': 'log_inverse_trig', 'expected_answer_expression': 2*Integral(asin(x), x)}

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
min_animals = 2
max_animals = 5
max_instances = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: How many legs are there in total if you have 1 sea slug, 1 deer?
Answer: 4
Metadata: {'animals': {'sea slug': 1, 'deer': 1}, 'total_legs': 4}

Example 2:
Question: How many legs are there in total if you have 2 sheeps, 2 dogs?
Answer: 16
Metadata: {'animals': {'sheep': 2, 'dog': 2}, 'total_legs': 16}

Example 3:
Question: How many legs are there in total if you have 1 crab, 2 lobsters, 1 human, 1 cow, 1 bee?
Answer: 42
Metadata: {'animals': {'crab': 1, 'lobster': 2, 'human': 1, 'cow': 1, 'bee': 1}, 'total_legs': 42}

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
Question: Unscramble these words: ew hsall eb ebla ot puodrce
Answer: we shall be able to produce
Metadata: {'num_words': 6, 'corruption_level': 0.12000860417813355, 'scrambled_words': ['ew', 'hsall', 'eb', 'ebla', 'ot', 'puodrce'], 'original_words': ['we', 'shall', 'be', 'able', 'to', 'produce']}

Example 2:
Question: Unscramble these words: ni oiurnalmsj Well Cahs
Answer: in journalism Well Cash
Metadata: {'num_words': 4, 'corruption_level': 0.3288673442377109, 'scrambled_words': ['ni', 'oiurnalmsj', 'Well', 'Cahs'], 'original_words': ['in', 'journalism', 'Well', 'Cash']}

Example 3:
Question: Unscramble these words: dear rchAdbali keep no nSice yrstyedae atnhks ot oyu rheet si a gain fo sucrbbisesr rM
Answer: dear Archibald keep on Since yesterday thanks to you there is a gain of subscribers Mr
Metadata: {'num_words': 16, 'corruption_level': 0.516016391169858, 'scrambled_words': ['dear', 'rchAdbali', 'keep', 'no', 'nSice', 'yrstyedae', 'atnhks', 'ot', 'oyu', 'rheet', 'si', 'a', 'gain', 'fo', 'sucrbbisesr', 'rM'], 'original_words': ['dear', 'Archibald', 'keep', 'on', 'Since', 'yesterday', 'thanks', 'to', 'you', 'there', 'is', 'a', 'gain', 'of', 'subscribers', 'Mr']}

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
Question: Solve this 4x4 Mini Sudoku puzzle:
_ _ _ _
_ _ _ _
_ 1 3 _
_ 4 _ 1
Answer: 4 2 1 3
1 3 4 2
2 1 3 4
3 4 2 1
Metadata: {'puzzle': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 3, 0], [0, 4, 0, 1]], 'solution': [[4, 2, 1, 3], [1, 3, 4, 2], [2, 1, 3, 4], [3, 4, 2, 1]], 'num_empty': 12}

Example 2:
Question: Solve this 4x4 Mini Sudoku puzzle:
3 _ _ _
_ _ 4 _
4 2 _ _
_ _ _ 4
Answer: 3 4 1 2
2 1 4 3
4 2 3 1
1 3 2 4
Metadata: {'puzzle': [[3, 0, 0, 0], [0, 0, 4, 0], [4, 2, 0, 0], [0, 0, 0, 4]], 'solution': [[3, 4, 1, 2], [2, 1, 4, 3], [4, 2, 3, 1], [1, 3, 2, 4]], 'num_empty': 11}

Example 3:
Question: Solve this 4x4 Mini Sudoku puzzle:
_ _ _ _
1 3 4 _
3 1 2 4
4 _ _ _
Answer: 2 4 1 3
1 3 4 2
3 1 2 4
4 2 3 1
Metadata: {'puzzle': [[0, 0, 0, 0], [1, 3, 4, 0], [3, 1, 2, 4], [4, 0, 0, 0]], 'solution': [[2, 4, 1, 3], [1, 3, 4, 2], [3, 1, 2, 4], [4, 2, 3, 1]], 'num_empty': 8}

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
Question: Solve this N Queens puzzle:
_ _ _ _ _ _ Q _
_ Q _ _ _ _ _ _
_ _ _ Q _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ Q
_ _ _ _ Q _ _ _
_ _ Q _ _ _ _ _
_ _ _ _ _ Q _ _

The board size is 8x8 and your job is to place 1 queen(s) on the board such that no two queens attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

Place a queen by replacing an underscore (_) with a Q.

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
Question: Solve this N Queens puzzle:
_ Q _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ Q _ _
_ _ _ _ _ _ _ Q
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ Q _
_ _ _ _ Q _ _ _

The board size is 8x8 and your job is to place 3 queen(s) on the board such that no two queens attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

Place a queen by replacing an underscore (_) with a Q.

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
Question: Solve this N Queens puzzle:
_ _ _ _ _ _ _ _
_ Q _ _ _ _ _ _
_ _ _ _ _ _ _ _
Q _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ Q _ _

The board size is 8x8 and your job is to place 5 queen(s) on the board such that no two queens attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

Place a queen by replacing an underscore (_) with a Q.

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
Answer: ['-51.0', '47.2942', '-82.612']
Metadata: {'original_numbers': ['-95.00', '-51.0', '47.2942', '-82.612'], 'filter_value': '-90', 'operation': 'keep_larger', 'result': ['-51.0', '47.2942', '-82.612']}

Example 2:
Question: Remove all numbers larger than 18.236 in this list: ['-42.8', '91.88', '34']
Answer: ['-42.8']
Metadata: {'original_numbers': ['-42.8', '91.88', '34'], 'filter_value': '18.236', 'operation': 'remove_larger', 'result': ['-42.8']}

Example 3:
Question: Keep all numbers larger than 19.8962 in this list: ['4', '-64.7', '-42.1', '-77', '-79.9640', '37.76', '38.702', '18.20', '-28.34']
Answer: ['37.76', '38.702']
Metadata: {'original_numbers': ['4', '-64.7', '-42.1', '-77', '-79.9640', '37.76', '38.702', '18.20', '-28.34'], 'filter_value': '19.8962', 'operation': 'keep_larger', 'result': ['37.76', '38.702']}

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
Answer: ['-80', '-72', '-51', '48']
Metadata: {'original_numbers': ['48', '-51', '-72', '-80'], 'direction': 'ascending', 'sorted_numbers': ['-80', '-72', '-51', '48']}

Example 2:
Question: Sort these numbers in ascending order: 39.2, -71.2, -7.5
Answer: ['-71.2', '-7.5', '39.2']
Metadata: {'original_numbers': ['39.2', '-71.2', '-7.5'], 'direction': 'ascending', 'sorted_numbers': ['-71.2', '-7.5', '39.2']}

Example 3:
Question: Sort these numbers in descending order: 8.39, 72.41, -64.67, -54.97, -94.18, -76.67, -98.24, -68.66, 2.74
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
Question: Rearrange these letters to form a palindrome. A palindrome is a word, phrase, or sequence that reads the same forward and backward.

For example, if the letters are: a, a, b — a valid palindrome is: aba.

Your letters: h, a, h, a

What palindrome can you form from these letters?
Answer: ahha
Metadata: {'letters': ['h', 'a', 'h', 'a'], 'generated_palindrome': 'ahha'}

Example 2:
Question: Rearrange these letters to form a palindrome. A palindrome is a word, phrase, or sequence that reads the same forward and backward.

For example, if the letters are: a, a, b — a valid palindrome is: aba.

Your letters: h, y, h

What palindrome can you form from these letters?
Answer: hyh
Metadata: {'letters': ['h', 'y', 'h'], 'generated_palindrome': 'hyh'}

Example 3:
Question: Rearrange these letters to form a palindrome. A palindrome is a word, phrase, or sequence that reads the same forward and backward.

For example, if the letters are: a, a, b — a valid palindrome is: aba.

Your letters: n, j, n, j, d, j, s, s, d

What palindrome can you form from these letters?
Answer: nsdjjjdsn
Metadata: {'letters': ['n', 'j', 'n', 'j', 'd', 'j', 's', 's', 'd'], 'generated_palindrome': 'nsdjjjdsn'}

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
Question: Find the real value(s) of u in the equation: -127*u = 0
Answer: [0.0]
Metadata: {'polynomial_expr': '-127*u', 'variable': 'u', 'degree': 1, 'real_solutions': [0.0]}

Example 2:
Question: Determine the real value(s) of b tha satisfies: 86*b**2 - 2*b - 13 = 0
Answer: [-0.3773425275273891, 0.4005983414808775]
Metadata: {'polynomial_expr': '86*b**2 - 2*b - 13', 'variable': 'b', 'degree': 2, 'real_solutions': [-0.3773425275273891, 0.4005983414808775]}

Example 3:
Question: Determine the real value(s) of n tha satisfies: 71*n**3 - 2*n - 29 = 0
Answer: [0.7546129960163634]
Metadata: {'polynomial_expr': '71*n**3 - 2*n - 29', 'variable': 'n', 'degree': 3, 'real_solutions': [0.7546129960163634]}

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
Question: Given:
1. R
2. Q
What can we conclude?
Answer: (P ∨ Q)
Metadata: {'premises': ['R', 'Q'], 'variables': ['P', 'Q', 'R', 'S'], 'complexity': 3}

Example 2:
Question: Given:
1. ((Q → P) ∨ (Q → P))
2. ((Q ↔ Q) → (P → P))
3. P
What can we conclude?
Answer: (P → P)
Metadata: {'premises': ['((Q → P) ∨ (Q → P))', '((Q ↔ Q) → (P → P))', 'P'], 'variables': ['P', 'Q'], 'complexity': 3}

Example 3:
Question: Given:
1. ((Q ∨ P) ∧ ¬P)
2. P
3. ((P ∧ R) ∧ ¬R)
4. ((Q ↔ R) → ¬Q)
What can we conclude?
Answer: (Q ∧ Q)
Metadata: {'premises': ['((Q ∨ P) ∧ ¬P)', 'P', '((P ∧ R) ∧ ¬R)', '((Q ↔ R) → ¬Q)'], 'variables': ['P', 'Q', 'R'], 'complexity': 3}

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
Question: Given a convex polygon with 3 sides, its first 2 interior angles are: 16.0°, 80.0°. What is the measure of the remaining interior angle (in degrees)?
Answer: 84
Metadata: {'n_sides': 3, 'known_angles': [16.0, 80.0], 'sum_of_known_angles': 96.0, 'missing_angle_raw': 84.0, 'missing_angle_rounded': 84, 'total_interior_sum': 180}

Example 2:
Question: A convex polygon has 3 sides. The measures of the first 2 interior angles are: 83.0°, 46.0°. Find the measure of the last interior angle.
Answer: 51
Metadata: {'n_sides': 3, 'known_angles': [83.0, 46.0], 'sum_of_known_angles': 129.0, 'missing_angle_raw': 51.0, 'missing_angle_rounded': 51, 'total_interior_sum': 180}

Example 3:
Question: Given a convex polygon with 6 sides, its first 5 interior angles are: 143.0°, 148.0°, 39.0°, 55.0°, 107.0°. What is the measure of the remaining interior angle (in degrees)?
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
Answer: 10*x**7 + 4*x**3/5 + C
Metadata: {'integrand': '70*x**6 + 12*x**2/5', 'variable': 'x', 'expected_answer_expression': 10*x**7 + 4*x**3/5}

Example 2:
Question: Find the indefinite integral: ∫ 49*x**6/10 + 48*x**5 - 4*x - 10/9 dx
Answer: 7*x**7/10 + 8*x**6 - 2*x**2 - 10*x/9 + C
Metadata: {'integrand': '49*x**6/10 + 48*x**5 - 4*x - 10/9', 'variable': 'x', 'expected_answer_expression': 7*x**7/10 + 8*x**6 - 2*x**2 - 10*x/9}

Example 3:
Question: Find the indefinite integral: ∫ -28*X**3 + 8*X dx
Answer: -7*X**4 + 4*X**2 + C
Metadata: {'integrand': '-28*X**3 + 8*X', 'variable': 'X', 'expected_answer_expression': -7*X**4 + 4*X**2}

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
4 _ _ _ 5 2 _ 3 _
_ _ 3 4 6 _ _ _ _
6 1 2 _ _ 8 4 _ _
1 _ _ _ _ _ 7 9 5
3 _ _ 7 1 _ _ 2 6
7 _ _ 5 _ _ _ _ 3
2 _ _ _ 7 5 _ _ _
_ 3 _ _ 4 1 _ _ _
_ _ _ 2 8 _ _ _ 4
Answer: 4 7 8 1 5 2 6 3 9
5 9 3 4 6 7 2 8 1
6 1 2 3 9 8 4 5 7
1 2 4 8 3 6 7 9 5
3 5 9 7 1 4 8 2 6
7 8 6 5 2 9 1 4 3
2 4 1 9 7 5 3 6 8
8 3 5 6 4 1 9 7 2
9 6 7 2 8 3 5 1 4
Metadata: {'puzzle': [[4, 0, 0, 0, 5, 2, 0, 3, 0], [0, 0, 3, 4, 6, 0, 0, 0, 0], [6, 1, 2, 0, 0, 8, 4, 0, 0], [1, 0, 0, 0, 0, 0, 7, 9, 5], [3, 0, 0, 7, 1, 0, 0, 2, 6], [7, 0, 0, 5, 0, 0, 0, 0, 3], [2, 0, 0, 0, 7, 5, 0, 0, 0], [0, 3, 0, 0, 4, 1, 0, 0, 0], [0, 0, 0, 2, 8, 0, 0, 0, 4]], 'solution': [[4, 7, 8, 1, 5, 2, 6, 3, 9], [5, 9, 3, 4, 6, 7, 2, 8, 1], [6, 1, 2, 3, 9, 8, 4, 5, 7], [1, 2, 4, 8, 3, 6, 7, 9, 5], [3, 5, 9, 7, 1, 4, 8, 2, 6], [7, 8, 6, 5, 2, 9, 1, 4, 3], [2, 4, 1, 9, 7, 5, 3, 6, 8], [8, 3, 5, 6, 4, 1, 9, 7, 2], [9, 6, 7, 2, 8, 3, 5, 1, 4]], 'num_empty': 48}

Example 2:
Question: Solve this Sudoku puzzle:
_ _ _ 1 3 2 6 4 5
_ 4 _ 7 _ _ _ 9 1
_ _ 1 8 _ 9 _ _ _
_ 8 9 _ _ _ 7 5 4
_ 3 _ 4 _ 1 9 8 _
4 6 _ 5 9 _ 1 2 3
5 _ 4 9 1 7 3 _ _
9 7 6 _ 8 4 5 1 _
8 _ 3 _ _ _ 4 7 _
Answer: 7 9 8 1 3 2 6 4 5
3 4 2 7 5 6 8 9 1
6 5 1 8 4 9 2 3 7
1 8 9 6 2 3 7 5 4
2 3 5 4 7 1 9 8 6
4 6 7 5 9 8 1 2 3
5 2 4 9 1 7 3 6 8
9 7 6 3 8 4 5 1 2
8 1 3 2 6 5 4 7 9
Metadata: {'puzzle': [[0, 0, 0, 1, 3, 2, 6, 4, 5], [0, 4, 0, 7, 0, 0, 0, 9, 1], [0, 0, 1, 8, 0, 9, 0, 0, 0], [0, 8, 9, 0, 0, 0, 7, 5, 4], [0, 3, 0, 4, 0, 1, 9, 8, 0], [4, 6, 0, 5, 9, 0, 1, 2, 3], [5, 0, 4, 9, 1, 7, 3, 0, 0], [9, 7, 6, 0, 8, 4, 5, 1, 0], [8, 0, 3, 0, 0, 0, 4, 7, 0]], 'solution': [[7, 9, 8, 1, 3, 2, 6, 4, 5], [3, 4, 2, 7, 5, 6, 8, 9, 1], [6, 5, 1, 8, 4, 9, 2, 3, 7], [1, 8, 9, 6, 2, 3, 7, 5, 4], [2, 3, 5, 4, 7, 1, 9, 8, 6], [4, 6, 7, 5, 9, 8, 1, 2, 3], [5, 2, 4, 9, 1, 7, 3, 6, 8], [9, 7, 6, 3, 8, 4, 5, 1, 2], [8, 1, 3, 2, 6, 5, 4, 7, 9]], 'num_empty': 34}

Example 3:
Question: Solve this Sudoku puzzle:
_ _ 1 2 3 _ _ _ 9
3 _ _ 1 8 5 6 7 2
_ _ _ 4 9 6 1 _ _
1 _ 5 7 _ _ 9 2 _
_ 4 _ _ 5 9 7 1 6
9 _ 6 _ 1 _ 4 5 3
_ _ 3 9 7 _ 2 8 4
_ _ 2 6 4 _ _ 9 1
_ 1 _ 5 2 8 3 _ _
Answer: 5 6 1 2 3 7 8 4 9
3 9 4 1 8 5 6 7 2
8 2 7 4 9 6 1 3 5
1 3 5 7 6 4 9 2 8
2 4 8 3 5 9 7 1 6
9 7 6 8 1 2 4 5 3
6 5 3 9 7 1 2 8 4
7 8 2 6 4 3 5 9 1
4 1 9 5 2 8 3 6 7
Metadata: {'puzzle': [[0, 0, 1, 2, 3, 0, 0, 0, 9], [3, 0, 0, 1, 8, 5, 6, 7, 2], [0, 0, 0, 4, 9, 6, 1, 0, 0], [1, 0, 5, 7, 0, 0, 9, 2, 0], [0, 4, 0, 0, 5, 9, 7, 1, 6], [9, 0, 6, 0, 1, 0, 4, 5, 3], [0, 0, 3, 9, 7, 0, 2, 8, 4], [0, 0, 2, 6, 4, 0, 0, 9, 1], [0, 1, 0, 5, 2, 8, 3, 0, 0]], 'solution': [[5, 6, 1, 2, 3, 7, 8, 4, 9], [3, 9, 4, 1, 8, 5, 6, 7, 2], [8, 2, 7, 4, 9, 6, 1, 3, 5], [1, 3, 5, 7, 6, 4, 9, 2, 8], [2, 4, 8, 3, 5, 9, 7, 1, 6], [9, 7, 6, 8, 1, 2, 4, 5, 3], [6, 5, 3, 9, 7, 1, 2, 8, 4], [7, 8, 2, 6, 4, 3, 5, 9, 1], [4, 1, 9, 5, 2, 8, 3, 6, 7]], 'num_empty': 33}

````

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
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Consider these statements:
1. No students are humans
2. No humans are chefs

Does it logically follow that:
No students are chefs?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'No students are humans', 'premise2': 'No humans are chefs', 'conclusion': 'No students are chefs', 'is_valid': True}

Example 2:
Question: Consider these statements:
1. Some children are not animals
2. Some animals are doctors

Does it logically follow that:
All children are doctors?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'Some children are not animals', 'premise2': 'Some animals are doctors', 'conclusion': 'All children are doctors', 'is_valid': True}

Example 3:
Question: Consider these statements:
1. All butterflies are tigers
2. No tigers are whales

Does it logically follow that:
Some butterflies are not whales?
(Answer Yes or No)
Answer: No
Metadata: {'premise1': 'All butterflies are tigers', 'premise2': 'No tigers are whales', 'conclusion': 'Some butterflies are not whales', 'is_valid': False}

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
Metadata: {'task_type': 'time', 'start_time': datetime.datetime(2025, 2, 5, 9, 44), 'end_time': datetime.datetime(2025, 2, 5, 12, 22), 'format': '%H:%M', 'expected_format': 'HH:MM'}

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
size = 50
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
Answer: ['Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 3 from Peg 1 to Peg 3', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 4 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 3 from Peg 3 to Peg 2', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 5 from Peg 1 to Peg 3', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 3 from Peg 2 to Peg 1', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 4 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 3 from Peg 1 to Peg 3', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 6 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 3 from Peg 3 to Peg 2', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 4 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 3 from Peg 2 to Peg 1', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 5 from Peg 3 to Peg 2', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 3 from Peg 1 to Peg 3', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 2 from Peg 2 to Peg 3', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 4 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2', 'Move disk 2 from Peg 3 to Peg 1', 'Move disk 1 from Peg 2 to Peg 1', 'Move disk 3 from Peg 3 to Peg 2', 'Move disk 1 from Peg 1 to Peg 3', 'Move disk 2 from Peg 1 to Peg 2', 'Move disk 1 from Peg 3 to Peg 2']
Metadata: {'num_disks': 6, 'num_pegs': 3, 'start_peg': 1, 'target_peg': 2, 'auxiliary_pegs': [3], 'solution_length': 63}

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
Answer: HAND,HARD,HERD,HEED,FEED,FLED,FLEE,GLEE
Metadata: {'start_word': 'HAND', 'end_word': 'GLEE', 'word_length': 4, 'chain_length': 8}

Example 2:
Question: Transform the word ladder 'JAZZ' to 'DORM' by changing one letter at a time.
Answer: JAZZ,JIZZ,FIZZ,FUZZ,FUZE,FAZE,FARE,FORE,FORM,DORM
Metadata: {'start_word': 'JAZZ', 'end_word': 'DORM', 'word_length': 4, 'chain_length': 10}

Example 3:
Question: Transform the word ladder 'SNOG' to 'SUQS' by changing one letter at a time.
Answer: SNOG,SNOW,SHOW,SHEW,SHES,SUES,SUQS
Metadata: {'start_word': 'SNOG', 'end_word': 'SUQS', 'word_length': 4, 'chain_length': 7}

````


### group_anagrams

Group anagrams together in a list of words. 

Default configuration
```python
anagram_groups: int = 10  # Groups of anagrams present in the input
max_words_per_group: int = 5  # Maximum number of words in a single anagram group
```

Example tasks:
```
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

--------------------------------------------------

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

--------------------------------------------------

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

--------------------------------------------------
```



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
Question: Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list:
DIRECT, given, exclaims, dreaming
Answer: DIRECT, dreaming, exclaims, given
Metadata: {'original_words': ['DIRECT', 'given', 'exclaims', 'dreaming'], 'transformed_words': ['DIRECT', 'given', 'exclaims', 'dreaming'], 'direction': 'ascending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['DIRECT', 'dreaming', 'exclaims', 'given']}

Example 2:
Question: Sort these words in descending order (using ASCII/Unicode ordering) and return them as a comma-separated list:
heat, begun, sometimes
Answer: sometimes, heat, begun
Metadata: {'original_words': ['heat', 'begun', 'sometimes'], 'transformed_words': ['heat', 'begun', 'sometimes'], 'direction': 'descending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['sometimes', 'heat', 'begun']}

Example 3:
Question: Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list:
violates, yes, already, completing, pages, duty, his, EXPRESS, duly
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

What is Name of the person who lives in House 1?
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

What is Name of the person who lives in House 1?
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

What is Name of the person who lives in House 1?
Answer: carol
Metadata: {'num_people': 4, 'num_characteristics': 4}

````


