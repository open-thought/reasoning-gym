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
- [word_ladder](#word-ladder)
- [word_sequence_reversal](#word-sequence-reversal)
- [word_sorting](#word-sorting)

## Dataset Examples
### base_conversion {base-conversion}
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
Question: Convert the base-5 number 21e to base-3
Answer: 21e
Metadata: {'decimal_value': 542, 'source_base': 5, 'target_base': 3, 'source_repr': '21e', 'target_repr': '21e'}

Example 2:
Question: Convert the base-6 number f7 to base-3
Answer: f7
Metadata: {'decimal_value': 247, 'source_base': 6, 'target_base': 3, 'source_repr': 'f7', 'target_repr': 'f7'}

Example 3:
Question: Convert the base-4 number 25d to base-14 (use lowercase letters a-z for digits above 9)
Answer: 25d
Metadata: {'decimal_value': 605, 'source_base': 4, 'target_base': 14, 'source_repr': '25d', 'target_repr': '25d'}

```

### basic_arithmetic {basic-arithmetic}
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
Question: 7035 / 1005 =
Answer: 7
Metadata: {'num_terms': 2, 'num_digits': 4, 'expression': '7035 / 1005'}

Example 2:
Question: -743 + 475 + 719 + -155 - -768 =
Answer: 1064
Metadata: {'num_terms': 5, 'num_digits': 3, 'expression': '-743 + 475 + 719 + -155 - -768'}

Example 3:
Question: 898 / 2 + 357 + -664 * -496 =
Answer: 330150
Metadata: {'num_terms': 5, 'num_digits': 3, 'expression': '898 / 2 + 357 + -664 * -496'}

```

### caesar_cipher {caesar-cipher}
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
Question: Decrypt this Caesar cipher text: B EJTDVTTJPO XBT HPJOH PO XIFO IF FOUFSFE
Answer: A DISCUSSION WAS GOING ON WHEN HE ENTERED
Metadata: {'rotation': 1, 'cipher_text': 'B EJTDVTTJPO XBT HPJOH PO XIFO IF FOUFSFE', 'clear_text': 'A DISCUSSION WAS GOING ON WHEN HE ENTERED'}

Example 2:
Question: Decrypt this Caesar cipher text: Q IU BQZML YCQBM BQZML LW GWC VWB BPQVS BPIB I JIBP EWCTL ZMNZMAP
Answer: I AM TIRED QUITE TIRED DO YOU NOT THINK THAT A BATH WOULD REFRESH
Metadata: {'rotation': 8, 'cipher_text': 'Q IU BQZML YCQBM BQZML LW GWC VWB BPQVS BPIB I JIBP EWCTL ZMNZMAP', 'clear_text': 'I AM TIRED QUITE TIRED DO YOU NOT THINK THAT A BATH WOULD REFRESH'}

Example 3:
Question: Decrypt this Caesar cipher text: Y IGLE GQ FC
Answer: A KING IS HE
Metadata: {'rotation': 24, 'cipher_text': 'Y IGLE GQ FC', 'clear_text': 'A KING IS HE'}

```

### chain_sum {chain-sum}
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
Question: 47 - 18 - 85 + 54 =
Answer: -2
Metadata: {'num_terms': 4, 'num_digits': 2, 'expression': '47 - 18 - 85 + 54'}

Example 2:
Question: 52 + 23 - 88 + 78 + 92 - 54 =
Answer: 103
Metadata: {'num_terms': 6, 'num_digits': 2, 'expression': '52 + 23 - 88 + 78 + 92 - 54'}

Example 3:
Question: 46 + 76 + 75 + 46 + 70 - 88 =
Answer: 225
Metadata: {'num_terms': 6, 'num_digits': 2, 'expression': '46 + 76 + 75 + 46 + 70 - 88'}

```

### color_cube_rotation {color-cube-rotation}
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
- a white top side
- a silver right side
- a brown front side
- a violet left side
- a pink back side
- a purple bottom side

The cube is rotated so that the side which was before at the left is now at the top.

Then the cube is rotated to bring the bottom side to the top.

Now the cube is rotated to place its back side at the top.

What is now the color of the top side of the cube?
Answer: brown
Metadata: {'initial_state': {'top': 'white', 'right': 'silver', 'front': 'brown', 'left': 'violet', 'back': 'pink', 'bottom': 'purple'}, 'rotations': ['left', 'bottom', 'back'], 'target_side': 'top', 'num_rotations': 3}

Example 2:
Question: A cube has:
- a violet top side
- a green right side
- a white front side
- a blue left side
- a gold back side
- a yellow bottom side

The cube is rotated so that the side which was before at the right is now at the top.

What is now the color of the bottom side of the cube?
Answer: blue
Metadata: {'initial_state': {'top': 'violet', 'right': 'green', 'front': 'white', 'left': 'blue', 'back': 'gold', 'bottom': 'yellow'}, 'rotations': ['right'], 'target_side': 'bottom', 'num_rotations': 1}

Example 3:
Question: A cube has:
- a magenta top side
- a green right side
- a brown front side
- a yellow left side
- a silver back side
- a violet bottom side

The cube is rotated so that the side which was before at the front is now at the top.

What is now the color of the bottom side of the cube?
Answer: silver
Metadata: {'initial_state': {'top': 'magenta', 'right': 'green', 'front': 'brown', 'left': 'yellow', 'back': 'silver', 'bottom': 'violet'}, 'rotations': ['front'], 'target_side': 'bottom', 'num_rotations': 1}

```

### countdown {countdown}
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
Question: Using the numbers 4, 15, 90, 49, 15, create an expression that equals 275.
You can only use each number once.
Answer: 49*90/15 - 15 - 4
Metadata: {'numbers': [4, 15, 90, 49, 15], 'target': 275, 'expression': '49*90/15 - 15 - 4'}

Example 2:
Question: Calculate 237 using the numbers 32, 56, 64, 23, 3, 100.
Each number may be used at most once.
Answer: 100*3 - 64 - 23 - 32 + 56
Metadata: {'numbers': [32, 56, 64, 23, 3, 100], 'target': 237, 'expression': '100*3 - 64 - 23 - 32 + 56'}

Example 3:
Question: Find a way to make 241 using some or all of these numbers: 87, 85, 82, 13.
Each number can only be used once.
Answer: 85 + 82 - 13 + 87
Metadata: {'numbers': [87, 85, 82, 13], 'target': 241, 'expression': '85 + 82 - 13 + 87'}

```

### family_relationships {family-relationships}
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
Question: Charles is married to Jessica. They have a child called Atlas. Atlas is married to Amelia. They have children called Patricia and Nova. David is married to Ava. They have a child called Amelia.

How is Jessica related to Nova?
Answer: grandmother
Metadata: {'person1': 'Jessica', 'person2': 'Nova', 'relationship': 'grandmother', 'family_size': 8}

Example 2:
Question: David is married to Charlotte. They have a child called Lucas. Lucas is married to Victoria. They have children called James and Abigail.

What is Victoria to Abigail?
Answer: mother
Metadata: {'person1': 'Victoria', 'person2': 'Abigail', 'relationship': 'mother', 'family_size': 6}

Example 3:
Question: Mason is married to Amelia. They have a child called James. James is married to Grace. They have a child called Abigail.

What relation is James to Amelia?
Answer: son
Metadata: {'person1': 'James', 'person2': 'Amelia', 'relationship': 'son', 'family_size': 5}

```

### figlet_font {figlet-font}
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
Question: What word does this say?

 __          _
(_    |_|   / \   | |   |\|
__)   | |   \_/   |^|   | |

Answer: SHOWN
Metadata: {'font': 'bigfig', 'space_letters': True}

Example 2:
Question: What word does this say?

  #####  ### ###     ##    ######    ######
 ## ###  ##   ##   #####   ###  ##  ######
##       ##   ##   ## ###  ##   ##     ##
##       #######  ##   ##  ##  ##      ##
##       ##   ##  #######  #####       ##
#####    ##   ##  ##  ##    ## ##      ##
 #####    #    #  #   #     ##  ##      #


Answer: CHART
Metadata: {'font': 'future_6', 'space_letters': True}

Example 3:
Question: Please read the following figlet font:

.dP"Y8     88  88     888888        db        88""Yb
`Ybo."     88  88     88__         dPYb       88__dP
o.`Y8b     888888     88""        dP__Yb      88"Yb
8bodP'     88  88     888888     dP""""Yb     88  Yb

Answer: SHEAR
Metadata: {'font': '4max', 'space_letters': True}

```

### fraction_simplification {fraction-simplification}
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
Question: Simplify the fraction $12054/36848$ to its lowest terms
Answer: $123/376$
Metadata: {'numerator': 12054, 'denominator': 36848, 'simplified_numerator': 123, 'simplified_denominator': 376, 'reduction_factor': 98, 'style': 'latex_inline'}

Example 2:
Question: Simplify the fraction 1218/28275 to its lowest terms
Answer: 14/325
Metadata: {'numerator': 1218, 'denominator': 28275, 'simplified_numerator': 14, 'simplified_denominator': 325, 'reduction_factor': 87, 'style': 'plain'}

Example 3:
Question: Simplify the fraction 21902/24111 to its lowest terms
Answer: 466/513
Metadata: {'numerator': 21902, 'denominator': 24111, 'simplified_numerator': 466, 'simplified_denominator': 513, 'reduction_factor': 47, 'style': 'plain'}

```

### gcd {gcd}
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
Question: Find the Greatest Common Divisor (GCD) of these numbers: 384, 414
Answer: 6
Metadata: {'numbers': [384, 414], 'result': 6}

Example 2:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 298, 803
Answer: 1
Metadata: {'numbers': [298, 803], 'result': 1}

Example 3:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 846, 550
Answer: 2
Metadata: {'numbers': [846, 550], 'result': 2}

```

### lcm {lcm}
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
Question: Find the Least Common Multiple (LCM) of these numbers: 33, 84
Answer: 924
Metadata: {'numbers': [33, 84], 'result': 924}

Example 2:
Question: Find the Least Common Multiple (LCM) of these numbers: 16, 23
Answer: 368
Metadata: {'numbers': [16, 23], 'result': 368}

Example 3:
Question: Find the Least Common Multiple (LCM) of these numbers: 66, 88
Answer: 264
Metadata: {'numbers': [66, 88], 'result': 264}

```

### leg_counting {leg-counting}
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
Question: How many legs are there in total if you have 2 scorpions, 3 sea slugs, 2 cockroachs, 2 fireflys?
Answer: 40
Metadata: {'animals': {'scorpion': 2, 'sea slug': 3, 'cockroach': 2, 'firefly': 2}, 'total_legs': 40}

Example 2:
Question: How many legs are there in total if you have 2 shrimps, 2 deers?
Answer: 28
Metadata: {'animals': {'shrimp': 2, 'deer': 2}, 'total_legs': 28}

Example 3:
Question: How many legs are there in total if you have 1 beetle, 3 spiders, 1 jellyfish?
Answer: 30
Metadata: {'animals': {'beetle': 1, 'spider': 3, 'jellyfish': 1}, 'total_legs': 30}

```

### letter_counting {letter-counting}
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
Question: How many times does the letter "s" appear in the text: "a varied assortment is always in readiness A subscription"?
Answer: 8
Metadata: {'span_length': 9, 'target_letter': 's', 'span': ['a', 'varied', 'assortment', 'is', 'always', 'in', 'readiness', 'A', 'subscription']}

Example 2:
Question: How many times does the letter "c" appear in the text: "exclaims every one present Yes answers"?
Answer: 1
Metadata: {'span_length': 6, 'target_letter': 'c', 'span': ['exclaims', 'every', 'one', 'present', 'Yes', 'answers']}

Example 3:
Question: How many times does the letter "f" appear in the text: "individual Project Gutenberg electronic work is derived from texts"?
Answer: 1
Metadata: {'span_length': 9, 'target_letter': 'f', 'span': ['individual', 'Project', 'Gutenberg', 'electronic', 'work', 'is', 'derived', 'from', 'texts']}

```

### letter_jumble {letter-jumble}
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
Question: Unscramble these words: Acemira bnleogs ot
Answer: America belongs to
Metadata: {'num_words': 3, 'corruption_level': 0.3687132105849005, 'scrambled_words': ['Acemira', 'bnleogs', 'ot'], 'original_words': ['America', 'belongs', 'to']}

Example 2:
Question: Unscramble these words: cubssribres ton noly
Answer: subscribers not only
Metadata: {'num_words': 3, 'corruption_level': 0.38741746634525664, 'scrambled_words': ['cubssribres', 'ton', 'noly'], 'original_words': ['subscribers', 'not', 'only']}

Example 3:
Question: Unscramble these words: yuo peerntd ttha yuo exepct ot cantmafuure a mnhau iegnb uot adn uot Wyh ton rM itSmh cndedaav
Answer: you pretend that you expect to manufacture a human being out and out Why not Mr Smith advanced
Metadata: {'num_words': 18, 'corruption_level': 0.5094277166629008, 'scrambled_words': ['yuo', 'peerntd', 'ttha', 'yuo', 'exepct', 'ot', 'cantmafuure', 'a', 'mnhau', 'iegnb', 'uot', 'adn', 'uot', 'Wyh', 'ton', 'rM', 'itSmh', 'cndedaav'], 'original_words': ['you', 'pretend', 'that', 'you', 'expect', 'to', 'manufacture', 'a', 'human', 'being', 'out', 'and', 'out', 'Why', 'not', 'Mr', 'Smith', 'advanced']}

```

### maze {maze}
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
Question: Navigate from 'a' (start) to ':' (goal):

```xxxxxxxxxx
xxxx?xx:xx
xxxx??x??x
xx????x??x
xxx?x???xx
x?x?????xx
x??ax???xx
x???xxx??x
x????x?xxx
xxxxxxxxxx```
Legend: 'x' = Wall, '?' = Passage

What is the minimum number of steps to reach the goal?
Answer: 9
Metadata: {'grid_size': 10, 'grid': ['xxxxxxxxxx', 'xxxx?xx:xx', 'xxxx??x??x', 'xx????x??x', 'xxx?x???xx', 'x?x?????xx', 'x??ax???xx', 'x???xxx??x', 'x????x?xxx', 'xxxxxxxxxx'], 'shortest_path_length': 9, 'start': 'a', 'goal': ':', 'wall': 'x', 'path': '?'}

Example 2:
Question: Navigate from '"' (start) to '}' (goal):

```444444444
4##4#4##4
44}444444
44##4#444
4#####"44
4##4####4
444#####4
4##4#4444
444444444```
Legend: '4' = Wall, '#' = Passage

What is the minimum number of steps to reach the goal?
Answer: 6
Metadata: {'grid_size': 9, 'grid': ['444444444', '4##4#4##4', '44}444444', '44##4#444', '4#####"44', '4##4####4', '444#####4', '4##4#4444', '444444444'], 'shortest_path_length': 6, 'start': '"', 'goal': '}', 'wall': '4', 'path': '#'}

Example 3:
Question: Navigate from '(' (start) to '$' (goal):

```eeeeeeeee
e(%%%%%ee
e%%%%%eee
ee%eee%ee
e%%%%%$%e
e%%%%e%ee
e%%%%%%%e
ee%%%e%%e
eeeeeeeee```
Legend: 'e' = Wall, '%' = Passage

What is the minimum number of steps to reach the goal?
Answer: 8
Metadata: {'grid_size': 9, 'grid': ['eeeeeeeee', 'e(%%%%%ee', 'e%%%%%eee', 'ee%eee%ee', 'e%%%%%$%e', 'e%%%%e%ee', 'e%%%%%%%e', 'ee%%%e%%e', 'eeeeeeeee'], 'shortest_path_length': 8, 'start': '(', 'goal': '$', 'wall': 'e', 'path': '%'}

```

### mini_sudoku {mini-sudoku}
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
_ 3 _ 1
2 1 _ _
_ _ _ 2
3 2 _ 4
Answer: 4 3 2 1
2 1 4 3
1 4 3 2
3 2 1 4
Metadata: {'puzzle': [[0, 3, 0, 1], [2, 1, 0, 0], [0, 0, 0, 2], [3, 2, 0, 4]], 'solution': [[4, 3, 2, 1], [2, 1, 4, 3], [1, 4, 3, 2], [3, 2, 1, 4]], 'num_empty': 8}

Example 2:
Question: Solve this 4x4 Mini Sudoku puzzle:
1 _ _ _
_ _ 1 _
2 _ _ _
3 4 _ _
Answer: 1 2 3 4
4 3 1 2
2 1 4 3
3 4 2 1
Metadata: {'puzzle': [[1, 0, 0, 0], [0, 0, 1, 0], [2, 0, 0, 0], [3, 4, 0, 0]], 'solution': [[1, 2, 3, 4], [4, 3, 1, 2], [2, 1, 4, 3], [3, 4, 2, 1]], 'num_empty': 11}

Example 3:
Question: Solve this 4x4 Mini Sudoku puzzle:
_ 2 4 3
_ 3 _ _
2 _ _ _
_ 1 2 _
Answer: 1 2 4 3
4 3 1 2
2 4 3 1
3 1 2 4
Metadata: {'puzzle': [[0, 2, 4, 3], [0, 3, 0, 0], [2, 0, 0, 0], [0, 1, 2, 0]], 'solution': [[1, 2, 4, 3], [4, 3, 1, 2], [2, 4, 3, 1], [3, 1, 2, 4]], 'num_empty': 9}

```

### number_filtering {number-filtering}
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
Question: Remove all numbers larger than 49.350 in this list: ['-96', '58.6', '39', '4.1432']
Answer: ['-96', '39', '4.1432']
Metadata: {'original_numbers': ['-96', '58.6', '39', '4.1432'], 'filter_value': '49.350', 'operation': 'remove_larger', 'result': ['-96', '39', '4.1432']}

Example 2:
Question: Remove all numbers larger than -58.8 in this list: ['42.685', '38.4878', '27.3', '29.6', '-41.16', '87.20', '-66.104', '57.848', '10.3373', '-45.7']
Answer: ['-66.104']
Metadata: {'original_numbers': ['42.685', '38.4878', '27.3', '29.6', '-41.16', '87.20', '-66.104', '57.848', '10.3373', '-45.7'], 'filter_value': '-58.8', 'operation': 'remove_larger', 'result': ['-66.104']}

Example 3:
Question: Keep all numbers smaller than -82.5 in this list: ['-27.517', '11.04', '61', '-95.59', '-89.6322', '84.9458', '-19.8']
Answer: ['-95.59', '-89.6322']
Metadata: {'original_numbers': ['-27.517', '11.04', '61', '-95.59', '-89.6322', '84.9458', '-19.8'], 'filter_value': '-82.5', 'operation': 'keep_smaller', 'result': ['-95.59', '-89.6322']}

```

### number_sequence {number-sequence}
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
Question: 7, 3, 1, 0, 0, 0, ?
Answer: 0
Metadata: {'rule': 'halve', 'complexity': 2, 'sequence': [7, 3, 1, 0, 0, 0, 0]}

Example 2:
Question: -5, -3, -2, -1, ?
Answer: -1
Metadata: {'rule': 'halve', 'complexity': 3, 'sequence': [-5, -3, -2, -1, -1]}

Example 3:
Question: 5, 5, 10, 15, 25, 40, 65, ?
Answer: 105
Metadata: {'rule': 'add previous', 'complexity': 1, 'sequence': [5, 5, 10, 15, 25, 40, 65, 105]}

```

### number_sorting {number-sorting}
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
Question: Sort these numbers in descending order: 34, 4, -49, -52, -19
Answer: ['34', '4', '-19', '-49', '-52']
Metadata: {'original_numbers': ['34', '4', '-49', '-52', '-19'], 'direction': 'descending', 'sorted_numbers': ['34', '4', '-19', '-49', '-52']}

Example 2:
Question: Sort these numbers in descending order: -4.44, 91.85, -86.58, -93.98, -92.88, 71.69, 25.88, 57.53, 89.65
Answer: ['91.85', '89.65', '71.69', '57.53', '25.88', '-4.44', '-86.58', '-92.88', '-93.98']
Metadata: {'original_numbers': ['-4.44', '91.85', '-86.58', '-93.98', '-92.88', '71.69', '25.88', '57.53', '89.65'], 'direction': 'descending', 'sorted_numbers': ['91.85', '89.65', '71.69', '57.53', '25.88', '-4.44', '-86.58', '-92.88', '-93.98']}

Example 3:
Question: Sort these numbers in descending order: -34.19, -85.95, -6.94, -74.52, 5.10, -18.09, -4.41
Answer: ['5.10', '-4.41', '-6.94', '-18.09', '-34.19', '-74.52', '-85.95']
Metadata: {'original_numbers': ['-34.19', '-85.95', '-6.94', '-74.52', '5.10', '-18.09', '-4.41'], 'direction': 'descending', 'sorted_numbers': ['5.10', '-4.41', '-6.94', '-18.09', '-34.19', '-74.52', '-85.95']}

```

### polynomial_equations {polynomial-equations}
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
Question: Find the real value(s) of q in the equation: -166*q**2 - 83*q = 0
Answer: [-0.5, 0.0]
Metadata: {'polynomial_expr': '-166*q**2 - 83*q', 'variable': 'q', 'degree': 2, 'real_solutions': [-0.5, 0.0]}

Example 2:
Question: Determine the real value(s) of i tha satisfies: -41*i = 0
Answer: [0.0]
Metadata: {'polynomial_expr': '-41*i', 'variable': 'i', 'degree': 1, 'real_solutions': [0.0]}

Example 3:
Question: Find the real value(s) of t in the equation: -153*t = 0
Answer: [0.0]
Metadata: {'polynomial_expr': '-153*t', 'variable': 't', 'degree': 1, 'real_solutions': [0.0]}

```

### prime_factorization {prime-factorization}
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
Question: Find the prime factorization of 139. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 139
Metadata: {'number': 139, 'factors': [139]}

Example 2:
Question: Find the prime factorization of 172. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 2 × 2 × 43
Metadata: {'number': 172, 'factors': [2, 2, 43]}

Example 3:
Question: Find the prime factorization of 562. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 2 × 281
Metadata: {'number': 562, 'factors': [2, 281]}

```

### propositional_logic {propositional-logic}
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
1. Q
2. S
3. P
What can we conclude?
Answer: (P ∧ S)
Metadata: {'premises': ['Q', 'S', 'P'], 'variables': ['P', 'Q', 'R', 'S'], 'complexity': 3}

Example 2:
Question: Given:
1. (P ∨ Q)
2. P
What can we conclude?
Answer: (P ∨ Q)
Metadata: {'premises': ['(P ∨ Q)', 'P'], 'variables': ['P', 'Q', 'R'], 'complexity': 3}

Example 3:
Question: Given:
1. Q
2. ((Q ↔ P) → (Q → Q))
3. ((Q → P) → (P ↔ Q))
What can we conclude?
Answer: (P → Q)
Metadata: {'premises': ['Q', '((Q ↔ P) → (Q → Q))', '((Q → P) → (P ↔ Q))'], 'variables': ['P', 'Q'], 'complexity': 3}

```

### quantum_lock {quantum-lock}
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
Target: 36
Buttons:
A: Add 3 (when any)
B: Multiply 3 (when any)
C: Multiply 3 (when red)
Answer: A → B → A → B
Metadata: {'difficulty': 10, 'solution_path': ['A', 'B', 'A', 'B'], 'target_value': 36, 'buttons': [{'name': 'A', 'type': 'add', 'value': 3, 'active_state': 'any'}, {'name': 'B', 'type': 'multiply', 'value': 3, 'active_state': 'any'}, {'name': 'C', 'type': 'multiply', 'value': 3, 'active_state': 'red'}], 'initial_state': 'red', 'initial_value': 0}

Example 2:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value.

Start: 0 (red)
Target: 30
Buttons:
A: Subtract 2 (when red)
B: Add 3 (when any)
C: Subtract 3 (when green)
Answer: B → B → B → B → B → B → B → B → B → B
Metadata: {'difficulty': 10, 'solution_path': ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'], 'target_value': 30, 'buttons': [{'name': 'A', 'type': 'subtract', 'value': 2, 'active_state': 'red'}, {'name': 'B', 'type': 'add', 'value': 3, 'active_state': 'any'}, {'name': 'C', 'type': 'subtract', 'value': 3, 'active_state': 'green'}], 'initial_state': 'red', 'initial_value': 0}

Example 3:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value.

Start: 0 (red)
Target: 38
Buttons:
A: Add 2 (when any)
B: Add 3 (when any)
C: Subtract 2 (when any)
Answer: A → B → B → B → B → B → B → B → B → B → B → B → B
Metadata: {'difficulty': 10, 'solution_path': ['A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'], 'target_value': 38, 'buttons': [{'name': 'A', 'type': 'add', 'value': 2, 'active_state': 'any'}, {'name': 'B', 'type': 'add', 'value': 3, 'active_state': 'any'}, {'name': 'C', 'type': 'subtract', 'value': 2, 'active_state': 'any'}], 'initial_state': 'red', 'initial_value': 0}

```

### rubiks_cube {rubiks-cube}
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
Question: You see a size 3 Rubik's cube. It is arranged this:

          R  R  R
          Y  Y  Y
          R  R  R
 W  R  W  G  G  G  Y  O  Y  B  B  B
 W  R  W  G  G  G  Y  O  Y  B  B  B
 B  B  B  W  R  W  G  G  G  Y  O  Y
          O  W  O
          O  W  O
          O  W  O


Please provide a solution to solve this cube.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': "B' F D", 'example_correct_answer': "B F D F' D' F' D B' D' R U' R' L U L' U' R' U R L U' L' U L U' L' U' B' U B U' U' F' U F U R U' R' U' B U' B' U' R' U R U' U' B' U B U L U' L' R U R' U R U U R' U U R U' L' U R' U' L U R U' L' U R' U' L U R' D' R D R' D' R D R' D' R D R' D' R D U R' D' R D R' D' R D R' D' R D R' D' R D U R' D' R D R' D' R D R' D' R D R' D' R D U"}

Example 2:
Question: You see a size 3 Rubik's cube. It is arranged this:

          B  O  G
          B  Y  G
          B  Y  G
 Y  Y  Y  O  G  W  O  O  O  Y  B  R
 R  R  R  Y  G  W  O  O  O  Y  B  W
 R  R  R  Y  G  R  W  W  W  O  B  W
          G  W  B
          G  W  B
          G  R  B


Please provide a solution to solve this cube.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': 'B L R', 'example_correct_answer': "R' B' U' L D F' D' U L U' L' U F U' F' U' L' U L U F U' F' U L' U L U F U' F' U' F' U F U R U' R' U' F' U F U R U' R' U F' U F U R U' R' F R U R' U' F' U R U R' U R U U R' L U' R' U L' U' R U L U' R' U L' U' D' R D R' D' R D U R' D' R D R' D' R D R' D' R D R' D' R D U R' D' R D R' D' R D R' D' R D R' D' R D U R' D' R D R' D' R D U"}

Example 3:
Question: You see a size 3 Rubik's cube. It is arranged this:

          O  O  O
          Y  Y  G
          Y  Y  G
 G  R  R  G  G  W  O  O  B  Y  Y  Y
 Y  R  R  G  G  W  O  O  W  B  B  B
 B  B  B  Y  R  R  G  G  W  O  O  W
          R  W  W
          R  W  W
          R  B  B


Please provide a solution to solve this cube.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': 'R B D', 'example_correct_answer': "B' F D F' D' R D R' B' D' L' U' L U R U' R' U' L U U L' F U' F' L' U U L U' B' U B U L U' L' L' U L U F U' F' U' F' U F U R U' R' U' U' R' U R U B U' B' U' U' B' U B U L U' L' U F R U R' U' R U R' U' F' U R U R' U R U U R' L U' R' U L' U' R U L U' R' U L' U' R U R' D' R D R' D' R D U R' D' R D R' D' R D U R' D' R D R' D' R D U"}

```

### sentence_reordering {sentence-reordering}
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
Question: Restore the correct order of words in the following sentence: about We think must it.
Answer: We must think about it.
Metadata: {'word_count': 5}

Example 2:
Question: Restore the correct order of words in the following sentence: 1 through 1.
Answer: 1 through 1.
Metadata: {'word_count': 3}

Example 3:
Question: Restore the correct order of words in the following sentence: lease Smith of great of a has falls obtained Niagara. the
Answer: Smith has obtained a lease of of the great falls Niagara.
Metadata: {'word_count': 11}

```

### simple_equations {simple-equations}
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
Question: Find the value of o in the equation: 84*o - 79 = 4625
Answer: 56
Metadata: {'equation': '84*o - 79 = 4625', 'variable': 'o'}

Example 2:
Question: Find the value of e in the equation: 2068*e = 198528
Answer: 96
Metadata: {'equation': '2068*e = 198528', 'variable': 'e'}

Example 3:
Question: Determine the value of g that satisfies: 71*g - 80 = 204
Answer: 4
Metadata: {'equation': '71*g - 80 = 204', 'variable': 'g'}

```

### spell_backward {spell-backward}
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
Question: Spell this word backward (example: sun -> nus): made
Answer: edam
Metadata: {'word': 'made', 'word_len': 4}

Example 2:
Question: Spell this word backward (example: sun -> nus): then
Answer: neht
Metadata: {'word': 'then', 'word_len': 4}

Example 3:
Question: Spell this word backward (example: sun -> nus): Europe
Answer: eporuE
Metadata: {'word': 'Europe', 'word_len': 6}

```

### sudoku {sudoku}
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
_ _ 2 _ _ _ 6 7 _
7 _ _ _ _ 9 _ _ _
3 _ _ _ 8 7 4 _ _
_ 8 4 _ 7 _ 9 _ _
_ _ _ _ _ _ 3 _ _
9 _ 3 1 _ _ _ 8 7
_ 1 8 4 9 _ _ 5 3
_ _ _ 8 5 1 2 9 4
4 5 9 _ _ 2 _ _ 6
Answer: 8 4 2 5 1 3 6 7 9
7 6 5 2 4 9 1 3 8
3 9 1 6 8 7 4 2 5
1 8 4 3 7 5 9 6 2
5 7 6 9 2 8 3 4 1
9 2 3 1 6 4 5 8 7
2 1 8 4 9 6 7 5 3
6 3 7 8 5 1 2 9 4
4 5 9 7 3 2 8 1 6
Metadata: {'puzzle': [[0, 0, 2, 0, 0, 0, 6, 7, 0], [7, 0, 0, 0, 0, 9, 0, 0, 0], [3, 0, 0, 0, 8, 7, 4, 0, 0], [0, 8, 4, 0, 7, 0, 9, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0, 0], [9, 0, 3, 1, 0, 0, 0, 8, 7], [0, 1, 8, 4, 9, 0, 0, 5, 3], [0, 0, 0, 8, 5, 1, 2, 9, 4], [4, 5, 9, 0, 0, 2, 0, 0, 6]], 'solution': [[8, 4, 2, 5, 1, 3, 6, 7, 9], [7, 6, 5, 2, 4, 9, 1, 3, 8], [3, 9, 1, 6, 8, 7, 4, 2, 5], [1, 8, 4, 3, 7, 5, 9, 6, 2], [5, 7, 6, 9, 2, 8, 3, 4, 1], [9, 2, 3, 1, 6, 4, 5, 8, 7], [2, 1, 8, 4, 9, 6, 7, 5, 3], [6, 3, 7, 8, 5, 1, 2, 9, 4], [4, 5, 9, 7, 3, 2, 8, 1, 6]], 'num_empty': 45}

Example 2:
Question: Solve this Sudoku puzzle:
3 5 _ _ _ _ _ _ _
_ 1 _ 3 _ 8 5 4 6
7 _ 8 9 _ _ _ 3 2
2 3 7 _ 4 _ _ 8 _
_ _ 1 8 _ 2 3 _ 4
_ _ 4 7 9 3 6 _ _
8 6 _ _ _ _ 2 _ _
_ 2 _ _ 8 7 _ _ _
_ _ _ 6 2 _ 8 5 _
Answer: 3 5 6 2 1 4 7 9 8
9 1 2 3 7 8 5 4 6
7 4 8 9 6 5 1 3 2
2 3 7 1 4 6 9 8 5
6 9 1 8 5 2 3 7 4
5 8 4 7 9 3 6 2 1
8 6 5 4 3 9 2 1 7
1 2 3 5 8 7 4 6 9
4 7 9 6 2 1 8 5 3
Metadata: {'puzzle': [[3, 5, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 3, 0, 8, 5, 4, 6], [7, 0, 8, 9, 0, 0, 0, 3, 2], [2, 3, 7, 0, 4, 0, 0, 8, 0], [0, 0, 1, 8, 0, 2, 3, 0, 4], [0, 0, 4, 7, 9, 3, 6, 0, 0], [8, 6, 0, 0, 0, 0, 2, 0, 0], [0, 2, 0, 0, 8, 7, 0, 0, 0], [0, 0, 0, 6, 2, 0, 8, 5, 0]], 'solution': [[3, 5, 6, 2, 1, 4, 7, 9, 8], [9, 1, 2, 3, 7, 8, 5, 4, 6], [7, 4, 8, 9, 6, 5, 1, 3, 2], [2, 3, 7, 1, 4, 6, 9, 8, 5], [6, 9, 1, 8, 5, 2, 3, 7, 4], [5, 8, 4, 7, 9, 3, 6, 2, 1], [8, 6, 5, 4, 3, 9, 2, 1, 7], [1, 2, 3, 5, 8, 7, 4, 6, 9], [4, 7, 9, 6, 2, 1, 8, 5, 3]], 'num_empty': 43}

Example 3:
Question: Solve this Sudoku puzzle:
2 _ 1 4 _ 5 6 _ _
_ 8 _ 6 _ 1 5 2 9
_ _ _ _ _ 2 _ 3 _
1 _ 4 2 _ _ _ _ 5
_ _ _ _ 4 _ _ 6 _
_ _ 9 _ _ _ 2 4 _
8 _ _ 5 1 6 3 _ 7
9 _ _ 7 _ 3 _ 1 2
3 _ _ 9 _ 4 _ _ 6
Answer: 2 9 1 4 3 5 6 7 8
4 8 3 6 7 1 5 2 9
7 5 6 8 9 2 1 3 4
1 3 4 2 6 7 9 8 5
5 2 8 1 4 9 7 6 3
6 7 9 3 5 8 2 4 1
8 4 2 5 1 6 3 9 7
9 6 5 7 8 3 4 1 2
3 1 7 9 2 4 8 5 6
Metadata: {'puzzle': [[2, 0, 1, 4, 0, 5, 6, 0, 0], [0, 8, 0, 6, 0, 1, 5, 2, 9], [0, 0, 0, 0, 0, 2, 0, 3, 0], [1, 0, 4, 2, 0, 0, 0, 0, 5], [0, 0, 0, 0, 4, 0, 0, 6, 0], [0, 0, 9, 0, 0, 0, 2, 4, 0], [8, 0, 0, 5, 1, 6, 3, 0, 7], [9, 0, 0, 7, 0, 3, 0, 1, 2], [3, 0, 0, 9, 0, 4, 0, 0, 6]], 'solution': [[2, 9, 1, 4, 3, 5, 6, 7, 8], [4, 8, 3, 6, 7, 1, 5, 2, 9], [7, 5, 6, 8, 9, 2, 1, 3, 4], [1, 3, 4, 2, 6, 7, 9, 8, 5], [5, 2, 8, 1, 4, 9, 7, 6, 3], [6, 7, 9, 3, 5, 8, 2, 4, 1], [8, 4, 2, 5, 1, 6, 3, 9, 7], [9, 6, 5, 7, 8, 3, 4, 1, 2], [3, 1, 7, 9, 2, 4, 8, 5, 6]], 'num_empty': 44}

```

### syllogism {syllogism}
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
1. Some programmers are cats
2. Some ... are not cats are engineers

Does it logically follow that:
No programmers are engineers?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'Some programmers are cats', 'premise2': 'Some ... are not cats are engineers', 'conclusion': 'No programmers are engineers', 'is_valid': True}

Example 2:
Question: Consider these statements:
1. All parents are cats
2. Some cats are lawyers

Does it logically follow that:
Some ... are not parents are lawyers?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'All parents are cats', 'premise2': 'Some cats are lawyers', 'conclusion': 'Some ... are not parents are lawyers', 'is_valid': True}

Example 3:
Question: Consider these statements:
1. No whales are birds
2. Some birds are teachers

Does it logically follow that:
All whales are teachers?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'No whales are birds', 'premise2': 'Some birds are teachers', 'conclusion': 'All whales are teachers', 'is_valid': True}

```

### word_ladder {word-ladder}
Generates word ladder transformation tasks

Default configuration:
```python
min_word_length = 4
max_word_length = 4
min_chain_length = -1
max_chain_length = -1
seed = None
size = 500
```

Example tasks:
```
Example 1:
Question: Transform the word ladder 'COLD' to 'WARM' by changing one letter at a time.
Answer: COLD,CORD,CARD,WARD,WARM
Metadata: {'start_word': 'COLD', 'end_word': 'WARM', 'word_length': 4, 'chain_length': 5}

Example 2:
Question: Transform the word ladder 'DARK' to 'LIGHT' by changing one letter at a time.
Answer: DARK,DARE,DATE,LATE,LITE,LIGHT
Metadata: {'start_word': 'DARK', 'end_word': 'LIGHT', 'word_length': 4, 'chain_length': 6}

Example 3:
Question: Transform the word ladder 'HEAD' to 'TAIL' by changing one letter at a time.
Answer: HEAD,HEAL,TEAL,TAIL
Metadata: {'start_word': 'HEAD', 'end_word': 'TAIL', 'word_length': 4, 'chain_length': 4}

```

### word_sequence_reversal {word-sequence-reversal}
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
Question: Reverse this list of words: upon, bold, what, of, have
Answer: have, of, what, bold, upon
Metadata: {'num_words': 5, 'words': ['upon', 'bold', 'what', 'of', 'have']}

Example 2:
Question: Reverse this list of words: years, WILL, Gutenberg, Nevertheless
Answer: Nevertheless, Gutenberg, WILL, years
Metadata: {'num_words': 4, 'words': ['years', 'WILL', 'Gutenberg', 'Nevertheless']}

Example 3:
Question: Reverse this list of words: or, of, With, no
Answer: no, With, of, or
Metadata: {'num_words': 4, 'words': ['or', 'of', 'With', 'no']}

```

### word_sorting {word-sorting}
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
believe, content, How, dedicated, seasons
Answer: seasons, dedicated, content, believe, How
Metadata: {'original_words': ['believe', 'content', 'How', 'dedicated', 'seasons'], 'transformed_words': ['believe', 'content', 'How', 'dedicated', 'seasons'], 'direction': 'descending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['seasons', 'dedicated', 'content', 'believe', 'How']}

Example 2:
Question: Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list:
owing, acute, included
Answer: acute, included, owing
Metadata: {'original_words': ['owing', 'acute', 'included'], 'transformed_words': ['owing', 'acute', 'included'], 'direction': 'ascending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['acute', 'included', 'owing']}

Example 3:
Question: Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list:
WARRANTY, tell, territory, Reckon, downloading
Answer: Reckon, WARRANTY, downloading, tell, territory
Metadata: {'original_words': ['WARRANTY', 'tell', 'territory', 'Reckon', 'downloading'], 'transformed_words': ['WARRANTY', 'tell', 'territory', 'Reckon', 'downloading'], 'direction': 'ascending', 'transformation': <TextTransformation.ORIGINAL: 'original'>, 'sorted_words': ['Reckon', 'WARRANTY', 'downloading', 'tell', 'territory']}

```
