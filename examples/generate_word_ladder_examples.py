# generates dataset of word ladder examples, and then generates simulated chain of thought reasoning for each example

import reasoning_gym
from openai import OpenAI
import os

# Configuration for the dataset
config = {
    'dataset_name': 'word_ladder',
    'dataset_config': {
        'min_word_length': 5,
        'max_word_length': 5,
        'min_chain_length':3, # set to -1 for shortest possible path, increase to generate more examples
        'max_chain_length':5,
        'size': 1,  # Generate a small dataset for demonstration
    }
}

system_prompt = """Word Ladder puzzles involve transforming a start word into an end word. 
You are allowed to change only one letter a time and you must keep the number of letters constant. 
Each time you change one letter the word in the chain must be forming one that's valid in English. 
Plurals are allowed, but not proper nouns. 
Given a start and an end word, generate a detailed step-by-step chain of thought reasoning of the transformation process.
You will be given the word ladder question, as well as the correct solution path. So you don't need to solve the problem, you have the solution.
Your task is to provide a perfectly simulated chain of thought reasoning exactly in the style of the example below, 
including considering multiple possibilities, validating words, and showing the final path. 
Use casual, verbose thinking with markdown formatting.
Example chain of thought reasoning: 
{sample_CoT}
"""

# sample CoT for demonstration, extracted from real inference data from deepseek r1
sample_CoT = """ 
Word Ladder puzzles involve transforming a start word into an end word. 
You are allowed to change only one letter a time and you must keep the number of letters constant. 
Each time you change one letter the word in the chain must be forming one that's valid in English. 
Plurals are allowed, but not proper nouns. 
Given a start and an end word, generate a detailed step-by-step chain of thought reasoning of the transformation process.
You will be given the word ladder question, as well as the correct solution path. So you don't need to solve the problem, you have the solution already, which should be your 'cheat sheet' to refer to as you go along.
Your task is to provide a perfectly simulated chain of thought reasoning exactly in the style of the example below, 
including considering multiple possibilities, validating words, and showing the final path. 
Use casual, verbose thinking with markdown formatting.
VERY IMPORTANT ***NEVER MENTION YOU HAVE BEEN PROVIDED WITH THE ANSWER, YOU NEED TO WRITE ONLY AS IF YOU'RE SOLVING THE QUESTION STEP BY STEP ***

Example chain of thought reasoning: 
<think>
Okay, so I need to figure out how to transform the word "CARK" into "ZIPS" by changing one letter at a time, making sure each intermediate step is a valid English word. Let me start by breaking down the problem.

First, both words are four letters long, so each step must maintain four letters. Each change can only alter one letter, either adding, removing, or replacing, but since the length must stay the same, it has to be a replacement each time. So, I need to find a path from CARK to ZIPS where each intermediate word is valid.

Let me write down the starting and target words:

CARK

...

ZIPS

I need to see which letters need to change. Let's compare the letters:

C -> Z

A -> I

R -> P

K -> S

So all four letters need to change. But since each step can only change one letter, I need to find a sequence where each single-letter change results in a valid word. Maybe I can approach this by changing one letter at a time, checking possible words each step.

Starting with CARK. Let's look at possible changes for each letter:

First letter C: changing to other letters to form a word. Possible options could be BARK (C to B), DARK (C to D), LARK (C to L), MARK (C to M), PARK (C to P), etc. Wait, but CARK is a valid word? Let me confirm. CARK means to worry or burden, so yes, it's valid.

So, possible first steps from CARK:

Change C to B: BARK

Change C to D: DARK

Change C to L: LARK

Change C to M: MARK

Change C to P: PARK

Change C to W: WARK (is that a word? Wark might be a Scottish term for work, but I'm not sure if it's accepted in standard English. Maybe not.)

Other possibilities: changing the A, R, or K. Let's see:

Second letter A: changing to other vowels or consonants. For example, CORK (changing A to O), but CORK is valid. Or CIRK (changing A to I; cirk is a variant of circus, but I think "cirk" isn't standard. Maybe CIRC? No, that's not right. Wait, maybe CIRK is a valid word in some contexts, like a circular trench in fortification? Hmm, not sure. Better to stick with more common words.

Third letter R: changing to another letter. For example, CAKE (changing R to K), but that would require changing both the third letter and the fourth letter? Wait, no: CARK to CAKE would be changing R to E and K to E? Wait, no. Wait, CARK is C-A-R-K. To get CAKE, you'd need to change R to K and K to E, which is two changes. So not possible in one step. Maybe changing R to another letter. For example, CASH (changing R to S and K to H? No, that's two letters again. Wait, no. Let's see: CARK. Changing the third letter R to another letter, like CASK (changing R to S). But that would be changing the third letter from R to S, resulting in CASK. Yes, that's a valid word. So CARK -> CASK.

Fourth letter K: changing to another letter. For example, CARD (changing K to D), CARP (changing K to P), CARS (changing K to S), CART (changing K to T). All of these are valid.

So possible first steps from CARK:

- BARK, DARK, LARK, MARK, PARK, CASK, CARD, CARP, CARS, CART.

Now, which of these would lead towards ZIPS? Let's think about the target word. ZIPS has a Z, which is the first letter. So maybe we need to get to a word starting with Z eventually. But changing from C to Z in one step would be a big jump. Alternatively, maybe we can change other letters first to reach intermediate steps that can approach Z.

Alternatively, maybe work on changing the last letters first. Let's see. For example, changing K to S (CARS) might be useful since the target ends with S. Let's try that.

CARK -> CARS (changing K to S). Now, from CARS, we need to change the third letter R to P and the first letter C to Z, and the second A to I. Let's see.

From CARS, possible changes:

Change C to Z: ZARS (is that a word? ZAR is a currency, plural ZARS? Maybe, but not sure. Alternatively, ZAPS (changing R to P and S to something? Wait, no. Let's check each letter.

Alternatively, from CARS, change A to I: CIRS (is that a word? CIR is a three-letter word, but CIRS? Maybe not. Or CURS (changing A to U), which is valid. But how does that help?

Alternatively, from CARS, change R to P: CAPS. Yes, that's valid. So CARS -> CAPS (changing R to P). Then from CAPS, we can change C to Z: ZAPS. Yes, ZAPS is a word. Then ZAPS -> ZIPS by changing A to I. So that would be a path:

CARK -> CARS -> CAPS -> ZAPS -> ZIPS.

Wait, let's check each step:

1. CARK to CARS: change K to S. Valid.

2. CARS to CAPS: change R to P. Valid.

3. CAPS to ZAPS: change C to Z. Valid.

4. ZAPS to ZIPS: change A to I. Valid.

That's four steps. But wait, starting from CARK (step 0), then steps 1-4 would be four changes. But the problem says "transform CARK into ZIPS by changing one letter at a time", so the number of steps would be the number of letters changed. Since all four letters are different, at minimum it would take four steps. But perhaps the path is longer if there are no shorter paths.

Wait, let me check if this works:

CARK → CARS → CAPS → ZAPS → ZIPS.

Yes, that's four steps. Each step changes one letter, and all intermediate words are valid. Let me verify each word:

- CARK: valid.

- CARS: valid.

- CAPS: valid.

- ZAPS: valid.

- ZIPS: valid.

Yes, that seems to work. But let me check if there's a shorter path or if this is correct.

Alternatively, is there a three-step path? Probably not since four letters need to change. Each step can only change one letter, so at least four steps are needed. So the path above is four steps, which is the minimum. Let me confirm each step again.

Another possible path: changing the first letter earlier. For example:

CARK → PARK (change C to P). Then PARK → PARKS (but that's five letters; can't do that. Alternatively, changing K to S: PARS. Then PARS -> PAWS (changing R to W?), but not helpful. Or PARK → PARS (change K to S), then PARS → PAIS (change R to I?), but PAIS is valid. Then PAIS → ZIPS (change P to Z and A to I? No, that's two letters. Hmm, not helpful.

Alternatively, CARK → DARK → DANK → DINK → ZINK → ZIPS? Let's see. CARK to DARK (C→D). DARK to DANK (R→N). DANK to DINK (A→I). DINK to ZINK (D→Z). ZINK is a variant spelling of zinc, but ZINK is a valid word. Then ZINK to ZIPS (K→S and N→P? Wait, no. ZINK is Z-I-N-K. To get to ZIPS, we need Z-I-P-S. So need to change N to P and K to S. That's two steps. So that path would be longer.

Another path: CARK → CARD → ZARD? ZARD isn't a word. CARD → WARD → WARD → ... not helpful.

Alternatively, CARK → CARK to CIRK (A→I). CIRK is a variant of circus, but maybe not standard. Then CIRK → ZIRK (C→Z). ZIRK is a surname or a brand, not sure if it's a valid word. Then ZIRK → ZIPS (R→P and K→S). But again, two letters at once. Not helpful.

So the first path I thought of seems viable: CARK → CARS → CAPS → ZAPS → ZIPS.

Another check: From CARK, can we go to CASK? CARK → CASK (R→S). Then CASK → ZASK (C→Z). ZASK? Not a word. Alternatively, CASK → CASK → CASH (K→H). Then CASH → ZASH (C→Z). Not a word. Not helpful.

Alternatively, CARK → CARP (K→P). CARP is valid. Then CARP → CARP → ZARP (C→Z). ZARP? Not a word. Or CARP → HARp (C→H), but HARP is valid. Then HARP → HIPS (A→I and R→P). But that's two changes. Not allowed.

So the initial path seems best. Let me confirm each step again:

1. CARK → CARS: change K to S. Valid.

2. CARS → CAPS: change R to P. Valid.

3. CAPS → ZAPS: change C to Z. Valid.

4. ZAPS → ZIPS: change A to I. Valid.

Yes, that's four steps, each changing one letter, all words are valid. So that should be the correct sequence.
</think>
"""

client = OpenAI(
  base_url="https://openrouter.ai/api/v1", # openrouter used in this example
  api_key=os.environ["OPENROUTER_API_KEY"], # set your key in environment variable
)


def generate_cot(question: str, answer: str) -> str:
    """Generate chain of thought reasoning for word ladder"""
    prompt = f"""The question is {question}. The correct solution is {answer}. 
    Provide the verbose chain of thought reasoning to transform the start word into the end word exactly in the style and length required."""
    
    completion = client.chat.completions.create(
        model="microsoft/phi-4", # choose model
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.6,
        max_tokens=10000
    )
    return completion.choices[0].message.content
# Create the word ladder dataset
dataset = reasoning_gym.create_dataset(config['dataset_name'], **config['dataset_config'])
print(f"Generated {len(dataset)} examples, moving on to generate CoT reasoning...")
# Generate and print examples with CoT
for item in dataset:
    # Generate CoT reasoning demo

    item['reasoning'] = generate_cot(item['question'],item['answer'])
    
    print("\n--- Example ---")
    print("Question:", item['question'])
    print("Answer:", item['answer'])
    print("\nChain of Thought:")
    print(item['reasoning'])
    print("\nMetadata:", item['metadata']) 