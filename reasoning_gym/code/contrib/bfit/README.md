<img width="200" height="200" align="left" style="float: left; margin: 0 10px 0 0;" alt="BF-it logo" src="images/logo.png?sanitize=true">

# **BF-it**
A compiler from a C-like language to Brainfuck

Always heard that Brainfuck is Turing complete but still had doubts about its computational power?
Those days are over


## What is this?
BF-it is a Compiler of a C-like language to Brainfuck, written in Python

You give it C-like code, and it gives you Brainfuck code with the same functionality :)


## Why did you create it?
There are several reasons why I chose to create this

1. I always heard that Brainfuck is Turing complete, and
   therefore as strong as any other language
    * I wanted to put that to the test, and find a concrete
      proof that it is true
2. I find it beautiful how a seemingly bunch of nonsense
   characters can have a logical meaning / functionality
    * I wanted to take it to the extreme
3. I wanted to practice my developing skills
4. It seemed to be like a real challenge (and it was!)
5. But mostly, and most importantly, I did it for teh lulz


## What can I use it for?
Create a Brainfuck code of your choice, send it to your friends, and tell them to run it (on some online interpreter, for example)

When they receive a bunch of nonsense which does something meaningful, they will (maybe) be amazed


## How do I use it?

1. Write a C-like code (example programs included) into a file
2. Open your favorite command line interface
3. Run `BF-it.py <path_to_code_file>`
    * This will compile your file and create a .bf file with
      Brainfuck code
    * (optional parameters: `-o` for output file, and `-r`
      for running the compiled file)
4. Run `Interpreter.py <path_to_bf_file>`, this will execute
   the Brainfuck code

Example:
```
$ cat helloworld.code
int main()
{
    print("Hello World!\n");
}

$ ./BF-it.py helloworld.code
Compiling file 'helloworld.code'...
Compiled successfully to 'helloworld.bf'
$ cat helloworld.bf
>[-]>[-]<>++++++++[-<+++++++++>]<.>++++[-<+++++++>]
<+.+++++++..+++.>++++++[-<------------->]<-.>+++++[
-<+++++++++++>]<.>++++[-<++++++>]<.+++.------.-----
---.>++++++[-<----------->]<-.>+++[-<------->]<--.<

$ ./Interpreter.py helloworld.bf
Hello World!
```

For a more interesting example, check out Tic-Tac-Toe!

<img alt="Tic-Tac-Toe" src="images/tic tac toe.gif?sanitize=true"><br/>

1. Take a look at tic_tac_toe.code in the 'examples/games' directory
2. Run ```./BF-it.py examples/games/tic_tac_toe.code -r```
3. Play around
4. If you're brave, take a look at the generated Brainfuck code
5. If you're very brave, try to understand it (I can't)

## How do I write code?
Please refer to the 'examples' directory.
It contains examples and explanations for syntax, operations, flow, and more


## I would like to add a feature / fix a bug
If you would like to add a feature yourself, perform a pull request and add your changes. I will review them

If you found a bug, or have an idea for a feature, open an issue



## References I used:
* https://esolangs.org/wiki/brainfuck_algorithms for basic Brainfuck algorithms
* https://www.iamcal.com/misc/bf_debug/ for debugging Brainfuck code
* https://fatiherikli.github.io/brainfuck-visualizer/ for debugging Brainfuck code
* https://godbolt.org/ for comparing my C-like compiler to other C compilers' implementations
* https://www.lysator.liu.se/c/ANSI-C-grammar-y.html for creating a grammar
* https://introcs.cs.princeton.edu/java/11precedence/ for operator precedence
* https://logomakr.com/ for creating a logo
* https://www.youtube.com/ for setting the mood
