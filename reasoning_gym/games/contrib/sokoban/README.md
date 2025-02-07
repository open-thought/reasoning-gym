# 📦 Sokoban Solver and Generator

This folder contains a minified version of Bruno Andrade's Sokoban game, all pygame dependencies were stripped.

The original version can be found here: [xbandrade/sokoban-solver-generator](https://github.com/xbandrade/sokoban-solver-generator)


This is a Sokoban puzzle generator and solver that uses BFS, A* and Dijkstra search algorithms.

`Sokoban` is a puzzle game in which the player pushes boxes around in a warehouse, trying to get every box to a goal.


### ❕Sokoban Puzzle
The puzzle states are stored in a matrix, and each element of the puzzle is represented by a single character in the matrix.
```
+ + + + + + +
+ * - @ - X +
+ + - @ - + +
+ X - - - $ +
+ + + + + + +
```
`*` - The player </br>
`%` - The player on a goal </br>
`@` - A box </br>
`X` - A goal </br>
`$` - A box on a goal </br>
`+` - A wall </br>
`-` - An empty position </br>

A box on a goal will have its color changed to green on the game window.


### ❕Sokoban Generator

The generator will initially create a puzzle with a random board size, then the player and the boxes on goals will be randomly placed on the board.
The player will only be able to pull boxes from their positions during the generation of a puzzle, breaking every wall on his way, so it is guaranteed that the puzzle will have a valid solution.


### ❕ Sokoban Solver

The algorithms used to implement the Sokoban puzzle solvers were `Breadth-First Search(BFS)` and `A*`.

The `BFS` solver uses a queue to store the next states of the puzzle it needs to visit. A visited state is stored in a hashset, and BFS won't try to visit the same state twice.

The `A*` algorithm is similar to the BFS algorithm, but it uses a priority queue instead of a queue, and it prioritizes moves that are more likely to solve the problem.
It does so by setting costs to the puzzle state and the player's movements, punishing the player with high costs for a bad move and rewarding the player with lower costs for a good move.
The state costs are defined by heuristic functions, and this solver was implemented with two different heuristics: the `Manhattan Distance` function and `Dijkstra` distance function.

All three implementations check for possible deadlocks (states that are impossible to solve) before adding the new state to the queue.


More about Sokoban: [Wikipedia Article](https://en.wikipedia.org/wiki/Sokoban)
