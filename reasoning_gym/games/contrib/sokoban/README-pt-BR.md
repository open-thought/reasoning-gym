# 📦 Solucionador e Gerador Sokoban

▶️ Vídeo mostrando as mecânicas do jogo, o gerador e o solucionador: [Sokoban Generator and Solver](https://www.youtube.com/watch?v=l0BHKkoViII)

Este é um gerador de puzzles e solucionador de puzzles Sokoban que utiliza os algoritmos de busca BFS, A* e Dijkstra.


`Sokoban` é um jogo de puzzle no qual o jogador empurra caixas em um armazém, com o objetivo de colocar todas as caixas em seus objetivos.


### ➡️ Setup 
```pip install -r requirements.txt```

```python -m sokoban```


### ❕Sokoban Puzzle
Os estados dos puzzles são armazenados em uma matriz, e cada elemento do puzzle é representado por um único caractere na matriz.
```
+ + + + + + +
+ * - @ - X +
+ + - @ - + +
+ X - - - $ +
+ + + + + + +
```
`*` - O player </br>
`%` - O player em cima de um objetivo </br>
`@` - Uma caixa </br>
`X` - Um objetivo </br>
`$` - Uma caixa sobre um objetivo </br>
`+` - Uma parede </br>
`-` - Uma posição vazia </br>

Uma caixa sobre um objetivo terá sua cor alterada para verde na tela do jogo.


### ❕Gerador Sokoban

Um puzzle pseudoaleatório e válido será gerado ao usar o botão `Random` na barra lateral.
Digitar um número de semente válida (1-99999) antes de usar o botão `Random` vai gerar o puzzle utilizando a semente especificada.

O gerador inicialmente vai criar um puzzle de tamanho aleatório, então o player e as caixas sobre os objetivos serão aleatoriamente posicionadas no puzzle.
Durante o período de geração do puzzle, o player só será capaz de puxas as caixas de suas posições, quebrando todas as paredes em seu caminho. Desta forma, o puzzle criado sempre possuirá uma solução válida.


### ❕ Solucionador Sokoban

<img src="https://raw.githubusercontent.com/xbandrade/sokoban-solver-generator/main/img/levelclear.gif" width=80% height=80%>

Os algoritmos de `Busca em largura(BFS)` e `A*` foram usados para implementar os solucionadores de puzzles Sokoban.

O solucionador `BFS` usa uma fila para armazenar os próximos estados do puzzle que ele deve visitar. Um estado que já foi visitado é armazenado em um hashset, então o BFS não vai tentar visitar o mesmo estado duas vezes.

O algoritmo `A*` é similar ao algoritmo BFS, mas ele usa uma fila de prioridade em vez de uma fila simples, priorizando movimentos que possuem mais chances de solucionar o problema.
Este algoritmo faz isso atribuindo custos aos estados do puzzle e aos movimentos do player, punindo o player com custos altos em um movimento ruim e recompensando o player com custos menores por um bom movimento.
Os custos de estado são definidos por funções heurísticas, e este solucionador foi implementado com duas heurísticas diferentes: a função `Distância de Manhattan` e a função de `Dijkstra`.

Todas as três implementações verificam por deadlocks (estados impossíveis de serem solucionados) antes de adicionar um novo estado à fila.


### ❕ Botões e Opções da Interface
- `Restart` Reinicia o nível atual para o estado inicial
- `Seed` Especifica uma semente para ser carregada com o botão `Random`
- `Random` Gera um puzzle pseudoaleatório válido
- `Solve BFS` Soluciona o puzzle atual usando Busca em Largura(BFS)
- `A* Manhattan` Soluciona o puzzle atual usando A* com heurística da Distância de Manhattan
- `Dijkstra` Soluciona o puzzle atual usando A* com heurística da distância de Dijkstra
- `Visualize` Exibe o processo de geração do puzzle e mostra o melhor caminho atual para as soluções
 

### ❕ Testes Unitários
Todos os testes unitários estão armazenados na pasta `/tests`, separados por categorias em classes e arquivos diferentes. Use `pytest` para rodas todos os testes unitários de uma vez.

Mais sobre Sokoban: [Artigo Wikipedia](https://en.wikipedia.org/wiki/Sokoban)
