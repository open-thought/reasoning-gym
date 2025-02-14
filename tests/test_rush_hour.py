import pytest

from reasoning_gym.games.rush_hour import Board


def test_perform_moves():
    b = Board("GBBoLoGHIoLMGHIAAMCCCKoMooJKDDEEJFFo")
    assert not b.solved
    incomplete_moves = "F+1 K+1 M-1 C+3 H+2 J-1 E+1 G+3 B-1 I-1 A-3 I+1 L+1 B+3 I-1 A+2 G-3"
    b.perform_moves(incomplete_moves)
    assert not b.solved
    solution = "E-1 H-3 A-1 J+1 C-3 M+1 B+1 K-4 A+1 C+2 D-1 F-1 H+3 A-1 K+1 B-1 M-1 C+1 J-1 E+1 G+3 A-1 I+1 B-3 I-1 A+1 G-1 E-1 J+1 C-1 K-1 L-1 M+3 A+3"
    b.perform_moves(solution)
    assert b.solved
