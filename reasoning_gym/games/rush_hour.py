import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random

from ..data import read_data_file
from ..factory import ProceduralDataset, register_dataset

TEST_STRING = "BBoKMxDDDKMoIAALooIoJLEEooJFFNoGGoxN"

@dataclass
class RushHourConfig:
    """Configuration for Rush Hour puzzle generation"""
    min_moves: int = 1
    max_moves: int = 50
    seed: Optional[int] = None 
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_moves > 0, "min_moves must be positive"
        assert self.max_moves >= self.min_moves, "max_moves must be >= min_moves"
        assert self.size > 0, "size must be positive"


BoardSize = 6
PrimaryRow = 2
PrimarySize = 2
MinPieceSize = 2
MaxPieceSize = 3
MinWalls = 0
MaxWalls = 0


BoardSize2 = BoardSize * BoardSize
Target = PrimaryRow * BoardSize + BoardSize - PrimarySize
H = 1  # horizontal stride
V = BoardSize  # vertical stride


# board boundary limits
def create_row_masks() -> list[int]:
    row_masks: list[int] = []
    for y in range(BoardSize):
        mask = 0
        for x in range(BoardSize):
            i = y * BoardSize + x
            mask |= 1 << i
        row_masks.append(mask)
    return row_masks


def create_column_masks() -> list[int]:
    column_masks: list[int] = []
    for x in range(BoardSize):
        mask = 0
        for y in range(BoardSize):
            i = y * BoardSize + x
            mask |= 1 << i
        column_masks.append(mask)
    return column_masks


ROW_MASKS = create_row_masks()
TOP_ROW = ROW_MASKS[0]
BOTTOM_ROW = ROW_MASKS[-1]

COLUMNS_MASKS = create_column_masks()
LEFT_COLUMN = COLUMNS_MASKS[0]
RIGHT_COLUMN = COLUMNS_MASKS[-1]


class Piece:
    def __init__(self, position: int, size: int, stride: int):
        self.position = position
        self.size = size
        self.stride = stride
        self.mask = 0

        p = position
        for i in range(size):
            self.mask |= 1 << p
            p += stride

    @property
    def fixed(self) -> bool:
        return self.size == 1

    def move(self, steps: int) -> None:
        d = self.stride * steps
        self.position += d
        if steps > 0:
            self.mask <<= d
        else:
            self.mask >>= -d


class RushHourDataset(ProceduralDataset):
    """Generates Rush Hour puzzle configurations from pre-computed database"""

    def __init__(self, config: RushHourConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        
        # Load and filter puzzles from data file
        self.puzzles: List[Tuple[str, int]] = []  # (board_config, min_moves)
        
        data = read_data_file("rush_18k.txt")
        for line in data.splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                min_moves = int(parts[0])
                board_config = parts[1]
                
                if config.min_moves <= min_moves <= config.max_moves:
                    self.puzzles.append((board_config, min_moves))
        
        if not self.puzzles:
            raise ValueError(
                f"No puzzles found with moves between {config.min_moves} and {config.max_moves}"
            )

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Rush Hour puzzle
        
        Args:
            idx: Index of the item to generate
            
        Returns:
            dict with keys:
                - question: str, the formatted board with instructions
                - answer: str, example solution (empty as multiple solutions exist)
                - metadata: dict with board config and min moves
        """
        # Create deterministic RNG from base seed and idx
        rng = random.Random(self.seed + idx)
        
        # Randomly select a puzzle meeting our criteria
        board_config, min_moves = rng.choice(self.puzzles)
        
        # Create board and get string representation
        board = Board(board_config)
        board_display = board.board_str()
        
        instructions = (
            "Move the red car (AA) to the exit on the right.\n"
            "Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...'\n"
            "where the letter is the vehicle and +/- number is spaces to move right/left or down/up."
        )
        
        return {
            "question": f"{instructions}\n\nBoard:\n{board_display}",
            "answer": "",  # Multiple valid solutions exist
            "metadata": {
                "board_config": board_config,
                "min_moves": min_moves,
            },
        }


class Board:
    def __init__(self, desc: str):
        self._horz_mask = 0
        self._vert_mask = 0
        self._pieces: list[Piece] = []

        if len(desc) != BoardSize2:
            raise ValueError("board string is wrong length")

        positions: dict[str, int] = {}
        for i, label in enumerate(desc):
            if label == "x" or label == "o":
                continue
            if label not in positions:
                positions[label] = []
            positions[label].append(i)

        labels = []
        for pair in positions:
            labels.append(pair[0])
        labels.sort()

        for label in labels:
            ps = positions[label]
            if len(ps) < MinPieceSize:
                raise ValueError("piece size < MinPieceSize")
            if len(ps) > MaxPieceSize:
                raise ValueError("piece size > MaxPieceSize")
            stride = ps[1] - ps[0]
            if stride != H and stride != V:
                raise ValueError("invalid piece shape")
            for i in range(2, len(ps)):
                if ps[i] - ps[i - 1] != stride:
                    raise ValueError("invalid piece shape")
            self.add_piece(Piece(ps[0], len(ps), stride))

    def add_piece(self, piece) -> None:
        self._pieces.append(piece)
        if piece.stride == H:
            self._horz_mask |= piece.mask
        else:
            self._vert_mask |= piece.mask

    def mask(self) -> int:
        return self._horz_mask | self._vert_mask

    # DoMove has no bounds checking
    def _do_move(self, i: int, steps: int) -> None:
        piece = self._pieces[i]
        if piece.stride == H:
            # Clears the current position from the horizontal mask
            self._horz_mask &= ~piece.mask
            piece.move(steps)
            # Adds the new position to the horizontal mask
            self._horz_mask |= piece.mask
        else:
            self._vert_mask &= ~piece.mask
            piece.move(steps)
            self._vert_mask |= piece.mask

    def move(self, target: str, dir: int) -> None:
        # The position of piecs are stored as bits,
        # it is compaired with the barrier (row/column) to confim the move being made is valid.
        # Example format:
        #                  1000001000000000000 Piece Mask
        #            1000000000000000000000000 Move Mask
        # 111000111100111101001111011111011110 Puzzle Board
        #
        boardMask = self.mask()
        # validate input before using index
        i = ord(target) - ord("A")
        if i < 0 or i > len(self._pieces):
            return

        piece = self._pieces[i]
        # boards increase difficulty by having unmovable blocks
        if piece.fixed:
            return

        for _step in range(abs(dir)):
            if piece.stride == H:
                # reverse / left (negative steps)
                if ((piece.mask & LEFT_COLUMN) == 0) and dir < 0:
                    mask = (piece.mask >> H) & ~piece.mask
                    # check pieces are intersected on a position
                    if (boardMask & mask) == 0:
                        self._do_move(i, -1)
                        continue

                # forward / right (positive steps)
                if ((piece.mask & RIGHT_COLUMN) == 0) and dir > 0:
                    mask = (piece.mask << H) & ~piece.mask
                    if (boardMask & mask) == 0:
                        self._do_move(i, 1)
                        continue

                # print("NOOP", target, dir)
            else:
                # reverse / up (negative steps)
                if ((piece.mask & TOP_ROW) == 0) and dir < 0:
                    mask = (piece.mask >> V) & ~piece.mask
                    if (boardMask & mask) == 0:
                        self._do_move(i, -1)
                        continue

                # forward / down (positive steps)
                if ((piece.mask & BOTTOM_ROW) == 0) and dir > 0:
                    mask = (piece.mask << V) & ~piece.mask
                    if (boardMask & mask) == 0:
                        # print("{0:36b}".format(piece.Mask))
                        # print("{0:36b}".format(mask))
                        # print("{0:36b}".format(boardMask))
                        self._do_move(i, 1)
                        continue

                # print("NOOP", target, dir)

    def perform_moves(self, ops: str) -> None:
        # This pattern matches:
        # - One or more letters (captured in group 1)
        # - A plus or minus sign (captured in group 2)
        # - One or more digits (captured in group 3)
        pattern = r"([A-Z]+)([+-])(\d+)"

        # Find all matches in the string
        matches = re.findall(pattern, ops)

        # Convert matches to list of tuples (character, number)
        # The number is converted to positive or negative based on the sign
        move_ops = [(chars, int(num) if sign == "+" else -int(num)) for chars, sign, num in matches]

        for target, dir in move_ops:
            self.move(target, dir)

    @property
    def solved(self) -> bool:
        return self._pieces[0].position == Target

    def __str__(self) -> str:
        s = ["."] * (BoardSize * (BoardSize + 1))
        for i in range(len(self._pieces)):
            piece = self._pieces[i]
            c = "x" if piece.fixed else chr(ord("A") + i)
            p = piece.position
            for i in range(piece.size):
                s[p] = c
                p += piece.stride
        return "".join(s)

    def board_str(self) -> str:
        s = ["."] * (BoardSize * (BoardSize + 1))
        for y in range(BoardSize):
            p = y * (BoardSize + 1) + BoardSize
            s[p] = "\n"
        for i in range(len(self._pieces)):
            piece = self._pieces[i]
            c = "x" if piece.fixed else chr(ord("A") + i)
            stride = piece.stride
            if stride == V:
                stride += 1
            y = piece.position // BoardSize
            x = piece.position % BoardSize
            p = y * (BoardSize + 1) + x
            for i in range(piece.size):
                s[p] = c
                p += stride
        return "".join(s)


# Register the dataset
register_dataset("rush_hour", RushHourDataset, RushHourConfig)

