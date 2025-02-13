# TODO
# DONE string of board state
# DONE display board
# DONE board bounds checking
# DONE LLM input
# wire up gym to reasoning-gym, so an LLM can call and display the board?

TEST_STRING = "BBoKMxDDDKMoIAALooIoJLEEooJFFNoGGoxN"


BoardSize = 6
PrimaryRow = 2
PrimarySize = 2
MinPieceSize = 2
MaxPieceSize = 3
MinWalls = 0
MaxWalls = 0


BoardSize2 = BoardSize * BoardSize
Target = PrimaryRow * BoardSize + BoardSize - PrimarySize
H = 1 # horizontal stride
V = BoardSize # vertical stride
DoWalls = MinPieceSize == 1

# board boundry limits
def create_row_masks():
    row_masks = []
    for y in range(BoardSize):
        mask = 0
        for x in range(BoardSize):
            i = y * BoardSize + x
            mask |= 1 << i
        row_masks.append(mask)
    return row_masks


def create_column_masks():
    column_masks = []
    for x in range(BoardSize):
        mask = 0
        for y in range(BoardSize):
            i = y * BoardSize + x
            mask |= 1 << i
        column_masks.append(mask)
    return column_masks

RowMasks = create_row_masks()
TopRow = RowMasks[0]
BottomRow = RowMasks[-1]

ColumnMasks = create_column_masks()
LeftColumn = ColumnMasks[0]
RightColumn = ColumnMasks[-1]



class Piece:
    def __init__(self, position: int, size: int, stride: int):
        self.Position = position
        self.Size = size
        self.Stride = stride
        self.Mask = 0

        p = position
        for i in range(size):
            self.Mask |= 1 << p
            p += stride
    
    def Fixed(self):
        return self.Size == 1
    
    def Move(self, steps: int):
        d = self.Stride * steps
        self.Position += d
        if (steps > 0):
            self.Mask <<= d
        else:
            self.Mask >>= -d


class Board:
    def __init__(self, desc):
        self.m_HorzMask = 0
        self.m_VertMask = 0
        self.m_Pieces = []

        if len(desc) != BoardSize2:
            raise ValueError("board string is wrong length")

        positions = {}
        for i, label in enumerate(desc):
            if label == 'x' or label == 'o':
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
            self.AddPiece(Piece(ps[0], len(ps), stride))

    def AddPiece(self, piece):
        self.m_Pieces.append(piece)
        if (piece.Stride == H):
            self.m_HorzMask |= piece.Mask
        else:
            self.m_VertMask |= piece.Mask

    def Mask(self):
        return self.m_HorzMask | self.m_VertMask
    
    # DoMove has no bounds checking
    def DoMove(self, i: int, steps: int):
        piece = self.m_Pieces[i]
        if (piece.Stride == H):
            # Clears the current position from the horizontal mask
            self.m_HorzMask &= ~piece.Mask
            piece.Move(steps)
            # Adds the new position to the horizontal mask
            self.m_HorzMask |= piece.Mask
        else:
            self.m_VertMask &= ~piece.Mask
            piece.Move(steps)
            self.m_VertMask |= piece.Mask 

    def Moves(self, target: str = "A", dir: str = "up"):
        # The position of piecs are stored as bits, 
        # it is compaired with the barrier (row/column) to confim the move being made is valid.
        # Example format:
        #                 1000001000000000000 Piece Mask
        #           1000000000000000000000000 Move Mask
        #111000111100111101001111011111011110 Puzzle Board
        #
        boardMask = self.Mask()
        i = ord(target) - ord('A')
        if i < 0 or i > 13:
            return
        
        piece = self.m_Pieces[i]
        # boards incress difficulty by having unmovable blocks
        if piece.Fixed():
            return
        if piece.Stride == H:
            # reverse / left (negative steps)
            if ((piece.Mask & LeftColumn) == 0) and dir.lower() == "left":
                mask = (piece.Mask >> H) & ~piece.Mask
                # check pieces are intersected on a position
                if (boardMask & mask) == 0:
                    self.DoMove(i, -1)
                    return
            
            # forward / right (positive steps)
            if ((piece.Mask & RightColumn) == 0) and dir.lower() == "right":
                mask = (piece.Mask << H) & ~piece.Mask
                if (boardMask & mask) == 0:
                    self.DoMove(i, 1)
                    return
                
            # print("NOOP")
        else:
            # reverse / up (negative steps)
            if ((piece.Mask & TopRow) == 0) and dir.lower() == "up":
                mask = (piece.Mask >> V) & ~piece.Mask
                if (boardMask & mask) == 0:
                    self.DoMove(i, -1)
                    return

            # forward / down (positive steps)
            if ((piece.Mask & BottomRow) == 0) and dir.lower() == "down":
                mask = (piece.Mask << V) & ~piece.Mask
                if (boardMask & mask) == 0:
                    # print("{0:36b}".format(piece.Mask))
                    # print("{0:36b}".format(mask))
                    # print("{0:36b}".format(boardMask))
                    self.DoMove(i, 1)
                    return
            
            # print("NOOP")

    def Solved(self):
        return self.m_Pieces[0].Position == Target

    def String(self):
        s = ['.'] * (BoardSize * (BoardSize + 1))
        for i in range(len(self.m_Pieces)):
            piece = self.m_Pieces[i]
            c = 'x' if piece.Fixed() else chr(ord('A') + i)
            p = piece.Position
            for i in range(piece.Size):
                s[p] = c
                p += piece.Stride
        return ''.join(s)

    def String2D(self):
        s = ['.'] * (BoardSize * (BoardSize + 1))
        for y in range(BoardSize):
            p = y * (BoardSize + 1) + BoardSize
            s[p] = '\n'
        for i in range(len(self.m_Pieces)):
            piece = self.m_Pieces[i]
            c = 'x' if piece.Fixed() else chr(ord('A') + i)
            stride = piece.Stride
            if (stride == V):
                stride += 1
            y = piece.Position // BoardSize
            x = piece.Position % BoardSize
            p = y * (BoardSize + 1) + x
            for i in range(piece.Size):
                s[p] = c
                p += stride
        return ''.join(s)
    

if __name__ == "__main__":
    # See it in action
    b = Board(TEST_STRING)
    print("-= Board String =-")
    print(b.String())
    
    print('\n' + "-= Board 2D =-")
    print("{}".format(b.String2D()))

    print("Number of Pieces")
    print(len(b.m_Pieces))
    
    print('\n' + "Lets move some pieces to test")
    b.Moves("B", "RIGHT")
    b.Moves("F", "right")
    b.Moves("F", "RIGHT")
    b.Moves("G", "down")
    b.Moves("G", "DOWN")
    # moves that should not update the board
    b.Moves("G", "down")
    b.Moves("J", "left")
    b.Moves("K", "right")

    print("\n" + "-= Updated Board2D =-")
    print("{}".format(b.String2D() ))

    print("Solved the board?")
    print(b.Solved())