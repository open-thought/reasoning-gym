# TODO
# DONE-string of board state
# DONE-display board
# board bounds checking
# human input?

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
            # print(ps)
            if len(ps) < MinPieceSize:
                raise ValueError("piece size < MinPieceSize")
            if len(ps) > MaxPieceSize:
                raise ValueError("piece size > MaxPieceSize")
            stride = ps[1] - ps[0]
            # print(stride)
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

    def Moves(self, i: int, steps: int = 1):
        # The position of piecs are stored as bits, 
        # it is compaired with the barrier (row/column) to confim the move being made is valid.
        # Example format
        #                                  11 Piece (B-car)
        #     1000001000001000001000001000001 LeftColumn Barrier
        #100110111100111101001111011111011011 Puzzle Board
        #
        boardMask = self.Mask()
        piece = self.m_Pieces[i]
        # boards incress difficulty by having unmovable blocks
        if piece.Fixed():
            return
        if piece.Stride == H:
            # reverse / left (negative steps)
            if ((piece.Mask & LeftColumn) == 0) and steps < 0:
                mask = (piece.Mask >> H) & ~piece.Mask
                # check pieces are intersected on a position
                if (boardMask & mask) == 0:
                    # shift to potential position
                    # mask >>= H
                    # board bounds check
                    if (mask & LeftColumn) != 0:
                        return
                    # update the board
                    self.DoMove(i, steps)
                    return
            
            # forward / right (positive steps)
            if ((piece.Mask & RightColumn) == 0) and steps > 0:
                mask = (piece.Mask << H) & ~piece.Mask
                if (boardMask & mask) == 0:
                    # mask <<= H
                    if (mask & RightColumn) != 0:
                        return
                    self.DoMove(i, steps)
                    return
                
            print("NOOP")
        else:
            # reverse / up (negative steps)
            if ((piece.Mask & TopRow) == 0) and steps < 0:
                mask = (piece.Mask >> V) & ~piece.Mask
                if (boardMask & mask) == 0:
                    # mask >>= V
                    if (mask & TopRow) != 0:
                        print("no up")
                        return
                    
                    print("Up")
                    self.DoMove(i, steps)
                    return

            # forward / down (positive steps)
            if ((piece.Mask & BottomRow) == 0) and steps > 0:
                mask = (piece.Mask << V) & ~piece.Mask
                if (boardMask & mask) == 0:
                    print("{0:36b}".format(piece.Mask))
                    # mask <<= V - 1
                    print("{0:36b}".format(mask))
                    print("{0:36b}".format(BottomRow))
                    # print("{0:36b}".format(boardMask))
                    # if (mask & BottomRow) != 0:
                    #     print("no down")x
                    #     return
                    
                    print("Down")
                    self.DoMove(i, steps)
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
    
    print('\n' + "Lets move B-car and two No-OP moves to test")
    # b.Moves(1, 1) # B
    b.Moves(6, 1) # G
    b.Moves(6, 1) # G
    # b.Moves(6, -1) # G
    # b.Moves(2,-1) # C 
    print("\n" + "-= Updated Board2D =-")
    print("{}".format(b.String2D() ))

    print("Solved the board?")
    print(b.Solved())