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
            self.m_HorzMask &= ~piece.Mask
            piece.Move(steps)
            self.m_HorzMask |= piece.Mask
        else:
            self.m_VertMask &= ~piece.Mask
            piece.Move(steps)
            self.m_VertMask |= piece.Mask 

    def Moves(self):
        boardMask = self.Mask
        for i in range(len(self.m_Pieces)):
            piece = self.m_Pieces[i]
            if piece.Fixed:
                continue
            if piece.Stride == H:
                # reverse / left (negative steps)
                if ((piece.Mask & LeftColumn) == 0):
                    mask = (piece.Mask >> H) & ~piece.Mask
                    steps = -1
                
                # forward / right (positive steps)
                if ((piece.Mask & RightColumn) == 0):
                    mask = (piece.Mask << H) & ~piece.Mask
                    steps = 1
            else:
                # reverse / up (negative steps)
                if ((piece.Mask & TopRow) == 0):
                    mask = (piece.Mask >> V) & ~piece.Mask
                    steps = -1

                # forward / down (positive steps)
                if ((piece.Mask & BottomRow) == 0):
                    mask = (piece.Mask << V) & ~piece.Mask
                    steps = 1
        return

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
    b = Board(TEST_STRING)
    print("-= Board =-")
    print("{}".format(b.String2D() ))
    print("-= Moved B =-")
    b.DoMove(1, 1)
    print("{}".format(b.String2D() ))