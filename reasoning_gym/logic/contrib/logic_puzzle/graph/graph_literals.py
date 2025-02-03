

def print_clue(clue_idxs, idx2clue, new_cell, first=False):
    if len(clue_idxs)==1:
        # print(type(idx2clue[clue_idxs[0]]))
        # if type(idx2clue[clue_idxs[0]]) == clues.found_at:
        return single_find_at(idx2clue[clue_idxs[0]], new_cell, first)
    else:
        if first:
            result= 'First combining clues: '
        else:
            result= 'Then combining clues: '
        for current_idx in clue_idxs:
            result += "<{}>".format(idx2clue[current_idx])
        result += ' Unique Values Rules and the fixed table structure. We know that '
        for cell in new_cell:
            result += cell
            result += '. '
        return result

def single_find_at(clue, new_cell, first=False):
    if first:
        result="First applying clue: "
    else:
        result="Then applying clue: "
    if len(new_cell)>1:
        result += "<{}> and Unique Values We know that ".format(clue)
    else:
        result += "<{}> We know that ".format(clue)
    for cell in new_cell:
        result += cell
        result += '. '
    return result