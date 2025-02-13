import random

from ..dsl import *
from ..utils import *


def generate_d0f5fe59(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, min(30, (h * w) // 9)))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    nfound = 0
    trials = 0
    maxtrials = nobjs * 5
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    while trials < maxtrials and nfound < nobjs:
        oh = unifint(rng, diff_lb, diff_ub, (1, 5))
        ow = unifint(rng, diff_lb, diff_ub, (1, 5))
        bx = asindices(canvas(-1, (oh, ow)))
        sp = rng.choice(totuple(bx))
        shp = {sp}
        dev = unifint(rng, diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = rng.choice((dev, oh * ow - dev))
        ncells = min(max(1, ncells), oh * ow - 1)
        for k in range(ncells):
            ij = rng.choice(totuple((bx - shp) & mapply(dneighbors, shp)))
            shp.add(ij)
        shp = normalize(shp)
        if len(inds) == 0:
            break
        loc = rng.choice(totuple(inds))
        plcd = shift(shp, loc)
        if plcd.issubset(inds):
            gi = fill(gi, fgc, plcd)
            inds = (inds - plcd) - mapply(neighbors, plcd)
            nfound += 1
        trials += 1
    go = canvas(bgc, (nfound, nfound))
    go = fill(go, fgc, connect((0, 0), (nfound - 1, nfound - 1)))
    return {"input": gi, "output": go}


def generate_6e82a1ae(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    b = frozenset({frozenset({ORIGIN, RIGHT}), frozenset({ORIGIN, DOWN})})
    c = frozenset(
        {
            frozenset({ORIGIN, DOWN, UNITY}),
            frozenset({ORIGIN, DOWN, RIGHT}),
            frozenset({UNITY, DOWN, RIGHT}),
            frozenset({UNITY, ORIGIN, RIGHT}),
            shift(frozenset({ORIGIN, UP, DOWN}), DOWN),
            shift(frozenset({ORIGIN, LEFT, RIGHT}), RIGHT),
        }
    )
    d = set()
    for k in range(100):
        shp = {(0, 0)}
        for jj in range(3):
            shp.add(rng.choice(totuple(mapply(dneighbors, shp) - shp)))
        shp = frozenset(normalize(shp))
        d.add(shp)
    d = frozenset(d)
    d, b, c = totuple(d), totuple(b), totuple(c)
    prs = [(b, 3), (c, 2), (d, 1)]
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    reminds = asindices(gi)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, ((h * w) // 2) // 3))
    maxtr = 10
    for k in range(nobjs):
        ntr = 0
        objs, col = rng.choice(prs)
        obj = rng.choice(objs)
        while ntr < maxtr:
            loc = rng.choice(totuple(reminds))
            olcd = shift(obj, loc)
            if olcd.issubset(reminds):
                gi = fill(gi, fgc, olcd)
                go = fill(go, col, olcd)
                reminds = (reminds - olcd) - mapply(dneighbors, olcd)
                break
            ntr += 1
    return {"input": gi, "output": go}


def generate_f2829549(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    remcols = remove(linc, remcols)
    acol = rng.choice(remcols)
    remcols = remove(acol, remcols)
    bcol = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numa = rng.choice((numadev, h * w - numadev))
    numb = rng.choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = rng.sample(inds, numa)
    bset = rng.sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    res = (set(inds) - set(aset)) & (set(inds) - set(bset))
    go = fill(c, 3, res)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_ce22a75a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    ndots = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 3))
    dots = rng.sample(totuple(asindices(c)), ndots)
    gi = fill(c, fgc, dots)
    go = fill(c, 1, mapply(neighbors, dots))
    go = fill(go, 1, dots)
    return {"input": gi, "output": go}


def generate_3c9b0459(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = rot180(gi)
    return {"input": gi, "output": go}


def generate_99b1bc43(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    remcols = remove(linc, remcols)
    acol = rng.choice(remcols)
    remcols = remove(acol, remcols)
    bcol = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numa = rng.choice((numadev, h * w - numadev))
    numb = rng.choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = rng.sample(inds, numa)
    bset = rng.sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    res = (set(bset) - set(aset)) | (set(aset) - set(bset))
    go = fill(c, 3, res)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_b6afb2da(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 4))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = rng.randint(3, 7)
        ow = rng.randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = rng.choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, 2, bd)
            go = fill(go, 4, box(bd))
            go = fill(go, 1, corners(bd))
            succ += 1
            indss = indss - bd
        tr += 1
    return {"input": gi, "output": go}


def generate_c8f0f002(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(7, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    numc = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    numo = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    orng = rng.sample(inds, numo)
    rem = difference(inds, orng)
    gi = fill(c, 7, orng)
    go = fill(c, 5, orng)
    for ij in rem:
        col = rng.choice(ccols)
        gi = fill(gi, col, {ij})
        go = fill(go, col, {ij})
    return {"input": gi, "output": go}


def generate_54d82841(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    nshps = unifint(rng, diff_lb, diff_ub, (1, w // 3))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    locs = interval(1, w - 1, 1)
    for k in range(nshps):
        if len(locs) == 0:
            break
        loc = rng.choice(locs)
        locs = remove(loc, locs)
        locs = remove(loc + 1, locs)
        locs = remove(loc - 1, locs)
        locs = remove(loc + 2, locs)
        locs = remove(loc - 2, locs)
        loci = rng.randint(1, h - 1)
        col = rng.choice(remcols)
        ij = (loci, loc)
        shp = neighbors(ij) - connect((loci + 1, loc - 1), (loci + 1, loc + 1))
        gi = fill(gi, col, shp)
        go = fill(go, col, shp)
        go = fill(go, 4, {(h - 1, loc)})
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_d631b094(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    bgc = 0
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    nc = unifint(rng, diff_lb, diff_ub, (1, min(30, (h * w) // 2 - 1)))
    c = canvas(bgc, (h, w))
    cands = totuple(asindices(c))
    cels = rng.sample(cands, nc)
    gi = fill(c, fgc, cels)
    go = canvas(fgc, (1, nc))
    return {"input": gi, "output": go}


def generate_7c008303(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 13))
    w = unifint(rng, diff_lb, diff_ub, (2, 13))
    h = h * 2
    w = w * 2
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    remcols = remove(linc, remcols)
    fgc = rng.choice(remcols)
    remcols = remove(fgc, remcols)
    fremcols = rng.sample(remcols, unifint(rng, diff_lb, diff_ub, (1, 4)))
    qc = [rng.choice(fremcols) for j in range(4)]
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    ncd = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    nc = rng.choice((ncd, h * w - ncd))
    nc = min(max(0, nc), h * w)
    cels = rng.sample(inds, nc)
    go = fill(c, fgc, cels)
    gi = canvas(bgc, (h + 3, w + 3))
    gi = paint(gi, shift(asobject(go), (3, 3)))
    gi = fill(gi, linc, connect((2, 0), (2, w + 2)))
    gi = fill(gi, linc, connect((0, 2), (h + 2, 2)))
    gi = fill(gi, qc[0], {(0, 0)})
    gi = fill(gi, qc[1], {(0, 1)})
    gi = fill(gi, qc[2], {(1, 0)})
    gi = fill(gi, qc[3], {(1, 1)})
    A = lefthalf(tophalf(go))
    B = righthalf(tophalf(go))
    C = lefthalf(bottomhalf(go))
    D = righthalf(bottomhalf(go))
    A2 = replace(A, fgc, qc[0])
    B2 = replace(B, fgc, qc[1])
    C2 = replace(C, fgc, qc[2])
    D2 = replace(D, fgc, qc[3])
    go = vconcat(hconcat(A2, B2), hconcat(C2, D2))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_dae9d2b5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    acol = rng.choice(remcols)
    remcols = remove(acol, remcols)
    bcol = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numa = rng.choice((numadev, h * w - numadev))
    numb = rng.choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = rng.sample(inds, numa)
    bset = rng.sample(inds, numb)
    if len(set(aset) & set(bset)) == 0:
        bset = bset[:-1] + [rng.choice(aset)]
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    res = set(aset) | set(bset)
    go = fill(c, 6, res)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_aedd82e4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(1, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 30))
    bgc = 0
    remcols = remove(bgc, colopts)
    c = canvas(bgc, (h, w))
    card_bounds = (0, max(0, (h * w) // 2 - 1))
    num = unifint(rng, diff_lb, diff_ub, card_bounds)
    numcols = unifint(rng, diff_lb, diff_ub, (0, min(8, num)))
    inds = totuple(asindices(c))
    chosinds = rng.sample(inds, num)
    choscols = rng.sample(remcols, numcols)
    locs = interval(0, len(chosinds), 1)
    choslocs = rng.sample(locs, numcols)
    gi = canvas(bgc, (h, w))
    for col, endidx in zip(choscols, sorted(choslocs)[::-1]):
        gi = fill(gi, col, chosinds[:endidx])
    objs = objects(gi, F, F, T)
    res = merge(sizefilter(objs, 1))
    go = fill(gi, 1, res)
    return {"input": gi, "output": go}


def generate_c9e6f938(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 15))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hconcat(gi, vmirror(gi))
    return {"input": gi, "output": go}


def generate_913fb3ed(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4, 6, 8))
    sr = (2, 3, 8)
    tr = (1, 6, 4)
    prs = list(zip(sr, tr))
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    numc = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    inds = asindices(gi)
    for k in range(numc):
        if len(inds) == 0:
            break
        loc = rng.choice(totuple(inds))
        a, b = rng.choice(prs)
        inds = (inds - neighbors(loc)) - outbox(neighbors(loc))
        inds = remove(loc, inds)
        gi = fill(gi, a, {loc})
        go = fill(go, a, {loc})
        go = fill(go, b, neighbors(loc))
    return {"input": gi, "output": go}


def generate_6430c8c4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    remcols = remove(linc, remcols)
    acol = rng.choice(remcols)
    remcols = remove(acol, remcols)
    bcol = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numa = rng.choice((numadev, h * w - numadev))
    numb = rng.choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = rng.sample(inds, numa)
    bset = rng.sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    res = (set(inds) - set(aset)) - set(bset)
    go = fill(c, 3, res)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_c0f76784(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (6, 7, 8))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, len(remcols)))
    ccols = rng.sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 20))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = rng.choice((3, 4, 5))
        ow = oh
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = rng.choice(ccols)
        if bd.issubset(indss):
            gi = fill(gi, col, bd)
            go = fill(go, col, bd)
            ccc = oh + 3
            bdx = backdrop(inbox(obj))
            gi = fill(gi, bgc, bdx)
            go = fill(go, ccc, bdx)
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {"input": gi, "output": go}


def generate_3af2c5a8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 15))
    w = unifint(rng, diff_lb, diff_ub, (1, 15))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hconcat(gi, vmirror(gi))
    go = vconcat(go, hmirror(go))
    return {"input": gi, "output": go}


def generate_496994bd(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 14))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    remcols = rng.sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(rng, diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {
        (rng.choice(remcols), rng.choice(totuple(sfilter(bx, lambda ij: ij[0] < h // 2)))),
        (rng.choice(remcols), rng.choice(totuple(sfilter(bx, lambda ij: ij[0] > h // 2)))),
    }
    for kk in range(nc - 2):
        dns = mapply(neighbors, toindices(obj))
        ch = rng.choice(totuple(bx & dns))
        obj.add((rng.choice(remcols), ch))
        bx = bx - {ch}
    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)
    flag = rng.choice((True, False))
    gi = hconcat(gix, canv if flag else hconcat(canvas(bgc, (h, 1)), canv))
    go = hconcat(gix, vmirror(gix) if flag else hconcat(canvas(bgc, (h, 1)), vmirror(gix)))
    if rng.choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    if rng.choice((True, False)):
        gi = hmirror(gi)
        go = hmirror(go)
    return {"input": gi, "output": go}


def generate_bd4472b8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 28))
    w = unifint(rng, diff_lb, diff_ub, (2, 8))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    ccols = rng.sample(remcols, w)
    cc = (tuple(ccols),)
    br = canvas(linc, (1, w))
    lp = canvas(bgc, (h, w))
    gi = vconcat(vconcat(cc, br), lp)
    go = vconcat(vconcat(cc, br), lp)
    pt = hupscale(dmirror(cc), w)
    pto = asobject(pt)
    idx = 2
    while idx < h + 3:
        go = paint(go, shift(pto, (idx, 0)))
        idx += w
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_fafffa47(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    acol = rng.choice(remcols)
    remcols = remove(acol, remcols)
    bcol = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numa = rng.choice((numadev, h * w - numadev))
    numb = rng.choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = rng.sample(inds, numa)
    bset = rng.sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    res = set(inds) - (set(aset) | set(bset))
    go = fill(c, 2, res)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_67e8384a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 14))
    w = unifint(rng, diff_lb, diff_ub, (1, 14))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 9))
    remcols = rng.sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(rng, diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(rng.choice(remcols), rng.choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = rng.choice(totuple(bx & dns))
        obj.add((rng.choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    go = paint(canv, obj)
    go = hconcat(go, vmirror(go))
    go = vconcat(go, hmirror(go))
    return {"input": gi, "output": go}


def generate_ed36ccf7(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = rot270(gi)
    return {"input": gi, "output": go}


def generate_67a3c6ac(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = vmirror(gi)
    return {"input": gi, "output": go}


def generate_a416b8f3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 15))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hconcat(gi, gi)
    return {"input": gi, "output": go}


def generate_d10ecb37(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = crop(gi, (0, 0), (2, 2))
    return {"input": gi, "output": go}


def generate_5bd6f4ac(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = rot90(crop(rot270(gi), (0, 0), (3, 3)))
    return {"input": gi, "output": go}


def generate_7b7f7511(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 15))
    bgc = rng.choice(cols)
    go = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, min(9, h * w - 1)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(go))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        go = fill(go, col, chos)
        inds = difference(inds, chos)
    if rng.choice((True, False)):
        go = dmirror(go)
        gi = vconcat(go, go)
    else:
        gi = hconcat(go, go)
    return {"input": gi, "output": go}


def generate_c59eb873(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 15))
    w = unifint(rng, diff_lb, diff_ub, (1, 15))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = upscale(gi, 2)
    return {"input": gi, "output": go}


def generate_b1948b0a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    npd = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    np = rng.choice((npd, h * w - npd))
    np = min(max(0, npd), h * w)
    gi = canvas(6, (h, w))
    inds = totuple(asindices(gi))
    pp = rng.sample(inds, np)
    npp = difference(inds, pp)
    for ij in npp:
        gi = fill(gi, rng.choice(cols), {ij})
    go = fill(gi, 2, pp)
    return {"input": gi, "output": go}


def generate_25ff71a9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    nc = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    ch = rng.choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc - 1):
        shp.add(rng.choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    go = fill(c, fgc, shift(plcd, (1, 0)))
    return {"input": gi, "output": go}


def generate_f25fbde4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    ncd = unifint(rng, diff_lb, diff_ub, (1, max(1, (min(15, h - 1) * min(15, w - 1)) // 2)))
    nc = rng.choice((ncd, (h - 1) * (w - 1) - ncd))
    nc = min(max(1, ncd), (h - 1) * (w - 1) - 1)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (min(15, h - 1), min(15, w - 1))))
    ch = rng.choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        shp.add(rng.choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    go = compress(gi)
    go = upscale(go, 2)
    return {"input": gi, "output": go}


def generate_a740d043(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    ncd = unifint(rng, diff_lb, diff_ub, (1, max(1, ((h - 1) * (w - 1)) // 2)))
    nc = rng.choice((ncd, (h - 1) * (w - 1) - ncd))
    nc = min(max(1, ncd), (h - 1) * (w - 1) - 1)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, len(remcols)))
    remcols = rng.sample(remcols, numc)
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (h - 1, w - 1)))
    ch = rng.choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        shp.add(rng.choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    obj = {(rng.choice(remcols), ij) for ij in plcd}
    gi = paint(c, obj)
    go = compress(gi)
    go = replace(go, bgc, 0)
    return {"input": gi, "output": go}


def generate_be94b721(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    no = unifint(rng, diff_lb, diff_ub, (3, max(3, (h * w) // 16)))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    nc = unifint(rng, diff_lb, diff_ub, (no + 1, max(no + 1, 2 * no)))
    inds = asindices(c)
    ch = rng.choice(totuple(inds))
    shp = {ch}
    inds = remove(ch, inds)
    for k in range(nc - 1):
        shp.add(rng.choice(totuple((inds - shp) & mapply(dneighbors, shp))))
    inds = (inds - shp) - mapply(neighbors, shp)
    trgc = rng.choice(remcols)
    gi = fill(c, trgc, shp)
    go = fill(canvas(bgc, shape(shp)), trgc, normalize(shp))
    for k in range(no):
        if len(inds) == 0:
            break
        ch = rng.choice(totuple(inds))
        shp = {ch}
        nc2 = unifint(rng, diff_lb, diff_ub, (1, nc - 1))
        for k in range(nc2 - 1):
            cands = totuple((inds - shp) & mapply(dneighbors, shp))
            if len(cands) == 0:
                break
            shp.add(rng.choice(cands))
        col = rng.choice(remcols)
        gi = fill(gi, col, shp)
        inds = (inds - shp) - mapply(neighbors, shp)
    return {"input": gi, "output": go}


def generate_44d8ac46(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, 10))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        tr += 1
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = rng.randint(5, 7)
        ow = rng.randint(5, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            continue
        loci, locj = rng.choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = rng.choice(remcols)
        if bd.issubset(indss):
            ensuresq = rng.choice((True, False))
            if ensuresq:
                dim = rng.randint(1, min(oh, ow) - 2)
                iloci = rng.randint(1, oh - dim - 1)
                ilocj = rng.randint(1, ow - dim - 1)
                inpart = backdrop({(loci + iloci, locj + ilocj), (loci + iloci + dim - 1, locj + ilocj + dim - 1)})
            else:
                cnds = backdrop(inbox(bd))
                ch = rng.choice(totuple(cnds))
                inpart = {ch}
                kk = unifint(rng, diff_lb, diff_ub, (1, len(cnds)))
                for k in range(kk - 1):
                    inpart.add(rng.choice(totuple((cnds - inpart) & mapply(dneighbors, inpart))))
            inpart = frozenset(inpart)
            hi, wi = shape(inpart)
            if hi == wi and len(inpart) == hi * wi:
                incol = 2
            else:
                incol = bgc
            gi = fill(gi, col, bd)
            go = fill(go, col, bd)
            gi = fill(gi, bgc, inpart)
            go = fill(go, incol, inpart)
            succ += 1
            indss = (indss - bd) - outbox(bd)
    return {"input": gi, "output": go}


def generate_3618c87e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc, linc, dotc = rng.sample(cols, 3)
    c = canvas(bgc, (h, w))
    ln = connect((0, 0), (0, w - 1))
    nlocs = unifint(rng, diff_lb, diff_ub, (1, w // 2))
    locs = []
    opts = interval(0, w, 1)
    for k in range(nlocs):
        if len(opts) == 0:
            break
        ch = rng.choice(opts)
        locs.append(ch)
        opts = remove(ch, opts)
        opts = remove(ch - 1, opts)
        opts = remove(ch + 1, opts)
    nlocs = len(opts)
    gi = fill(c, linc, ln)
    go = fill(c, linc, ln)
    for j in locs:
        hh = rng.randint(1, h - 3)
        lnx = connect((0, j), (hh, j))
        gi = fill(gi, linc, lnx)
        go = fill(go, linc, lnx)
        gi = fill(gi, dotc, {(hh + 1, j)})
        go = fill(go, dotc, {(0, j)})
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_b27ca6d3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc, dotc = rng.sample(cols, 2)
    c = canvas(bgc, (h, w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    ndots = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 5))
    nbars = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 12))
    dot = frozenset({(dotc, (1, 1))}) | recolor(bgc, dneighbors((1, 1)))
    bar1 = fill(canvas(bgc, (4, 3)), dotc, {(1, 1), (2, 1)})
    bar2 = dmirror(bar1)
    bar1 = asobject(bar1)
    bar2 = asobject(bar2)
    opts = [dot] * ndots + [rng.choice((bar1, bar2)) for k in range(nbars)]
    rng.shuffle(opts)
    inds = shift(asindices(canvas(-1, (h + 2, w + 2))), (-1, -1))
    for elem in opts:
        loc = (-1, -1)
        tr = 0
        while not toindices(shift(elem, loc)).issubset(inds) and tr < 5:
            loc = rng.choice(totuple(inds))
            tr += 1
        xx = shift(elem, loc)
        if toindices(xx).issubset(inds):
            gi = paint(gi, xx)
            if len(elem) == 12:
                go = paint(go, {cel if cel[0] != bgc else (3, cel[1]) for cel in xx})
            else:
                go = paint(go, xx)
            inds = inds - toindices(xx)
    return {"input": gi, "output": go}


def generate_46f33fce(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 7))
    w = unifint(rng, diff_lb, diff_ub, (2, 7))
    nc = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2 - 1))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    go = canvas(bgc, (h, w))
    gi = canvas(bgc, (h * 2, w * 2))
    inds = totuple(asindices(go))
    locs = rng.sample(inds, nc)
    objo = frozenset({(rng.choice(remcols), ij) for ij in locs})
    f = lambda cij: (cij[0], double(cij[1]))
    obji = shift(apply(f, objo), (1, 1))
    gi = paint(gi, obji)
    go = paint(go, objo)
    go = upscale(go, 4)
    return {"input": gi, "output": go}


def generate_a79310a0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    nc = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    ch = rng.choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc - 1):
        shp.add(rng.choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    go = fill(c, 2, shift(plcd, (1, 0)))
    return {"input": gi, "output": go}


def generate_dc1df850(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    nc = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2 - 1))
    nreddev = unifint(rng, diff_lb, diff_ub, (0, nc // 2))
    nred = rng.choice((nreddev, nc - nreddev))
    nred = min(max(0, nred), nc)
    inds = totuple(asindices(c))
    occ = rng.sample(inds, nc)
    reds = rng.sample(occ, nred)
    others = difference(occ, reds)
    c = fill(c, 2, reds)
    obj = frozenset({(rng.choice(remcols), ij) for ij in others})
    c = paint(c, obj)
    gi = tuple(r for r in c)
    go = underfill(c, 1, mapply(neighbors, frozenset(reds)))
    return {"input": gi, "output": go}


def generate_f76d97a5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, remove(5, interval(0, 10, 1)))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    col = rng.choice(cols)
    gi = canvas(5, (h, w))
    go = canvas(col, (h, w))
    numdev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    num = rng.choice((numdev, h * w - numdev))
    num = min(max(1, num), h * w)
    inds = totuple(asindices(gi))
    locs = rng.sample(inds, num)
    gi = fill(gi, col, locs)
    go = fill(go, 0, locs)
    return {"input": gi, "output": go}


def generate_0d3d703e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    incols = (1, 2, 3, 4, 5, 6, 8, 9)
    outcols = (5, 6, 4, 3, 1, 2, 9, 8)
    k = len(incols)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    gi = canvas(-1, (h, w))
    go = canvas(-1, (h, w))
    inds = asindices(gi)
    numc = unifint(rng, diff_lb, diff_ub, (1, k))
    idxes = rng.sample(interval(0, k, 1), numc)
    for ij in inds:
        idx = rng.choice(idxes)
        gi = fill(gi, incols[idx], {ij})
        go = fill(go, outcols[idx], {ij})
    return {"input": gi, "output": go}


def generate_445eab21(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    bigcol, area = 0, 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = rng.randint(3, 7)
        ow = rng.randint(3, 7)
        if oh * ow == area:
            continue
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = rng.choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, box(bd))
            succ += 1
            indss = indss - bd
            if oh * ow > area:
                bigcol, area = col, oh * ow
        tr += 1
    go = canvas(bigcol, (2, 2))
    return {"input": gi, "output": go}


def generate_b94a9452(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc, outer, inner = rng.sample(cols, 3)
    c = canvas(bgc, (h, w))
    oh = unifint(rng, diff_lb, diff_ub, (3, h - 1))
    ow = unifint(rng, diff_lb, diff_ub, (3, w - 1))
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    oh2d = unifint(rng, diff_lb, diff_ub, (0, oh // 2))
    ow2d = unifint(rng, diff_lb, diff_ub, (0, ow // 2))
    oh2 = rng.choice((oh2d, oh - oh2d))
    oh2 = min(max(1, oh2), oh - 2)
    ow2 = rng.choice((ow2d, ow - ow2d))
    ow2 = min(max(1, ow2), ow - 2)
    loci2 = rng.randint(loci + 1, loci + oh - oh2 - 1)
    locj2 = rng.randint(locj + 1, locj + ow - ow2 - 1)
    obj1 = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    obj2 = backdrop(frozenset({(loci2, locj2), (loci2 + oh2 - 1, locj2 + ow2 - 1)}))
    gi = fill(c, outer, obj1)
    gi = fill(gi, inner, obj2)
    go = compress(gi)
    go = switch(go, outer, inner)
    return {"input": gi, "output": go}


def generate_e9afcf9a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    numc = unifint(rng, diff_lb, diff_ub, (1, min(10, h)))
    colss = rng.sample(cols, numc)
    rr = tuple(rng.choice(colss) for k in range(h))
    rr2 = rr[::-1]
    gi = []
    go = []
    for k in range(w):
        gi.append(rr)
        if k % 2 == 0:
            go.append(rr)
        else:
            go.append(rr2)
    gi = dmirror(tuple(gi))
    go = dmirror(tuple(go))
    return {"input": gi, "output": go}


def generate_e9614598(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    r = rng.randint(0, h - 1)
    sizh = unifint(rng, diff_lb, diff_ub, (2, w // 2))
    siz = 2 * sizh + 1
    siz = min(max(5, siz), w)
    locj = rng.randint(0, w - siz)
    bgc, dotc = rng.sample(cols, 2)
    c = canvas(bgc, (h, w))
    A = (r, locj)
    B = (r, locj + siz - 1)
    gi = fill(c, dotc, {A, B})
    locc = (r, locj + siz // 2)
    go = fill(gi, 3, {locc})
    go = fill(go, 3, dneighbors(locc))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_d23f8c26(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    wh = unifint(rng, diff_lb, diff_ub, (1, 14))
    w = 2 * wh + 1
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    numn = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    numcols = unifint(rng, diff_lb, diff_ub, (1, 9))
    remcols = rng.sample(remcols, numcols)
    inds = totuple(asindices(gi))
    locs = rng.sample(inds, numn)
    for ij in locs:
        col = rng.choice(remcols)
        gi = fill(gi, col, {ij})
        a, b = ij
        if b == w // 2:
            go = fill(go, col, {ij})
    return {"input": gi, "output": go}


def generate_ce9e57f2(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    nbars = unifint(rng, diff_lb, diff_ub, (2, (w - 2) // 2))
    locopts = interval(1, w - 1, 1)
    barlocs = []
    for k in range(nbars):
        if len(locopts) == 0:
            break
        loc = rng.choice(locopts)
        barlocs.append(loc)
        locopts = remove(loc, locopts)
        locopts = remove(loc + 1, locopts)
        locopts = remove(loc - 1, locopts)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, 8))
    colss = rng.sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for j in barlocs:
        barloci = unifint(rng, diff_lb, diff_ub, (1, h - 2))
        fullbar = connect((0, j), (barloci, j))
        halfbar = connect((0, j), (barloci // 2 if barloci % 2 == 1 else (barloci - 1) // 2, j))
        barcol = rng.choice(colss)
        gi = fill(gi, barcol, fullbar)
        go = fill(go, barcol, fullbar)
        go = fill(go, 8, halfbar)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_b9b7f026(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    outcol = None
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = rng.randint(3, 7)
        ow = rng.randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = rng.choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            succ += 1
            indss = indss - bd
            if outcol is None:
                outcol = col
                cands = totuple(backdrop(inbox(bd)))
                bd2 = backdrop(frozenset(rng.sample(cands, 2)) if len(cands) > 2 else frozenset(cands))
                gi = fill(gi, bgc, bd2)
        tr += 1
    go = canvas(outcol, (1, 1))
    return {"input": gi, "output": go}


def generate_6d75e8bb(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    nc = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    ch = rng.choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc - 1):
        shp.add(rng.choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    go = fill(c, 2, backdrop(plcd))
    go = fill(go, fgc, plcd)
    return {"input": gi, "output": go}


def generate_3f7978a0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc, noisec, linec = rng.sample(cols, 3)
    c = canvas(bgc, (h, w))
    oh = unifint(rng, diff_lb, diff_ub, (4, max(4, int((2 / 3) * h))))
    oh = min(oh, h)
    ow = unifint(rng, diff_lb, diff_ub, (4, max(4, int((2 / 3) * w))))
    ow = min(ow, w)
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    nnoise = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 4))
    inds = totuple(asindices(c))
    noise = rng.sample(inds, nnoise)
    gi = fill(c, noisec, noise)
    ulc = (loci, locj)
    lrc = (loci + oh - 1, locj + ow - 1)
    llc = (loci + oh - 1, locj)
    urc = (loci, locj + ow - 1)
    gi = fill(gi, linec, connect(ulc, llc))
    gi = fill(gi, linec, connect(urc, lrc))
    crns = {ulc, lrc, llc, urc}
    gi = fill(gi, noisec, crns)
    go = subgrid(crns, gi)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_e76a88a6(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    objh = unifint(rng, diff_lb, diff_ub, (2, 5))
    objw = unifint(rng, diff_lb, diff_ub, (2, 5))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {rng.choice(totuple(bounds))}
    nc = unifint(rng, diff_lb, diff_ub, (2, len(bounds) - 2))
    for j in range(nc):
        ij = rng.choice(totuple((bounds - shp) & mapply(dneighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    dmyc = rng.choice(remcols)
    remcols = remove(dmyc, remcols)
    oh, ow = shape(shp)
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
    numco = unifint(rng, diff_lb, diff_ub, (2, 8))
    colll = rng.sample(remcols, numco)
    shppc = frozenset({(rng.choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(rng.choice(colll), ij) for ij in shpp})
    shppcn = normalize(shppc)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shppc)
    go = tuple(e for e in gi)
    ub = ((h * w) / (oh * ow)) // 2
    ub = max(1, ub)
    numlocs = unifint(rng, diff_lb, diff_ub, (1, ub))
    cnt = 0
    fails = 0
    maxfails = 5 * numlocs
    idns = (asindices(gi) - shpp) - mapply(dneighbors, shpp)
    idns = sfilter(idns, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
    while cnt < numlocs and fails < maxfails:
        if len(idns) == 0:
            break
        loc = rng.choice(totuple(idns))
        plcd = shift(shppcn, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(idns):
            go = paint(go, plcd)
            gi = fill(gi, dmyc, plcdi)
            cnt += 1
            idns = (idns - plcdi) - mapply(dneighbors, plcdi)
        else:
            fails += 1
    return {"input": gi, "output": go}


def generate_a61f2674(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, remove(1, interval(0, 10, 1)))
    w = unifint(rng, diff_lb, diff_ub, (5, 28))
    h = unifint(rng, diff_lb, diff_ub, (w // 2 + 1, 30))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nbars = unifint(rng, diff_lb, diff_ub, (2, w // 2))
    barlocs = []
    options = interval(0, w, 1)
    while len(options) > 0 and len(barlocs) < nbars:
        loc = rng.choice(options)
        barlocs.append(loc)
        options = remove(loc, options)
        options = remove(loc + 1, options)
        options = remove(loc - 1, options)
    barheights = rng.sample(interval(0, h, 1), nbars)
    for j, bh in zip(barlocs, barheights):
        gi = fill(gi, fgc, connect((0, j), (bh, j)))
        if bh == max(barheights):
            go = fill(go, 1, connect((0, j), (bh, j)))
        if bh == min(barheights):
            go = fill(go, 2, connect((0, j), (bh, j)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_ce4f8723(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    barcol = rng.choice(remcols)
    remcols = remove(barcol, remcols)
    cola = rng.choice(remcols)
    colb = rng.choice(remove(cola, remcols))
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(rng, diff_lb, diff_ub, devrng)
    devb = unifint(rng, diff_lb, diff_ub, devrng)
    sgna = rng.choice((+1, -1))
    sgnb = rng.choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = rng.sample(inds, numa)
    b = rng.sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    go = fill(canv, 3, set(a) | set(b))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_caa06a1f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    vp = unifint(rng, diff_lb, diff_ub, (2, h // 2 - 1))
    hp = unifint(rng, diff_lb, diff_ub, (2, w // 2 - 1))
    bgc = rng.choice(cols)
    numc = unifint(rng, diff_lb, diff_ub, (2, min(8, max(2, hp * vp))))
    remcols = remove(bgc, cols)
    ccols = rng.sample(remcols, numc)
    remcols = difference(remcols, ccols)
    tric = rng.choice(remcols)
    obj = {(rng.choice(ccols), ij) for ij in asindices(canvas(-1, (vp, hp)))}
    go = canvas(bgc, (h, w))
    gi = canvas(bgc, (h, w))
    for a in range(-vp, h + 1, vp):
        for b in range(-hp, w + 1, hp):
            go = paint(go, shift(obj, (a, b + 1)))
    for a in range(-vp, h + 1, vp):
        for b in range(-hp, w + 1, hp):
            gi = paint(gi, shift(obj, (a, b)))
    ioffs = unifint(rng, diff_lb, diff_ub, (1, h - 2 * vp))
    joffs = unifint(rng, diff_lb, diff_ub, (1, w - 2 * hp))
    for a in range(ioffs):
        gi = fill(gi, tric, connect((a, 0), (a, w - 1)))
    for b in range(joffs):
        gi = fill(gi, tric, connect((0, b), (h - 1, b)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_94f9d214(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    acol = rng.choice(remcols)
    remcols = remove(acol, remcols)
    bcol = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numa = rng.choice((numadev, h * w - numadev))
    numb = rng.choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = rng.sample(inds, numa)
    bset = rng.sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    res = (set(inds) - set(aset)) & (set(inds) - set(bset))
    go = fill(c, 2, res)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_feca6190(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    w = unifint(rng, diff_lb, diff_ub, (2, 6))
    bgc = 0
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (1, min(w, 5)))
    ccols = rng.sample(remcols, ncols)
    cands = interval(0, w, 1)
    locs = rng.sample(cands, ncols)
    gi = canvas(bgc, (1, w))
    go = canvas(bgc, (w * ncols, w * ncols))
    for col, j in zip(ccols, locs):
        gi = fill(gi, col, {(0, j)})
        go = fill(go, col, shoot((w * ncols - 1, j), UP_RIGHT))
    return {"input": gi, "output": go}


def generate_d5d6de2d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, 16))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = rng.randint(1, 7)
        ow = rng.randint(1, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = rng.choice(remcols)
        if bd.issubset(indss):
            gi = fill(gi, col, box(bd))
            if oh > 2 and ow > 2:
                go = fill(go, 3, backdrop(inbox(bd)))
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {"input": gi, "output": go}


def generate_4612dd53(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    ih = unifint(rng, diff_lb, diff_ub, (5, h - 1))
    iw = unifint(rng, diff_lb, diff_ub, (5, w - 1))
    bgc, col = rng.sample(cols, 2)
    loci = rng.randint(0, h - ih)
    locj = rng.randint(0, w - iw)
    bx = box(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    if rng.choice((True, False)):
        locc = rng.randint(loci + 2, loci + ih - 3)
        br = connect((locc, locj + 1), (locc, locj + iw - 2))
    else:
        locc = rng.randint(locj + 2, locj + iw - 3)
        br = connect((loci + 1, locc), (loci + ih - 2, locc))
    c = canvas(bgc, (h, w))
    crns = rng.sample(totuple(corners(bx)), 3)
    onbx = totuple(crns)
    rembx = difference(bx, crns)
    onbr = rng.sample(totuple(br), 2)
    rembr = difference(br, onbr)
    noccbx = unifint(rng, diff_lb, diff_ub, (0, len(rembx)))
    noccbr = unifint(rng, diff_lb, diff_ub, (0, len(rembr)))
    occbx = rng.sample(totuple(rembx), noccbx)
    occbr = rng.sample(totuple(rembr), noccbr)
    c = fill(c, col, bx)
    c = fill(c, col, br)
    gi = fill(c, bgc, occbx)
    gi = fill(gi, bgc, occbr)
    go = fill(c, 2, occbx)
    go = fill(go, 2, occbr)
    if rng.choice((True, False)):
        gi = fill(gi, bgc, br)
        go = fill(go, bgc, br)
    return {"input": gi, "output": go}


def generate_1f642eb9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    ih = unifint(rng, diff_lb, diff_ub, (2, min(h - 4, 2 * (h // 3))))
    iw = unifint(rng, diff_lb, diff_ub, (2, min(w - 4, 2 * (w // 3))))
    loci = rng.randint(2, h - ih - 2)
    locj = rng.randint(2, w - iw - 2)
    bgc, sqc = rng.sample(cols, 2)
    remcols = difference(cols, (bgc, sqc))
    numcells = unifint(rng, diff_lb, diff_ub, (1, 2 * ih + 2 * iw - 4))
    outs = []
    ins = []
    c1 = rng.choice((True, False))
    c2 = rng.choice((True, False))
    c3 = rng.choice((True, False))
    c4 = rng.choice((True, False))
    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0))
        ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, w - 1))
        ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - (c3)):
        outs.append((0, b))
        ins.append((loci, b))
    for b in range(locj + (c2), locj + iw - (c4)):
        outs.append((h - 1, b))
        ins.append((loci + ih - 1, b))
    inds = interval(0, 2 * ih + 2 * iw - 4, 1)
    locs = rng.sample(inds, numcells)
    numc = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numc)
    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]
    c = canvas(bgc, (h, w))
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    gi = fill(c, sqc, bd)
    seq = [rng.choice(ccols) for k in range(numcells)]
    for c, loc in zip(seq, outs):
        gi = fill(gi, c, {loc})
    go = tuple(e for e in gi)
    for c, loc in zip(seq, ins):
        go = fill(go, c, {loc})
    return {"input": gi, "output": go}


def generate_681b3aeb(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    fullsuc = False
    while not fullsuc:
        hi = unifint(rng, diff_lb, diff_ub, (2, 8))
        wi = unifint(rng, diff_lb, diff_ub, (2, 8))
        h = unifint(rng, diff_lb, diff_ub, ((3 * hi, 30)))
        w = unifint(rng, diff_lb, diff_ub, ((3 * wi, 30)))
        c = canvas(-1, (hi, hi))
        bgc, ca, cb = rng.sample(cols, 3)
        gi = canvas(bgc, (h, w))
        conda, condb = True, True
        while conda and condb:
            inds = totuple(asindices(c))
            pa = rng.choice(inds)
            reminds = remove(pa, inds)
            pb = rng.choice(reminds)
            reminds = remove(pb, reminds)
            A = {pa}
            B = {pb}
            for k in range(len(reminds)):
                acands = set(reminds) & mapply(dneighbors, A)
                bcands = set(reminds) & mapply(dneighbors, B)
                opts = []
                if len(acands) > 0:
                    opts.append(0)
                if len(bcands) > 0:
                    opts.append(1)
                idx = rng.choice(opts)
                if idx == 0:
                    loc = rng.choice(totuple(acands))
                    A.add(loc)
                else:
                    loc = rng.choice(totuple(bcands))
                    B.add(loc)
                reminds = remove(loc, reminds)
            conda = len(A) == height(A) * width(A)
            condb = len(B) == height(B) * width(B)
        go = fill(c, ca, A)
        go = fill(go, cb, B)
        fullocs = totuple(asindices(gi))
        A = normalize(A)
        B = normalize(B)
        ha, wa = shape(A)
        hb, wb = shape(B)
        minisuc = False
        if not (ha > h or wa > w):
            for kkk in range(10):
                locai = rng.randint(0, h - ha)
                locaj = rng.randint(0, w - wa)
                plcda = shift(A, (locaj, locaj))
                remlocs = difference(fullocs, plcda)
                remlocs2 = sfilter(remlocs, lambda ij: ij[0] <= h - hb and ij[1] <= w - wb)
                if len(remlocs2) == 0:
                    continue
                ch = rng.choice(remlocs2)
                plcdb = shift(B, (ch))
                if set(plcdb).issubset(set(remlocs2)):
                    minisuc = True
                    break
        if minisuc:
            fullsuc = True
    gi = fill(gi, ca, plcda)
    gi = fill(gi, cb, plcdb)
    return {"input": gi, "output": go}


def generate_d364b489(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 6, 7, 8))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 5))
    res = set()
    for j in range(num):
        if len(inds) == 0:
            break
        r = rng.choice(inds)
        inds = remove(r, inds)
        inds = difference(inds, neighbors(r))
        inds = difference(inds, totuple(shift(apply(rbind(multiply, TWO), dneighbors(ORIGIN)), r)))
        res.add(r)
    gi = fill(gi, fgc, res)
    go = fill(gi, 7, shift(res, LEFT))
    go = fill(go, 6, shift(res, RIGHT))
    go = fill(go, 8, shift(res, DOWN))
    go = fill(go, 2, shift(res, UP))
    return {"input": gi, "output": go}


def generate_25d8a9c8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    gi = []
    go = []
    ncols = unifint(rng, diff_lb, diff_ub, (2, 10))
    ccols = rng.sample(cols, ncols)
    for k in range(h):
        singlecol = rng.choice((True, False))
        col = rng.choice(ccols)
        row = repeat(col, w)
        if singlecol:
            gi.append(row)
            go.append(repeat(5, w))
        else:
            remcols = remove(col, ccols)
            nothercinv = unifint(rng, diff_lb, diff_ub, (1, w - 1))
            notherc = w - 1 - nothercinv
            notherc = min(max(1, notherc), w - 1)
            row = list(row)
            indss = interval(0, w, 1)
            for j in rng.sample(indss, notherc):
                row[j] = rng.choice(remcols)
            gi.append(tuple(row))
            go.append(repeat(0, w))
    gi = tuple(gi)
    go = tuple(go)
    return {"input": gi, "output": go}


def generate_bda2d7a6(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 14))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    ncols = unifint(rng, diff_lb, diff_ub, (2, 10))
    cols = rng.sample(colopts, ncols)
    colord = [rng.choice(cols) for j in range(min(h, w))]
    shp = (h * 2, w * 2)
    gi = canvas(0, shp)
    for idx, (ci, co) in enumerate(zip(colord, colord[-1:] + colord[:-1])):
        ulc = (idx, idx)
        lrc = (h * 2 - 1 - idx, w * 2 - 1 - idx)
        bx = box(frozenset({ulc, lrc}))
        gi = fill(gi, ci, bx)
    I = gi
    objso = order(objects(I, T, F, F), compose(maximum, shape))
    if color(objso[0]) == color(objso[-1]):
        objso = (combine(objso[0], objso[-1]),) + objso[1:-1]
    res = mpapply(recolor, apply(color, objso), (objso[-1],) + objso[:-1])
    go = paint(gi, res)
    return {"input": gi, "output": go}


def generate_a5f85a15(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(4, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    startlocs = apply(toivec, interval(h - 1, 0, -1)) + apply(tojvec, interval(0, w, 1))
    cands = interval(0, h + w - 1, 1)
    num = unifint(rng, diff_lb, diff_ub, (1, (h + w - 1) // 3))
    locs = []
    for k in range(num):
        if len(cands) == 0:
            break
        loc = rng.choice(cands)
        locs.append(loc)
        cands = remove(loc, cands)
        cands = remove(loc - 1, cands)
        cands = remove(loc + 1, cands)
    locs = set([startlocs[loc] for loc in locs])
    bgc, fgc = rng.sample(colopts, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for loc in locs:
        ln = order(shoot(loc, (1, 1)), first)
        gi = fill(gi, fgc, ln)
        go = fill(go, fgc, ln)
        go = fill(go, 4, ln[1::2])
    return {"input": gi, "output": go}


def generate_32597951(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    ih = unifint(rng, diff_lb, diff_ub, (2, h // 2))
    iw = unifint(rng, diff_lb, diff_ub, (2, w // 2))
    bgc, noisec, fgc = rng.sample(cols, 3)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    ndev = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 2))
    num = rng.choice((ndev, h * w - ndev))
    num = min(max(num, 0), h * w)
    ofc = rng.sample(inds, num)
    c = fill(c, noisec, ofc)
    loci = rng.randint(0, h - ih)
    locj = rng.randint(0, w - iw)
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    tofillfc = bd & ofcolor(c, bgc)
    gi = fill(c, fgc, tofillfc)
    if len(tofillfc) > 0:
        go = fill(gi, 3, backdrop(tofillfc) & ofcolor(gi, noisec))
    else:
        go = gi
    return {"input": gi, "output": go}


def generate_cf98881b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 9))
    bgc, barcol, cola, colb, colc = rng.sample(cols, 5)
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(rng, diff_lb, diff_ub, devrng)
    devb = unifint(rng, diff_lb, diff_ub, devrng)
    devc = unifint(rng, diff_lb, diff_ub, devrng)
    sgna = rng.choice((+1, -1))
    sgnb = rng.choice((+1, -1))
    sgnc = rng.choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    devc = sgnc * devc
    numa = mp + deva
    numb = mp + devb
    numc = mp + devc
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    numc = max(min(h * w - 1, numc), 1)
    a = rng.sample(inds, numa)
    b = rng.sample(inds, numb)
    c = rng.sample(inds, numc)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gic = fill(canv, colc, c)
    gi = hconcat(hconcat(hconcat(gia, gbar), hconcat(gib, gbar)), gic)
    go = fill(gic, colb, b)
    go = fill(go, cola, a)
    return {"input": gi, "output": go}


def generate_41e4d17e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 16))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    bx = box(frozenset({(0, 0), (4, 4)}))
    bd = backdrop(bx)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = rng.choice(totuple(inds))
        bxs = shift(bx, loc)
        if bxs.issubset(set(inds)):
            gi = fill(gi, fgc, bxs)
            go = fill(go, fgc, bxs)
            cen = center(bxs)
            frns = hfrontier(cen) | vfrontier(cen)
            kep = frns & ofcolor(go, bgc)
            go = fill(go, 6, kep)
            inds = difference(inds, shift(bd, loc))
            succ += 1
        tr += 1
    return {"input": gi, "output": go}


def generate_91714a58(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    bgc, targc = rng.sample(cols, 2)
    remcols = remove(bgc, cols)
    nnoise = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 2))
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    noise = rng.sample(inds, nnoise)
    ih = rng.randint(2, h // 2)
    iw = rng.randint(2, w // 2)
    loci = rng.randint(0, h - ih)
    locj = rng.randint(0, w - iw)
    loc = (loci, locj)
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    go = fill(gi, targc, bd)
    for ij in noise:
        col = rng.choice(remcols)
        gi = fill(gi, col, {ij})
    gi = fill(gi, targc, bd)
    return {"input": gi, "output": go}


def generate_b60334d2(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 9))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    bx = box(frozenset({(0, 0), (2, 2)}))
    bd = backdrop(bx)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = rng.choice(totuple(inds))
        bxs = shift(bx, loc)
        if bxs.issubset(set(inds)):
            cen = center(bxs)
            gi = fill(gi, fgc, {cen})
            go = fill(go, fgc, ineighbors(cen))
            go = fill(go, 1, dneighbors(cen))
            inds = difference(inds, shift(bd, loc))
            succ += 1
        tr += 1
    return {"input": gi, "output": go}


def generate_952a094c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    ih = unifint(rng, diff_lb, diff_ub, (4, h - 2))
    iw = unifint(rng, diff_lb, diff_ub, (4, w - 2))
    loci = rng.randint(1, h - ih - 1)
    locj = rng.randint(1, w - iw - 1)
    sp = (loci, locj)
    ep = (loci + ih - 1, locj + iw - 1)
    bx = box(frozenset({sp, ep}))
    bgc, fgc, a, b, c, d = rng.sample(cols, 6)
    canv = canvas(bgc, (h, w))
    canvv = fill(canv, fgc, bx)
    gi = tuple(e for e in canvv)
    go = tuple(e for e in canvv)
    gi = fill(gi, a, {(loci + 1, locj + 1)})
    go = fill(go, a, {(loci + ih, locj + iw)})
    gi = fill(gi, b, {(loci + 1, locj + iw - 2)})
    go = fill(go, b, {(loci + ih, locj - 1)})
    gi = fill(gi, c, {(loci + ih - 2, locj + 1)})
    go = fill(go, c, {(loci - 1, locj + iw)})
    gi = fill(gi, d, {(loci + ih - 2, locj + iw - 2)})
    go = fill(go, d, {(loci - 1, locj - 1)})
    return {"input": gi, "output": go}


def generate_b8cdaf2b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc, linc, dotc = rng.sample(cols, 3)
    lin = connect((0, 0), (0, w - 1))
    winv = unifint(rng, diff_lb, diff_ub, (2, w - 1))
    w2 = w - winv
    w2 = min(max(w2, 1), w - 2)
    locj = rng.randint(1, w - w2 - 1)
    bar2 = connect((0, locj), (0, locj + w2 - 1))
    c = canvas(bgc, (h, w))
    gi = fill(c, linc, lin)
    gi = fill(gi, dotc, bar2)
    gi = fill(gi, linc, shift(bar2, (1, 0)))
    go = fill(gi, dotc, shoot((2, locj - 1), (1, -1)))
    go = fill(go, dotc, shoot((2, locj + w2), (1, 1)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_b548a754(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    hi = unifint(rng, diff_lb, diff_ub, (4, h - 1))
    wi = unifint(rng, diff_lb, diff_ub, (3, w - 1))
    loci = rng.randint(0, h - hi)
    locj = rng.randint(0, w - wi)
    bx = box(frozenset({(loci, locj), (loci + hi - 1, locj + wi - 1)}))
    ins = backdrop(inbox(bx))
    bgc, boxc, inc, dotc = rng.sample(cols, 4)
    c = canvas(bgc, (h, w))
    go = fill(c, boxc, bx)
    go = fill(go, inc, ins)
    cutoff = rng.randint(loci + 2, loci + hi - 2)
    bx2 = box(frozenset({(loci, locj), (cutoff, locj + wi - 1)}))
    ins2 = backdrop(inbox(bx2))
    gi = fill(c, boxc, bx2)
    gi = fill(gi, inc, ins2)
    locc = rng.choice(totuple(connect((loci + hi - 1, locj), (loci + hi - 1, locj + wi - 1))))
    gi = fill(gi, dotc, {locc})
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_95990924(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4))
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 16))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    bx = box(frozenset({(0, 0), (3, 3)}))
    bd = backdrop(bx)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = rng.choice(totuple(inds))
        bxs = shift(bx, loc)
        if bxs.issubset(set(inds)):
            gi = fill(gi, fgc, inbox(bxs))
            go = fill(go, fgc, inbox(bxs))
            go = fill(go, 1, {loc})
            go = fill(go, 2, {add(loc, (0, 3))})
            go = fill(go, 3, {add(loc, (3, 0))})
            go = fill(go, 4, {add(loc, (3, 3))})
            inds = difference(inds, shift(bd, loc))
            succ += 1
        tr += 1
    return {"input": gi, "output": go}


def generate_f1cefba8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (7, 30))
    ih = unifint(rng, diff_lb, diff_ub, (6, h - 1))
    iw = unifint(rng, diff_lb, diff_ub, (6, w - 1))
    loci = rng.randint(0, h - ih)
    locj = rng.randint(0, w - iw)
    bgc, ringc, inc = rng.sample(cols, 3)
    obj = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    ring1 = box(obj)
    ring2 = inbox(obj)
    bd = backdrop(obj)
    c = canvas(bgc, (h, w))
    c = fill(c, inc, bd)
    c = fill(c, ringc, ring1 | ring2)
    cands = totuple(ring2 - corners(ring2))
    numc = unifint(rng, diff_lb, diff_ub, (1, len(cands) // 2))
    locs = rng.sample(cands, numc)
    gi = fill(c, inc, locs)
    lm = lowermost(ring2)
    hori = sfilter(locs, lambda ij: ij[0] > loci + 1 and ij[0] < lm)
    verti = difference(locs, hori)
    hlines = mapply(hfrontier, hori)
    vlines = mapply(vfrontier, verti)
    fulllocs = set(hlines) | set(vlines)
    topaintinc = fulllocs & ofcolor(c, bgc)
    topaintringc = fulllocs & ofcolor(c, inc)
    go = fill(c, inc, topaintinc)
    go = fill(go, ringc, topaintringc)
    return {"input": gi, "output": go}


def generate_c444b776(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 9))
    w = unifint(rng, diff_lb, diff_ub, (2, 9))
    nh = unifint(rng, diff_lb, diff_ub, (1, 3))
    nw = unifint(rng, diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
    bgclinc = rng.sample(cols, 2)
    bgc, linc = bgclinc
    remcols = difference(cols, bgclinc)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    inds = totuple(asindices(smallc))
    numcol = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numcol)
    numcels = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 2))
    cels = rng.sample(inds, numcels)
    obj = {(rng.choice(ccols), ij) for ij in cels}
    smallcpainted = paint(smallc, obj)
    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = rng.choice(llocs)
    obj = asobject(smallcpainted)
    gi = paint(c, shift(obj, srcloc))
    remlocs = remove(srcloc, llocs)
    bobj = asobject(smallc)
    for rl in remlocs:
        gi = paint(gi, shift(bobj, rl))
    go = tuple(e for e in gi)
    for rl in remlocs:
        go = paint(go, shift(obj, rl))
    return {"input": gi, "output": go}


def generate_97999447(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    opts = interval(0, h, 1)
    num = unifint(rng, diff_lb, diff_ub, (1, h))
    locs = rng.sample(opts, num)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for idx in locs:
        col = rng.choice(ccols)
        j = rng.randint(0, w - 1)
        dot = (idx, j)
        gi = fill(gi, col, {dot})
        go = fill(go, col, {(idx, x) for x in range(j, w, 2)})
        go = fill(go, 5, {(idx, x) for x in range(j + 1, w, 2)})
    return {"input": gi, "output": go}


def generate_d89b689b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc, sqc, a, b, c, d = rng.sample(cols, 6)
    loci = rng.randint(1, h - 3)
    locj = rng.randint(1, w - 3)
    canv = canvas(bgc, (h, w))
    go = fill(canv, a, {(loci, locj)})
    go = fill(go, b, {(loci, locj + 1)})
    go = fill(go, c, {(loci + 1, locj)})
    go = fill(go, d, {(loci + 1, locj + 1)})
    inds = totuple(asindices(canv))
    aopts = sfilter(inds, lambda ij: ij[0] < loci and ij[1] < locj)
    bopts = sfilter(inds, lambda ij: ij[0] < loci and ij[1] > locj + 1)
    copts = sfilter(inds, lambda ij: ij[0] > loci + 1 and ij[1] < locj)
    dopts = sfilter(inds, lambda ij: ij[0] > loci + 1 and ij[1] > locj + 1)
    aopts = order(aopts, lambda ij: manhattan({ij}, {(loci, locj)}))
    bopts = order(bopts, lambda ij: manhattan({ij}, {(loci, locj + 1)}))
    copts = order(copts, lambda ij: manhattan({ij}, {(loci + 1, locj)}))
    dopts = order(dopts, lambda ij: manhattan({ij}, {(loci + 1, locj + 1)}))
    aidx = unifint(rng, diff_lb, diff_ub, (0, len(aopts) - 1))
    bidx = unifint(rng, diff_lb, diff_ub, (0, len(bopts) - 1))
    cidx = unifint(rng, diff_lb, diff_ub, (0, len(copts) - 1))
    didx = unifint(rng, diff_lb, diff_ub, (0, len(dopts) - 1))
    loca = aopts[aidx]
    locb = bopts[bidx]
    locc = copts[cidx]
    locd = dopts[didx]
    gi = fill(canv, sqc, backdrop({(loci, locj), (loci + 1, locj + 1)}))
    gi = fill(gi, a, {loca})
    gi = fill(gi, b, {locb})
    gi = fill(gi, c, {locc})
    gi = fill(gi, d, {locd})
    return {"input": gi, "output": go}


def generate_543a7ed5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3, 4))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, 7))
    ccols = rng.sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 25))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = rng.randint(4, 8)
        ow = rng.randint(4, 8)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = rng.choice(ccols)
        if bd.issubset(indss):
            bdibd = backdrop(frozenset({(loci + 1, locj + 1), (loci + oh - 2, locj + ow - 2)}))
            go = fill(go, col, bdibd)
            go = fill(go, 3, box(bd))
            gi = fill(gi, col, bdibd)
            if oh > 5 and ow > 5 and rng.randint(1, 10) != 1:
                ulci, ulcj = ulcorner(bdibd)
                lrci, lrcj = lrcorner(bdibd)
                aa = rng.randint(ulci + 1, lrci - 1)
                aa = rng.randint(ulci + 1, aa)
                bb = rng.randint(ulcj + 1, lrcj - 1)
                bb = rng.randint(ulcj + 1, bb)
                cc = rng.randint(aa, lrci - 1)
                dd = rng.randint(bb, lrcj - 1)
                cc = rng.randint(cc, lrci - 1)
                dd = rng.randint(dd, lrcj - 1)
                ins = backdrop({(aa, bb), (cc, dd)})
                go = fill(go, 4, ins)
                gi = fill(gi, bgc, ins)
            succ += 1
            indss = indss - bd
        tr += 1
    return {"input": gi, "output": go}


def generate_a2fd1cf0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3, 8))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    gloci = unifint(rng, diff_lb, diff_ub, (1, h - 1))
    glocj = unifint(rng, diff_lb, diff_ub, (1, w - 1))
    gloc = (gloci, glocj)
    bgc = rng.choice(cols)
    g = canvas(bgc, (h, w))
    g = fill(g, 3, {gloc})
    g = rot180(g)
    glocinv = center(ofcolor(g, 3))
    glocinvi, glocinvj = glocinv
    rloci = unifint(rng, diff_lb, diff_ub, (glocinvi + 1, h - 1))
    rlocj = unifint(rng, diff_lb, diff_ub, (glocinvj + 1, w - 1))
    rlocinv = (rloci, rlocj)
    g = fill(g, 2, {rlocinv})
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(g)
    a, b = center(ofcolor(gi, 2))
    c, d = center(ofcolor(gi, 3))
    go = fill(gi, 8, connect((a, b), (a, d)))
    go = fill(go, 8, connect((a, d), (c, d)))
    go = fill(go, 2, {(a, b)})
    go = fill(go, 3, {(c, d)})
    return {"input": gi, "output": go}


def generate_cdecee7f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    numc = unifint(rng, diff_lb, diff_ub, (1, min(9, w)))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, numcols)
    inds = interval(0, w, 1)
    locs = rng.sample(inds, numc)
    locs = order(locs, identity)
    gi = canvas(bgc, (h, w))
    go = []
    for j in locs:
        iloc = rng.randint(0, h - 1)
        col = rng.choice(ccols)
        gi = fill(gi, col, {(iloc, j)})
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go = tuple(go)
    go = tuple([go[:3], go[3:6][::-1], go[6:]])
    return {"input": gi, "output": go}


def generate_0962bcdd(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3, 4))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (2, 7))
    ccols = rng.sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 25))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    oh, ow = 5, 5
    subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        obj = frozenset({(loci, locj), (loci + 4, locj + 4)})
        bd = backdrop(obj)
        col = rng.choice(ccols)
        if bd.issubset(indss):
            ca, cb = rng.sample(ccols, 2)
            cp = (loci + 2, locj + 2)
            lins1 = connect((loci, locj), (loci + 4, locj + 4))
            lins2 = connect((loci + 4, locj), (loci, locj + 4))
            lins12 = lins1 | lins2
            lins3 = connect((loci + 2, locj), (loci + 2, locj + 4))
            lins4 = connect((loci, locj + 2), (loci + 4, locj + 2))
            lins34 = lins3 | lins4
            go = fill(go, cb, lins34)
            go = fill(go, ca, lins12)
            gi = fill(gi, ca, {cp})
            gi = fill(gi, cb, dneighbors(cp))
            succ += 1
            indss = indss - bd
        tr += 1
    return {"input": gi, "output": go}


def generate_dc0a314f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 15))
    w = h
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    remcols = rng.sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(rng, diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(rng.choice(remcols), rng.choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = rng.choice(totuple(bx & dns))
        obj.add((rng.choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))
    locidev = unifint(rng, diff_lb, diff_ub, (1, 2 * h))
    locjdev = unifint(rng, diff_lb, diff_ub, (1, w))
    loci = 2 * h - locidev
    locj = w - locjdev
    loci2 = unifint(rng, diff_lb, diff_ub, (loci, 2 * h - 1))
    locj2 = unifint(rng, diff_lb, diff_ub, (locj, w - 1))
    bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
    go = subgrid(bd, gi)
    gi = fill(gi, 3, bd)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_29623171(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 6))
    w = unifint(rng, diff_lb, diff_ub, (2, 6))
    nh = unifint(rng, diff_lb, diff_ub, (2, 4))
    nw = unifint(rng, diff_lb, diff_ub, (2, 4))
    bgc, linc, fgc = rng.sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    inds = totuple(asindices(smallc))
    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = rng.choice(llocs)
    nmostc = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    mostc = rng.sample(inds, nmostc)
    srcg = fill(smallc, fgc, mostc)
    obj = asobject(srcg)
    shftd = shift(obj, srcloc)
    gi = paint(c, shftd)
    go = fill(c, fgc, shftd)
    remlocs = remove(srcloc, llocs)
    gg = asobject(fill(smallc, bgc, inds))
    for rl in remlocs:
        noth = unifint(rng, diff_lb, diff_ub, (0, nmostc))
        otherg = fill(smallc, fgc, rng.sample(inds, noth))
        gi = paint(gi, shift(asobject(otherg), rl))
        if noth == nmostc:
            go = fill(go, fgc, shift(obj, rl))
        else:
            go = paint(go, shift(gg, rl))
    return {"input": gi, "output": go}


def generate_d4a91cb9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 4, 8))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    gloci = unifint(rng, diff_lb, diff_ub, (1, h - 1))
    glocj = unifint(rng, diff_lb, diff_ub, (1, w - 1))
    gloc = (gloci, glocj)
    bgc = rng.choice(cols)
    g = canvas(bgc, (h, w))
    g = fill(g, 8, {gloc})
    g = rot180(g)
    glocinv = center(ofcolor(g, 8))
    glocinvi, glocinvj = glocinv
    rloci = unifint(rng, diff_lb, diff_ub, (glocinvi + 1, h - 1))
    rlocj = unifint(rng, diff_lb, diff_ub, (glocinvj + 1, w - 1))
    rlocinv = (rloci, rlocj)
    g = fill(g, 2, {rlocinv})
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(g)
    a, b = center(ofcolor(gi, 2))
    c, d = center(ofcolor(gi, 8))
    go = fill(gi, 4, connect((a, b), (a, d)))
    go = fill(go, 4, connect((a, d), (c, d)))
    go = fill(go, 2, {(a, b)})
    go = fill(go, 8, {(c, d)})
    return {"input": gi, "output": go}


def generate_60b61512(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(7, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 20))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = rng.randint(2, 7)
        ow = rng.randint(2, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        indsss = asindices(canvas(-1, (oh, ow)))
        chch = rng.choice(totuple(indsss))
        obj = {chch}
        indsss = remove(chch, indsss)
        numcd = unifint(rng, diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = rng.choice((numcd, oh * ow - numcd))
        numc = min(max(2, numc), oh * ow - 1)
        for k in range(numc):
            obj.add(rng.choice(totuple(indsss & mapply(neighbors, obj))))
            indsss = indsss - obj
        oh, ow = shape(obj)
        obj = shift(obj, (loci, locj))
        bd = backdrop(obj)
        col = rng.choice(ccols)
        if bd.issubset(indss):
            gi = fill(gi, col, obj)
            go = fill(go, 7, bd)
            go = fill(go, col, obj)
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {"input": gi, "output": go}


def generate_4938f0c2(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 31))
    w = unifint(rng, diff_lb, diff_ub, (10, 31))
    oh = unifint(rng, diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(rng, diff_lb, diff_ub, (2, (w - 3) // 2))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    cc = rng.choice(remcols)
    remcols = remove(cc, remcols)
    objc = rng.choice(remcols)
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    ncells = unifint(rng, diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
    cells = rng.sample(reminds, ncells)
    while ncells == 4 and shape(cells) == (2, 2):
        ncells = unifint(rng, diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        cells = rng.sample(reminds, ncells)
    sg = fill(sg, objc, cells)
    G1 = sg
    G2 = vmirror(sg)
    G3 = hmirror(sg)
    G4 = vmirror(hmirror(sg))
    vbar = canvas(bgc, (oh, 1))
    hbar = canvas(bgc, (1, ow))
    cp = canvas(cc, (1, 1))
    topg = hconcat(hconcat(G1, vbar), G2)
    botg = hconcat(hconcat(G3, vbar), G4)
    ggm = hconcat(hconcat(hbar, cp), hbar)
    GG = vconcat(vconcat(topg, ggm), botg)
    gg = asobject(GG)
    canv = canvas(bgc, (h, w))
    loci = rng.randint(0, h - 2 * oh - 1)
    locj = rng.randint(0, w - 2 * ow - 1)
    loc = (loci, locj)
    go = paint(canv, shift(gg, loc))
    gi = paint(canv, shift(asobject(sg), loc))
    gi = fill(gi, cc, ofcolor(go, cc))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    ccpi, ccpj = center(ofcolor(gi, cc))
    gi = gi[:ccpi] + gi[ccpi + 1 :]
    gi = tuple(r[:ccpj] + r[ccpj + 1 :] for r in gi)
    go = go[:ccpi] + go[ccpi + 1 :]
    go = tuple(r[:ccpj] + r[ccpj + 1 :] for r in go)
    return {"input": gi, "output": go}


def generate_a8d7556c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 2))
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    fgc = rng.choice(cols)
    c = canvas(fgc, (h, w))
    numblacks = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 3 * 2))
    inds = totuple(asindices(c))
    blacks = rng.sample(inds, numblacks)
    gi = fill(c, 0, blacks)
    numsq = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 10))
    sqlocs = rng.sample(inds, numsq)
    gi = fill(gi, 0, shift(sqlocs, (0, 0)))
    gi = fill(gi, 0, shift(sqlocs, (0, 1)))
    gi = fill(gi, 0, shift(sqlocs, (1, 0)))
    gi = fill(gi, 0, shift(sqlocs, (1, 1)))
    go = tuple(e for e in gi)
    for a in range(h - 1):
        for b in range(w - 1):
            if gi[a][b] == 0 and gi[a + 1][b] == 0 and gi[a][b + 1] == 0 and gi[a + 1][b + 1] == 0:
                go = fill(go, 2, {(a, b), (a + 1, b), (a, b + 1), (a + 1, b + 1)})
    return {"input": gi, "output": go}


def generate_007bbfb7(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    c = canvas(0, (h, w))
    numcd = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numc = rng.choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    inds = totuple(asindices(c))
    locs = rng.sample(inds, numc)
    fgc = rng.choice(cols)
    gi = fill(c, fgc, locs)
    go = canvas(0, (h**2, w**2))
    for loc in locs:
        go = fill(go, fgc, shift(locs, multiply(loc, (h, w))))
    return {"input": gi, "output": go}


def generate_b190f7f5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    fullcols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    bgc = rng.choice(fullcols)
    cols = remove(bgc, fullcols)
    c = canvas(bgc, (h, w))
    numcd = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numc = rng.choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    numcd2 = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numc2 = rng.choice((numcd2, h * w - numcd2))
    numc2 = min(max(2, numc2), h * w - 1)
    inds = totuple(asindices(c))
    srclocs = rng.sample(inds, numc)
    srccol = rng.choice(cols)
    remcols = remove(srccol, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (2, 8))
    trglocs = rng.sample(inds, numc2)
    ccols = rng.sample(remcols, numcols)
    fixc1 = rng.choice(ccols)
    trgobj = [(fixc1, trglocs[0]), (rng.choice(remove(fixc1, ccols)), trglocs[1])] + [
        (rng.choice(ccols), ij) for ij in trglocs[2:]
    ]
    trgobj = frozenset(trgobj)
    gisrc = fill(c, srccol, srclocs)
    gitrg = paint(c, trgobj)
    catf = rng.choice((hconcat, vconcat))
    ordd = rng.choice(([gisrc, gitrg], [gitrg, gisrc]))
    gi = catf(*ordd)
    go = canvas(bgc, (h**2, w**2))
    for loc in trglocs:
        a, b = loc
        go = fill(go, gitrg[a][b], shift(srclocs, multiply(loc, (h, w))))
    return {"input": gi, "output": go}


def generate_2bcee788(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 20))
    w = unifint(rng, diff_lb, diff_ub, (2, 10))
    bgc, sepc, objc = rng.sample(cols, 3)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    spi = rng.randint(0, h - 1)
    sp = (spi, w - 1)
    shp = {sp}
    numcellsd = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numc = rng.choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)
    reminds = set(remove(sp, inds))
    for k in range(numc):
        shp.add(rng.choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    while width(shp) == 1:
        shp.add(rng.choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    c2 = fill(c, objc, shp)
    borderinds = sfilter(shp, lambda ij: ij[1] == w - 1)
    c3 = fill(c, sepc, borderinds)
    gimini = asobject(hconcat(c2, vmirror(c3)))
    gomini = asobject(hconcat(c2, vmirror(c2)))
    fullh = unifint(rng, diff_lb, diff_ub, (h + 1, 30))
    fullw = unifint(rng, diff_lb, diff_ub, (2 * w + 1, 30))
    fullg = canvas(bgc, (fullh, fullw))
    loci = rng.randint(0, fullh - h)
    locj = rng.randint(0, fullw - 2 * w)
    loc = (loci, locj)
    gi = paint(fullg, gimini)
    go = paint(fullg, gomini)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    go = replace(go, bgc, 3)
    return {"input": gi, "output": go}


def generate_a3df8b1e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    w = unifint(rng, diff_lb, diff_ub, (2, 10))
    h = unifint(rng, diff_lb, diff_ub, (w + 1, 30))
    bgc, linc = rng.sample(cols, 2)
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    go = tuple(e for e in gi)
    changing = True
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        go2 = fill(go, linc, {sp})
        if go2 == go:
            break
        go = go2
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    gix = tuple(e for e in gi)
    gox = tuple(e for e in go)
    numlins = unifint(rng, diff_lb, diff_ub, (1, 4))
    if numlins > 1:
        gi = fill(gi, linc, ofcolor(hmirror(gix), linc))
        go = fill(go, linc, ofcolor(hmirror(gox), linc))
    if numlins > 2:
        gi = fill(gi, linc, ofcolor(vmirror(gix), linc))
        go = fill(go, linc, ofcolor(vmirror(gox), linc))
    if numlins > 3:
        gi = fill(gi, linc, ofcolor(hmirror(vmirror(gix)), linc))
        go = fill(go, linc, ofcolor(hmirror(vmirror(gox)), linc))
    return {"input": gi, "output": go}


def generate_80af3007(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    fullcols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    bgc = rng.choice(fullcols)
    cols = remove(bgc, fullcols)
    c = canvas(bgc, (h, w))
    numcd = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    numc = rng.choice((numcd, h * w - numcd))
    numc = min(max(0, numc), h * w)
    inds = totuple(asindices(c))
    locs = tuple(set(rng.sample(inds, numc)) | set(rng.sample(totuple(corners(inds)), 3)))
    fgc = rng.choice(cols)
    gi = fill(c, fgc, locs)
    go = canvas(bgc, (h**2, w**2))
    for loc in locs:
        go = fill(go, fgc, shift(locs, multiply(loc, (h, w))))
    fullh = unifint(rng, diff_lb, diff_ub, (h**2 + 2, 30))
    fullw = unifint(rng, diff_lb, diff_ub, (w**2 + 2, 30))
    fullg = canvas(bgc, (fullh, fullw))
    loci = rng.randint(1, fullh - h**2 - 1)
    locj = rng.randint(1, fullw - w**2 - 1)
    loc = (loci, locj)
    giups = hupscale(vupscale(gi, h), w)
    gi = paint(fullg, shift(asobject(giups), loc))
    return {"input": gi, "output": go}


def generate_e50d258f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    padcol = rng.choice(remcols)
    remcols = remove(padcol, remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, 10))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    bound = None
    go = None
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = rng.randint(3, 8)
        ow = rng.randint(3, 8)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        if bd.issubset(indss):
            numcc = unifint(rng, diff_lb, diff_ub, (1, 7))
            ccols = rng.sample(remcols, numcc)
            if succ == 0:
                numred = unifint(rng, diff_lb, diff_ub, (1, oh * ow))
                bound = numred
            else:
                numred = unifint(rng, diff_lb, diff_ub, (0, min(oh * ow, bound - 1)))
            cc = canvas(rng.choice(ccols), (oh, ow))
            cci = asindices(cc)
            subs = rng.sample(totuple(cci), numred)
            obj1 = {(rng.choice(ccols), ij) for ij in cci - set(subs)}
            obj2 = {(2, ij) for ij in subs}
            obj = obj1 | obj2
            gi = paint(gi, shift(obj, (loci, locj)))
            if go is None:
                go = paint(cc, obj)
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {"input": gi, "output": go}


def generate_0e206a2e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc, acol, bcol, ccol, Dcol = rng.sample(cols, 5)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nsrcs = unifint(rng, diff_lb, diff_ub, (1, min(h, w) // 5))
    srcs = []
    abclist = []
    maxtrforsrc = 5 * nsrcs
    trforsrc = 0
    srcsucc = 0
    while trforsrc < maxtrforsrc and srcsucc < nsrcs:
        trforsrc += 1
        objsize = unifint(rng, diff_lb, diff_ub, (5, 20))
        bb = asindices(canvas(-1, (7, 7)))
        sp = rng.choice(totuple(bb))
        bb = remove(sp, bb)
        shp = {sp}
        for k in range(objsize - 1):
            shp.add(rng.choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while 1 in shape(shp):
            shp.add(rng.choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while len(set([x - y for x, y in shp])) == 1 or len(set([x + y for x, y in shp])) == 1:
            shp.add(rng.choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        shp = normalize(shp)
        shp = list(shp)
        rng.shuffle(shp)
        a, b, c = shp[:3]
        while 1 in shape({a, b, c}) or (
            len(set([x - y for x, y in {a, b, c}])) == 1 or len(set([x + y for x, y in {a, b, c}])) == 1
        ):
            rng.shuffle(shp)
            a, b, c = shp[:3]
        if sorted(shape({a, b, c})) in abclist:
            continue
        D = shp[3:]
        markers = {(acol, a), (bcol, b), (ccol, c)}
        obj = markers | {(Dcol, ij) for ij in D}
        obj = frozenset(obj)
        oh, ow = shape(obj)
        opts = sfilter(inds, lambda ij: shift(set(shp), ij).issubset(inds))
        if len(opts) == 0:
            continue
        loc = rng.choice(totuple(opts))
        srcsucc += 1
        gi = paint(gi, shift(obj, loc))
        shpplcd = shift(set(shp), loc)
        go = fill(go, -1, shpplcd)
        inds = (inds - shpplcd) - mapply(neighbors, shpplcd)
        srcs.append((obj, markers))
        abclist.append(sorted(shape({a, b, c})))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 30))
    maxtrials = 10 * num
    tr = 0
    succ = 0
    while succ < num and tr < maxtrials:
        mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
        fn = rng.choice(mfs)
        gi = fn(gi)
        go = fn(go)
        aigo = asindices(go)
        fullinds = ofcolor(go, bgc) - mapply(neighbors, aigo - ofcolor(go, bgc))
        obj, markers = rng.choice(srcs)
        shp = toindices(obj)
        if len(fullinds) == 0:
            break
        loctr = rng.choice(totuple(fullinds))
        xx = shift(shp, loctr)
        if xx.issubset(fullinds):
            succ += 1
            gi = paint(gi, shift(markers, loctr))
            go = paint(go, shift(obj, loctr))
        tr += 1
    go = replace(go, -1, bgc)
    return {"input": gi, "output": go}


def generate_b230c067(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    while True:
        h = unifint(rng, diff_lb, diff_ub, (10, 30))
        w = unifint(rng, diff_lb, diff_ub, (10, 30))
        oh = unifint(rng, diff_lb, diff_ub, (2, h // 3 - 1))
        ow = unifint(rng, diff_lb, diff_ub, (2, w // 3 - 1))
        numcd = unifint(rng, diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = rng.choice((numcd, oh * ow - numcd))
        numca = min(max(2, numc), oh * ow - 2)
        bounds = asindices(canvas(-1, (oh, ow)))
        sp = rng.choice(totuple(bounds))
        shp = {sp}
        for k in range(numca):
            ij = rng.choice(totuple((bounds - shp) & mapply(neighbors, shp)))
            shp.add(ij)
        shpa = normalize(shp)
        shpb = set(normalize(shp))
        mxnch = oh * ow - len(shpa)
        nchinv = unifint(rng, diff_lb, diff_ub, (1, mxnch))
        nch = mxnch - nchinv
        nch = min(max(1, nch), mxnch)
        for k in range(nch):
            ij = rng.choice(totuple((bounds - shpb) & mapply(neighbors, shpb)))
            shpb.add(ij)
        if rng.choice((True, False)):
            shpa, shpb = shpb, shpa
        bgc, fgc = rng.sample(cols, 2)
        c = canvas(bgc, (h, w))
        inds = asindices(c)
        acands = sfilter(inds, lambda ij: ij[0] <= h - height(shpa) and ij[1] <= w - width(shpa))
        aloc = rng.choice(totuple(acands))
        aplcd = shift(shpa, aloc)
        gi = fill(c, fgc, aplcd)
        go = fill(c, 2, aplcd)
        maxtrials = 10
        tr = 0
        succ = 0
        inds = (inds - aplcd) - mapply(neighbors, aplcd)
        inds = sfilter(inds, lambda ij: ij[0] <= h - height(shpb) and ij[1] <= w - width(shpb))
        while succ < 2 and tr <= maxtrials:
            if len(inds) == 0:
                break
            loc = rng.choice(totuple(inds))
            plcbd = shift(shpb, loc)
            if plcbd.issubset(inds):
                gi = fill(gi, fgc, plcbd)
                go = fill(go, 1, plcbd)
                succ += 1
                inds = (inds - plcbd) - mapply(neighbors, plcbd)
            tr += 1
        if succ == 2:
            break
    return {"input": gi, "output": go}


def generate_db93a21d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 3))
    h = unifint(rng, diff_lb, diff_ub, (12, 31))
    w = unifint(rng, diff_lb, diff_ub, (12, 32))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 25))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = rng.randint(1, h // 4)
        ow = oh
        fullh = 4 * oh
        fullw = 4 * ow
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - fullh and ij[1] < w - fullw))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = rng.choice(subs)
        bigobj = backdrop(frozenset({(loci, locj), (loci + fullh - 1, locj + fullw - 1)}))
        smallobj = backdrop(frozenset({(loci + oh, locj + ow), (loci + fullh - 1 - oh, locj + fullw - 1 - ow)}))
        if bigobj.issubset(indss | ofcolor(go, 3)):
            gi = fill(gi, fgc, smallobj)
            go = fill(go, 3, bigobj)
            go = fill(go, fgc, smallobj)
            strp = mapply(rbind(shoot, (1, 0)), connect(lrcorner(smallobj), llcorner(smallobj)))
            go = fill(go, 1, ofcolor(go, bgc) & strp)
            succ += 1
            indss = indss - bigobj
        tr += 1
    gi = gi[1:]
    go = go[1:]
    gi = tuple(r[1:-1] for r in gi)
    go = tuple(r[1:-1] for r in go)
    return {"input": gi, "output": go}


def generate_1e32b0e9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 6))
    w = unifint(rng, diff_lb, diff_ub, (4, 6))
    nh = unifint(rng, diff_lb, diff_ub, (1, 4))
    nw = unifint(rng, diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
    bgc, linc, fgc = rng.sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = rng.choice(llocs)
    remlocs = remove(srcloc, llocs)
    ncells = unifint(rng, diff_lb, diff_ub, (0, (h - 2) * (w - 2) - 1))
    smallc2 = canvas(bgc, (h - 2, w - 2))
    inds = asindices(smallc2)
    sp = rng.choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(ncells):
        ij = rng.choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = shift(shp, (1, 1))
    gg = asobject(fill(smallc, fgc, shp))
    gg2 = asobject(fill(smallc, linc, shp))
    gi = paint(c, shift(gg, srcloc))
    go = tuple(e for e in gi)
    ncc = ncells + 1
    for rl in remlocs:
        nleft = rng.randint(0, ncc)
        subobj = rng.sample(totuple(shp), nleft)
        sg2 = asobject(fill(smallc, fgc, subobj))
        gi = paint(gi, shift(sg2, rl))
        go = paint(go, shift(gg2, rl))
        go = fill(go, fgc, shift(subobj, rl))
    return {"input": gi, "output": go}


def generate_6773b310(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    nh = unifint(rng, diff_lb, diff_ub, (2, 5))
    nw = unifint(rng, diff_lb, diff_ub, (2, 5))
    bgc, linc, fgc = rng.sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    nbldev = unifint(rng, diff_lb, diff_ub, (0, (nh * nw) // 2))
    nbl = rng.choice((nbldev, nh * nw - nbldev))
    nbl = min(max(1, nbl), nh * nw - 1)
    bluelocs = rng.sample(llocs, nbl)
    bglocs = difference(llocs, bluelocs)
    inds = totuple(asindices(smallc))
    gi = tuple(e for e in c)
    go = canvas(bgc, (nh, nw))
    for ij in bluelocs:
        subg = asobject(fill(smallc, fgc, rng.sample(inds, 2)))
        gi = paint(gi, shift(subg, ij))
        a, b = ij
        loci = a // (h + 1)
        locj = b // (w + 1)
        go = fill(go, 1, {(loci, locj)})
    for ij in bglocs:
        subg = asobject(fill(smallc, fgc, rng.sample(inds, 1)))
        gi = paint(gi, shift(subg, ij))
    return {"input": gi, "output": go}


def generate_6ecd11f4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 7))
    w = unifint(rng, diff_lb, diff_ub, (2, 7))
    bgc, fgc = rng.sample(cols, 2)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, ncols)
    inds = asindices(canvas(bgc, (h, w)))
    nlocsd = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    nlocs = rng.choice((nlocsd, h * w - nlocsd))
    nlocs = min(max(3, nlocs), h * w - 1)
    sp = rng.choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(nlocs):
        ij = rng.choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    h, w = shape(shp)
    canv = canvas(bgc, (h, w))
    objbase = fill(canv, fgc, shp)
    maxhscf = (2 * h + h + 1) // h
    maxwscf = (2 * w + w + 1) // w
    hscf = unifint(rng, diff_lb, diff_ub, (2, maxhscf))
    wscf = unifint(rng, diff_lb, diff_ub, (2, maxwscf))
    obj = asobject(hupscale(vupscale(objbase, hscf), wscf))
    oh, ow = shape(obj)
    inds = asindices(canv)
    objx = {(rng.choice(ccols), ij) for ij in inds}
    if len(palette(objx)) == 1:
        objxodo = first(objx)
        objx = insert((rng.choice(remove(objxodo[0], ccols)), objxodo[1]), remove(objxodo, objx))
    fullh = unifint(rng, diff_lb, diff_ub, (hscf * h + h + 1, 30))
    fullw = unifint(rng, diff_lb, diff_ub, (wscf * w + w + 1, 30))
    gi = canvas(bgc, (fullh, fullw))
    fullinds = asindices(gi)
    while True:
        loci = rng.randint(0, fullh - oh)
        locj = rng.randint(0, fullw - ow)
        loc = (loci, locj)
        gix = paint(gi, shift(obj, loc))
        ofc = ofcolor(gix, fgc)
        delt = fullinds - ofc
        delt2 = delt - mapply(neighbors, ofc)
        scands = sfilter(delt2, lambda ij: ij[0] <= fullh - oh and ij[1] <= fullw - ow)
        if len(scands) == 0:
            continue
        locc = rng.choice(totuple(scands))
        shftd = shift(objx, locc)
        if toindices(shftd).issubset(delt2):
            gi = paint(gix, shftd)
            break
    go = paint(canv, objx)
    go = fill(go, bgc, ofcolor(objbase, bgc))
    return {"input": gi, "output": go}
