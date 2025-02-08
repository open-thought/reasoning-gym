import random

from ..dsl import *
from ..utils import *


def generate_11852cab(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (7, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = shift(asindices(trim(gi)), UNITY)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - 5 and ij[0] <= w - 5)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        plcd = shift(bx, loc)
        if plcd.issubset(inds):
            inds = (inds - plcd) - outbox(plcd)
            ringcols = [rng.choice(ccols) for k in range(4)]
            plcdrings = [shift(r, loc) for r in rings]
            gi = fill(gi, ringcols[0], plcdrings[0])
            go = fill(go, ringcols[0], plcdrings[0])
            idx = rng.randint(1, 3)
            gi = fill(gi, ringcols[idx], plcdrings[idx])
            go = fill(go, ringcols[idx], plcdrings[idx])
            remrings = plcdrings[1:idx] + plcdrings[idx + 1 :]
            remringcols = ringcols[1:idx] + ringcols[idx + 1 :]
            numrs = unifint(rng, diff_lb, diff_ub, (1, 2))
            locs = rng.sample((0, 1), numrs)
            remrings = [rr for j, rr in enumerate(remrings) if j in locs]
            remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
            tofillgi = merge(
                frozenset(
                    recolor(col, frozenset(rng.sample(totuple(remring), 4 - unifint(rng, diff_lb, diff_ub, (0, 3)))))
                    for remring, col in zip(remrings, remringcols)
                )
            )
            tofillgo = merge(frozenset(recolor(col, remring) for remring, col in zip(remrings, remringcols)))
            if min(shape(tofillgi)) == 5:
                succ += 1
                gi = paint(gi, tofillgi)
                go = paint(go, tofillgo)
    return {"input": gi, "output": go}


def generate_025d127b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, numcols)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = rng.randint(3, 6)
        ow = rng.randint(3, 6)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        topl = connect((0, 0), (0, ow - 1))
        leftl = connect((1, 0), (oh - 2, oh - 3))
        rightl = connect((1, ow), (oh - 2, ow + oh - 3))
        botl = connect((oh - 1, oh - 2), (oh - 1, oh - 3 + ow))
        inobj = topl | leftl | rightl | botl
        outobj = (
            shift(topl, (0, 1))
            | botl
            | shift(leftl, (0, 1))
            | connect((1, ow + 1), (oh - 3, ow + oh - 3))
            | {(oh - 2, ow + oh - 3)}
        )
        outobj = sfilter(outobj, lambda ij: ij[1] <= rightmost(inobj))
        fullobj = inobj | outobj
        inobj = shift(inobj, loc)
        outobj = shift(outobj, loc)
        fullobj = shift(fullobj, loc)
        if fullobj.issubset(inds):
            inds = (inds - fullobj) - mapply(neighbors, fullobj)
            succ += 1
            col = rng.choice(ccols)
            gi = fill(gi, col, inobj)
            go = fill(go, col, outobj)
    return {"input": gi, "output": go}


def generate_045e512c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (11, 30))
    w = unifint(rng, diff_lb, diff_ub, (11, 30))
    while True:
        oh = unifint(rng, diff_lb, diff_ub, (2, min(4, (h - 2) // 3)))
        ow = unifint(rng, diff_lb, diff_ub, (2, min(4, (w - 2) // 3)))
        bounds = asindices(canvas(-1, (oh, ow)))
        c1 = rng.choice(totuple(connect((0, 0), (oh - 1, 0))))
        c2 = rng.choice(totuple(connect((0, 0), (0, ow - 1))))
        c3 = rng.choice(totuple(connect((oh - 1, ow - 1), (oh - 1, 0))))
        c4 = rng.choice(totuple(connect((oh - 1, ow - 1), (0, ow - 1))))
        obj = {c1, c2, c3, c4}
        remcands = totuple(bounds - obj)
        ncells = unifint(rng, diff_lb, diff_ub, (0, len(remcands)))
        for k in range(ncells):
            loc = rng.choice(remcands)
            obj.add(loc)
            remcands = remove(loc, remcands)
        objt = normalize(obj)
        cc = canvas(0, shape(obj))
        cc = fill(cc, 1, objt)
        if len(colorfilter(objects(cc, T, T, F), 1)) == 1:
            break
    loci = rng.randint(oh + 1, h - 2 * oh - 1)
    locj = rng.randint(ow + 1, w - 2 * ow - 1)
    loc = (loci, locj)
    bgc, objc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(objc, cols))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    obj = shift(recolor(objc, obj), loc)
    gi = paint(gi, obj)
    go = paint(go, obj)
    options = totuple(neighbors((0, 0)))
    ndirs = unifint(rng, diff_lb, diff_ub, (1, 8))
    dirs = rng.sample(options, ndirs)
    dcols = [rng.choice(ccols) for k in range(ndirs)]
    hbars = hfrontier((loci - 2, 0)) | hfrontier((loci + oh + 1, 0))
    vbars = vfrontier((0, locj - 2)) | vfrontier((0, locj + ow + 1))
    bars = hbars | vbars
    ofs = increment((oh, ow))
    for direc, col in zip(dirs, dcols):
        indicatorobj = shift(obj, multiply(direc, increment((oh, ow))))
        indicatorobj = sfilter(indicatorobj, lambda cij: cij[1] in bars)
        nindsd = unifint(rng, diff_lb, diff_ub, (0, len(indicatorobj) - 1))
        ninds = len(indicatorobj) - nindsd
        indicatorobj = set(rng.sample(totuple(indicatorobj), ninds))
        if len(indicatorobj) > 0 and len(indicatorobj) < len(obj):
            gi = fill(gi, col, indicatorobj)
            for k in range(1, 10):
                go = fill(go, col, shift(obj, multiply(multiply(k, direc), ofs)))
    return {"input": gi, "output": go}


def generate_1b60fb0c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    odh = unifint(rng, diff_lb, diff_ub, (2, min(h, w) // 2))
    loci = rng.randint(0, h - 2 * odh)
    locj = rng.randint(0, w - 2 * odh)
    loc = (loci, locj)
    bgc, objc = rng.sample(cols, 2)
    quad = canvas(bgc, (odh, odh))
    ncellsd = unifint(rng, diff_lb, diff_ub, (0, odh**2 // 2))
    ncells = rng.choice((ncellsd, odh**2 - ncellsd))
    ncells = min(max(1, ncells), odh**2 - 1)
    cells = rng.sample(totuple(asindices(canvas(-1, (odh, odh)))), ncells)
    g1 = fill(quad, objc, cells)
    g2 = rot90(g1)
    g3 = rot90(g2)
    g4 = rot90(g3)
    c1 = shift(ofcolor(g1, objc), (0, 0))
    c2 = shift(ofcolor(g2, objc), (0, odh))
    c3 = shift(ofcolor(g3, objc), (odh, odh))
    c4 = shift(ofcolor(g4, objc), (odh, 0))
    shftamt = rng.randint(0, odh)
    c1 = shift(c1, (0, shftamt))
    c2 = shift(c2, (shftamt, 0))
    c3 = shift(c3, (0, -shftamt))
    c4 = shift(c4, (-shftamt, 0))
    cs = (c1, c2, c3, c4)
    rempart = rng.choice(cs)
    inobjparts = remove(rempart, cs)
    inobj = merge(set(inobjparts))
    rempart = rempart - inobj
    inobj = shift(inobj, loc)
    rempart = shift(rempart, loc)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, inobj)
    go = fill(gi, 2, rempart)
    return {"input": gi, "output": go}


def generate_1f0c79e5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc, objc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 24))
    inds = asindices(gi)
    obj = ((0, 0), (0, 1), (1, 0), (1, 1))
    for k in range(nobjs):
        cands = sfilter(inds, lambda ij: shift(set(obj), ij).issubset(inds))
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        nred = unifint(rng, diff_lb, diff_ub, (1, 3))
        reds = rng.sample(totuple(plcd), nred)
        gi = fill(gi, objc, plcd)
        gi = fill(gi, 2, reds)
        for idx in reds:
            direc = decrement(multiply(2, add(idx, invert(loc))))
            go = fill(go, objc, mapply(rbind(shoot, direc), frozenset(plcd)))
        inds = (inds - plcd) - mapply(dneighbors, set(plcd))
    return {"input": gi, "output": go}


def generate_1f876c06(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    nlns = unifint(rng, diff_lb, diff_ub, (1, min(min(h, w), 9)))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ccols = rng.sample(remcols, nlns)
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    direcs = ineighbors((0, 0))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nlns and tr < maxtr:
        tr += 1
        if len(inds) == 0:
            break
        loc = rng.choice(totuple(inds))
        lns = []
        for direc in direcs:
            ln = [loc]
            ofs = 1
            while True:
                nextpix = add(loc, multiply(ofs, direc))
                ofs += 1
                if nextpix not in inds:
                    break
                ln.append(nextpix)
            if len(ln) > 2:
                lns.append(ln)
        if len(lns) > 0:
            succ += 1
            lns = sorted(lns, key=len)
            idx = unifint(rng, diff_lb, diff_ub, (0, len(lns) - 1))
            ln = lns[idx]
            col = ccols[0]
            ccols = ccols[1:]
            gi = fill(gi, col, {ln[0], ln[-1]})
            go = fill(go, col, set(ln))
            inds = inds - set(ln)
    return {"input": gi, "output": go}


def generate_22233c11(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 10))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, ncols)
    while succ < nobjs and tr < maxtr:
        if len(inds) == 0:
            break
        tr += 1
        od = rng.randint(1, 3)
        fulld = 4 * od
        g = canvas(bgc, (4, 4))
        g = fill(g, 8, {(0, 3), (3, 0)})
        col = rng.choice(ccols)
        g = fill(g, col, {(1, 1), (2, 2)})
        if rng.choice((True, False)):
            g = hmirror(g)
        g = upscale(g, od)
        inobj = recolor(col, ofcolor(g, col))
        outobj = inobj | recolor(8, ofcolor(g, 8))
        loc = rng.choice(totuple(inds))
        outobj = shift(outobj, loc)
        inobj = shift(inobj, loc)
        outobji = toindices(outobj)
        if toindices(inobj).issubset(inds) and (outobji & fullinds).issubset(inds):
            succ += 1
            inds = (inds - outobji) - mapply(neighbors, outobji)
            gi = paint(gi, inobj)
            go = paint(go, outobj)
    return {"input": gi, "output": go}


def generate_264363fd(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    cp = (2, 2)
    neighs = neighbors(cp)
    o1 = shift(frozenset({(0, 1), (-1, 1)}), (1, 1))
    o2 = shift(frozenset({(1, 0), (1, -1)}), (1, 1))
    o3 = shift(frozenset({(2, 1), (3, 1)}), (1, 1))
    o4 = shift(frozenset({(1, 2), (1, 3)}), (1, 1))
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}
    h = unifint(rng, diff_lb, diff_ub, (15, 30))
    w = unifint(rng, diff_lb, diff_ub, (15, 30))
    bgc, sqc, linc = rng.sample(cols, 3)
    remcols = difference(cols, (bgc, sqc, linc))
    cpcol = rng.choice(remcols)
    nbhcol = rng.choice(remcols)
    nspikes = rng.randint(1, 4)
    spikes = rng.sample((o1, o2, o3, o4), nspikes)
    lns = merge(set(spikes))
    obj = {(cpcol, cp)} | recolor(linc, lns) | recolor(nbhcol, neighs - lns)
    loci = rng.randint(0, h - 5)
    locj = rng.randint(0, w - 5)
    loc = (loci, locj)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = paint(gi, shift(obj, loc))
    numsq = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 100))
    succ = 0
    tr = 0
    maxtr = 10 * numsq
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(shift(obj, loc)))
    while succ < numsq and tr < maxtr:
        tr += 1
        gh = rng.randint(5, h // 2 + 1)
        gw = rng.randint(5, w // 2 + 1)
        cands = sfilter(inds, lambda ij: ij[0] <= h - gh and ij[1] <= w - gw)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        g1 = canvas(sqc, (gh, gw))
        g2 = canvas(sqc, (gh, gw))
        ginds = asindices(g1)
        gindsfull = asindices(g1)
        bck = shift(ginds, loc)
        if bck.issubset(inds):
            noccs = unifint(rng, diff_lb, diff_ub, (1, (gh * gw) // 25))
            succ2 = 0
            tr2 = 0
            maxtr2 = 5 * noccs
            while succ2 < noccs and tr2 < maxtr2:
                tr2 += 1
                cands2 = sfilter(ginds, lambda ij: ij[0] <= gh - 5 and ij[1] <= gw - 5)
                if len(cands2) == 0:
                    break
                loc2 = rng.choice(totuple(cands2))
                lns2 = merge(frozenset({shoot(add(cp, add(loc2, mpr[spike])), mpr[spike]) for spike in spikes}))
                lns2 = lns2 & gindsfull
                plcd2 = shift(obj, loc2)
                plcd2i = toindices(plcd2)
                if plcd2i.issubset(ginds) and lns2.issubset(ginds | ofcolor(g2, linc)) and len(lns2 - plcd2i) > 0:
                    succ2 += 1
                    ginds = ((ginds - plcd2i) - mapply(neighbors, plcd2i)) - lns2
                    g1 = fill(g1, cpcol, {add(cp, loc2)})
                    g2 = paint(g2, plcd2)
                    g2 = fill(g2, linc, lns2)
            if succ2 > 0:
                succ += 1
                inds = (inds - bck) - outbox(bck)
                objfull1 = shift(asobject(g1), loc)
                objfull2 = shift(asobject(g2), loc)
                gi = paint(gi, objfull1)
                go = paint(go, objfull2)
    return {"input": gi, "output": go}


def generate_29ec7d0e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    hp = unifint(rng, diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(rng, diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    bgc, noisec = rng.sample(cols, 2)
    remcols = remove(noisec, cols)
    numc = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, numc)
    pobj = frozenset({(rng.choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            loci = (a + 1) + hp * a
            locj = (b + 1) + wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 20))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = rng.randint(2, 6)
        pw = rng.randint(2, 6)
        loci = rng.randint(0, h - ph)
        locj = rng.randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if (
                len(sfilter(gi2, lambda r: noisec not in r)) >= 2
                and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2
            ):
                succ += 1
                gi = gi2
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_3bd67248(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 4))
    h = unifint(rng, diff_lb, diff_ub, (3, 15))
    w = unifint(rng, diff_lb, diff_ub, (3, 15))
    bgc, linc = rng.sample(cols, 2)
    fac = unifint(rng, diff_lb, diff_ub, (1, 30 // max(h, w)))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, linc, connect((0, 0), (h - 1, 0)))
    go = fill(gi, 4, connect((h - 1, 1), (h - 1, w - 1)))
    go = fill(go, 2, shoot((h - 2, 1), (-1, 1)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    gi = upscale(gi, fac)
    go = upscale(go, fac)
    return {"input": gi, "output": go}


def generate_484b58aa(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    hp = unifint(rng, diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(rng, diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    noisec = rng.choice(cols)
    remcols = remove(noisec, cols)
    numc = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, numc)
    pobj = frozenset({(rng.choice(ccols), ij) for ij in pinds})
    go = canvas(-1, (h, w))
    locs = set()
    ofs = rng.randint(1, hp - 1)
    for a in range(2 * (h // hp + 1)):
        for b in range(w // wp + 1):
            loci = hp * a - ofs * b
            locj = wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 20))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = rng.randint(2, 6)
        pw = rng.randint(2, 6)
        loci = rng.randint(0, h - ph)
        locj = rng.randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if (
                len(sfilter(gi2, lambda r: noisec not in r)) >= 2
                and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2
            ):
                succ += 1
                gi = gi2
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_6aa20dc0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    od = unifint(rng, diff_lb, diff_ub, (2, 4))
    ncellsextra = rng.randint(1, max(1, (od**2 - 2) // 2))
    sinds = asindices(canvas(-1, (od, od)))
    extracells = set(rng.sample(totuple(sinds - {(0, 0), (od - 1, od - 1)}), ncellsextra))
    extracells.add(rng.choice(totuple(dneighbors((0, 0)) & sinds)))
    extracells.add(rng.choice(totuple(dneighbors((od - 1, od - 1)) & sinds)))
    extracells = frozenset(extracells)
    bgc, fgc, c1, c2 = rng.sample(cols, 4)
    obj = frozenset({(c1, (0, 0)), (c2, (od - 1, od - 1))}) | recolor(fgc, extracells)
    obj = obj | dmirror(obj)
    if rng.choice((True, False)):
        obj = hmirror(obj)
    gi = canvas(bgc, (h, w))
    loci = rng.randint(0, h - od)
    locj = rng.randint(0, w - od)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = asindices(gi)
    inds = inds - backdrop(outbox(plcd))
    nocc = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // (od**2 * 2))))
    succ = 0
    tr = 0
    maxtr = 4 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = rng.randint(1, 4)
        mf1 = rng.choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf2 = rng.choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf = compose(mf2, mf1)
        cobj = normalize(upscale(mf(obj), fac))
        ohx, owx = shape(cobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - ohx and ij[1] <= w - owx)
        if len(cands) == 0:
            continue
        locc = rng.choice(totuple(cands))
        cobjo = shift(cobj, locc)
        cobji = sfilter(cobjo, lambda cij: cij[0] != fgc)
        cobjoi = toindices(cobjo)
        if cobjoi.issubset(inds):
            succ += 1
            inds = inds - backdrop(outbox(cobjoi))
            gi = paint(gi, cobji)
            go = paint(go, cobjo)
    return {"input": gi, "output": go}


def generate_6855a6e4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    fullh = unifint(rng, diff_lb, diff_ub, (10, h))
    fullw = unifint(rng, diff_lb, diff_ub, (3, w))
    bgc, objc, boxc = rng.sample(cols, 3)
    bcanv = canvas(bgc, (h, w))
    loci = rng.randint(0, h - fullh)
    locj = rng.randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(rng, diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = rng.choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(rng.sample(totuple(cands), ncells))
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_39a8645d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (15, 30))
    w = unifint(rng, diff_lb, diff_ub, (15, 30))
    oh = rng.randint(2, 4)
    ow = rng.randint(2, 4)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, oh + ow))
    ccols = rng.sample(remcols, nobjs + 1)
    mxcol = ccols[0]
    rcols = ccols[1:]
    maxnocc = unifint(rng, diff_lb, diff_ub, (nobjs + 2, max(nobjs + 2, (h * w) // 16)))
    tr = 0
    maxtr = 10 * maxnocc
    succ = 0
    allobjs = []
    bounds = asindices(canvas(-1, (oh, ow)))
    for k in range(nobjs + 1):
        while True:
            ncells = rng.randint(oh + ow - 1, oh * ow)
            cobj = {rng.choice(totuple(bounds))}
            while shape(cobj) != (oh, ow) and len(cobj) < ncells:
                cobj.add(rng.choice(totuple((bounds - cobj) & mapply(neighbors, cobj))))
            if cobj not in allobjs:
                break
        allobjs.append(frozenset(cobj))
    mcobj = normalize(allobjs[0])
    remobjs = apply(normalize, allobjs[1:])
    mxobjcounter = 0
    remobjcounter = {robj: 0 for robj in remobjs}
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    while tr < maxtr and succ < maxnocc:
        tr += 1
        candobjs = [robj for robj, cnt in remobjcounter.items() if cnt + 1 < mxobjcounter]
        if len(candobjs) == 0 or rng.randint(0, 100) / 100 > diff_lb:
            obj = mcobj
            col = mxcol
        else:
            obj = rng.choice(candobjs)
            col = rcols[remobjs.index(obj)]
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds - mapply(neighbors, ofcolor(gi, col))):
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            gi = fill(gi, col, plcd)
            if obj in remobjcounter:
                remobjcounter[obj] += 1
            else:
                mxobjcounter += 1
    go = fill(canvas(bgc, shape(mcobj)), mxcol, mcobj)
    return {"input": gi, "output": go}


def generate_150deff5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 8))
    bo = {(0, 0), (0, 1), (1, 0), (1, 1)}
    ro1 = {(0, 0), (0, 1), (0, 2)}
    ro2 = {(0, 0), (1, 0), (2, 0)}
    boforb = set()
    reforb = set()
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    noccs = unifint(rng, diff_lb, diff_ub, (2, (h * w) // 10))
    inds = asindices(gi)
    needsbgc = []
    for k in range(noccs):
        obj, col = rng.choice(((bo, 8), (rng.choice((ro1, ro2)), 2)))
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow and shift(obj, ij).issubset(inds))
        if col == 8:
            cands = sfilter(cands, lambda ij: ij not in boforb)
        else:
            cands = sfilter(cands, lambda ij: ij not in reforb)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        if col == 8:
            boforb.add(add(loc, (-2, 0)))
            boforb.add(add(loc, (2, 0)))
            boforb.add(add(loc, (0, 2)))
            boforb.add(add(loc, (0, -2)))
        if col == 2:
            if obj == ro1:
                reforb.add(add(loc, (0, 3)))
                reforb.add(add(loc, (0, -3)))
            else:
                reforb.add(add(loc, (1, 0)))
                reforb.add(add(loc, (-1, 0)))
        plcd = shift(obj, loc)
        gi = fill(gi, fgc, plcd)
        go = fill(go, col, plcd)
        inds = inds - plcd
    return {"input": gi, "output": go}


def generate_239be575(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    sq = {(0, 0), (1, 1), (0, 1), (1, 0)}
    cols = interval(1, 10, 1)
    while True:
        h = unifint(rng, diff_lb, diff_ub, (6, 30))
        w = unifint(rng, diff_lb, diff_ub, (6, 30))
        c = canvas(0, (h, w))
        fullcands = totuple(asindices(canvas(0, (h - 1, w - 1))))
        a = rng.choice(fullcands)
        b = rng.choice(remove(a, fullcands))
        mindist = unifint(rng, diff_lb, diff_ub, (3, min(h, w) - 3))
        while not manhattan({a}, {b}) > mindist:
            a = rng.choice(fullcands)
            b = rng.choice(remove(a, fullcands))
        markcol, sqcol = rng.sample(cols, 2)
        aset = shift(sq, a)
        bset = shift(sq, b)
        gi = fill(c, sqcol, aset | bset)
        cands = totuple(ofcolor(gi, 0))
        num = unifint(rng, diff_lb, diff_ub, (int(0.25 * len(cands)), int(0.75 * len(cands))))
        mc = rng.sample(cands, num)
        gi = fill(gi, markcol, mc)
        bobjs = colorfilter(objects(gi, T, F, F), markcol)
        ss = sfilter(bobjs, fork(both, rbind(adjacent, aset), rbind(adjacent, bset)))
        shoudlhaveconn = rng.choice((True, False))
        if shoudlhaveconn and len(ss) == 0:
            while len(ss) == 0:
                opts2 = totuple(ofcolor(gi, 0))
                if len(opts2) == 0:
                    break
                gi = fill(gi, markcol, {rng.choice(opts2)})
                bobjs = colorfilter(objects(gi, T, F, F), markcol)
                ss = sfilter(bobjs, fork(both, rbind(adjacent, aset), rbind(adjacent, bset)))
        elif not shoudlhaveconn and len(ss) > 0:
            while len(ss) > 0:
                opts2 = totuple(ofcolor(gi, markcol))
                if len(opts2) == 0:
                    break
                gi = fill(gi, 0, {rng.choice(opts2)})
                bobjs = colorfilter(objects(gi, T, F, F), markcol)
                ss = sfilter(bobjs, fork(both, rbind(adjacent, aset), rbind(adjacent, bset)))
        if len(palette(gi)) == 3:
            break
    oc = markcol if shoudlhaveconn else 0
    go = canvas(oc, (1, 1))
    return {"input": gi, "output": go}


def generate_0dfd9992(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    hp = unifint(rng, diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(rng, diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    bgc, noisec = rng.sample(cols, 2)
    remcols = remove(noisec, cols)
    numc = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, numc)
    pobj = frozenset({(rng.choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            loci = hp * a
            locj = wp * b
            locs.add((loci, locj))
            mf1 = identity if a % 2 == 0 else hmirror
            mf2 = identity if b % 2 == 0 else vmirror
            mf = compose(mf1, mf2)
            go = paint(go, shift(mf(pobj), (loci, locj)))
    numpatches = unifint(rng, diff_lb, diff_ub, (1, int((h * w) ** 0.5 // 2)))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = rng.randint(2, 6)
        pw = rng.randint(2, 6)
        loci = rng.randint(0, h - ph)
        locj = rng.randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        candset = apply(normalize, apply(rbind(toobject, gi2), places))
        if (
            len(sfilter(gi2, lambda r: noisec not in r)) >= 2
            and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2
            and (
                pobj in candset
                or hmirror(pobj) in candset
                or vmirror(pobj) in candset
                or hmirror(vmirror(pobj)) in candset
            )
        ):
            succ += 1
            gi = gi2
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_d06dbe63(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    obj1 = mapply(lbind(shift, frozenset({(-1, 0), (-2, 0), (-2, 1), (-2, 2)})), {(-k * 2, 2 * k) for k in range(15)})
    obj2 = mapply(lbind(shift, frozenset({(1, 0), (2, 0), (2, -1), (2, -2)})), {(2 * k, -k * 2) for k in range(15)})
    obj = obj1 | obj2
    objf = lambda ij: shift(obj, ij)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    ndots = unifint(rng, diff_lb, diff_ub, (1, min(h, w)))
    succ = 0
    tr = 0
    maxtr = 4 * ndots
    bgc, dotc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    while tr < maxtr and succ < ndots:
        tr += 1
        if len(inds) == 0:
            break
        loc = rng.choice(totuple(inds))
        objx = objf(loc)
        if (objx & fullinds).issubset(inds):
            succ += 1
            inds = (inds - objx) - {loc}
            gi = fill(gi, dotc, {loc})
            go = fill(go, dotc, {loc})
            go = fill(go, 5, objx)
    return {"input": gi, "output": go}


def generate_a3325580(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, 9))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ccols = rng.sample(remcols, nobjs)
    gi = canvas(bgc, (h, w))
    lmocc = set()
    inds = asindices(gi)
    succ = 0
    tr = 0
    maxtr = 4 * nobjs
    seenobjs = set()
    mxncells = rng.randint(nobjs + 1, 30)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = rng.randint(1, 6)
        ow = rng.randint(1, 6)
        while oh * ow < mxncells:
            oh = rng.randint(1, 6)
            ow = rng.randint(1, 6)
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = rng.randint(1, oh * ow)
        ncells = unifint(rng, diff_lb, diff_ub, (1, min(oh * ow, mxncells)))
        ncells = unifint(rng, diff_lb, diff_ub, (ncells, min(oh * ow, mxncells)))
        sp = rng.choice(totuple(bounds))
        obj = {sp}
        for k in range(ncells - 1):
            obj.add(rng.choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        if obj in seenobjs:
            continue
        obj = normalize(obj)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow and ij[1] not in lmocc)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            gi = fill(gi, ccols[succ], plcd)
            succ += 1
            lmocc.add(loc[1])
    objs = objects(gi, T, F, T)
    mxncells = valmax(objs, size)
    objs = sfilter(objs, matcher(size, mxncells))
    objs = order(objs, leftmost)
    go = canvas(-1, (mxncells, len(objs)))
    for idx, o in enumerate(objs):
        go = fill(go, color(o), connect((0, idx), (mxncells - 1, idx)))
    return {"input": gi, "output": go}


def generate_1fad071e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nbl = rng.randint(0, 5)
    nobjs = unifint(rng, diff_lb, diff_ub, (nbl, max(nbl, (h * w) // 10)))
    bgc, otherc = rng.sample(cols, 2)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    bcount = 0
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    ofcfrbinds = {1: set(), otherc: set()}
    while succ < nobjs and tr < maxtr:
        tr += 1
        col = rng.choice((1, otherc))
        oh = rng.randint(1, 3)
        ow = rng.randint(1, 3)
        if bcount < nbl:
            col = 1
            oh, ow = 2, 2
        else:
            while col == 1 and oh == ow == 2:
                col = rng.choice((1, otherc))
                oh = rng.randint(1, 3)
                ow = rng.randint(1, 3)
        bd = backdrop(frozenset({(0, 0), (oh - 1, ow - 1)}))
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        bd = shift(bd, loc)
        if bd.issubset(inds) and len(mapply(dneighbors, bd) & ofcfrbinds[col]) == 0:
            succ += 1
            inds = inds - bd
            ofcfrbinds[col] = ofcfrbinds[col] | mapply(dneighbors, bd) | bd
            gi = fill(gi, col, bd)
            if col == 1 and oh == ow == 2:
                bcount += 1
    go = (repeat(1, bcount) + repeat(bgc, 5 - bcount),)
    return {"input": gi, "output": go}


def generate_27a28665(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    mapping = [
        (1, {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}),
        (2, {(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)}),
        (3, {(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)}),
        (6, {(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)}),
    ]
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    col, obj = rng.choice(mapping)
    bgc, objc = rng.sample(cols, 2)
    fac = unifint(rng, diff_lb, diff_ub, (1, min(h, w) // 3))
    go = canvas(col, (1, 1))
    gi = canvas(bgc, (h, w))
    canv = canvas(bgc, (3, 3))
    canv = fill(canv, objc, obj)
    canv = upscale(canv, fac)
    obj = asobject(canv)
    loci = rng.randint(0, h - 3 * fac)
    locj = rng.randint(0, w - 3 * fac)
    loc = (loci, locj)
    gi = paint(gi, shift(obj, loc))
    return {"input": gi, "output": go}


def generate_b775ac94(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    gi = canvas(0, (1, 1))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = rng.randint(2, 5)
        ow = rng.randint(2, 5)
        canv = canvas(bgc, (oh, ow))
        c1, c2, c3, c4 = rng.sample(remcols, 4)
        obj = {(0, 0)}
        ncellsd = unifint(rng, diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = rng.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        bounds = asindices(canv)
        for k in range(ncells):
            obj.add(rng.choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        gLR = fill(canv, c1, obj)
        gLL = replace(vmirror(gLR), c1, c2)
        gUR = replace(hmirror(gLR), c1, c3)
        gUL = replace(vmirror(hmirror(gLR)), c1, c4)
        gU = hconcat(gUL, gUR)
        gL = hconcat(gLL, gLR)
        g = vconcat(gU, gL)
        g2 = canvas(bgc, (oh * 2, ow * 2))
        g2 = fill(g2, c1, shift(obj, (oh, ow)))
        nkeepcols = unifint(rng, diff_lb, diff_ub, (1, 3))
        keepcols = rng.sample((c2, c3, c4), nkeepcols)
        for cc in (c2, c3, c4):
            if cc not in keepcols:
                g = replace(g, cc, bgc)
            else:
                ofsi = -1 if cc in (c3, c4) else 0
                ofsj = -1 if cc in (c2, c4) else 0
                g2 = fill(g2, cc, {(oh + ofsi, ow + ofsj)})
        rotf = rng.choice((identity, rot90, rot180, rot270))
        g = rotf(g)
        g2 = rotf(g2)
        obji = asobject(g2)
        objo = asobject(g)
        objo = sfilter(objo, lambda cij: cij[0] != bgc)
        obji = sfilter(obji, lambda cij: cij[0] != bgc)
        tonorm = invert(ulcorner(objo))
        obji = shift(obji, tonorm)
        objo = shift(objo, tonorm)
        oh, ow = shape(objo)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        plcdi = shift(obji, loc)
        plcdo = shift(objo, loc)
        plcdoi = toindices(plcdo)
        if plcdoi.issubset(inds):
            succ += 1
            inds = (inds - plcdoi) - mapply(neighbors, plcdoi)
            gi = paint(gi, plcdi)
            go = paint(go, plcdo)
    return {"input": gi, "output": go}


def generate_6f8cd79b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, ncols)
    ncells = unifint(rng, diff_lb, diff_ub, (0, h * w))
    inds = asindices(gi)
    cells = rng.sample(totuple(inds), ncells)
    obj = {(rng.choice(ccols), ij) for ij in cells}
    gi = paint(gi, obj)
    brd = box(inds)
    go = fill(gi, 8, brd)
    return {"input": gi, "output": go}


def generate_de1cd16c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    noisec = rng.choice(cols)
    remcols = remove(noisec, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, ncols)
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = canvas(starterc, (h, w))
    for k in range(ncols - 1):
        objs = objects(gi, T, F, F)
        objs = sfilter(objs, lambda o: height(o) > 5 or width(o) > 5)
        if len(objs) == 0:
            break
        objs = totuple(objs)
        obj = rng.choice(objs)
        if height(obj) > 5 and width(obj) > 5:
            ax = rng.choice((0, 1))
        elif height(obj) > 5:
            ax = 0
        elif width(obj) > 5:
            ax = 1
        if ax == 0:
            loci = rng.randint(uppermost(obj) + 3, lowermost(obj) - 2)
            newobj = sfilter(toindices(obj), lambda ij: ij[0] >= loci)
        elif ax == 1:
            locj = rng.randint(leftmost(obj) + 3, rightmost(obj) - 2)
            newobj = sfilter(toindices(obj), lambda ij: ij[1] >= locj)
        gi = fill(gi, ccols[k], newobj)
    objs = order(objects(gi, T, F, F), size)
    allowances = [max(1, ((height(o) - 2) * (width(o) - 2)) // 2) for o in objs]
    meann = max(1, int(sum(allowances) / len(allowances)))
    chosens = [rng.randint(0, min(meann, allowed)) for allowed in allowances]
    while max(chosens) == 0:
        chosens = [rng.randint(0, min(meann, allowed)) for allowed in allowances]
    mx = max(chosens)
    fixinds = [idx for idx, cnt in enumerate(chosens) if cnt == mx]
    gogoind = fixinds[0]
    gogocol = color(objs[gogoind])
    fixinds = fixinds[1:]
    for idx in fixinds:
        chosens[idx] -= 1
    for obj, cnt in zip(objs, chosens):
        locs = rng.sample(totuple(backdrop(inbox(toindices(obj)))), cnt)
        gi = fill(gi, noisec, locs)
    go = canvas(gogocol, (1, 1))
    return {"input": gi, "output": go}


def generate_6cf79266(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 1))
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    nfgcs = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(cols, nfgcs)
    gi = canvas(-1, (h, w))
    fgcobj = {(rng.choice(ccols), ij) for ij in asindices(gi)}
    gi = paint(gi, fgcobj)
    num = unifint(rng, diff_lb, diff_ub, (int(0.25 * h * w), int(0.6 * h * w)))
    inds = asindices(gi)
    locs = rng.sample(totuple(inds), num)
    gi = fill(gi, 0, locs)
    noccs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 16))
    cands = asindices(canvas(-1, (h - 2, w - 2)))
    locs = rng.sample(totuple(cands), noccs)
    mini = asindices(canvas(-1, (3, 3)))
    for ij in locs:
        gi = fill(gi, 0, shift(mini, ij))
    trg = recolor(0, mini)
    occs = occurrences(gi, trg)
    go = tuple(e for e in gi)
    for occ in occs:
        go = fill(go, 1, shift(mini, occ))
    return {"input": gi, "output": go}


def generate_a85d4709(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3, 4))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w3 = unifint(rng, diff_lb, diff_ub, (1, 10))
    w = w3 * 3
    bgc, dotc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for ii in range(h):
        loc = rng.randint(0, w3 - 1)
        dev = unifint(rng, diff_lb, diff_ub, (0, w3 // 2 + 1))
        loc = w3 // 3 + rng.choice((+dev, -dev))
        loc = min(max(0, loc), w3 - 1)
        ofs, col = rng.choice(((0, 2), (1, 4), (2, 3)))
        loc += ofs * w3
        gi = fill(gi, dotc, {(ii, loc)})
        ln = connect((ii, 0), (ii, w - 1))
        go = fill(go, col, ln)
    return {"input": gi, "output": go}


def generate_f8a8fe49(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    fullh = unifint(rng, diff_lb, diff_ub, (10, h))
    fullw = unifint(rng, diff_lb, diff_ub, (3, w))
    bgc, objc, boxc = rng.sample(cols, 3)
    bcanv = canvas(bgc, (h, w))
    loci = rng.randint(0, h - fullh)
    locj = rng.randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(rng, diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = rng.choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(rng.sample(totuple(cands), ncells))
        cells = insert(rng.choice(totuple(sfilter(cands, lambda ij: ij[0] == lowermost(cands)))), cells)
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    go, gi = gi, go
    return {"input": gi, "output": go}


def generate_f8c80d96(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    ow = rng.randint(1, 3 if h > 10 else 2)
    oh = rng.randint(1, 3 if w > 10 else 2)
    loci = rng.randint(-oh + 1, h - 1)
    locj = rng.randint(-ow + 1, w - 1)
    obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    bgc, linc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(5, (h, w))
    ln1 = outbox(obj)
    ulci, ulcj = decrement(ulcorner(obj))
    lrci, lrcj = increment(lrcorner(obj))
    hoffs = rng.randint(2, 4 if h > 12 else 3)
    woffs = rng.randint(2, 4 if w > 12 else 3)
    lns = []
    for k in range(max(h, w) // min(hoffs, woffs) + 1):
        lnx = box(frozenset({(ulci - hoffs * k, ulcj - woffs * k), (lrci + hoffs * k, lrcj + woffs * k)}))
        lns.append(lnx)
    inds = asindices(gi)
    lns = sfilter(lns, lambda ln: len(ln & inds) > 0)
    nlns = len(lns)
    nmissing = unifint(rng, diff_lb, diff_ub, (0, nlns - 2))
    npresent = nlns - nmissing
    for k in range(npresent):
        gi = fill(gi, linc, lns[k])
    for ln in lns:
        go = fill(go, linc, ln)
    return {"input": gi, "output": go}


def generate_f35d900a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc, c1, c2 = rng.sample(cols, 3)
    oh = unifint(rng, diff_lb, diff_ub, (4, h))
    ow = unifint(rng, diff_lb, diff_ub, (4, w))
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, c1, {ulcorner(bx), lrcorner(bx)})
    gi = fill(gi, c2, {urcorner(bx), llcorner(bx)})
    go = fill(go, c1, {ulcorner(bx), lrcorner(bx)})
    go = fill(go, c2, {urcorner(bx), llcorner(bx)})
    go = fill(go, c1, neighbors(urcorner(bx)) | neighbors(llcorner(bx)))
    go = fill(go, c2, neighbors(ulcorner(bx)) | neighbors(lrcorner(bx)))
    crns = corners(bx)
    for c in crns:
        cobj = {c}
        remcorns = remove(c, crns)
        belongto = sfilter(bx, lambda ij: manhattan(cobj, {ij}) <= valmin(remcorns, lambda cc: manhattan({ij}, {cc})))
        valids = sfilter(belongto, lambda ij: manhattan(cobj, {ij}) > 1 and manhattan(cobj, {ij}) % 2 == 0)
        go = fill(go, 5, valids)
    return {"input": gi, "output": go}


def generate_ec883f72(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    ohi = unifint(rng, diff_lb, diff_ub, (0, h - 6))
    owi = unifint(rng, diff_lb, diff_ub, (0, w - 6))
    oh = h - 5 - ohi
    ow = w - 5 - owi
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    bgc, sqc, linc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, sqc, obj)
    obob = outbox(outbox(obj))
    gi = fill(gi, linc, obob)
    ln1 = shoot(lrcorner(obob), (1, 1))
    ln2 = shoot(ulcorner(obob), (-1, -1))
    ln3 = shoot(llcorner(obob), (1, -1))
    ln4 = shoot(urcorner(obob), (-1, 1))
    lns = (ln1 | ln2 | ln3 | ln4) & ofcolor(gi, bgc)
    go = fill(gi, sqc, lns)
    return {"input": gi, "output": go}


def generate_ea786f4a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 14))
    w = unifint(rng, diff_lb, diff_ub, (1, 14))
    mp = (h, w)
    h = 2 * h + 1
    w = 2 * w + 1
    linc = rng.choice(cols)
    remcols = remove(linc, cols)
    gi = canvas(linc, (h, w))
    inds = remove(mp, asindices(gi))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, ncols)
    obj = {(rng.choice(ccols), ij) for ij in inds}
    gi = paint(gi, obj)
    ln1 = shoot(mp, (-1, -1))
    ln2 = shoot(mp, (1, 1))
    ln3 = shoot(mp, (-1, 1))
    ln4 = shoot(mp, (1, -1))
    go = fill(gi, linc, ln1 | ln2 | ln3 | ln4)
    return {"input": gi, "output": go}


def generate_ded97339(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc, linc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    ndots = unifint(rng, diff_lb, diff_ub, (2, (h * w) // 9))
    inds = asindices(gi)
    dots = set()
    if rng.choice((True, False)):
        idxi = rng.randint(0, h - 1)
        locj1 = rng.randint(0, w - 3)
        locj2 = rng.randint(locj1 + 2, w - 1)
        dots.add((idxi, locj1))
        dots.add((idxi, locj2))
    else:
        idxj = rng.randint(0, w - 1)
        loci1 = rng.randint(0, h - 3)
        loci2 = rng.randint(loci1 + 2, h - 1)
        dots.add((loci1, idxj))
        dots.add((loci2, idxj))
    for k in range(ndots - 2):
        if len(inds) == 0:
            break
        loc = rng.choice(totuple(inds))
        dots.add(loc)
        inds = (inds - {loc}) - neighbors(loc)
    gi = fill(gi, linc, dots)
    go = tuple(e for e in gi)
    for ii, r in enumerate(gi):
        if r.count(linc) > 1:
            a = r.index(linc)
            b = w - r[::-1].index(linc) - 1
            go = fill(go, linc, connect((ii, a), (ii, b)))
    go = dmirror(go)
    gi = dmirror(gi)
    for ii, r in enumerate(gi):
        if r.count(linc) > 1:
            a = r.index(linc)
            b = h - r[::-1].index(linc) - 1
            go = fill(go, linc, connect((ii, a), (ii, b)))
    return {"input": gi, "output": go}


def generate_d687bc17(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc, c1, c2, c3, c4 = rng.sample(cols, 5)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, c1, connect((0, 0), (0, w - 1)))
    gi = fill(gi, c2, connect((0, 0), (h - 1, 0)))
    gi = fill(gi, c3, connect((h - 1, w - 1), (0, w - 1)))
    gi = fill(gi, c4, connect((h - 1, w - 1), (h - 1, 0)))
    inds = asindices(gi)
    gi = fill(gi, bgc, corners(inds))
    go = tuple(e for e in gi)
    cands = backdrop(inbox(inbox(inds)))
    ndots = unifint(rng, diff_lb, diff_ub, (1, min(len(cands), h + h + w + w)))
    dots = rng.sample(totuple(cands), ndots)
    dots = {(rng.choice((c1, c2, c3, c4)), ij) for ij in dots}
    n1 = toindices(sfilter(dots, lambda cij: cij[0] == c1))
    n1coverage = apply(last, n1)
    if len(n1coverage) == w - 4 and w > 5:
        n1coverage = remove(rng.choice(totuple(n1coverage)), n1coverage)
    for jj in n1coverage:
        loci = rng.choice([ij[0] for ij in sfilter(n1, lambda ij: ij[1] == jj)])
        gi = fill(gi, c1, {(loci, jj)})
        go = fill(go, c1, {(1, jj)})
    n2 = toindices(sfilter(dots, lambda cij: cij[0] == c2))
    n2coverage = apply(first, n2)
    if len(n2coverage) == h - 4 and h > 5:
        n2coverage = remove(rng.choice(totuple(n2coverage)), n2coverage)
    for ii in n2coverage:
        locj = rng.choice([ij[1] for ij in sfilter(n2, lambda ij: ij[0] == ii)])
        gi = fill(gi, c2, {(ii, locj)})
        go = fill(go, c2, {(ii, 1)})
    n3 = toindices(sfilter(dots, lambda cij: cij[0] == c4))
    n3coverage = apply(last, n3)
    if len(n3coverage) == w - 4 and w > 5:
        n3coverage = remove(rng.choice(totuple(n3coverage)), n3coverage)
    for jj in n3coverage:
        loci = rng.choice([ij[0] for ij in sfilter(n3, lambda ij: ij[1] == jj)])
        gi = fill(gi, c4, {(loci, jj)})
        go = fill(go, c4, {(h - 2, jj)})
    n4 = toindices(sfilter(dots, lambda cij: cij[0] == c3))
    n4coverage = apply(first, n4)
    if len(n4coverage) == h - 4 and h > 5:
        n4coverage = remove(rng.choice(totuple(n4coverage)), n4coverage)
    for ii in n4coverage:
        locj = rng.choice([ij[1] for ij in sfilter(n4, lambda ij: ij[0] == ii)])
        gi = fill(gi, c3, {(ii, locj)})
        go = fill(go, c3, {(ii, w - 2)})
    noisecands = ofcolor(gi, bgc)
    noisecols = difference(cols, (bgc, c1, c2, c3, c4))
    nnoise = unifint(rng, diff_lb, diff_ub, (0, len(noisecands)))
    ub = ((h * w) - 2 * h - 2 * (w - 2)) // 2 - ndots - 1
    nnoise = unifint(rng, diff_lb, diff_ub, (0, max(0, ub)))
    noise = rng.sample(totuple(noisecands), nnoise)
    noiseobj = {(rng.choice(noisecols), ij) for ij in noise}
    gi = paint(gi, noiseobj)
    return {"input": gi, "output": go}


def generate_d90796e8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (8, 2, 3))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc, noisec = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    nocc = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 3))
    inds = asindices(gi)
    locs = rng.sample(totuple(inds), nocc)
    obj = frozenset({(rng.choice((noisec, 2, 3)), ij) for ij in locs})
    gi = paint(gi, obj)
    fixloc = rng.choice(totuple(inds))
    fixloc2 = rng.choice(totuple(dneighbors(fixloc) & inds))
    gi = fill(gi, 2, {fixloc})
    gi = fill(gi, 3, {fixloc2})
    go = tuple(e for e in gi)
    reds = ofcolor(gi, 2)
    greens = ofcolor(gi, 3)
    tocover = set()
    tolblue = set()
    for r in reds:
        inters = dneighbors(r) & greens
        if len(inters) > 0:
            tocover.add(r)
            tolblue = tolblue | inters
    go = fill(go, bgc, tocover)
    go = fill(go, 8, tolblue)
    return {"input": gi, "output": go}


def generate_a68b268e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 14))
    w = unifint(rng, diff_lb, diff_ub, (2, 4))
    bgc, linc, c1, c2, c3, c4 = rng.sample(cols, 6)
    canv = canvas(bgc, (h, w))
    inds = asindices(canv)
    nc1d = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    nc1 = rng.choice((nc1d, h * w - nc1d))
    nc1 = min(max(1, nc1), h * w - 1)
    nc2d = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    nc2 = rng.choice((nc2d, h * w - nc2d))
    nc2 = min(max(1, nc2), h * w - 1)
    nc3d = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    nc3 = rng.choice((nc3d, h * w - nc3d))
    nc3 = min(max(1, nc3), h * w - 1)
    nc4d = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    nc4 = rng.choice((nc4d, h * w - nc4d))
    nc4 = min(max(1, nc4), h * w - 1)
    ofc1 = rng.sample(totuple(inds), nc1)
    ofc2 = rng.sample(totuple(inds), nc2)
    ofc3 = rng.sample(totuple(inds), nc3)
    ofc4 = rng.sample(totuple(inds), nc4)
    go = fill(canv, c1, ofc1)
    go = fill(go, c2, ofc2)
    go = fill(go, c3, ofc3)
    go = fill(go, c4, ofc4)
    LR = asobject(fill(canv, c1, ofc1))
    LL = asobject(fill(canv, c2, ofc2))
    UR = asobject(fill(canv, c3, ofc3))
    UL = asobject(fill(canv, c4, ofc4))
    gi = canvas(linc, (2 * h + 1, 2 * w + 1))
    gi = paint(gi, shift(LR, (h + 1, w + 1)))
    gi = paint(gi, shift(LL, (h + 1, 0)))
    gi = paint(gi, shift(UR, (0, w + 1)))
    gi = paint(gi, shift(UL, (0, 0)))
    return {"input": gi, "output": go}


def generate_ea32f347(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 4))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    max_l = min(h, w)

    a = unifint(rng, diff_lb, diff_ub, (3, max_l))
    b = unifint(rng, diff_lb, diff_ub, (2, a))
    c = unifint(rng, diff_lb, diff_ub, (1, b))

    if c - a == 2:
        if a > 1:
            a -= 1
        elif c < min(h, w):
            c += 1

    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)

    for col, l in zip((1, 4, 2), (a, b, c)):
        ln1 = connect((0, 0), (0, l - 1))
        ln2 = connect((0, 0), (l - 1, 0))
        tmpg = fill(gi, -1, asindices(gi) - inds)

        occs1 = occurrences(tmpg, recolor(bgc, ln1))
        occs2 = occurrences(tmpg, recolor(bgc, ln2))
        pool = []
        if len(occs1) > 0:
            pool.append((ln1, occs1))
        if len(occs2) > 0:
            pool.append((ln2, occs2))

        ln, occs = rng.choice(pool)
        loc = rng.choice(totuple(occs))
        plcd = shift(ln, loc)

        gi = fill(gi, rng.choice(remcols), plcd)
        go = fill(go, col, plcd)
        inds = (inds - plcd) - mapply(dneighbors, plcd)

    return {"input": gi, "output": go}


def generate_e179c5f4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
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
    go = replace(go, bgc, 8)
    return {"input": gi, "output": go}


def generate_aba27056(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    bgc, sqc = rng.sample(cols, 2)
    canv = canvas(bgc, (h, w))
    oh = rng.randint(3, h)
    ow = unifint(rng, diff_lb, diff_ub, (5, w - 1))
    loci = unifint(rng, diff_lb, diff_ub, (0, h - oh))
    locj = rng.randint(0, w - ow)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    maxk = (ow - 4) // 2
    k = rng.randint(0, maxk)
    hole = connect((loci, locj + 2 + k), (loci, locj + ow - 3 - k))
    gi = fill(canv, sqc, bx)
    gi = fill(gi, bgc, hole)
    go = fill(canv, 4, backdrop(bx))
    go = fill(go, sqc, bx)
    bar = mapply(rbind(shoot, (-1, 0)), hole)
    go = fill(go, 4, bar)
    go = fill(go, 4, shoot(add((-1, 1), urcorner(hole)), (-1, 1)))
    go = fill(go, 4, shoot(add((-1, -1), ulcorner(hole)), (-1, -1)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_e40b9e2f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    d = unifint(rng, diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = rng.randint(0, h - d)
    locj = rng.randint(0, w - d)
    loc = (loci, locj)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, numcols)
    subg = canvas(bgc, (d, d))
    inds = asindices(subg)
    if d % 2 == 0:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] < d // 2)
        cp = {(d // 2 - 1, d // 2 - 1), (d // 2, d // 2 - 1), (d // 2 - 1, d // 2), (d // 2, d // 2)}
    else:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] <= d // 2)
        cp = {(d // 2, d // 2)} | ineighbors((d // 2, d // 2))
    nrings = unifint(rng, diff_lb, diff_ub, (1, max(1, (d - 2) // 2)))
    rings = set()
    for k in range(nrings):
        ring = box({(k, k), (d - k - 1, d - k - 1)})
        rings = rings | ring
    qin = q - rings
    qout = rings & q
    ntailobjcells = unifint(rng, diff_lb, diff_ub, (1, len(q)))
    tailobjcells = rng.sample(totuple(q), ntailobjcells)
    tailobjcells = set(tailobjcells) | {rng.choice(totuple(qin))} | {rng.choice(totuple(qout))}
    tailobj = {(rng.choice(ccols), ij) for ij in tailobjcells}
    while hmirror(tailobj) == tailobj and vmirror(tailobj) == tailobj:
        ntailobjcells = unifint(rng, diff_lb, diff_ub, (1, len(q)))
        tailobjcells = rng.sample(totuple(q), ntailobjcells)
        tailobjcells = set(tailobjcells) | {rng.choice(totuple(qin))} | {rng.choice(totuple(qout))}
        tailobj = {(rng.choice(ccols), ij) for ij in tailobjcells}
    for k in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)
    fxobj = recolor(rng.choice(ccols), cp)
    subg = paint(subg, fxobj)
    subgi = subg
    subgo = tuple(e for e in subgi)
    subgi = fill(subgi, bgc, rings)
    nsplits = unifint(rng, diff_lb, diff_ub, (1, 4))
    splits = [set() for k in range(nsplits)]
    for idx, cel in enumerate(tailobj):
        splits[idx % nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, splits[jj])
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    subgi = rotf(subgi)
    subgo = rotf(subgo)
    gi = paint(canvas(bgc, (h, w)), shift(asobject(subgi), loc))
    go = paint(canvas(bgc, (h, w)), shift(asobject(subgo), loc))
    return {"input": gi, "output": go}


def generate_e8dc4411(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (9, 30))
    w = unifint(rng, diff_lb, diff_ub, (9, 30))
    d = unifint(rng, diff_lb, diff_ub, (3, min(h, w) // 2 - 1))
    bgc, objc, remc = rng.sample(cols, 3)
    c = canvas(bgc, (d, d))
    inds = sfilter(asindices(c), lambda ij: ij[0] >= d // 2 and ij[1] >= d // 2)
    ncd = unifint(rng, diff_lb, diff_ub, (1, len(inds) // 2))
    nc = rng.choice((ncd, len(inds) - ncd))
    nc = min(max(2, nc), len(inds) - 1)
    cells = rng.sample(totuple(inds), nc)
    cells = set(cells) | {rng.choice(((d // 2, d // 2), (d // 2, d // 2 - 1)))}
    cells = cells | {(jj, ii) for ii, jj in cells}
    for k in range(4):
        c = fill(c, objc, cells)
        c = rot90(c)
    while palette(toobject(box(asindices(c)), c)) == frozenset({bgc}) and height(c) > 3:
        c = trim(c)
    obj = ofcolor(c, objc)
    od = height(obj)
    loci = rng.randint(1, h - 2 * od)
    locj = rng.randint(1, w - 2 * od)
    obj = shift(obj, (loci, locj))
    bd = backdrop(obj)
    p = 0
    while len(shift(obj, (p, p)) & bd) > 0:
        p += 1
    obj2 = shift(obj, (p, p))
    nbhs = mapply(neighbors, obj)
    while len(obj2 & nbhs) == 0:
        nbhs = mapply(neighbors, nbhs)
    indic = obj2 & nbhs
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    gi = fill(gi, remc, indic)
    go = tuple(e for e in gi)
    for k in range(30):
        newg = fill(go, remc, shift(obj, (p * (k + 1), p * (k + 1))))
        if newg == go:
            break
        go = newg
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_ddf7fa4f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nocc = unifint(rng, diff_lb, diff_ub, (1, min(w // 3, (h * w) // 36)))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    succ = 0
    tr = 0
    maxtr = 10 * nocc
    inds = asindices(gi)
    inds = sfilter(inds, lambda ij: ij[0] > 1)
    while succ < nocc and tr < maxtr:
        tr += 1
        oh = rng.randint(2, 7)
        ow = rng.randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        hastobein = {cidx for cidx, col in enumerate(gi[0]) if col == bgc}
        cantbein = {cidx for cidx, col in enumerate(gi[0]) if col != bgc}
        jopts = [
            j
            for j in range(w)
            if len(set(interval(j, j + ow, 1)) & hastobein) > 0 and len(set(interval(j, j + ow, 1)) & cantbein) == 0
        ]
        cands = sfilter(cands, lambda ij: ij[1] in jopts)
        if len(cands) == 0:
            continue
        loci, locj = rng.choice(totuple(cands))
        locat = rng.choice(sfilter(interval(locj, locj + ow, 1), lambda jj: jj in hastobein))
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - mapply(dneighbors, sq)
            col = rng.choice(remcols)
            gr = rng.choice(remove(col, remcols))
            gi = fill(gi, col, {(0, locat)})
            go = fill(go, col, {(0, locat)})
            gi = fill(gi, gr, sq)
            go = fill(go, col, sq)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_d07ae81c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    lnf = lambda ij: shoot(ij, (1, 1)) | shoot(ij, (-1, -1)) | shoot(ij, (-1, 1)) | shoot(ij, (1, -1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    c1, c2, c3, c4 = rng.sample(cols, 4)
    magiccol = 0
    gi = canvas(0, (h, w))
    ndivi = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 10))
    for k in range(ndivi):
        objs = objects(gi, T, F, F)
        objs = sfilter(objs, lambda o: min(shape(o)) > 3 and max(shape(o)) > 4)
        objs = sfilter(objs, lambda o: height(o) * width(o) == len(o))
        if len(objs) == 0:
            break
        obj = rng.choice(totuple(objs))
        if rng.choice((True, False)):
            loci = rng.randint(uppermost(obj) + 2, lowermost(obj) - 1)
            newobj = backdrop(frozenset({(loci, leftmost(obj)), lrcorner(obj)}))
        else:
            locj = rng.randint(leftmost(obj) + 2, rightmost(obj) - 1)
            newobj = backdrop(frozenset({(uppermost(obj), locj), lrcorner(obj)}))
        magiccol += 1
        gi = fill(gi, magiccol, newobj)
    objs = objects(gi, T, F, F)
    for ii, obj in enumerate(objs):
        col = c1 if ii == 0 else (c2 if ii == 1 else rng.choice((c1, c2)))
        gi = fill(gi, col, toindices(obj))
    ofc1 = ofcolor(gi, c1)
    ofc2 = ofcolor(gi, c2)
    mn = min(len(ofc1), len(ofc2))
    n1 = unifint(rng, diff_lb, diff_ub, (1, max(1, int(mn**0.5))))
    n2 = unifint(rng, diff_lb, diff_ub, (1, max(1, int(mn**0.5))))
    srcs1 = set()
    for k in range(n1):
        cands = totuple((ofc1 - srcs1) - mapply(neighbors, srcs1))
        if len(cands) == 0:
            break
        srcs1.add(rng.choice(cands))
    srcs2 = set()
    for k in range(n2):
        cands = totuple((ofc2 - srcs2) - mapply(neighbors, srcs2))
        if len(cands) == 0:
            break
        srcs2.add(rng.choice(cands))
    gi = fill(gi, c3, srcs1)
    gi = fill(gi, c4, srcs2)
    lns = mapply(lnf, srcs1) | mapply(lnf, srcs2)
    ofc3 = ofc1 & lns
    ofc4 = ofc2 & lns
    go = fill(gi, c3, ofc3)
    go = fill(go, c4, ofc4)
    return {"input": gi, "output": go}


def generate_b2862040(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (8,))
    while True:
        h = unifint(rng, diff_lb, diff_ub, (10, 30))
        w = unifint(rng, diff_lb, diff_ub, (10, 30))
        nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 16))
        succ = 0
        tr = 0
        maxtr = 10 * nobjs
        bgc = rng.choice(cols)
        remcols = remove(bgc, cols)
        gi = canvas(bgc, (h, w))
        inds = asindices(gi)
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = rng.randint(3, 6)
            ow = rng.randint(3, 6)
            obj = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
            if rng.choice((True, False)):
                nkeep = unifint(rng, diff_lb, diff_ub, (2, len(obj) - 1))
                nrem = len(obj) - nkeep
                obj = remove(rng.choice(totuple(obj - corners(obj))), obj)
                for k in range(nrem - 1):
                    xx = sfilter(obj, lambda ij: len(dneighbors(ij) & obj) == 1)
                    if len(xx) == 0:
                        break
                    obj = remove(rng.choice(totuple(xx)), obj)
            npert = unifint(rng, diff_lb, diff_ub, (0, oh + ow))
            objcands = outbox(obj) | outbox(outbox(obj)) | outbox(outbox(outbox(obj)))
            obj = set(obj)
            for k in range(npert):
                obj.add(rng.choice(totuple((objcands - obj) & (mapply(dneighbors, obj) & objcands))))
            obj = normalize(obj)
            oh, ow = shape(obj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = rng.choice(totuple(cands))
            plcd = shift(obj, loc)
            if plcd.issubset(inds):
                gi = fill(gi, rng.choice(remcols), plcd)
                succ += 1
                inds = (inds - plcd) - mapply(neighbors, plcd)
        objs = objects(gi, T, F, F)
        bobjs = colorfilter(objs, bgc)
        objsm = mfilter(bobjs, compose(flip, rbind(bordering, gi)))
        if len(objsm) > 0:
            res = mfilter(objs - bobjs, rbind(adjacent, objsm))
            go = fill(gi, 8, res)
            break
    return {"input": gi, "output": go}


def generate_a61ba2ce(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 15))
    w = unifint(rng, diff_lb, diff_ub, (4, 15))
    lociL = rng.randint(2, h - 2)
    lociR = rng.randint(2, h - 2)
    locjT = rng.randint(2, w - 2)
    locjB = rng.randint(2, w - 2)
    bgc, c1, c2, c3, c4 = rng.sample(cols, 5)
    ulco = connect((0, 0), (lociL - 1, 0)) | connect((0, 0), (0, locjT - 1))
    urco = connect((0, w - 1), (0, locjT)) | connect((0, w - 1), (lociR - 1, w - 1))
    llco = connect((h - 1, 0), (lociL, 0)) | connect((h - 1, 0), (h - 1, locjB - 1))
    lrco = connect((h - 1, w - 1), (h - 1, locjB)) | connect((h - 1, w - 1), (lociR, w - 1))
    go = canvas(bgc, (h, w))
    go = fill(go, c1, ulco)
    go = fill(go, c2, urco)
    go = fill(go, c3, llco)
    go = fill(go, c4, lrco)
    fullh = unifint(rng, diff_lb, diff_ub, (2 * h, 30))
    fullw = unifint(rng, diff_lb, diff_ub, (2 * w, 30))
    gi = canvas(bgc, (fullh, fullw))
    objs = (ulco, urco, llco, lrco)
    ocols = (c1, c2, c3, c4)
    while True:
        inds = asindices(gi)
        locs = []
        for o, c in zip(objs, ocols):
            cands = sfilter(inds, lambda ij: shift(o, ij).issubset(inds))
            if len(cands) == 0:
                break
            loc = rng.choice(totuple(cands))
            locs.append(loc)
            inds = inds - shift(o, loc)
        if len(locs) == 4:
            break
    for o, c, l in zip(objs, ocols, locs):
        gi = fill(gi, c, shift(o, l))
    return {"input": gi, "output": go}


def generate_bbc9ae5d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    w = unifint(rng, diff_lb, diff_ub, (2, 15))
    w = w * 2
    locinv = unifint(rng, diff_lb, diff_ub, (2, w))
    locj = w - locinv
    loc = (0, locj)
    c1 = rng.choice(cols)
    remcols = remove(c1, cols)
    ln1 = connect((0, 0), (0, locj))
    remobj = connect((0, locj + 1), (0, w - 1))
    numc = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, numc)
    remobj = {(rng.choice(ccols), ij) for ij in remobj}
    gi = canvas(-1, (1, w))
    go = canvas(-1, (w // 2, w))
    ln2 = shoot(loc, (1, 1))
    gi = fill(gi, c1, ln1)
    gi = paint(gi, remobj)
    go = fill(go, c1, mapply(rbind(shoot, (0, -1)), ln2))
    for c, ij in remobj:
        go = fill(go, c, shoot(ij, (1, 1)))
    return {"input": gi, "output": go}


def generate_9edfc990(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(2, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    namt = unifint(rng, diff_lb, diff_ub, (int(0.4 * h * w), int(0.7 * h * w)))
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    locs = rng.sample(totuple(inds), namt)
    noise = {(rng.choice(cols), ij) for ij in locs}
    gi = paint(gi, noise)
    remlocs = inds - set(locs)
    numc = unifint(rng, diff_lb, diff_ub, (1, max(1, len(remlocs) // 10)))
    blocs = rng.sample(totuple(remlocs), numc)
    gi = fill(gi, 1, blocs)
    objs = objects(gi, T, F, F)
    objs = colorfilter(objs, 0)
    res = mfilter(objs, rbind(adjacent, blocs))
    go = fill(gi, 1, res)
    return {"input": gi, "output": go}


def generate_a78176bb(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    nlns = unifint(rng, diff_lb, diff_ub, (1, (h + w) // 8))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    spopts = []
    for idx in range(h - 5, -1, -1):
        spopts.append((idx, 0))
    for idx in range(1, w - 4, 1):
        spopts.append((0, idx))
    while succ < nlns and tr < maxtr:
        tr += 1
        if len(spopts) == 0:
            break
        sp = rng.choice(spopts)
        ln = shoot(sp, (1, 1)) & fullinds
        if not ln.issubset(inds):
            continue
        lno = sorted(ln, key=lambda x: x[0])
        trid1 = rng.randint(2, min(5, len(lno) - 3))
        trid2 = rng.randint(2, min(5, len(lno) - 3))
        tri1 = sfilter(asindices(canvas(-1, (trid1, trid1))), lambda ij: ij[1] >= ij[0])
        triloc1 = add(rng.choice(lno[1 : -trid1 - 1]), (0, 1))
        tri2 = dmirror(sfilter(asindices(canvas(-1, (trid2, trid2))), lambda ij: ij[1] >= ij[0]))
        triloc2 = add(rng.choice(lno[1 : -trid2 - 1]), (1, 0))
        spo2 = add(sp, (0, -trid2 - 2))
        nexlin2 = {add(spo2, (k, k)) for k in range(max(h, w))} & fullinds
        spo1 = add(sp, (-trid1 - 2, 0))
        nexlin1 = {add(spo1, (k, k)) for k in range(max(h, w))} & fullinds
        for idx, (tri, triloc, nexlin) in enumerate(
            rng.sample([(tri1, triloc1, nexlin1), (tri2, triloc2, nexlin2)], 2)
        ):
            tri = shift(tri, triloc)
            fullobj = ln | tri | nexlin
            if idx == 0:
                lncol, tricol = rng.sample(remcols, 2)
            else:
                tricol = rng.choice(remove(lncol, remcols))
            if fullobj.issubset(inds) if idx == 0 else (tri | nexlin).issubset(fullobj):
                succ += 1
                inds = (inds - fullobj) - mapply(neighbors, fullobj)
                gi = fill(gi, tricol, tri)
                gi = fill(gi, lncol, ln)
                go = fill(go, lncol, ln)
                go = fill(go, lncol, nexlin)
    if rng.choice((True, False)):
        gi = hmirror(gi)
        go = hmirror(go)
    return {"input": gi, "output": go}


def generate_995c5fa3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    o1 = asindices(canvas(-1, (4, 4)))
    o2 = box(asindices(canvas(-1, (4, 4))))
    o3 = asindices(canvas(-1, (4, 4))) - {(1, 0), (2, 0), (1, 3), (2, 3)}
    o4 = o1 - shift(asindices(canvas(-1, (2, 2))), (2, 1))
    mpr = [(o1, 2), (o2, 8), (o3, 3), (o4, 4)]
    num = unifint(rng, diff_lb, diff_ub, (1, 6))
    h = 4
    w = 4 * num + num - 1
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    ccols = []
    for k in range(num):
        col = rng.choice(remcols)
        obj, outcol = rng.choice(mpr)
        locj = 5 * k
        gi = fill(gi, col, shift(obj, (0, locj)))
        ccols.append(outcol)
    go = tuple(repeat(c, num) for c in ccols)
    return {"input": gi, "output": go}


def generate_9aec4887(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (12, 30))
    w = unifint(rng, diff_lb, diff_ub, (12, 30))
    oh = unifint(rng, diff_lb, diff_ub, (4, h // 2 - 2))
    ow = unifint(rng, diff_lb, diff_ub, (4, w // 2 - 2))
    bgc, pc, c1, c2, c3, c4 = rng.sample(cols, 6)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (oh, ow))
    ln1 = connect((1, 0), (oh - 2, 0))
    ln2 = connect((1, ow - 1), (oh - 2, ow - 1))
    ln3 = connect((0, 1), (0, ow - 2))
    ln4 = connect((oh - 1, 1), (oh - 1, ow - 2))
    go = fill(go, c1, ln1)
    go = fill(go, c2, ln2)
    go = fill(go, c3, ln3)
    go = fill(go, c4, ln4)
    objB = asobject(go)
    bounds = asindices(canvas(-1, (oh - 2, ow - 2)))
    objA = {rng.choice(totuple(bounds))}
    ncells = unifint(rng, diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for k in range(ncells - 1):
        objA.add(rng.choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    while shape(objA) != (oh - 2, ow - 2):
        objA.add(rng.choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    fullinds = asindices(gi)
    loci = rng.randint(0, h - 2 * oh + 2)
    locj = rng.randint(0, w - ow)
    plcdB = shift(objB, (loci, locj))
    plcdi = toindices(plcdB)
    rems = sfilter(fullinds - plcdi, lambda ij: loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = rng.choice(totuple(rems))
    plcdA = shift(objA, loc)
    gi = paint(gi, plcdB)
    gi = fill(gi, pc, plcdA)
    objA = shift(objA, (1, 1))
    objs = objects(go, T, F, T)
    for ij in objA:
        manhs = {obj: manhattan(obj, {ij}) for obj in objs}
        manhsl = list(manhs.values())
        mmh = min(manhsl)
        if manhsl.count(mmh) == 1:
            col = color([o for o, mnh in manhs.items() if mmh == mnh][0])
        else:
            col = pc
        go = fill(go, col, {ij})
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_846bdb03(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (12, 30))
    w = unifint(rng, diff_lb, diff_ub, (12, 30))
    oh = unifint(rng, diff_lb, diff_ub, (4, h // 2 - 2))
    ow = unifint(rng, diff_lb, diff_ub, (4, w // 2 - 2))
    bgc, dotc, c1, c2 = rng.sample(cols, 4)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (oh, ow))
    ln1 = connect((1, 0), (oh - 2, 0))
    ln2 = connect((1, ow - 1), (oh - 2, ow - 1))
    go = fill(go, c1, ln1)
    go = fill(go, c2, ln2)
    go = fill(go, dotc, corners(asindices(go)))
    objB = asobject(go)
    bounds = asindices(canvas(-1, (oh - 2, ow - 2)))
    objA = {rng.choice(totuple(bounds))}
    ncells = unifint(rng, diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for k in range(ncells - 1):
        objA.add(rng.choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    while shape(objA) != (oh - 2, ow - 2):
        objA.add(rng.choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    fullinds = asindices(gi)
    loci = rng.randint(0, h - 2 * oh + 2)
    locj = rng.randint(0, w - ow)
    plcdB = shift(objB, (loci, locj))
    plcdi = toindices(plcdB)
    rems = sfilter(fullinds - plcdi, lambda ij: loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = rng.choice(totuple(rems))
    plcdA = shift(objA, loc)
    mp = center(plcdA)[1]
    plcdAL = sfilter(plcdA, lambda ij: ij[1] < mp)
    plcdAR = plcdA - plcdAL
    plcdA = recolor(c1, plcdAL) | recolor(c2, plcdAR)
    gi = paint(gi, plcdB)
    ism = rng.choice((True, False))
    gi = paint(gi, vmirror(plcdA) if ism else plcdA)
    objA = shift(normalize(plcdA), (1, 1))
    objs = objects(go, T, F, T)
    go = paint(go, objA)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_2dd70a9a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    if rng.choice((True, False)):
        oh = unifint(rng, diff_lb, diff_ub, (5, h - 2))
        ow = unifint(rng, diff_lb, diff_ub, (3, w - 2))
        loci = rng.randint(1, h - oh - 1)
        locj = rng.randint(1, w - ow - 1)
        hli = rng.randint(loci + 2, loci + oh - 3)
        sp = {(loci + oh - 1, locj), (loci + oh - 2, locj)}
        ep = {(loci, locj + ow - 1), (loci + 1, locj + ow - 1)}
        bp1 = (hli - 1, locj)
        bp2 = (hli, locj + ow)
        ln1 = connect((loci + oh - 1, locj), (hli, locj))
        ln2 = connect((hli, locj), (hli, locj + ow - 1))
        ln3 = connect((hli, locj + ow - 1), (loci + 2, locj + ow - 1))
    else:
        oh = unifint(rng, diff_lb, diff_ub, (3, h - 2))
        ow = unifint(rng, diff_lb, diff_ub, (3, w - 2))
        loci = rng.randint(1, h - oh - 1)
        locj = rng.randint(1, w - ow - 1)
        if rng.choice((True, False)):
            sp1j = rng.randint(locj, locj + ow - 3)
            ep1j = locj
        else:
            ep1j = rng.randint(locj, locj + ow - 3)
            sp1j = locj
        sp = {(loci, sp1j), (loci, sp1j + 1)}
        ep = {(loci + oh - 1, ep1j), (loci + oh - 1, ep1j + 1)}
        bp1 = (loci, locj + ow)
        bp2 = (loci + oh, locj + ow - 1)
        ln1 = connect((loci, sp1j + 2), (loci, locj + ow - 1))
        ln2 = connect((loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1))
        ln3 = connect((loci + oh - 1, ep1j + 2), (loci + oh - 1, locj + ow - 1))
    gi = fill(gi, 3, sp)
    gi = fill(gi, 2, ep)
    go = fill(go, 3, sp)
    go = fill(go, 2, ep)
    lns = ln1 | ln2 | ln3
    bps = {bp1, bp2}
    gi = fill(gi, fgc, bps)
    go = fill(go, fgc, bps)
    go = fill(go, 3, lns)
    inds = ofcolor(go, bgc)
    namt = unifint(rng, diff_lb, diff_ub, (0, len(inds) // 2))
    noise = rng.sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = fill(go, fgc, noise)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_36fdfd69(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (4,))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 30))
    bgc, fgc, objc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    namt = rng.randint(int(0.35 * h * w), int(0.65 * h * w))
    noise = rng.sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = tuple(e for e in gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = rng.randint(2, 7)
        ow = rng.randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if bd.issubset(inds):
            ncells = rng.randint(2, oh * ow - 1)
            obj = {rng.choice(totuple(bd))}
            for k in range(ncells - 1):
                obj.add(rng.choice(totuple((bd - obj) & mapply(neighbors, mapply(dneighbors, obj)))))
            while len(obj) == height(obj) * width(obj):
                obj = {rng.choice(totuple(bd))}
                for k in range(ncells - 1):
                    obj.add(rng.choice(totuple((bd - obj) & mapply(neighbors, mapply(dneighbors, obj)))))
            obj = normalize(obj)
            oh, ow = shape(obj)
            obj = shift(obj, loc)
            bd = backdrop(obj)
            gi2 = fill(gi, fgc, bd)
            gi2 = fill(gi2, objc, obj)
            if colorcount(gi2, objc) < min(colorcount(gi2, fgc), colorcount(gi2, bgc)):
                succ += 1
                inds = (inds - bd) - (outbox(bd) | outbox(outbox(bd)))
                gi = gi2
                go = fill(go, 4, bd)
                go = fill(go, objc, obj)
    return {"input": gi, "output": go}


def generate_28e73c20(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3,))
    direcmapper = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    sp = (0, w - 1)
    direc = (1, 0)
    ncols = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(cols, ncols)
    gi = canvas(-1, (h, w))
    inds = asindices(gi)
    obj = {(rng.choice(ccols), ij) for ij in inds}
    gi = paint(gi, obj)
    go = fill(gi, 3, connect((0, 0), sp))
    lw = w
    lh = h
    ld = h
    isverti = False
    while ld > 0:
        lw -= 1
        lh -= 1
        ep = add(sp, multiply(direc, ld - 1))
        ln = connect(sp, ep)
        go = fill(go, 3, ln)
        direc = direcmapper[direc]
        if isverti:
            ld = lh
        else:
            ld = lw
        isverti = not isverti
        sp = ep
    gi = dmirror(dmirror(gi)[1:])
    go = dmirror(dmirror(go)[1:])
    return {"input": gi, "output": go}


def generate_3eda0437(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(1, 10, 1), (6,))
    h = unifint(rng, diff_lb, diff_ub, (3, 8))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    if rng.choice((True, False)):
        h, w = w, h
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    fgcs = rng.sample(cols, ncols)
    gi = canvas(-1, (h, w))
    gi = paint(gi, {(rng.choice(fgcs), ij) for ij in asindices(gi)})
    spac = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 3 * 2))
    inds = asindices(gi)
    obj = rng.sample(totuple(inds), spac)
    gi = fill(gi, 0, obj)
    locx = (rng.randint(0, h - 1), rng.randint(0, w - 1))
    gi = fill(gi, 0, {locx, add(locx, RIGHT), add(locx, DOWN), add(locx, UNITY)})
    maxsiz = -1
    mapper = dict()
    maxpossw = max([r.count(0) for r in gi])
    maxpossh = max([c.count(0) for c in dmirror(gi)])
    for a in range(2, maxpossh + 1):
        for b in range(2, maxpossw + 1):
            siz = a * b
            if siz < maxsiz:
                continue
            objx = recolor(0, asindices(canvas(-1, (a, b))))
            occs = occurrences(gi, objx)
            if len(occs) > 0:
                if siz == maxsiz:
                    mapper[objx] = occs
                elif siz > maxsiz:
                    mapper = {objx: occs}
                    maxsiz = siz
    go = tuple(e for e in gi)
    for obj, locs in mapper.items():
        go = fill(go, 6, mapply(lbind(shift, obj), locs))
    return {"input": gi, "output": go}


def generate_7447852a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    w = unifint(rng, diff_lb, diff_ub, (2, 8))
    h = unifint(rng, diff_lb, diff_ub, (w + 1, 30))
    bgc, linc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(linc, cols))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        gi2 = fill(gi, linc, {sp})
        if gi2 == gi:
            break
        gi = gi2
    gi = rot90(gi)
    objs = objects(gi, T, F, F)
    inds = ofcolor(gi, bgc)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 7))
    ccols = rng.sample(remcols, numcols)
    ncells = unifint(rng, diff_lb, diff_ub, (0, len(inds)))
    locs = rng.sample(totuple(inds), ncells)
    obj = {(rng.choice(ccols), ij) for ij in locs}
    gi = paint(gi, obj)
    go = tuple(e for e in gi)
    objs = order(colorfilter(objs, bgc), leftmost)
    objs = merge(set(objs[0::3]))
    go = fill(go, 4, objs)
    return {"input": gi, "output": go}


def generate_6b9890af(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    oh = unifint(rng, diff_lb, diff_ub, (2, 5))
    ow = unifint(rng, diff_lb, diff_ub, (2, 5))
    h = unifint(rng, diff_lb, diff_ub, (2 * oh + 2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2 * ow + 2, 30))
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {rng.choice(totuple(bounds))}
    while shape(obj) != (oh, ow):
        obj.add(rng.choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    maxfac = 1
    while oh * maxfac + 2 <= h - oh and ow * maxfac + 2 <= w - ow:
        maxfac += 1
    maxfac -= 1
    fac = unifint(rng, diff_lb, diff_ub, (1, maxfac))
    bgc, sqc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numc)
    obj = {(rng.choice(ccols), ij) for ij in obj}
    gi = canvas(bgc, (h, w))
    sq = box(frozenset({(0, 0), (oh * fac + 1, ow * fac + 1)}))
    loci = rng.randint(0, h - (oh * fac + 2) - oh)
    locj = rng.randint(0, w - (ow * fac + 2))
    gi = fill(gi, sqc, shift(sq, (loci, locj)))
    loci = rng.randint(loci + oh * fac + 2, h - oh)
    locj = rng.randint(0, w - ow)
    objp = shift(obj, (loci, locj))
    gi = paint(gi, objp)
    go = canvas(bgc, (oh * fac + 2, ow * fac + 2))
    go = fill(go, sqc, sq)
    go2 = paint(canvas(bgc, (oh, ow)), obj)
    upscobj = asobject(upscale(go2, fac))
    go = paint(go, shift(upscobj, (1, 1)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_963e52fc(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 15))
    p = unifint(rng, diff_lb, diff_ub, (2, w // 2))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, numc)
    obj = set()
    for j in range(p):
        ub = unifint(rng, diff_lb, diff_ub, (0, h // 2))
        ub = h // 2 - ub
        lb = unifint(rng, diff_lb, diff_ub, (ub, h - 1))
        numcells = unifint(rng, diff_lb, diff_ub, (1, lb - ub + 1))
        for ii in rng.sample(interval(ub, lb + 1, 1), numcells):
            loc = (ii, j)
            col = rng.choice(ccols)
            cell = (col, loc)
            obj.add(cell)
    go = canvas(bgc, (h, w * 2))
    minobj = obj | shift(obj, (0, p))
    addonw = rng.randint(0, p)
    addon = sfilter(obj, lambda cij: cij[1][1] < addonw)
    fullobj = minobj | addon
    leftshift = rng.randint(0, addonw)
    fullobj = shift(fullobj, (0, -leftshift))
    go = paint(go, fullobj)
    for j in range((2 * w) // (2 * p) + 1):
        go = paint(go, shift(fullobj, (0, j * 2 * p)))
    gi = lefthalf(go)
    return {"input": gi, "output": go}


def generate_3e980e27(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3))
    h = unifint(rng, diff_lb, diff_ub, (11, 30))
    w = unifint(rng, diff_lb, diff_ub, (11, 30))
    bgc, rcol, gcol = rng.sample(cols, 3)
    objs = []
    for fixc, remc in ((2, rcol), (3, gcol)):
        oh = unifint(rng, diff_lb, diff_ub, (2, 5))
        ow = unifint(rng, diff_lb, diff_ub, (2, 5))
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {rng.choice(totuple(bounds))}
        ncellsd = unifint(rng, diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = rng.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(2, ncells), oh * ow)
        for k in range(ncells - 1):
            obj.add(rng.choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        fixp = rng.choice(totuple(obj))
        rem = remove(fixp, obj)
        obj = {(fixc, fixp)} | recolor(remc, rem)
        objs.append(obj)
    robj, gobj = objs
    obj1, obj2 = rng.sample(objs, 2)
    loci1 = rng.randint(0, h - height(obj1) - height(obj2) - 1)
    locj1 = rng.randint(0, w - width(obj1))
    loci2 = rng.randint(loci1 + height(obj1) + 1, h - height(obj2))
    locj2 = rng.randint(0, w - width(obj2))
    gi = canvas(bgc, (h, w))
    obj1p = shift(obj1, (loci1, locj1))
    obj2p = shift(obj2, (loci2, locj2))
    gi = paint(gi, obj1p)
    gi = paint(gi, obj2p)
    noccs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // int(1.5 * (len(robj) + len(gobj)))))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    robj = vmirror(robj)
    inds = ofcolor(gi, bgc) - (mapply(neighbors, toindices(obj1p)) | mapply(neighbors, toindices(obj2p)))
    go = tuple(e for e in gi)
    objopts = [robj, gobj]
    while tr < maxtr and succ < noccs:
        tr += 1
        obj = rng.choice(objopts)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
            gi = paint(gi, sfilter(plcd, lambda cij: cij[0] in (2, 3)))
            go = paint(go, plcd)
    if unifint(rng, diff_lb, diff_ub, (1, 100)) < 30:
        c = rng.choice((2, 3))
        giobjs = objects(gi, F, T, T)
        goobjs = objects(go, F, T, T)
        gi = fill(gi, bgc, mfilter(giobjs, lambda o: c in palette(o)))
        go = fill(go, bgc, mfilter(goobjs, lambda o: c in palette(o)))
    return {"input": gi, "output": go}


def generate_a8c38be5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    goh = unifint(rng, diff_lb, diff_ub, (9, 20))
    gow = unifint(rng, diff_lb, diff_ub, (9, 20))
    h = unifint(rng, diff_lb, diff_ub, (goh + 4, 30))
    w = unifint(rng, diff_lb, diff_ub, (gow + 4, 30))
    bgc, sqc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numc)
    go = canvas(sqc, (goh, gow))
    go = fill(go, bgc, box(asindices(go)))
    loci1 = rng.randint(2, goh - 7)
    loci2 = rng.randint(loci1 + 4, goh - 3)
    locj1 = rng.randint(2, gow - 7)
    locj2 = rng.randint(locj1 + 4, gow - 3)
    f1 = hfrontier((loci1, 0))
    f2 = hfrontier((loci2, 0))
    f3 = vfrontier((0, locj1))
    f4 = vfrontier((0, locj2))
    fs = f1 | f2 | f3 | f4
    go = fill(go, sqc, fs)
    go = fill(go, bgc, {((loci1 + loci2) // 2, 1)})
    go = fill(go, bgc, {((loci1 + loci2) // 2, gow - 2)})
    go = fill(go, bgc, {(1, (locj1 + locj2) // 2)})
    go = fill(go, bgc, {(goh - 2, (locj1 + locj2) // 2)})
    objs = objects(go, T, F, T)
    objs = merge(set(recolor(rng.choice(ccols), obj) for obj in objs))
    go = paint(go, objs)
    gi = go
    hdelt = h - goh
    hdelt1 = rng.randint(1, hdelt - 3)
    hdelt2 = rng.randint(1, hdelt - hdelt1 - 2)
    hdelt3 = rng.randint(1, hdelt - hdelt1 - hdelt2 - 1)
    hdelt4 = hdelt - hdelt1 - hdelt2 - hdelt3
    wdelt = w - gow
    wdelt1 = rng.randint(1, wdelt - 3)
    wdelt2 = rng.randint(1, wdelt - wdelt1 - 2)
    wdelt3 = rng.randint(1, wdelt - wdelt1 - wdelt2 - 1)
    wdelt4 = wdelt - wdelt1 - wdelt2 - wdelt3
    gi = gi[:loci2] + repeat(repeat(bgc, gow), hdelt2) + gi[loci2:]
    gi = gi[: loci1 + 1] + repeat(repeat(bgc, gow), hdelt3) + gi[loci1 + 1 :]
    gi = repeat(repeat(bgc, gow), hdelt1) + gi + repeat(repeat(bgc, gow), hdelt4)
    gi = dmirror(gi)
    gi = gi[:locj2] + repeat(repeat(bgc, h), wdelt2) + gi[locj2:]
    gi = gi[: locj1 + 1] + repeat(repeat(bgc, h), wdelt3) + gi[locj1 + 1 :]
    gi = repeat(repeat(bgc, h), wdelt1) + gi + repeat(repeat(bgc, h), wdelt4)
    gi = dmirror(gi)
    nswitcheroos = unifint(rng, diff_lb, diff_ub, (0, 10))
    if rng.choice((True, False)):
        gi = gi[loci1 + hdelt1 + 1 :] + gi[: loci1 + hdelt1 + 1]
    if rng.choice((True, False)):
        gi = dmirror(gi)
        gi = gi[locj1 + wdelt1 + 1 :] + gi[: locj1 + wdelt1 + 1]
        gi = dmirror(gi)
    for k in range(nswitcheroos):
        o = asobject(gi)
        tmpc = canvas(bgc, (h + 12, w + 12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, T, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        mpr = dict()
        for obj in objs:
            shp = shape(obj)
            if shp in mpr:
                mpr[shp].append(obj)
            else:
                mpr[shp] = [obj]
        if max([len(x) for x in mpr.values()]) == 1:
            break
        ress = [(kk, v) for kk, v in mpr.items() if len(v) > 1]
        res, abc = rng.choice(ress)
        a, b = rng.sample(abc, 2)
        ulca = ulcorner(a)
        ulcb = ulcorner(b)
        ap = shift(normalize(a), ulcb)
        bp = shift(normalize(b), ulca)
        gi = paint(gi, ap | bp)
    nshifts = unifint(rng, diff_lb, diff_ub, (0, 30))
    for k in range(nshifts):
        o = asobject(gi)
        tmpc = canvas(bgc, (h + 12, w + 12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, F, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        objs = sfilter(objs, compose(flip, rbind(bordering, gi)))
        if len(objs) == 0:
            break
        obj = rng.choice(totuple(objs))
        direc1 = (rng.randint(-1, 1), rng.randint(-1, 1))
        direc2 = position({(h // 2, w // 2)}, {center(obj)})
        direc = rng.choice((direc1, direc2))
        gi = fill(gi, bgc, obj)
        gi = paint(gi, shift(obj, direc))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_6c434453(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, ncols)
    nobjs = unifint(rng, diff_lb, diff_ub, (2, (h * w) // 16))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        if rng.choice((True, False)):
            oh = rng.choice((3, 5))
            ow = rng.choice((3, 5))
            obji = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
        else:
            oh = rng.randint(1, 5)
            ow = rng.randint(1, 5)
            bounds = asindices(canvas(-1, (oh, ow)))
            ncells = rng.randint(1, oh * ow)
            obji = {rng.choice(totuple(bounds))}
            for k in range(ncells - 1):
                obji.add(rng.choice(totuple((bounds - obji) & mapply(dneighbors, obji))))
            obji = normalize(obji)
        oh, ow = shape(obji)
        flag = obji == box(obji) and set(shape(obji)).issubset({3, 5})
        if flag:
            objo = connect((0, ow // 2), (oh - 1, ow // 2)) | connect((oh // 2, 0), (oh // 2, ow - 1))
            tocover = backdrop(obji)
        else:
            objo = obji
            tocover = obji
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        loc = rng.choice(totuple(cands))
        plcdi = shift(obji, loc)
        if plcdi.issubset(inds):
            plcdo = shift(objo, loc)
            succ += 1
            tocoveri = shift(tocover, loc)
            inds = (inds - tocoveri) - mapply(dneighbors, tocoveri)
            col = rng.choice(ccols)
            gi = fill(gi, col, plcdi)
            go = fill(go, 2 if flag else col, plcdo)
    return {"input": gi, "output": go}


def generate_7837ac64(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    oh = unifint(rng, diff_lb, diff_ub, (2, 6))
    ow = unifint(rng, diff_lb, diff_ub, (2, 6))
    bgc, linc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(linc, cols))
    numcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numcols)
    go = canvas(bgc, (oh, ow))
    inds = asindices(go)
    fullinds = asindices(go)
    nocc = unifint(rng, diff_lb, diff_ub, (1, oh * ow))
    for k in range(nocc):
        mpr = {
            cc: sfilter(
                inds | mapply(neighbors, ofcolor(go, cc)),
                lambda ij: (neighbors(ij) & fullinds).issubset(inds | ofcolor(go, cc)),
            )
            for cc in ccols
        }
        mpr = [(kk, vv) for kk, vv in mpr.items() if len(vv) > 0]
        if len(mpr) == 0:
            break
        col, locs = rng.choice(mpr)
        loc = rng.choice(totuple(locs))
        go = fill(go, col, {loc})
        inds = remove(loc, inds)
    obj = fullinds - ofcolor(go, bgc)
    go = subgrid(obj, go)
    oh, ow = shape(go)
    sqsizh = unifint(rng, diff_lb, diff_ub, (2, (30 - oh - 1) // oh))
    sqsizw = unifint(rng, diff_lb, diff_ub, (2, (30 - ow - 1) // ow))
    fullh = oh + 1 + oh * sqsizh
    fullw = ow + 1 + ow * sqsizw
    gi = canvas(linc, (fullh, fullw))
    sq = backdrop(frozenset({(0, 0), (sqsizh - 1, sqsizw - 1)}))
    obj = asobject(go)
    for col, ij in obj:
        plcd = shift(sq, add((1, 1), multiply(ij, (sqsizh + 1, sqsizw + 1))))
        gi = fill(gi, bgc, plcd)
        if col != bgc:
            gi = fill(gi, col, corners(outbox(plcd)))
    gih = unifint(rng, diff_lb, diff_ub, (fullh, 30))
    giw = unifint(rng, diff_lb, diff_ub, (fullw, 30))
    loci = rng.randint(0, gih - fullh)
    locj = rng.randint(0, giw - fullw)
    gigi = canvas(bgc, (gih, giw))
    plcd = shift(asobject(gi), (loci, locj))
    gigi = paint(gigi, plcd)
    ulci, ulcj = ulcorner(plcd)
    lrci, lrcj = lrcorner(plcd)
    for a in range(ulci, gih + 1, sqsizh + 1):
        gigi = fill(gigi, linc, hfrontier((a, 0)))
    for a in range(ulci, -1, -sqsizh - 1):
        gigi = fill(gigi, linc, hfrontier((a, 0)))
    for b in range(ulcj, giw + 1, sqsizw + 1):
        gigi = fill(gigi, linc, vfrontier((0, b)))
    for b in range(ulcj, -1, -sqsizw - 1):
        gigi = fill(gigi, linc, vfrontier((0, b)))
    gi = paint(gigi, plcd)
    gop = palette(go)
    while True:
        go2 = identity(go)
        for c in set(ccols) & gop:
            o1 = frozenset({(c, ORIGIN), (bgc, RIGHT), (c, (0, 2))})
            o2 = dmirror(o1)
            go2 = fill(go2, c, combine(shift(occurrences(go, o1), RIGHT), shift(occurrences(go, o2), DOWN)))
        if go2 == go:
            break
        go = go2
    return {"input": gi, "output": go}


def generate_5ad4f10b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = apply(lbind(shift, nbh), {(0, 0), (-1, 0), (0, -1), (-1, -1)})
    oh = unifint(rng, diff_lb, diff_ub, (2, 6))
    ow = unifint(rng, diff_lb, diff_ub, (2, 6))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(rng, diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = rng.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(rng.sample(totuple(bounds), ncells))
    while len(sfilter(obj, lambda ij: sum([len(obj & shift(nbh, ij)) < 4 for nbh in nbhs]) > 0)) == 0:
        ncellsd = unifint(rng, diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = rng.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow)
        obj = set(rng.sample(totuple(bounds), ncells))
    obj = normalize(obj)
    oh, ow = shape(obj)
    bgc, noisec, objc = rng.sample(cols, 3)
    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)
    fac = unifint(rng, diff_lb, diff_ub, (2, min(28 // oh, 28 // ow)))
    gobj = asobject(upscale(replace(go, noisec, objc), fac))
    oh, ow = shape(gobj)
    h = unifint(rng, diff_lb, diff_ub, (oh + 2, 30))
    w = unifint(rng, diff_lb, diff_ub, (ow + 2, 30))
    loci = rng.randint(1, h - oh - 1)
    locj = rng.randint(1, w - ow - 1)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(gobj, (loci, locj)))
    cands = ofcolor(gi, bgc)
    namt = unifint(rng, diff_lb, diff_ub, (2, max(1, len(cands) // 4)))
    noise = rng.sample(totuple(cands), namt)
    gi = fill(gi, noisec, noise)
    return {"input": gi, "output": go}


def generate_7df24a62(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (12, 32))
    w = unifint(rng, diff_lb, diff_ub, (12, 32))
    oh = unifint(rng, diff_lb, diff_ub, (3, min(7, h // 3)))
    ow = unifint(rng, diff_lb, diff_ub, (3, min(7, w // 3)))
    bgc, noisec, sqc = rng.sample(cols, 3)
    tmpg = canvas(sqc, (oh, ow))
    inbounds = backdrop(inbox(asindices(tmpg)))
    obj = {rng.choice(totuple(inbounds))}
    while shape(obj) != (oh - 2, ow - 2):
        obj.add(rng.choice(totuple(inbounds - obj)))
    pat = fill(tmpg, noisec, obj)
    targ = asobject(fill(canvas(bgc, (oh, ow)), noisec, obj))
    sour = asobject(pat)
    gi = canvas(bgc, (h, w))
    loci = rng.randint(1, h - oh - 1)
    locj = rng.randint(1, w - ow - 1)
    plcddd = shift(sour, (loci, locj))
    gi = paint(gi, plcddd)
    inds = ofcolor(gi, bgc) & shift(asindices(canvas(-1, (h - 2, w - 2))), (1, 1))
    inds = inds - (toindices(plcddd) | mapply(dneighbors, toindices(plcddd)))
    namt = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
    noise = rng.sample(totuple(inds), namt)
    gi = fill(gi, noisec, noise)
    targs = []
    sours = []
    for fn1 in (identity, dmirror, cmirror, hmirror, vmirror):
        for fn2 in (identity, dmirror, cmirror, hmirror, vmirror):
            targs.append(normalize(fn1(fn2(targ))))
            sours.append(normalize(fn1(fn2(sour))))
    noccs = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // ((oh * ow * 4)))))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        t = rng.choice(targs)
        hh, ww = shape(t)
        cands = sfilter(inds, lambda ij: 1 <= ij[0] <= h - hh - 1 and 1 <= ij[1] <= w - ww - 1)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        tp = shift(t, loc)
        tpi = toindices(tp)
        if tpi.issubset(inds):
            succ += 1
            inds = inds - tpi
            gi = paint(gi, tp)
    go = replace(gi, sqc, bgc)
    go = paint(go, plcddd)
    res = set()
    for t, s in zip(targs, sours):
        res |= mapply(lbind(shift, s), occurrences(go, t))
    go = paint(go, res)
    gi = trim(gi)
    go = trim(go)
    return {"input": gi, "output": go}


def generate_539a4f51(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    d = unifint(rng, diff_lb, diff_ub, (2, 15))
    h, w = d, d
    gi = canvas(0, (h, w))
    numc = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(cols, numc)
    numocc = unifint(rng, diff_lb, diff_ub, (1, d))
    arr = [rng.choice(ccols) for k in range(numocc)]
    while len(set(arr)) == 1:
        arr = [rng.choice(ccols) for k in range(d)]
    for j, col in enumerate(arr):
        gi = fill(gi, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    go = canvas(0, (2 * d, 2 * d))
    for j in range(2 * d):
        col = arr[j % len(arr)]
        go = fill(go, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_ce602527(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (12, 30))
    w = unifint(rng, diff_lb, diff_ub, (12, 30))
    bgc, c1, c2, c3 = rng.sample(cols, 4)
    while True:
        objs = []
        for k in range(2):
            oh1 = unifint(rng, diff_lb, diff_ub, (3, h // 3 - 1))
            ow1 = unifint(rng, diff_lb, diff_ub, (3, w // 3 - 1))
            cc1 = canvas(bgc, (oh1, ow1))
            bounds1 = asindices(cc1)
            numcd1 = unifint(rng, diff_lb, diff_ub, (0, (oh1 * ow1) // 2 - 4))
            numc1 = rng.choice((numcd1, oh1 * ow1 - numcd1))
            numc1 = min(max(3, numc1), oh1 * ow1 - 3)
            obj1 = {rng.choice(totuple(bounds1))}
            while len(obj1) < numc1 or shape(obj1) != (oh1, ow1):
                obj1.add(rng.choice(totuple((bounds1 - obj1) & mapply(dneighbors, obj1))))
            objs.append(normalize(obj1))
        a, b = objs
        ag = fill(canvas(0, shape(a)), 1, a)
        bg = fill(canvas(0, shape(b)), 1, b)
        maxinh = min(height(a), height(b)) // 2 + 1
        maxinw = min(width(a), width(b)) // 2 + 1
        maxshp = (maxinh, maxinw)
        ag = crop(ag, (0, 0), maxshp)
        bg = crop(bg, (0, 0), maxshp)
        if ag != bg:
            break
    a, b = objs
    trgo = rng.choice(objs)
    trgo2 = ofcolor(upscale(fill(canvas(0, shape(trgo)), 1, trgo), 2), 1)
    staysinh = unifint(rng, diff_lb, diff_ub, (maxinh * 2, height(trgo) * 2))
    nout = height(trgo2) - staysinh
    loci = h - height(trgo2) + nout
    locj = rng.randint(0, w - maxinw * 2)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, c3, shift(trgo2, (loci, locj)))
    (lcol, lobj), (rcol, robj) = rng.sample([(c1, a), (c2, b)], 2)
    cands = ofcolor(gi, bgc) - box(asindices(gi))
    lca = sfilter(cands, lambda ij: ij[1] < w // 3 * 2)
    rca = sfilter(cands, lambda ij: ij[1] > w // 3)
    lcands = sfilter(lca, lambda ij: shift(lobj, ij).issubset(lca))
    rcands = sfilter(rca, lambda ij: shift(robj, ij).issubset(rca))
    while True:
        lloc = rng.choice(totuple(lcands))
        rloc = rng.choice(totuple(lcands))
        lplcd = shift(lobj, lloc)
        rplcd = shift(robj, rloc)
        if lplcd.issubset(cands) and rplcd.issubset(cands) and len(lplcd & rplcd) == 0:
            break
    gi = fill(gi, lcol, shift(lobj, lloc))
    gi = fill(gi, rcol, shift(robj, rloc))
    go = fill(canvas(bgc, shape(trgo)), c1 if trgo == a else c2, trgo)
    mfs = (identity, rot90, rot180, rot270, cmirror, dmirror, hmirror, vmirror)
    mf = rng.choice(mfs)
    gi, go = mf(gi), mf(go)
    return {"input": gi, "output": go}


def generate_c8cbb738(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    gh = unifint(rng, diff_lb, diff_ub, (3, 10))
    gw = unifint(rng, diff_lb, diff_ub, (3, 10))
    h = unifint(rng, diff_lb, diff_ub, (gh * 2, 30))
    w = unifint(rng, diff_lb, diff_ub, (gw * 2, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (gh, gw))
    goinds = asindices(go)
    ring = box(goinds)
    crns = corners(ring)
    remring = ring - crns
    nrr = len(remring)
    sc = ccols[0]
    go = fill(go, sc, crns)
    loci = rng.randint(0, h - gh)
    locj = rng.randint(0, w - gw)
    gi = fill(gi, sc, shift(crns, (loci, locj)))
    ccols = ccols[1:]
    issucc = True
    bL = connect((0, 0), (gh - 1, 0))
    bR = connect((0, gw - 1), (gh - 1, gw - 1))
    bT = connect((0, 0), (0, gw - 1))
    bB = connect((gh - 1, 0), (gh - 1, gw - 1))
    validpairs = [(bL, bT), (bL, bB), (bR, bT), (bR, bB)]
    for c in ccols:
        if len(remring) < 3:
            break
        obj = set(
            rng.sample(
                totuple(remring), unifint(rng, diff_lb, diff_ub, (3, max(3, min(len(remring), nrr // len(ccols)))))
            )
        )
        flag = False
        for b1, b2 in validpairs:
            if len(obj & b1) > 0 and len(obj & b2) > 0:
                flag = True
                break
        if flag:
            oh, ow = shape(obj)
            locs = ofcolor(gi, bgc)
            cands = sfilter(locs, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) > 0:
                objn = normalize(obj)
                cands2 = sfilter(cands, lambda ij: shift(objn, ij).issubset(locs))
                if len(cands2) > 0:
                    loc = rng.choice(totuple(cands2))
                    gi = fill(gi, c, shift(objn, loc))
                    go = fill(go, c, obj)
                    remring -= obj
    return {"input": gi, "output": go}


def generate_b527c5c6(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    noccs = unifint(rng, diff_lb, diff_ub, (1, 10))
    tr = 0
    succ = 0
    maxtr = 10 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        d1 = rng.randint(3, rng.randint(3, (min(h, w)) // 2 - 1))
        d2 = rng.randint(d1 * 2 + 1, rng.randint(d1 * 2 + 1, min(h, w) - 1))
        oh, ow = rng.sample([d1, d2], 2)
        cands = sfilter(inds, lambda ij: 1 <= ij[0] <= h - oh - 1 and 1 <= ij[1] <= w - ow - 1)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bd = backdrop(bx)
        if ow < oh:
            lrflag = True
            dcands1 = connect((loci + ow - 1, locj), (loci + oh - 1 - ow + 1, locj))
            dcands2 = shift(dcands1, (0, ow - 1))
        else:
            lrflag = False
            dcands1 = connect((loci, locj + oh - 1), (loci, locj + ow - 1 - oh + 1))
            dcands2 = shift(dcands1, (oh - 1, 0))
        dcands = dcands1 | dcands2
        loc = rng.choice(totuple(dcands))
        sgnflag = -1 if loc in dcands1 else 1
        direc = (sgnflag * (0 if lrflag else 1), sgnflag * (0 if not lrflag else 1))
        ln = shoot(loc, direc)
        shell = set()
        for k in range(min(oh, ow) - 1):
            shell |= power(outbox, k + 1)(ln)
        sqc, dotc = rng.sample(ccols, 2)
        giobj = recolor(sqc, remove(loc, bd)) | {(dotc, loc)}
        goobj = recolor(sqc, (bd | shell) - ln) | recolor(dotc, ln)
        goobj = sfilter(goobj, lambda cij: cij[1] in fullinds)
        goobji = toindices(goobj)
        if goobji.issubset(inds):
            succ += 1
            inds = (inds - goobji) - mapply(dneighbors, bd)
            gi = paint(gi, giobj)
            go = paint(go, goobj)
    return {"input": gi, "output": go}


def generate_228f6490(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nsq = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 50))
    succ = 0
    tr = 0
    maxtr = 5 * nsq
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    sqc = rng.choice(remcols)
    remcols = remove(sqc, remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    forbidden = []
    while tr < maxtr and succ < nsq:
        tr += 1
        oh = rng.randint(3, 6)
        ow = rng.randint(3, 6)
        bd = asindices(canvas(-1, (oh, ow)))
        bounds = shift(asindices(canvas(-1, (oh - 2, ow - 2))), (1, 1))
        obj = {rng.choice(totuple(bounds))}
        ncells = rng.randint(1, (oh - 2) * (ow - 2))
        for k in range(ncells - 1):
            obj.add(rng.choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        sqcands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(sqcands) == 0:
            continue
        loc = rng.choice(totuple(sqcands))
        bdplcd = shift(bd, loc)
        if bdplcd.issubset(inds):
            tmpinds = inds - bdplcd
            inobjn = normalize(obj)
            oh, ow = shape(obj)
            inobjcands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(inobjcands) == 0:
                continue
            loc2 = rng.choice(totuple(inobjcands))
            inobjplcd = shift(inobjn, loc2)
            bdnorm = bd - obj
            if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
                forbidden.append(bdnorm)
                forbidden.append(inobjn)
                succ += 1
                inds = (inds - (bdplcd | inobjplcd)) - mapply(dneighbors, inobjplcd)
                col = rng.choice(remcols)
                oplcd = shift(obj, loc)
                gi = fill(gi, sqc, bdplcd - oplcd)
                go = fill(go, sqc, bdplcd)
                go = fill(go, col, oplcd)
                gi = fill(gi, col, inobjplcd)
    nremobjs = unifint(rng, diff_lb, diff_ub, (0, len(inds) // 25))
    succ = 0
    tr = 0
    maxtr = 10 * nremobjs
    while tr < maxtr and succ < nremobjs:
        tr += 1
        oh = rng.randint(1, 4)
        ow = rng.randint(1, 4)
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {rng.choice(totuple(bounds))}
        ncells = rng.randint(1, oh * ow)
        for k in range(ncells - 1):
            obj.add(rng.choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        if obj in forbidden:
            continue
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            col = rng.choice(remcols)
            gi = fill(gi, col, plcd)
            go = fill(go, col, plcd)
    return {"input": gi, "output": go}


def generate_93b581b8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, numcols)
    numocc = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 50))
    succ = 0
    tr = 0
    maxtr = 10 * numocc
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    while tr < maxtr and succ < numocc:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - 2 and ij[1] <= w - 2)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        c1, c2, c3, c4 = [rng.choice(ccols) for k in range(4)]
        q = {(0, 0), (0, 1), (1, 0), (1, 1)}
        inobj = {(c1, (0, 0)), (c2, (0, 1)), (c3, (1, 0)), (c4, (1, 1))}
        outobj = (
            inobj
            | recolor(c4, shift(q, (-2, -2)))
            | recolor(c3, shift(q, (-2, 2)))
            | recolor(c2, shift(q, (2, -2)))
            | recolor(c1, shift(q, (2, 2)))
        )
        inobjplcd = shift(inobj, loc)
        outobjplcd = shift(outobj, loc)
        outobjplcd = sfilter(outobjplcd, lambda cij: cij[1] in fullinds)
        outobjplcdi = toindices(outobjplcd)
        if outobjplcdi.issubset(inds):
            succ += 1
            inds = (inds - outobjplcdi) - mapply(dneighbors, toindices(inobjplcd))
            gi = paint(gi, inobjplcd)
            go = paint(go, outobjplcd)
    return {"input": gi, "output": go}


def generate_447fd412(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (12, 30))
    w = unifint(rng, diff_lb, diff_ub, (12, 30))
    bgc, indic, mainc = rng.sample(cols, 3)
    oh = unifint(rng, diff_lb, diff_ub, (1, 4))
    ow = unifint(rng, diff_lb, diff_ub, (1, 4))
    if oh * ow < 3:
        if rng.choice((True, False)):
            oh = unifint(rng, diff_lb, diff_ub, (3, 4))
        else:
            ow = unifint(rng, diff_lb, diff_ub, (3, 4))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = unifint(rng, diff_lb, diff_ub, (3, oh * ow))
    obj = {rng.choice(totuple(bounds))}
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    objt = totuple(obj)
    kk = len(obj)
    nindic = rng.randint(1, kk // 2 if kk % 2 == 1 else kk // 2 - 1)
    indicobj = set(rng.sample(objt, nindic))
    mainobj = obj - indicobj
    obj = recolor(indic, indicobj) | recolor(mainc, mainobj)
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = paint(go, plcd)
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
    fullinds = asindices(gi)
    noccs = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(plcd)))))
    tr = 0
    maxtr = 5 * noccs
    succ = 0
    while succ < noccs and tr < maxtr:
        tr += 1
        fac = rng.randint(1, min(5, min(h, w) // max(oh, ow) - 1))
        outobj = upscale(obj, fac)
        inobj = sfilter(outobj, lambda cij: cij[0] == indic)
        hh, ww = shape(outobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        inobjp = shift(inobj, loc)
        outobjp = shift(outobj, loc)
        outobjp = sfilter(outobjp, lambda cij: cij[1] in fullinds)
        outobjpi = toindices(outobjp)
        if outobjpi.issubset(inds):
            succ += 1
            inds = (inds - outobjpi) - mapply(neighbors, toindices(inobjp))
            gi = paint(gi, inobjp)
            go = paint(go, outobjp)
    return {"input": gi, "output": go}


def generate_50846271(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    cf1 = lambda d: {(d // 2, 0), (d // 2, d - 1)} | set(
        rng.sample(totuple(connect((d // 2, 0), (d // 2, d - 1))), rng.randint(1, d))
    )
    cf2 = lambda d: {(0, d // 2), (d - 1, d // 2)} | set(
        rng.sample(totuple(connect((0, d // 2), (d - 1, d // 2))), rng.randint(1, d))
    )
    cf3 = lambda d: set(
        rng.sample(totuple(remove((d // 2, d // 2), connect((d // 2, 0), (d // 2, d - 1)))), rng.randint(1, d - 1))
    ) | set(rng.sample(totuple(remove((d // 2, d // 2), connect((0, d // 2), (d - 1, d // 2)))), rng.randint(1, d - 1)))
    cf = lambda d: rng.choice((cf1, cf2, cf3))(d)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    dim = unifint(rng, diff_lb, diff_ub, (1, 3))
    dim = 2 * dim + 1
    cross = connect((dim // 2, 0), (dim // 2, dim - 1)) | connect((0, dim // 2), (dim - 1, dim // 2))
    bgc, crossc, noisec = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    namt = unifint(rng, diff_lb, diff_ub, (int(0.35 * h * w), int(0.65 * h * w)))
    inds = asindices(gi)
    noise = rng.sample(totuple(inds), namt)
    gi = fill(gi, noisec, noise)
    initcross = rng.choice((cf1, cf2))(dim)
    loci = rng.randint(0, h - dim)
    locj = rng.randint(0, w - dim)
    delt = shift(cross - initcross, (loci, locj))
    gi = fill(gi, crossc, shift(initcross, (loci, locj)))
    gi = fill(gi, noisec, delt)
    go = fill(gi, 8, delt)
    plcd = shift(cross, (loci, locj))
    bd = backdrop(plcd)
    nbhs = mapply(neighbors, plcd)
    inds = (inds - plcd) - nbhs
    nbhs2 = mapply(neighbors, nbhs)
    inds = inds - nbhs2
    inds = inds - mapply(neighbors, nbhs2)
    noccs = unifint(rng, diff_lb, diff_ub, (1, (h * w) / (10 * dim)))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - dim and ij[1] <= w - dim)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        marked = shift(cf(dim), loc)
        full = shift(cross, loc)
        unmarked = full - marked
        inobj = recolor(noisec, unmarked) | recolor(crossc, marked)
        outobj = recolor(8, unmarked) | recolor(crossc, marked)
        outobji = toindices(outobj)
        if outobji.issubset(inds):
            dnbhs = mapply(neighbors, outobji)
            dnbhs2 = mapply(neighbors, dnbhs)
            inds = (inds - outobji) - (dnbhs | dnbhs2 | mapply(neighbors, dnbhs2))
            succ += 1
            gi = paint(gi, inobj)
            go = paint(go, outobj)
    return {"input": gi, "output": go}


def generate_ae3edfdc(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3, 7))
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    bgc = rng.choice(cols)
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    rdi = rng.randint(1, h - 2)
    rdj = rng.randint(1, w - 2)
    rd = (rdi, rdj)
    reminds = inds - ({rd} | neighbors(rd))
    reminds = sfilter(reminds, lambda ij: 1 <= ij[0] <= h - 2 and 1 <= ij[1] <= w - 2)
    bd = rng.choice(totuple(reminds))
    bdi, bdj = bd
    go = fill(go, 2, {rd})
    go = fill(go, 1, {bd})
    ngd = unifint(rng, diff_lb, diff_ub, (1, 8))
    gd = rng.sample(totuple(neighbors(rd)), ngd)
    nod = unifint(rng, diff_lb, diff_ub, (1, 8))
    od = rng.sample(totuple(neighbors(bd)), nod)
    go = fill(go, 3, gd)
    go = fill(go, 7, od)
    gdmapper = {d: (3, position({rd}, {d})) for d in gd}
    odmapper = {d: (7, position({bd}, {d})) for d in od}
    mpr = {**gdmapper, **odmapper}
    ub = (len(gd) + len(od)) * ((h + w) // 5)
    ndist = unifint(rng, diff_lb, diff_ub, (1, ub))
    gi = tuple(e for e in go)
    fullinds = asindices(gi)
    for k in range(ndist):
        options = []
        for loc, (col, direc) in mpr.items():
            ii, jj = add(loc, direc)
            if (ii, jj) in fullinds and gi[ii][jj] == bgc:
                options.append((loc, col, direc))
        if len(options) == 0:
            break
        loc, col, direc = rng.choice(options)
        del mpr[loc]
        newloc = add(loc, direc)
        mpr[newloc] = (col, direc)
        gi = fill(gi, bgc, {loc})
        gi = fill(gi, col, {newloc})
    return {"input": gi, "output": go}


def generate_469497ad(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 6))
    w = unifint(rng, diff_lb, diff_ub, (3, 6))
    bgc, sqc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    sqh = rng.randint(1, h - 2)
    sqw = rng.randint(1, w - 2)
    sqloci = rng.randint(0, h - sqh - 2)
    sqlocj = rng.randint(0, w - sqw - 2)
    sq = backdrop(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    gi = fill(gi, sqc, sq)
    numcub = min(min(min(h, w) + 1, 30 // (max(h, w))), 7)
    numc = unifint(rng, diff_lb, diff_ub, (2, numcub))
    numaccc = numc - 1
    remcols = remove(bgc, remove(sqc, cols))
    ccols = rng.sample(remcols, numaccc)
    gi = rot180(gi)
    locs = rng.sample(interval(1, min(h, w), 1), numaccc - 1)
    locs = [0] + sorted(locs)
    for c, l in zip(ccols, locs):
        gi = fill(gi, c, shoot((0, l), (0, 1)))
        gi = fill(gi, c, shoot((l, 0), (1, 0)))
    gi = rot180(gi)
    go = upscale(gi, numc)
    rect = ofcolor(go, sqc)
    l1 = shoot(lrcorner(rect), (1, 1))
    l2 = shoot(ulcorner(rect), (-1, -1))
    l3 = shoot(urcorner(rect), (-1, 1))
    l4 = shoot(llcorner(rect), (1, -1))
    ll = l1 | l2 | l3 | l4
    go = fill(go, 2, ll & ofcolor(go, bgc))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_97a05b5b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (15, 30))
    w = unifint(rng, diff_lb, diff_ub, (15, 30))
    sgh = rng.randint(h // 3, h // 3 * 2)
    sgw = rng.randint(w // 3, w // 3 * 2)
    bgc, sqc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    gi = canvas(bgc, (h, w))
    oh = rng.randint(2, sgh // 2)
    ow = rng.randint(2, sgw // 2)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, 8))
    objs = set()
    cands = asindices(canvas(-1, (oh, ow)))
    forbidden = set()
    tr = 0
    maxtr = 4 * nobjs
    while len(objs) != nobjs and tr < maxtr:
        tr += 1
        obj = {rng.choice(totuple(cands))}
        ncells = rng.randint(1, oh * ow - 1)
        for k in range(ncells - 1):
            obj.add(rng.choice(totuple((cands - obj) & mapply(neighbors, obj))))
        obj |= rng.choice((dmirror, cmirror, vmirror, hmirror))(obj)
        if len(obj) == height(obj) * width(obj):
            continue
        obj = frozenset(obj)
        objn = normalize(obj)
        if objn not in forbidden:
            objs.add(objn)
        for augmf1 in (identity, dmirror, cmirror, hmirror, vmirror):
            for augmf2 in (identity, dmirror, cmirror, hmirror, vmirror):
                forbidden.add(augmf1(augmf2(objn)))
    tr = 0
    maxtr = 5 * nobjs
    succ = 0
    loci = rng.randint(0, h - sgh)
    locj = rng.randint(0, w - sgw)
    bd = backdrop(frozenset({(loci, locj), (loci + sgh - 1, locj + sgw - 1)}))
    gi = fill(gi, sqc, bd)
    go = canvas(sqc, (sgh, sgw))
    goinds = asindices(go)
    giinds = asindices(gi) - shift(goinds, (loci, locj))
    giinds = giinds - mapply(neighbors, shift(goinds, (loci, locj)))
    while succ < nobjs and tr < maxtr and len(objs) > 0:
        tr += 1
        obj = rng.choice(totuple(objs))
        col = rng.choice(remcols)
        subgi = fill(canvas(col, shape(obj)), sqc, obj)
        if len(palette(subgi)) == 1:
            continue
        f1 = rng.choice((identity, dmirror, vmirror, cmirror, hmirror))
        f2 = rng.choice((identity, dmirror, vmirror, cmirror, hmirror))
        f = compose(f1, f2)
        subgo = f(subgi)
        giobj = asobject(subgi)
        goobj = asobject(subgo)
        ohi, owi = shape(giobj)
        oho, owo = shape(goobj)
        gocands = sfilter(goinds, lambda ij: ij[0] <= sgh - oho and ij[1] <= sgw - owo)
        if len(gocands) == 0:
            continue
        goloc = rng.choice(totuple(gocands))
        goplcd = shift(goobj, goloc)
        goplcdi = toindices(goplcd)
        if goplcdi.issubset(goinds):
            gicands = sfilter(giinds, lambda ij: ij[0] <= h - ohi and ij[1] <= owi)
            if len(gicands) == 0:
                continue
            giloc = rng.choice(totuple(gicands))
            giplcd = shift(giobj, giloc)
            giplcdi = toindices(giplcd)
            if giplcdi.issubset(giinds):
                succ += 1
                remcols = remove(col, remcols)
                objs = remove(obj, objs)
                goinds = goinds - goplcdi
                giinds = (giinds - giplcdi) - mapply(neighbors, giplcdi)
                gi = paint(gi, giplcd)
                gi = fill(gi, bgc, sfilter(shift(goplcd, (loci, locj)), lambda cij: cij[0] == sqc))
                go = paint(go, goplcd)
    return {"input": gi, "output": go}


def generate_a5313dff(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    noccs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    inds = shift(asindices(canvas(-1, (h + 2, w + 2))), (-1, -1))
    while (tr < maxtr and succ < noccs) or len(
        sfilter(colorfilter(objects(gi, T, F, F), bgc), compose(flip, rbind(bordering, gi)))
    ) == 0:
        tr += 1
        oh = rng.randint(3, 8)
        ow = rng.randint(3, 8)
        bx = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
        ins = backdrop(inbox(bx))
        loc = rng.choice(totuple(inds))
        plcdins = shift(ins, loc)
        if len(plcdins & ofcolor(gi, fgc)) == 0:
            succ += 1
            gi = fill(gi, fgc, shift(bx, loc))
            if rng.choice((True, True, False)):
                ss = rng.sample(totuple(plcdins), rng.randint(1, max(1, len(ins) // 2)))
                gi = fill(gi, fgc, ss)
    res = mfilter(colorfilter(objects(gi, T, F, F), bgc), compose(flip, rbind(bordering, gi)))
    go = fill(gi, 1, res)
    return {"input": gi, "output": go}


def generate_780d0b14(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nh = unifint(rng, diff_lb, diff_ub, (2, 6))
    nw = unifint(rng, diff_lb, diff_ub, (2, 6))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (3, 9))
    ccols = rng.sample(remcols, ncols)
    go = canvas(-1, (nh, nw))
    obj = {(rng.choice(ccols), ij) for ij in asindices(go)}
    go = paint(go, obj)
    while len(dedupe(go)) < nh or len(dedupe(dmirror(go))) < nw:
        obj = {(rng.choice(ccols), ij) for ij in asindices(go)}
        go = paint(go, obj)
    h = unifint(rng, diff_lb, diff_ub, (2 * nh + nh - 1, 30))
    w = unifint(rng, diff_lb, diff_ub, (2 * nw + nw - 1, 30))
    hdist = [2 for k in range(nh)]
    for k in range(h - 2 * nh - nh + 1):
        idx = rng.randint(0, nh - 1)
        hdist[idx] += 1
    wdist = [2 for k in range(nw)]
    for k in range(w - 2 * nw - nw + 1):
        idx = rng.randint(0, nw - 1)
        wdist[idx] += 1
    gi = merge(tuple(repeat(r, c) + (repeat(bgc, nw),) for r, c in zip(go, hdist)))[:-1]
    gi = dmirror(merge(tuple(repeat(r, c) + (repeat(bgc, h),) for r, c in zip(dmirror(gi), wdist)))[:-1])
    objs = objects(gi, T, F, F)
    bgobjs = colorfilter(objs, bgc)
    objs = objs - bgobjs
    for obj in objs:
        gi = fill(gi, bgc, rng.sample(totuple(toindices(obj)), unifint(rng, diff_lb, diff_ub, (1, len(obj) // 2))))
    return {"input": gi, "output": go}


def generate_57aa92db(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    oh = rng.randint(2, 5)
    ow = rng.randint(2, 5)
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {rng.choice(totuple(bounds))}
    ncellsd = unifint(rng, diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = rng.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fixp = rng.choice(totuple(obj))
    bgc, fixc, mainc = rng.sample(cols, 3)
    remcols = difference(cols, (bgc, fixc, mainc))
    gi = canvas(bgc, (h, w))
    obj = {(fixc, fixp)} | recolor(mainc, remove(fixp, obj))
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
    nocc = unifint(rng, diff_lb, diff_ub, (1, (h * w) // (4 * len(obj))))
    tr = 0
    succ = 0
    maxtr = 5 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = rng.randint(1, 4)
        objups = upscale(obj, fac)
        hh, ww = shape(objups)
        cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        objupsplcd = shift(objups, loc)
        objupsplcdi = toindices(objupsplcd)
        if objupsplcdi.issubset(inds):
            succ += 1
            newc = rng.choice(remcols)
            fixp2 = sfilter(objupsplcd, lambda cij: cij[0] == fixc)
            inds = inds - mapply(neighbors, objupsplcdi)
            gi = paint(gi, fixp2)
            go = paint(go, fixp2)
            remobjfull = toindices(objupsplcd - fixp2)
            ntorem = unifint(rng, diff_lb, diff_ub, (0, max(0, len(remobjfull) - 1)))
            ntokeep = len(remobjfull) - ntorem
            tokeep = {rng.choice(totuple(remobjfull & outbox(fixp2)))}
            fixp2i = toindices(fixp2)
            for k in range(ntokeep - 1):
                fullopts = remobjfull & mapply(neighbors, tokeep | fixp2i)
                remopts = fullopts - tokeep
                tokeep.add(rng.choice(totuple(remopts)))
            gi = fill(gi, newc, tokeep)
            go = fill(go, newc, remobjfull)
    return {"input": gi, "output": go}


def generate_53b68214(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(rng, diff_lb, diff_ub, (2, 6))
        w = unifint(rng, diff_lb, diff_ub, (8, 30))
        bgc = rng.choice(cols)
        remcols = remove(bgc, cols)
        ncols = unifint(rng, diff_lb, diff_ub, (1, 9))
        ccols = rng.sample(remcols, ncols)
        oh = unifint(rng, diff_lb, diff_ub, (1, h // 2))
        ow = unifint(rng, diff_lb, diff_ub, (1, w // 2 - 1))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(rng, diff_lb, diff_ub, (1, oh * ow))
        obj = rng.sample(totuple(bounds), ncells)
        obj = {(rng.choice(ccols), ij) for ij in obj}
        obj = normalize(obj)
        oh, ow = shape(obj)
        locj = rng.randint(0, w // 2)
        plcd = shift(obj, (0, locj))
        go = canvas(bgc, (10, w))
        hoffs = rng.randint(0, ow // 2 + 1)
        for k in range(10 // oh + 1):
            go = paint(go, shift(plcd, (k * oh, k * hoffs)))
        if len(palette(go[h:])) > 1:
            break
    gi = go[:h]
    if rng.choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    return {"input": gi, "output": go}


def generate_39e1d7f9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 10))
    w = unifint(rng, diff_lb, diff_ub, (5, 10))
    bgc, linc, dotc = rng.sample(cols, 3)
    remcols = difference(cols, (bgc, linc, dotc))
    gi = canvas(bgc, (h, w))
    loci = rng.randint(1, h - 2)
    locj = rng.randint(1, w - 2)
    if h == 5:
        loci = rng.choice((1, h - 2))
    if w == 5:
        locj = rng.choice((1, w - 2))
    npix = unifint(rng, diff_lb, diff_ub, (1, 8))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 7))
    ccols = rng.sample(remcols, ncols)
    candsss = neighbors((loci, locj))
    pixs = {(loci, locj)}
    for k in range(npix):
        pixs.add(rng.choice(totuple((mapply(dneighbors, pixs) & candsss) - pixs)))
    pixs = totuple(remove((loci, locj), pixs))
    obj = {(rng.choice(ccols), ij) for ij in pixs}
    gi = fill(gi, dotc, {(loci, locj)})
    gi = paint(gi, obj)
    go = tuple(e for e in gi)
    noccs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // (2 * len(pixs) + 1)))
    succ = 0
    tr = 0
    maxtr = 6 * noccs
    inds = ofcolor(gi, bgc) - mapply(dneighbors, neighbors((loci, locj)))
    objn = shift(obj, (-loci, -locj))
    triedandfailed = set()
    while (tr < maxtr and succ < noccs) or succ == 0:
        lopcands = totuple(inds - triedandfailed)
        if len(lopcands) == 0:
            break
        tr += 1
        loci, locj = rng.choice(lopcands)
        plcd = shift(objn, (loci, locj))
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            inds = inds - (plcdi | {(loci, locj)})
            succ += 1
            gi = fill(gi, dotc, {(loci, locj)})
            go = fill(go, dotc, {(loci, locj)})
            go = paint(go, plcd)
        else:
            triedandfailed.add((loci, locj))
    hfac = unifint(rng, diff_lb, diff_ub, (1, (30 - h + 1) // h))
    wfac = unifint(rng, diff_lb, diff_ub, (1, (30 - w + 1) // w))
    fullh = hfac * h + h - 1
    fullw = wfac * w + w - 1
    gi2 = canvas(linc, (fullh, fullw))
    go2 = canvas(linc, (fullh, fullw))
    bd = asindices(canvas(-1, (hfac, wfac)))
    for a in range(h):
        for b in range(w):
            c = gi[a][b]
            gi2 = fill(gi2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    for a in range(h):
        for b in range(w):
            c = go[a][b]
            go2 = fill(go2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    gi, go = gi2, go2
    return {"input": gi, "output": go}


def generate_017c7c7b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 2))
    h = unifint(rng, diff_lb, diff_ub, (3, 10))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    h += h
    fgc = rng.choice(cols)
    go = canvas(0, (h + h // 2, w))
    oh = unifint(rng, diff_lb, diff_ub, (1, h // 3 * 2))
    ow = unifint(rng, diff_lb, diff_ub, (1, w))
    locj = rng.randint(0, w - ow)
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    ncells = rng.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow)
    obj = rng.sample(totuple(bounds), ncells)
    for k in range((2 * h) // oh):
        go = fill(go, 2, shift(obj, (k * oh, 0)))
    gi = replace(go[:h], 2, fgc)
    return {"input": gi, "output": go}


def generate_8a004b2b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    oh = unifint(rng, diff_lb, diff_ub, (2, h // 5))
    ow = unifint(rng, diff_lb, diff_ub, (2, w // 5))
    bounds = asindices(canvas(-1, (oh, ow)))
    bgc, cornc, ac1, ac2, objc = rng.sample(cols, 5)
    gi = canvas(bgc, (h, w))
    obj = {rng.choice(totuple(bounds))}
    ncellsd = unifint(rng, diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = rng.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fp1 = rng.choice(totuple(obj))
    fp2 = rng.choice(remove(fp1, totuple(obj)))
    remobj = obj - {fp1, fp2}
    obj = recolor(objc, remobj) | {(ac1, fp1), (ac2, fp2)}
    maxhscf = (h - oh - 4) // oh
    maxwscf = (w - ow - 4) // ow
    hscf = unifint(rng, diff_lb, diff_ub, (1, maxhscf))
    wscf = unifint(rng, diff_lb, diff_ub, (1, maxwscf))
    loci = rng.randint(0, 2)
    locj = rng.randint(0, 2)
    oplcd = shift(obj, (loci, locj))
    gi = paint(gi, oplcd)
    inh = hscf * oh
    inw = wscf * ow
    sqh = unifint(rng, diff_lb, diff_ub, (inh + 2, h - oh - 2))
    sqw = unifint(rng, diff_lb, diff_ub, (inw + 2, w))
    sqloci = rng.randint(loci + oh, h - sqh)
    sqlocj = rng.randint(0, w - sqw)
    crns = corners(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    gi = fill(gi, cornc, crns)
    gomini = subgrid(oplcd, gi)
    goo = vupscale(hupscale(gomini, wscf), hscf)
    goo = asobject(goo)
    gloci = rng.randint(sqloci + 1, sqloci + sqh - 1 - height(goo))
    glocj = rng.randint(sqlocj + 1, sqlocj + sqw - 1 - width(goo))
    gooplcd = shift(goo, (gloci, glocj))
    go = paint(gi, gooplcd)
    go = subgrid(crns, go)
    indic = sfilter(gooplcd, lambda cij: cij[0] in (ac1, ac2))
    gi = paint(gi, indic)
    if rng.choice((True, False)) and len(obj) > 3:
        idx = rng.choice(totuple(toindices(sfilter(obj, lambda cij: cij[0] == objc))))
        idxi, idxj = idx
        xx = shift(asindices(canvas(-1, (hscf, wscf))), (gloci + idxi * hscf, glocj + idxj * wscf))
        gi = fill(gi, objc, xx)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_49d1d64f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 28))
    w = unifint(rng, diff_lb, diff_ub, (2, 28))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 10))
    ccols = rng.sample(cols, ncols)
    gi = canvas(-1, (h, w))
    obj = {(rng.choice(ccols), ij) for ij in asindices(gi)}
    gi = paint(gi, obj)
    go = canvas(0, (h + 2, w + 2))
    go = paint(go, shift(asobject(gi), (1, 1)))
    ts = sfilter(obj, lambda cij: cij[1][0] == 0)
    bs = sfilter(obj, lambda cij: cij[1][0] == h - 1)
    ls = sfilter(obj, lambda cij: cij[1][1] == 0)
    rs = sfilter(obj, lambda cij: cij[1][1] == w - 1)
    ts = shift(ts, (1, 1))
    bs = shift(bs, (1, 1))
    ls = shift(ls, (1, 1))
    rs = shift(rs, (1, 1))
    go = paint(go, shift(ts, (-1, 0)))
    go = paint(go, shift(bs, (1, 0)))
    go = paint(go, shift(ls, (0, -1)))
    go = paint(go, shift(rs, (0, 1)))
    return {"input": gi, "output": go}


def generate_890034e9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    oh = rng.randint(2, h // 4)
    ow = rng.randint(2, w // 4)
    markercol = rng.choice(cols)
    remcols = remove(markercol, cols)
    numbgc = unifint(rng, diff_lb, diff_ub, (1, 8))
    bgcols = rng.sample(remcols, numbgc)
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    obj = {(rng.choice(bgcols), ij) for ij in inds}
    gi = paint(gi, obj)
    numbl = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    blacks = rng.sample(totuple(inds), numbl)
    gi = fill(gi, 0, blacks)
    patt = asindices(canvas(-1, (oh, ow)))
    tocover = set()
    for occ in occurrences(gi, recolor(0, patt)):
        tocover.add(rng.choice(totuple(shift(patt, occ))))
    tocover = {(rng.choice(bgcols), ij) for ij in tocover}
    gi = paint(gi, tocover)
    noccs = unifint(rng, diff_lb, diff_ub, (2, (h * w) // ((oh + 2) * (ow + 2))))
    tr = 0
    succ = 0
    maxtr = 5 * noccs
    go = tuple(e for e in gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        bd = shift(patt, loc)
        plcd = outbox(bd)
        if plcd.issubset(inds):
            succ += 1
            inds = inds - plcd
            gi = fill(gi, 0, bd)
            go = fill(go, 0, bd)
            if succ == 1:
                gi = fill(gi, markercol, plcd)
            go = fill(go, markercol, plcd)
            loci, locj = loc
            ln1 = connect((loci - 1, locj), (loci - 1, locj + ow - 1))
            ln2 = connect((loci + oh, locj), (loci + oh, locj + ow - 1))
            ln3 = connect((loci, locj - 1), (loci + oh - 1, locj - 1))
            ln4 = connect((loci, locj + ow), (loci + oh - 1, locj + ow))
            if succ > 1:
                fixxer = {(rng.choice(bgcols), rng.choice(totuple(xx))) for xx in [ln1, ln2, ln3, ln4]}
                gi = paint(gi, fixxer)
    return {"input": gi, "output": go}


def generate_776ffc46(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc, sqc, inc, outc = rng.sample(cols, 4)
    gi = canvas(bgc, (h, w))
    sqh = rng.randint(3, h // 3 + 1)
    sqw = rng.randint(3, w // 3 + 1)
    loci = rng.randint(0, 3)
    locj = rng.randint(0, w - sqw)
    bx = box(frozenset({(loci, locj), (loci + sqh - 1, locj + sqw - 1)}))
    bounds = asindices(canvas(-1, (sqh - 2, sqw - 2)))
    obj = {rng.choice(totuple(bounds))}
    ncells = rng.randint(1, (sqh - 2) * (sqw - 2))
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    objp = shift(obj, (loci + 1 + rng.randint(0, sqh - oh - 2), locj + 1 + rng.randint(0, sqw - ow - 2)))
    gi = fill(gi, sqc, bx)
    gi = fill(gi, inc, objp)
    inds = (ofcolor(gi, bgc) - backdrop(bx)) - mapply(neighbors, backdrop(bx))
    cands = sfilter(inds, lambda ij: shift(obj, ij).issubset(inds))
    loc = rng.choice(totuple(cands))
    plcd = shift(obj, loc)
    gi = fill(gi, outc, plcd)
    inds = (inds - plcd) - mapply(neighbors, plcd)
    noccs = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    fullinds = asindices(gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        if rng.choice((True, False)):
            sqh = rng.randint(3, h // 3 + 1)
            sqw = rng.randint(3, w // 3 + 1)
            bx = box(frozenset({(0, 0), (sqh - 1, sqw - 1)}))
            bounds = asindices(canvas(-1, (sqh - 2, sqw - 2)))
            obj2 = {rng.choice(totuple(bounds))}
            ncells = rng.randint(1, (sqh - 2) * (sqw - 2))
            for k in range(ncells - 1):
                obj2.add(rng.choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
            if normalize(obj2) == obj:
                if len(obj2) < (sqh - 2) * (sqw - 2):
                    obj2.add(rng.choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
                else:
                    continue
            obj2 = normalize(obj2)
            ooh, oow = shape(obj2)
            cands1 = connect((-1, -1), (-1, w - sqw + 1))
            cands2 = connect((h - sqh + 1, -1), (h - sqh + 1, w - sqw + 1))
            cands3 = connect((-1, -1), (h - sqh + 1, -1))
            cands4 = connect((-1, w - sqw + 1), (h - sqh + 1, w - sqw + 1))
            cands = cands1 | cands2 | cands3 | cands4
            if len(cands) == 0:
                continue
            loc = rng.choice(totuple(cands))
            sloci, slocj = loc
            plcdbx = shift(bx, loc)
            if (backdrop(plcdbx) & fullinds).issubset(inds):
                succ += 1
                oloci = rng.randint(sloci + 1, sloci + 1 + rng.randint(0, sqh - ooh - 2))
                olocj = rng.randint(slocj + 1, slocj + 1 + rng.randint(0, sqw - oow - 2))
                gi = fill(gi, sqc, plcdbx)
                gi = fill(gi, inc, shift(obj2, (oloci, olocj)))
                inds = inds - backdrop(outbox(plcdbx))
        else:
            ooh = rng.randint(1, h // 3 - 1)
            oow = rng.randint(1, w // 3 - 1)
            bounds = asindices(canvas(-1, (ooh, oow)))
            obj2 = {rng.choice(totuple(bounds))}
            ncells = rng.randint(1, oow * ooh)
            for k in range(ncells - 1):
                obj2.add(rng.choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
            if normalize(obj2) == obj:
                if len(obj2) < ooh * oow:
                    obj2.add(rng.choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
                else:
                    continue
        if rng.choice((True, False, False)):
            obj2 = obj
        obj2 = normalize(obj2)
        ooh, oow = shape(obj2)
        for kk in range(rng.randint(1, 3)):
            cands = sfilter(inds, lambda ij: ij[0] <= h - ooh and ij[1] <= w - oow)
            if len(cands) == 0:
                continue
            loc = rng.choice(totuple(cands))
            plcd = shift(obj2, loc)
            if plcd.issubset(inds):
                succ += 1
                inds = (inds - plcd) - mapply(neighbors, plcd)
                gi = fill(gi, outc, plcd)
    objs = objects(gi, T, F, F)
    objs = colorfilter(objs, outc)
    objs = mfilter(objs, lambda o: equality(normalize(toindices(o)), obj))
    go = fill(gi, inc, objs)
    return {"input": gi, "output": go}


def generate_e6721834(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 15))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    bgc1, bgc2, sqc = rng.sample(cols, 3)
    remcols = difference(cols, (bgc1, bgc2, sqc))
    gi1 = canvas(bgc1, (h, w))
    gi2 = canvas(bgc2, (h, w))
    noccs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 16))
    tr = 0
    succ = 0
    maxtr = 5 * noccs
    gi1inds = asindices(gi1)
    gi2inds = asindices(gi2)
    go = canvas(bgc2, (h, w))
    seen = []
    while tr < maxtr and succ < noccs:
        tr += 1
        oh = rng.randint(2, min(6, h // 2))
        ow = rng.randint(2, min(6, w // 2))
        cands = sfilter(gi1inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        bounds = shift(asindices(canvas(-1, (oh, ow))), loc)
        ncells = unifint(rng, diff_lb, diff_ub, (1, (oh * ow) // 2))
        obj = set(rng.sample(totuple(bounds), ncells))
        objc = rng.choice(remcols)
        objn = normalize(obj)
        if (objn, objc) in seen:
            continue
        seen.append(((objn, objc)))
        if bounds.issubset(gi1inds):
            succ += 1
            gi1inds = (gi1inds - bounds) - mapply(neighbors, bounds)
            gi1 = fill(gi1, sqc, bounds)
            gi1 = fill(gi1, objc, obj)
            cands2 = sfilter(gi2inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands2) == 0:
                continue
            loc2 = rng.choice(totuple(cands2))
            bounds2 = shift(shift(bounds, invert(loc)), loc2)
            obj2 = shift(shift(obj, invert(loc)), loc2)
            if bounds2.issubset(gi2inds):
                gi2inds = (gi2inds - bounds2) - mapply(neighbors, bounds2)
                gi2 = fill(gi2, objc, obj2)
                go = fill(go, sqc, bounds2)
                go = fill(go, objc, obj2)
    gi = vconcat(gi1, gi2)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_ef135b50(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(9, interval(0, 10, 1))
    while True:
        h = unifint(rng, diff_lb, diff_ub, (8, 30))
        w = unifint(rng, diff_lb, diff_ub, (8, 30))
        bgc = rng.choice(cols)
        remcols = remove(bgc, cols)
        numc = unifint(rng, diff_lb, diff_ub, (1, 8))
        ccols = rng.sample(remcols, numc)
        gi = canvas(bgc, (h, w))
        nsq = unifint(rng, diff_lb, diff_ub, (2, (h * w) // 30))
        succ = 0
        tr = 0
        maxtr = 5 * nsq
        inds = asindices(gi)
        pats = set()
        while tr < maxtr and succ < nsq:
            tr += 1
            oh = rng.randint(1, (h // 3 * 2))
            ow = rng.randint(1, (w // 3 * 2))
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = rng.choice(totuple(cands))
            loci, locj = loc
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            if bd.issubset(inds):
                succ += 1
                inds = (inds - bd) - mapply(neighbors, bd)
                gi = fill(gi, rng.choice(ccols), bd)
                pats.add(bd)
        res = set()
        ofc = ofcolor(gi, bgc)
        for pat1 in pats:
            for pat2 in remove(pat1, pats):
                if hmatching(pat1, pat2):
                    um = max(uppermost(pat1), uppermost(pat2))
                    bm = min(lowermost(pat1), lowermost(pat2))
                    lm = min(rightmost(pat1), rightmost(pat2)) + 1
                    rm = max(leftmost(pat1), leftmost(pat2)) - 1
                    res = res | backdrop(frozenset({(um, lm), (bm, rm)}))
        res = (res & ofc) - box(asindices(gi))
        go = fill(gi, 9, res)
        if go != gi:
            break
    return {"input": gi, "output": go}


def generate_794b24be(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    mpr = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 1)}
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    nblue = rng.randint(1, 4)
    go = canvas(bgc, (3, 3))
    for k in range(nblue):
        go = fill(go, 2, {mpr[k + 1]})
    gi = canvas(bgc, (h, w))
    locs = rng.sample(totuple(asindices(gi)), nblue)
    gi = fill(gi, 1, locs)
    remlocs = ofcolor(gi, bgc)
    namt = unifint(rng, diff_lb, diff_ub, (0, len(remlocs) // 2 - 1))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, 7))
    ccols = rng.sample(remcols, numc)
    noise = rng.sample(totuple(remlocs), namt)
    noise = {(rng.choice(ccols), ij) for ij in noise}
    gi = paint(gi, noise)
    return {"input": gi, "output": go}


def generate_ff28f65a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    mpr = {1: (0, 0), 2: (0, 2), 3: (1, 1), 4: (2, 0), 5: (2, 2)}
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    nred = rng.randint(1, 5)
    gi = canvas(bgc, (h, w))
    succ = 0
    tr = 0
    maxtr = 5 * nred
    inds = asindices(gi)
    while tr < maxtr and succ < nred:
        tr += 1
        oh = rng.randint(1, h // 2 + 1)
        ow = rng.randint(1, w // 2 + 1)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if bd.issubset(inds):
            succ += 1
            inds = (inds - bd) - mapply(dneighbors, bd)
            gi = fill(gi, 2, bd)
    nblue = succ
    namt = unifint(rng, diff_lb, diff_ub, (0, nred * 2))
    succ = 0
    tr = 0
    maxtr = 5 * namt
    remcols = remove(bgc, cols)
    tr += 1
    while tr < maxtr and succ < namt:
        tr += 1
        oh = rng.randint(1, h // 2 + 1)
        ow = rng.randint(1, w // 2 + 1)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if bd.issubset(inds):
            succ += 1
            inds = (inds - bd) - mapply(dneighbors, bd)
            gi = fill(gi, rng.choice(remcols), bd)
    go = canvas(bgc, (3, 3))
    for k in range(nblue):
        go = fill(go, 1, {mpr[k + 1]})
    return {"input": gi, "output": go}


def generate_73251a56(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        d = unifint(rng, diff_lb, diff_ub, (10, 30))
        h, w = d, d
        noisec = rng.choice(cols)
        remcols = remove(noisec, cols)
        nsl = unifint(rng, diff_lb, diff_ub, (2, min(9, h // 2)))
        slopes = [0] + sorted(rng.sample(interval(1, h - 1, 1), nsl - 1))
        ccols = rng.sample(cols, nsl)
        gi = canvas(-1, (h, w))
        inds = asindices(gi)
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            locs = sfilter(inds, lambda ij: slope * ij[1] <= ij[0])
            gi = fill(gi, col, locs)
        ln = connect((0, 0), (d - 1, d - 1))
        gi = fill(gi, ccols[-2], ln)
        obj = asobject(gi)
        obj = sfilter(obj, lambda cij: cij[1][1] >= cij[1][0])
        gi = paint(gi, dmirror(obj))
        cf1 = lambda g: ccols[-2] in palette(toobject(ln, g))
        cf2 = lambda g: len((ofcolor(g, noisec) & frozenset({ij[::-1] for ij in ofcolor(g, noisec)})) - ln) == 0
        ndist = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 15))
        tr = 0
        succ = 0
        maxtr = 10 * ndist
        go = tuple(e for e in gi)
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = rng.randint(1, 5)
            ow = rng.randint(1, 5)
            loci = rng.randint(1, h - oh - 1)
            locj = rng.randint(1, w - ow - 1)
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            gi2 = fill(gi, noisec, bd)
            if cf1(gi2) and cf2(gi2):
                succ += 1
                gi = gi2
        if gi != go:
            break
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_3631a71a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 15))
    w = h
    bgc, patchcol = rng.sample(cols, 2)
    patchcol = rng.choice(cols)
    bgc = rng.choice(remove(patchcol, cols))
    remcols = difference(cols, (bgc, patchcol))
    c = canvas(bgc, (h, w))
    inds = sfilter(asindices(c), lambda ij: ij[0] >= ij[1])
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, ncols)
    ncells = unifint(rng, diff_lb, diff_ub, (1, len(inds)))
    cells = set(rng.sample(totuple(inds), ncells))
    obj = {(rng.choice(ccols), ij) for ij in cells}
    c = paint(dmirror(paint(c, obj)), obj)
    c = hconcat(c, vmirror(c))
    c = vconcat(c, hmirror(c))
    cutoff = 2
    go = dmirror(dmirror(c[:-cutoff])[:-cutoff])
    gi = tuple(e for e in go)
    forbidden = asindices(canvas(-1, (cutoff, cutoff)))
    dmirrareaL = shift(asindices(canvas(-1, (h * 2 - 2 * cutoff, cutoff))), (cutoff, 0))
    dmirrareaT = shift(asindices(canvas(-1, (cutoff, 2 * w - 2 * cutoff))), (0, cutoff))
    inds1 = sfilter(asindices(gi), lambda ij: cutoff <= ij[0] < h and cutoff <= ij[1] < w and ij[0] >= ij[1])
    inds2 = dmirror(inds1)
    inds3 = shift(hmirror(inds1), (h - cutoff, 0))
    inds4 = shift(hmirror(inds2), (h - cutoff, 0))
    inds5 = shift(vmirror(inds1), (0, w - cutoff))
    inds6 = shift(vmirror(inds2), (0, w - cutoff))
    inds7 = shift(hmirror(vmirror(inds1)), (h - cutoff, w - cutoff))
    inds8 = shift(hmirror(vmirror(inds2)), (h - cutoff, w - cutoff))
    f1 = identity
    f2 = dmirror
    f3 = lambda x: hmirror(shift(x, invert((h - cutoff, 0))))
    f4 = lambda x: dmirror(hmirror(shift(x, invert((h - cutoff, 0)))))
    f5 = lambda x: vmirror(shift(x, invert((0, w - cutoff))))
    f6 = lambda x: dmirror(vmirror(shift(x, invert((0, w - cutoff)))))
    f7 = lambda x: vmirror(hmirror(shift(x, invert((h - cutoff, w - cutoff)))))
    f8 = lambda x: dmirror(vmirror(hmirror(shift(x, invert((h - cutoff, w - cutoff))))))
    indsarr = [inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8]
    farr = [f1, f2, f3, f4, f5, f6, f7, f8]
    ndist = unifint(rng, diff_lb, diff_ub, (1, int((2 * h * 2 * w) ** 0.5)))
    succ = 0
    tr = 0
    maxtr = 10 * ndist
    fullh, fullw = shape(gi)
    while succ < ndist and tr < maxtr:
        tr += 1
        oh = rng.randint(2, h // 2 + 1)
        ow = rng.randint(2, w // 2 + 1)
        loci = rng.randint(0, fullh - oh)
        locj = rng.randint(0, fullw - ow)
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        isleft = set()
        gi2 = fill(gi, patchcol, bd)
        if patchcol in palette(toobject(forbidden, gi2)):
            continue
        oo1 = toindices(sfilter(toobject(dmirrareaL, gi2), lambda cij: cij[0] != patchcol))
        oo2 = toindices(sfilter(toobject(dmirrareaT, gi2), lambda cij: cij[0] != patchcol))
        oo2 = frozenset({(ij[1], ij[0]) for ij in oo2})
        if oo1 | oo2 != dmirrareaL:
            continue
        for ii, ff in zip(indsarr, farr):
            oo = toobject(ii, gi2)
            rem = toindices(sfilter(oo, lambda cij: cij[0] != patchcol))
            if len(rem) > 0:
                isleft = isleft | ff(rem)
        if isleft != inds1:
            continue
        succ += 1
        gi = gi2
    return {"input": gi, "output": go}


def generate_234bbc79(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(rng, diff_lb, diff_ub, (5, 30))
        w = unifint(rng, diff_lb, diff_ub, (6, 20))
        bgc, dotc = rng.sample(cols, 2)
        remcols = difference(cols, (bgc, dotc))
        go = canvas(bgc, (h, 30))
        ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
        ccols = rng.sample(remcols, ncols)
        spi = rng.randint(0, h - 1)
        snek = [(spi, 0)]
        gi = fill(go, dotc, {(spi, 0)})
        while True:
            previ, prevj = snek[-1]
            if prevj == w - 1:
                if rng.choice((True, False, False)):
                    break
            options = []
            if previ < h - 1:
                if go[previ + 1][prevj] == bgc:
                    options.append((previ + 1, prevj))
            if previ > 0:
                if go[previ - 1][prevj] == bgc:
                    options.append((previ - 1, prevj))
            if prevj < w - 1:
                options.append((previ, prevj + 1))
            if len(options) == 0:
                break
            loc = rng.choice(options)
            snek.append(loc)
            go = fill(go, dotc, {loc})
        objs = []
        cobj = []
        for idx, cel in enumerate(snek):
            if len(cobj) > 2 and width(frozenset(cobj)) > 1 and snek[idx - 1] == add(cel, (0, -1)):
                objs.append(cobj)
                cobj = [cel]
            else:
                cobj.append(cel)
        objs[-1] += cobj
        nobjs = len(objs)
        if nobjs < 2:
            continue
        ntokeep = unifint(rng, diff_lb, diff_ub, (2, nobjs))
        ntorem = nobjs - ntokeep
        for k in range(ntorem):
            idx = rng.randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx + 1]] + objs[idx + 2 :]
        inobjs = []
        for idx, obj in enumerate(objs):
            col = rng.choice(ccols)
            go = fill(go, col, set(obj))
            centerpart = recolor(col, set(obj[1:-1]))
            leftpart = {(dotc if idx > 0 else col, obj[0])}
            rightpart = {(dotc if idx < len(objs) - 1 else col, obj[-1])}
            inobj = centerpart | leftpart | rightpart
            inobjs.append(inobj)
        spacings = [1 for idx in range(len(inobjs) - 1)]
        fullw = unifint(rng, diff_lb, diff_ub, (w, 30))
        for k in range(fullw - w - len(inobjs) - 1):
            idx = rng.randint(0, len(spacings) - 1)
            spacings[idx] += 1
        lspacings = [0] + spacings
        gi = canvas(bgc, (h, fullw))
        ofs = 0
        for i, (lsp, obj) in enumerate(zip(lspacings, inobjs)):
            obj = set(obj)
            if i == 0:
                ulc = ulcorner(obj)
            else:
                ulci = rng.randint(0, h - height(obj))
                ulcj = ofs + lsp
                ulc = (ulci, ulcj)
            ofs += width(obj) + lsp
            plcd = shift(normalize(obj), ulc)
            gi = paint(gi, plcd)
        break
    ins = size(merge(fgpartition(gi)))
    while True:
        go2 = dmirror(dmirror(go)[:-1])
        if size(sfilter(asobject(go2), lambda cij: cij[0] != bgc)) < ins:
            break
        else:
            go = go2
    return {"input": gi, "output": go}


def generate_cbded52d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    oh = unifint(rng, diff_lb, diff_ub, (1, 4))
    ow = unifint(rng, diff_lb, diff_ub, (1, 4))
    numh = unifint(rng, diff_lb, diff_ub, (3, 31 // (oh + 1)))
    numw = unifint(rng, diff_lb, diff_ub, (3, 31 // (ow + 1)))
    bgc, linc = rng.sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    ncols = unifint(rng, diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = rng.sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = canvas(linc, (fullh, fullw))
    sgi = asindices(canvas(bgc, (oh, ow)))
    for a in range(numh):
        for b in range(numw):
            gi = fill(gi, bgc, shift(sgi, (a * (oh + 1), b * (ow + 1))))
    go = tuple(e for e in gi)
    for col in ccols:
        inds = ofcolor(go, bgc)
        if len(inds) == 0:
            break
        loc = rng.choice(totuple(inds))
        narms = rng.randint(1, 4)
        armdirs = rng.sample(totuple(dneighbors((0, 0))), narms)
        succ = 0
        for armdir in armdirs:
            x, y = armdir
            arm = []
            for k in range(1, max(numh, numw)):
                nextloc = add(loc, (k * x * (oh + 1), k * y * (ow + 1)))
                if nextloc not in inds:
                    break
                arm.append(nextloc)
            if len(arm) < 2:
                continue
            aidx = unifint(rng, diff_lb, diff_ub, (1, len(arm) - 1))
            endp = arm[aidx]
            gi = fill(gi, col, {endp})
            go = fill(go, col, set(arm[: aidx + 1]))
            succ += 1
        if succ > 0:
            gi = fill(gi, col, {loc})
            go = fill(go, col, {loc})
    return {"input": gi, "output": go}


def generate_06df4c85(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    oh = unifint(rng, diff_lb, diff_ub, (1, 4))
    ow = unifint(rng, diff_lb, diff_ub, (1, 4))
    numh = unifint(rng, diff_lb, diff_ub, (3, 31 // (oh + 1)))
    numw = unifint(rng, diff_lb, diff_ub, (3, 31 // (ow + 1)))
    bgc, linc = rng.sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    ncols = unifint(rng, diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = rng.sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = canvas(linc, (fullh, fullw))
    sgi = asindices(canvas(bgc, (oh, ow)))
    for a in range(numh):
        for b in range(numw):
            gi = fill(gi, bgc, shift(sgi, (a * (oh + 1), b * (ow + 1))))
    go = tuple(e for e in gi)
    sinds = asindices(canvas(-1, (oh, ow)))
    for col in ccols:
        inds = occurrences(go, recolor(bgc, sinds))
        if len(inds) == 0:
            break
        loc = rng.choice(totuple(inds))
        narms = rng.randint(1, 4)
        armdirs = rng.sample(totuple(dneighbors((0, 0))), narms)
        succ = 0
        for armdir in armdirs:
            x, y = armdir
            arm = []
            for k in range(1, max(numh, numw)):
                nextloc = add(loc, (k * x * (oh + 1), k * y * (ow + 1)))
                if nextloc not in inds:
                    break
                arm.append(nextloc)
            if len(arm) < 2:
                continue
            aidx = unifint(rng, diff_lb, diff_ub, (1, len(arm) - 1))
            endp = arm[aidx]
            gi = fill(gi, col, shift(sinds, endp))
            go = fill(go, col, mapply(lbind(shift, sinds), set(arm[: aidx + 1])))
            succ += 1
        gi = fill(gi, col, shift(sinds, loc))
        go = fill(go, col, shift(sinds, loc))
    return {"input": gi, "output": go}


def generate_90f3ed37(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    while True:
        h = unifint(rng, diff_lb, diff_ub, (8, 30))
        w = unifint(rng, diff_lb, diff_ub, (8, 30))
        pathh = unifint(rng, diff_lb, diff_ub, (1, max(1, h // 4)))
        pathh = unifint(rng, diff_lb, diff_ub, (pathh, max(1, h // 4)))
        Lpatper = unifint(rng, diff_lb, diff_ub, (1, w // 7))
        Rpatper = unifint(rng, diff_lb, diff_ub, (1, w // 7))
        hh = rng.randint(1, pathh)
        Linds = asindices(canvas(-1, (hh, Lpatper)))
        Rinds = asindices(canvas(-1, (hh, Rpatper)))
        lpatsd = unifint(rng, diff_lb, diff_ub, (0, (hh * Lpatper) // 2))
        rpatsd = unifint(rng, diff_lb, diff_ub, (0, (hh * Rpatper) // 2))
        lpats = rng.choice((lpatsd, hh * Lpatper - lpatsd))
        rpats = rng.choice((rpatsd, hh * Rpatper - rpatsd))
        lpats = min(max(Lpatper, lpats), hh * Lpatper)
        rpats = min(max(Rpatper, rpats), hh * Rpatper)
        lpat = set(rng.sample(totuple(Linds), lpats))
        rpat = set(rng.sample(totuple(Rinds), rpats))
        midpatw = rng.randint(0, w - 2 * Lpatper - 2 * Rpatper)
        if midpatw == 0 or Lpatper == hh == 1:
            midpat = set()
            midpatw = 0
        else:
            midpat = set(
                rng.sample(totuple(asindices(canvas(-1, (hh, midpatw)))), rng.randint(midpatw, (hh * midpatw)))
            )
        if shift(midpat, (0, 2 * Lpatper - midpatw)).issubset(lpat):
            midpat = set()
            midpatw = 0
        loci = rng.randint(0, h - pathh)
        lplac = shift(lpat, (loci, 0)) | shift(lpat, (loci, Lpatper))
        mplac = shift(midpat, (loci, 2 * Lpatper))
        rplac = shift(rpat, (loci, 2 * Lpatper + midpatw)) | shift(rpat, (loci, 2 * Lpatper + midpatw + Rpatper))
        sp = 2 * Lpatper + midpatw + Rpatper
        for k in range(w // Lpatper + 1):
            lplac |= shift(lpat, (loci, -k * Lpatper))
        for k in range(w // Rpatper + 1):
            rplac |= shift(rpat, (loci, sp + k * Rpatper))
        pat = lplac | mplac | rplac
        patn = shift(pat, (-loci, 0))
        bgc, fgc = rng.sample(cols, 2)
        gi = canvas(bgc, (h, w))
        gi = fill(gi, fgc, pat)
        options = interval(0, h - pathh + 1, 1)
        options = difference(options, interval(loci - pathh - 1, loci + 2 * pathh, 1))
        nplacements = unifint(rng, diff_lb, diff_ub, (1, max(1, len(options) // pathh)))
        go = tuple(e for e in gi)
        for k in range(nplacements):
            if len(options) == 0:
                break
            locii = rng.choice(options)
            options = difference(options, interval(locii - pathh - 1, locii + 2 * pathh, 1))
            hoffs = rng.randint(0, max(Rpatper, w - sp - 2))
            cutoffopts = interval(2 * Lpatper + midpatw, 2 * Lpatper + midpatw + hoffs + 1, 1)
            cutoffopts = cutoffopts[::-1]
            idx = unifint(rng, diff_lb, diff_ub, (0, len(cutoffopts) - 1))
            cutoff = cutoffopts[idx]
            patnc = sfilter(patn, lambda ij: ij[1] <= cutoff)
            go = fill(go, 1, shift(patn, (locii, hoffs)))
            gi = fill(gi, fgc, shift(patnc, (locii, hoffs)))
            go = fill(go, fgc, shift(patnc, (locii, hoffs)))
        if 1 in palette(go):
            break
    return {"input": gi, "output": go}


def generate_36d67576(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(rng, diff_lb, diff_ub, (10, 30))
        w = unifint(rng, diff_lb, diff_ub, (10, 30))
        bgc, mainc, markerc = rng.sample(cols, 3)
        remcols = difference(cols, (bgc, mainc, markerc))
        ncols = unifint(rng, diff_lb, diff_ub, (1, len(remcols)))
        ccols = rng.sample(remcols, ncols)
        gi = canvas(bgc, (h, w))
        oh = unifint(rng, diff_lb, diff_ub, (2, 5))
        ow = unifint(rng, diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if rng.choice((True, False)):
            oh, ow = ow, oh
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(rng, diff_lb, diff_ub, (4, len(bounds)))
        obj = {rng.choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(rng.choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        ntocompc = unifint(rng, diff_lb, diff_ub, (1, ncells - 3))
        markercell = rng.choice(totuple(obj))
        remobj = remove(markercell, obj)
        markercellobj = {(markerc, markercell)}
        tocompc = set(rng.sample(totuple(remobj), ntocompc))
        mainpart = (obj - {markercell}) - tocompc
        mainpartobj = recolor(mainc, mainpart)
        tocompcobj = {(rng.choice(remcols), ij) for ij in tocompc}
        obj = tocompcobj | mainpartobj | markercellobj
        smobj = mainpartobj | markercellobj
        smobjn = normalize(smobj)
        isfakesymm = False
        for symmf in [dmirror, cmirror, hmirror, vmirror]:
            if symmf(smobjn) == smobjn and symmf(obj) != obj:
                isfakesymm = True
                break
        if isfakesymm:
            continue
        loci = rng.randint(0, h - oh)
        locj = rng.randint(0, w - ow)
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (asindices(gi) - plcdi) - mapply(neighbors, plcdi)
        noccs = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ = 0
        tr = 0
        maxtr = noccs * 5
        go = tuple(e for e in gi)
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = rng.choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf2 = rng.choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf = compose(mf1, mf2)
            outobj = normalize(mf(obj))
            inobj = sfilter(outobj, lambda cij: cij[0] in [mainc, markerc])
            oh, ow = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = rng.choice(totuple(cands))
            outobjp = shift(outobj, loc)
            inobjp = shift(inobj, loc)
            outobjpi = toindices(outobjp)
            if outobjpi.issubset(inds):
                succ += 1
                inds = (inds - outobjpi) - mapply(neighbors, outobjpi)
                gi = paint(gi, inobjp)
                go = paint(go, outobjp)
        break
    return {"input": gi, "output": go}


def generate_4522001f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 10))
    w = unifint(rng, diff_lb, diff_ub, (3, 10))
    bgc, sqc, dotc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (3 * h, 3 * w))
    sqi = {(dotc, (1, 1))} | recolor(sqc, {(0, 0), (0, 1), (1, 0)})
    sqo = backdrop(frozenset({(0, 0), (3, 3)}))
    sqo |= shift(sqo, (4, 4))
    loci = rng.randint(0, min(h - 2, 3 * h - 8))
    locj = rng.randint(0, min(w - 2, 3 * w - 8))
    loc = (loci, locj)
    plcdi = shift(sqi, loc)
    plcdo = shift(sqo, loc)
    gi = paint(gi, plcdi)
    go = fill(go, sqc, plcdo)
    noccs = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 9))
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    iinds = ofcolor(gi, bgc) - mapply(dneighbors, toindices(plcdi))
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = sfilter(iinds, lambda ij: ij[0] <= h - 2 and ij[1] <= w - 2)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        plcdi = shift(sqi, loc)
        plcdo = shift(sqo, loc)
        plcdii = toindices(plcdi)
        if plcdii.issubset(iinds):
            succ += 1
            iinds = (iinds - plcdii) - mapply(dneighbors, plcdii)
            gi = paint(gi, plcdi)
            go = fill(go, sqc, plcdo)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_72322fa7(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, 4))
    ccols = rng.sample(remcols, 2 * nobjs)
    cpairs = list(zip(ccols[:nobjs], ccols[nobjs:]))
    objs = []
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    for ca, cb in cpairs:
        oh = unifint(rng, diff_lb, diff_ub, (1, 4))
        ow = unifint(rng, diff_lb, diff_ub, (2 if oh == 1 else 1, 4))
        if rng.choice((True, False)):
            oh, ow = ow, oh
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {rng.choice(totuple(bounds))}
        ncells = rng.randint(2, oh * ow)
        for k in range(ncells - 1):
            obj.add(rng.choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        objn = normalize(obj)
        objt = totuple(objn)
        apart = rng.sample(objt, rng.randint(1, len(objt) - 1))
        bpart = difference(objt, apart)
        obj = recolor(ca, set(apart)) | recolor(cb, set(bpart))
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: shift(objn, ij).issubset(inds))
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (inds - plcdi) - mapply(neighbors, plcdi)
        objs.append(obj)
    avgs = sum([len(o) for o in objs]) / len(objs)
    ub = max(1, (h * w) // (avgs * 2))
    noccs = unifint(rng, diff_lb, diff_ub, (1, ub))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    go = tuple(e for e in gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        obj = rng.choice(objs)
        ca, cb = list(palette(obj))
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
            go = paint(go, plcd)
            col = rng.choice((ca, cb))
            gi = paint(gi, sfilter(plcd, lambda cij: cij[0] == col))
    return {"input": gi, "output": go}


def generate_4290ef0e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        d = unifint(rng, diff_lb, diff_ub, (2, 7))
        h, w = d, d
        fullh = unifint(rng, diff_lb, diff_ub, (4 * d, 30))
        fullw = unifint(rng, diff_lb, diff_ub, (4 * d, 30))
        bgc = rng.choice(cols)
        remcols = remove(bgc, cols)
        ccols = rng.sample(remcols, d)
        quad = canvas(bgc, (d + 1, d + 1))
        for idx, c in enumerate(ccols):
            linlen = rng.randint(2, w - idx + 1)
            quad = fill(quad, c, (connect((idx, idx), (idx + linlen - 1, idx))))
            quad = fill(quad, c, (connect((idx, idx), (idx, idx + linlen - 1))))
        go = canvas(bgc, (d + 1, 2 * d + 1))
        qobj1 = asobject(quad)
        qobj2 = shift(asobject(vmirror(quad)), (0, d))
        go = paint(go, qobj1)
        go = paint(go, qobj2)
        go = vconcat(go, hmirror(go)[1:])
        if rng.choice((True, False)):
            go = fill(go, rng.choice(difference(remcols, ccols)), {center(asindices(go))})
        objs = partition(go)
        objs = sfilter(objs, lambda o: color(o) != bgc)
        gi = canvas(bgc, (fullh, fullw))
        objs = order(objs, width)
        fullinds = asindices(gi)
        inds = asindices(gi)
        fullsuc = True
        for obj in objs:
            objn = normalize(obj)
            obji = toindices(objn)
            d = width(obj)
            dh = max(0, d // 2 - 1)
            cands = sfilter(fullinds, lambda ij: ij[0] <= fullh - d and ij[1] <= fullw - d)
            cands = (
                cands | shift(cands, (-dh, 0)) | shift(cands, (0, -dh)) | shift(cands, (dh, 0)) | shift(cands, (0, dh))
            )
            maxtr = 10
            tr = 0
            succ = False
            if len(cands) == 0:
                break
            while tr < maxtr and not succ:
                tr += 1
                loc = rng.choice(totuple(cands))
                if (shift(obji, loc) & fullinds).issubset(inds):
                    succ = True
                    break
            if not succ:
                fullsuc = False
                break
            gi = paint(gi, shift(objn, loc))
            inds = inds - shift(obji, loc)
        if not fullsuc:
            continue
        break
    return {"input": gi, "output": go}


def generate_6a1e5592(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (9, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    barh = rng.randint(3, h // 3)
    maxobjh = h - barh - 1
    nobjs = unifint(rng, diff_lb, diff_ub, (1, w // 3))
    barc, bgc, objc = rng.sample(cols, 3)
    c1 = canvas(barc, (barh, w))
    c2 = canvas(bgc, (h - barh, w))
    gi = vconcat(c1, c2)
    go = tuple(e for e in gi)
    tr = 0
    succ = 0
    maxtr = 10 * nobjs
    placopts = interval(1, w - 1, 1)
    iinds = ofcolor(gi, bgc)
    oinds = asindices(go)
    barinds = ofcolor(gi, barc)
    forbmarkers = set()
    while tr < maxtr and succ < nobjs:
        tr += 1
        oh = rng.randint(1, maxobjh)
        ow = rng.randint(1, min(4, w // 2))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = rng.randint(1, oh * ow)
        sp = rng.choice(totuple(connect((0, 0), (0, ow - 1))))
        obj = {sp}
        for k in range(ncells - 1):
            obj.add(rng.choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        markerh = rng.randint(1, min(oh, barh - 1))
        markpart = sfilter(obj, lambda ij: ij[0] < markerh)
        markpartn = normalize(markpart)
        isinvalid = False
        for k in range(1, markerh + 1):
            if normalize(sfilter(markpartn, lambda ij: ij[0] < k)) in forbmarkers:
                isinvalid = True
        if isinvalid:
            continue
        for k in range(1, markerh + 1):
            forbmarkers.add(normalize(sfilter(markpartn, lambda ij: ij[0] < k)))
        placoptcands = sfilter(placopts, lambda jj: set(interval(jj, jj + ow + 1, 1)).issubset(set(placopts)))
        if len(placoptcands) == 0:
            continue
        jloc = rng.choice(placoptcands)
        iloc = barh - markerh
        oplcd = shift(obj, (iloc, jloc))
        if oplcd.issubset(oinds):
            icands = sfilter(iinds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(icands) == 0:
                continue
            loc = rng.choice(totuple(icands))
            iplcd = shift(obj, loc)
            if iplcd.issubset(iinds):
                succ += 1
                iinds = (iinds - iplcd) - mapply(neighbors, iplcd)
                oinds = oinds - oplcd
                gi = fill(gi, objc, iplcd)
                gi = fill(gi, bgc, oplcd & barinds)
                go = fill(go, 1, oplcd)
                jm = apply(last, ofcolor(go, 1))
                placopts = sorted(difference(placopts, jm | apply(decrement, jm) | apply(increment, jm)))
        if len(placopts) == 0:
            break
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_e73095fd(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    while True:
        h = unifint(rng, diff_lb, diff_ub, (10, 32))
        w = unifint(rng, diff_lb, diff_ub, (10, 32))
        bgc, fgc = rng.sample(cols, 2)
        gi = canvas(bgc, (h, w))
        nsplits = unifint(rng, diff_lb, diff_ub, (2, min(h, w) // 3))
        for k in range(nsplits):
            objs = objects(gi, T, F, F)
            objs = colorfilter(objs, bgc)
            objs = apply(toindices, objs)
            hobjs = sfilter(objs, lambda o: height(o) > 6)
            wobjs = sfilter(objs, lambda o: width(o) > 6)
            if len(hobjs) == 0 and len(wobjs) == 0:
                break
            cgroups = [(g, ax) for g, ax in zip([hobjs, wobjs], [0, 1]) if len(g) > 0]
            g, ax = rng.choice(cgroups)
            obj = rng.choice(totuple(g))
            ulci, ulcj = ulcorner(obj)
            oh, ow = shape(obj)
            if ax == 0:
                iloc = rng.randint(ulci + 3, ulci + oh - 3)
                bar = sfilter(obj, lambda ij: ij[0] == iloc)
            else:
                jloc = rng.randint(ulcj + 3, ulcj + ow - 3)
                bar = sfilter(obj, lambda ij: ij[1] == jloc)
            gi = fill(gi, fgc, bar)
        copts = sfilter(
            ofcolor(gi, fgc), lambda ij: len(sfilter(toobject(dneighbors(ij), gi), lambda cij: cij[0] == fgc)) > 2
        )
        copts = sfilter(
            copts, lambda ij: len(sfilter(toobject(outbox(outbox({ij})), gi), lambda cij: cij[0] == fgc)) in {3, 4}
        )
        if len(copts) == 0:
            continue
        noccs = unifint(rng, diff_lb, diff_ub, (1, len(copts)))
        noccs = unifint(rng, diff_lb, diff_ub, (noccs, len(copts)))
        occs = rng.sample(totuple(copts), noccs)
        go = tuple(e for e in gi)
        forb = set()
        for occ in occs:
            ulci, ulcj = decrement(occ)
            lrci, lrcj = increment(occ)
            if len(sfilter(toobject(box({(ulci, ulcj), (lrci, lrcj)}), gi), lambda cij: cij[0] == fgc)) in {3, 4}:
                boptions = []
                for ulcioffs in [-2, -1, 0]:
                    for ulcjoffs in [-2, -1, 0]:
                        for lrcioffs in [0, 1, 2]:
                            for lrcjoffs in [0, 1, 2]:
                                bx = box({(ulci + ulcioffs, ulcj + ulcjoffs), (lrci + lrcioffs, lrcj + lrcjoffs)})
                                bxobj = toobject(bx, gi)
                                if len(sfilter(toobject(bxobj, gi), lambda cij: cij[0] == fgc)) in {3, 4} and len(
                                    sfilter(toobject(outbox(bxobj), gi), lambda cij: cij[0] == fgc)
                                ) in {3, 4}:
                                    boptions.append(bx)
                boptions = sfilter(boptions, lambda bx: len(backdrop(bx) & forb) == 0)
                if len(boptions) > 0:
                    bx = rng.choice(boptions)
                    bd = backdrop(bx)
                    gi = fill(gi, bgc, bd)
                    gi = fill(gi, fgc, bx)
                    go = fill(go, 4, bd)
                    go = fill(go, fgc, bx)
                    forb |= bd
        gi = trim(gi)
        go = trim(go)
        if 4 in palette(go):
            break
    return {"input": gi, "output": go}
