import random

from ..dsl import *
from ..utils import *


def generate_8403a5d5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    loccinv = unifint(rng, diff_lb, diff_ub, (1, w - 1))
    locc = w - loccinv
    bgc, fgc = rng.sample(cols, 2)
    c = canvas(bgc, (h, w))
    idx = (h - 1, locc)
    gi = fill(c, fgc, {idx})
    go = canvas(bgc, (h, w))
    for j in range(locc, w, 2):
        go = fill(go, fgc, connect((0, j), (h - 1, j)))
    for j in range(locc + 1, w, 4):
        go = fill(go, 5, {(0, j)})
    for j in range(locc + 3, w, 4):
        go = fill(go, 5, {(h - 1, j)})
    return {"input": gi, "output": go}


def generate_941d9a10(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    opts = interval(2, (h - 1) // 2 + 1, 2)
    nhidx = unifint(rng, diff_lb, diff_ub, (0, len(opts) - 1))
    nh = opts[nhidx]
    opts = interval(2, (w - 1) // 2 + 1, 2)
    nwidx = unifint(rng, diff_lb, diff_ub, (0, len(opts) - 1))
    nw = opts[nwidx]
    bgc, fgc = rng.sample(cols, 2)
    hgrid = canvas(bgc, (2 * nh + 1, w))
    for j in range(1, h, 2):
        hgrid = fill(hgrid, fgc, connect((j, 0), (j, w)))
    for k in range(h - (2 * nh + 1)):
        loc = rng.randint(0, height(hgrid) - 1)
        hgrid = hgrid[:loc] + canvas(bgc, (1, w)) + hgrid[loc:]
    wgrid = canvas(bgc, (2 * nw + 1, h))
    for j in range(1, w, 2):
        wgrid = fill(wgrid, fgc, connect((j, 0), (j, h)))
    for k in range(w - (2 * nw + 1)):
        loc = rng.randint(0, height(wgrid) - 1)
        wgrid = wgrid[:loc] + canvas(bgc, (1, h)) + wgrid[loc:]
    wgrid = dmirror(wgrid)
    gi = canvas(bgc, (h, w))
    fronts = ofcolor(hgrid, fgc) | ofcolor(wgrid, fgc)
    gi = fill(gi, fgc, fronts)
    objs = objects(gi, T, T, F)
    objs = colorfilter(objs, bgc)
    blue = argmin(objs, lambda o: leftmost(o) + uppermost(o))
    green = argmax(objs, lambda o: leftmost(o) + uppermost(o))
    f1 = lambda o: len(sfilter(objs, lambda o2: leftmost(o2) < leftmost(o))) == len(
        sfilter(objs, lambda o2: leftmost(o2) > leftmost(o))
    )
    f2 = lambda o: len(sfilter(objs, lambda o2: uppermost(o2) < uppermost(o))) == len(
        sfilter(objs, lambda o2: uppermost(o2) > uppermost(o))
    )
    red = extract(objs, lambda o: f1(o) and f2(o))
    go = fill(gi, 1, blue)
    go = fill(go, 3, green)
    go = fill(go, 2, red)
    return {"input": gi, "output": go}


def generate_b0c4d837(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    oh = unifint(rng, diff_lb, diff_ub, (3, h - 1))
    ow = unifint(rng, diff_lb, diff_ub, (3, w - 1))
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    bgc, boxc, fillc = rng.sample(cols, 3)
    subg = canvas(boxc, (oh, ow))
    subg2 = canvas(fillc, (oh - 1, ow - 2))
    ntofill = unifint(rng, diff_lb, diff_ub, (1, min(9, oh - 2)))
    for j in range(ntofill):
        subg2 = fill(subg2, bgc, connect((j, 0), (j, ow - 2)))
    subg = paint(subg, shift(asobject(subg2), (0, 1)))
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(asobject(subg), (loci, locj)))
    go = repeat(fillc, ntofill) + repeat(bgc, 9 - ntofill)
    go = (go[:3], go[3:6][::-1], go[6:])
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    return {"input": gi, "output": go}


def generate_0a938d79(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 29))
    w = unifint(rng, diff_lb, diff_ub, (h + 1, 30))
    bgc, cola, colb = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    locja = unifint(rng, diff_lb, diff_ub, (3, w - 2))
    locjb = unifint(rng, diff_lb, diff_ub, (1, locja - 2))
    locia = rng.choice((0, h - 1))
    locib = rng.choice((0, h - 1))
    gi = fill(gi, cola, {(locia, locja)})
    gi = fill(gi, colb, {(locib, locjb)})
    ofs = -2 * (locja - locjb)
    for aa in range(locja, -1, ofs):
        go = fill(go, cola, connect((0, aa), (h - 1, aa)))
    for bb in range(locjb, -1, ofs):
        go = fill(go, colb, connect((0, bb), (h - 1, bb)))
    rotf = rng.choice((rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_b7249182(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    ih = unifint(rng, diff_lb, diff_ub, (3, (h - 1) // 2))
    bgc, ca, cb = rng.sample(cols, 3)
    subg = canvas(bgc, (ih, 5))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    subg = fill(subg, ca, connect((0, 2), (ih - 2, 2)))
    subg = fill(subg, ca, connect((ih - 2, 0), (ih - 2, 4)))
    subg = fill(subg, ca, {(ih - 1, 0)})
    subga = fill(subg, ca, {(ih - 1, 4)})
    subgb = replace(subga, ca, cb)
    subg = vconcat(subga, hmirror(subgb))
    loci = rng.randint(0, h - 2 * ih)
    locj = rng.randint(0, w - 5)
    obj = asobject(subg)
    obj = shift(obj, (loci, locj))
    gi = fill(gi, ca, {(loci, locj + 2)})
    gi = fill(gi, cb, {(loci + 2 * ih - 1, locj + 2)})
    go = paint(go, obj)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_7b6016b9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, remove(2, interval(0, 10, 1)))
    while True:
        h = unifint(rng, diff_lb, diff_ub, (5, 30))
        w = unifint(rng, diff_lb, diff_ub, (5, 30))
        bgc, fgc = rng.sample(cols, 2)
        numl = unifint(rng, diff_lb, diff_ub, (4, min(h, w)))
        gi = canvas(bgc, (h, w))
        jint = interval(0, w, 1)
        iint = interval(0, h, 1)
        iopts = interval(1, h - 1, 1)
        jopts = interval(1, w - 1, 1)
        numlh = rng.randint(numl // 3, numl // 3 * 2)
        numlw = numl - numlh
        for k in range(numlh):
            if len(iopts) == 0:
                continue
            loci = rng.choice(iopts)
            iopts = remove(loci, iopts)
            iopts = remove(loci + 1, iopts)
            iopts = remove(loci - 1, iopts)
            a, b = rng.sample(jint, 2)
            a = rng.randint(0, a)
            b = rng.randint(b, w - 1)
            gi = fill(gi, fgc, connect((loci, a), (loci, b)))
        for k in range(numlw):
            if len(jopts) == 0:
                continue
            locj = rng.choice(jopts)
            jopts = remove(locj, jopts)
            jopts = remove(locj + 1, jopts)
            jopts = remove(locj - 1, jopts)
            a, b = rng.sample(iint, 2)
            a = rng.randint(0, a)
            b = rng.randint(b, h - 1)
            gi = fill(gi, fgc, connect((a, locj), (b, locj)))
        objs = objects(gi, T, F, F)
        bgobjs = colorfilter(objs, bgc)
        tofill = toindices(mfilter(bgobjs, compose(flip, rbind(bordering, gi))))
        if len(tofill) > 0:
            break
    tofix = mapply(neighbors, tofill) - tofill
    gi = fill(gi, fgc, tofix)
    go = fill(gi, 2, tofill)
    go = replace(go, bgc, 3)
    return {"input": gi, "output": go}


def generate_72ca375d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 25))
    srcobjh = unifint(rng, diff_lb, diff_ub, (2, 8))
    srcobjwh = unifint(rng, diff_lb, diff_ub, (1, 4))
    bnds = asindices(canvas(-1, (srcobjh, srcobjwh)))
    spi = rng.randint(0, srcobjh - 1)
    sp = (spi, srcobjwh - 1)
    srcobj = {sp}
    bnds = remove(sp, bnds)
    ncellsd = unifint(rng, diff_lb, diff_ub, (0, (srcobjh * srcobjwh) // 2))
    ncells1 = rng.choice((ncellsd, srcobjh * srcobjwh - ncellsd))
    ncells2 = unifint(rng, diff_lb, diff_ub, (1, srcobjh * srcobjwh))
    ncells = (ncells1 + ncells2) // 2
    ncells = min(max(1, ncells), srcobjh * srcobjwh, (h * w) // 2 - 1)
    for k in range(ncells - 1):
        srcobj.add(rng.choice(totuple((bnds - srcobj) & mapply(neighbors, srcobj))))
    srcobj = normalize(srcobj)
    srcobj = srcobj | shift(vmirror(srcobj), (0, width(srcobj)))
    srcobjh, srcobjw = shape(srcobj)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    trgc = rng.choice(remcols)
    go = canvas(bgc, (srcobjh, srcobjw))
    go = fill(go, trgc, srcobj)
    loci = rng.randint(0, h - srcobjh)
    locj = rng.randint(0, w - srcobjw)
    locc = (loci, locj)
    gi = canvas(bgc, (h, w))
    shftd = shift(srcobj, locc)
    gi = fill(gi, trgc, shftd)
    indss = asindices(gi)
    indss = (indss - shftd) - mapply(neighbors, shftd)
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    remcands = asindices(canvas(-1, (8, 8))) - srcobj
    while succ < nobjs and tr <= maxtrials:
        if len(indss) == 0:
            break
        while True:
            newobj = {e for e in srcobj}
            numperti = unifint(rng, diff_lb, diff_ub, (1, 63))
            numpert = 64 - numperti
            for np in range(numpert):
                isadd = rng.choice((True, False))
                if isadd and len(newobj) < 64:
                    cndds = totuple((remcands - newobj) & mapply(neighbors, newobj))
                    if len(cndds) == 0:
                        break
                    newobj.add(rng.choice(cndds))
                if not isadd and len(newobj) > 2:
                    newobj = remove(rng.choice(totuple(newobj)), newobj)
            newobj = normalize(newobj)
            a, b = shape(newobj)
            cc = canvas(-1, (a + 2, b + 2))
            cc2 = compress(fill(cc, -2, shift(newobj, (1, 1))))
            newobj = toindices(argmax(colorfilter(objects(cc2, T, T, F), -2), size))
            if newobj != vmirror(newobj):
                break
        col = rng.choice(remcols)
        loccands = sfilter(indss, lambda ij: shift(newobj, ij).issubset(indss))
        if len(loccands) == 0:
            tr += 1
            continue
        locc = rng.choice(totuple(loccands))
        newobj = shift(newobj, locc)
        gi = fill(gi, col, newobj)
        succ += 1
        indss = (indss - newobj) - mapply(neighbors, newobj)
    return {"input": gi, "output": go}


def generate_673ef223(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    barh = unifint(rng, diff_lb, diff_ub, (2, (h - 1) // 2))
    ncells = unifint(rng, diff_lb, diff_ub, (1, barh))
    bgc, barc, dotc = rng.sample(cols, 3)
    sg = canvas(bgc, (barh, w))
    topsgi = fill(sg, barc, connect((0, 0), (barh - 1, 0)))
    botsgi = vmirror(topsgi)
    topsgo = tuple(e for e in topsgi)
    botsgo = tuple(e for e in botsgi)
    iloccands = interval(0, barh, 1)
    ilocs = rng.sample(iloccands, ncells)
    for k in ilocs:
        jloc = rng.randint(2, w - 2)
        topsgi = fill(topsgi, dotc, {(k, jloc)})
        topsgo = fill(topsgo, 4, {(k, jloc)})
        topsgo = fill(topsgo, dotc, connect((k, 1), (k, jloc - 1)))
        botsgo = fill(botsgo, dotc, connect((k, 0), (k, w - 2)))
    outpi = (topsgi, botsgi)
    outpo = (topsgo, botsgo)
    rr = canvas(bgc, (1, w))
    while len(merge(outpi)) < h:
        idx = rng.randint(0, len(outpi) - 1)
        outpi = outpi[:idx] + (rr,) + outpi[idx:]
        outpo = outpo[:idx] + (rr,) + outpo[idx:]
    gi = merge(outpi)
    go = merge(outpo)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_868de0fa(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 7))
    h = unifint(rng, diff_lb, diff_ub, (9, 30))
    w = unifint(rng, diff_lb, diff_ub, (9, 30))
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
        if len(indss) == 0:
            break
        oh = rng.randint(3, 8)
        ow = oh
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
            if oh % 2 == 1:
                go = fill(go, 7, bd)
            else:
                go = fill(go, 2, bd)
            go = fill(go, col, box(bd))
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {"input": gi, "output": go}


def generate_40853293(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    nlines = unifint(rng, diff_lb, diff_ub, (2, min(8, (h * w) // 2)))
    nhorilines = rng.randint(1, nlines - 1)
    nvertilines = nlines - nhorilines
    ilocs = interval(0, h, 1)
    ilocs = rng.sample(ilocs, min(nhorilines, len(ilocs)))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for ii in ilocs:
        llen = unifint(rng, diff_lb, diff_ub, (2, w - 1))
        js = rng.randint(0, w - llen)
        je = js + llen - 1
        a = (ii, js)
        b = (ii, je)
        hln = connect(a, b)
        col = rng.choice(remcols)
        remcols = remove(col, remcols)
        gi = fill(gi, col, {a, b})
        go = fill(go, col, hln)
    jlocs = interval(0, w, 1)
    gim = dmirror(gi)
    jlocs = sfilter(jlocs, lambda j: sum(1 for e in gim[j] if e == bgc) > 1)
    nvertilines = min(nvertilines, len(jlocs))
    jlocs = rng.sample(jlocs, nvertilines)
    for jj in jlocs:
        jcands = [idx for idx, e in enumerate(gim[jj]) if e == bgc]
        kk = len(jcands)
        locopts = interval(0, kk, 1)
        llen = unifint(rng, diff_lb, diff_ub, (2, kk))
        sp = rng.randint(0, kk - llen)
        ep = sp + llen - 1
        sp = jcands[sp]
        ep = jcands[ep]
        a = (sp, jj)
        b = (ep, jj)
        vln = connect(a, b)
        col = rng.choice(remcols)
        remcols = remove(col, remcols)
        gi = fill(gi, col, {a, b})
        go = fill(go, col, vln)
    return {"input": gi, "output": go}


def generate_6e19193c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    dirs = (((0, 0), (-1, -1)), ((0, 1), (-1, 1)), ((1, 0), (1, -1)), ((1, 1), (1, 1)))
    base = ((0, 0), (1, 0), (0, 1), (1, 1))
    candsi = [set(base) - {dr[0]} for dr in dirs]
    candso = [(set(base) | shoot(dr[0], dr[1])) - {dr[0]} for dr in dirs]
    cands = list(zip(candsi, candso))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 8))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    fullinds = asindices(gi)
    inds = asindices(canvas(-1, (h, w)))
    kk, tr = 0, 0
    maxtrials = num * 4
    while kk < num and tr < maxtrials:
        if len(inds) == 0:
            break
        loc = rng.choice(totuple(inds))
        obji, objo = rng.choice(cands)
        obji = shift(obji, loc)
        objo = shift(objo, loc)
        objo = objo & fullinds
        if objo.issubset(inds) and obji.issubset(objo):
            col = rng.choice(remcols)
            gi = fill(gi, col, obji)
            go = fill(go, col, objo)
            inds = (inds - objo) - mapply(dneighbors, obji)
            kk += 1
        tr += 1
    return {"input": gi, "output": go}


def generate_8731374e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    inh = rng.randint(5, h - 2)
    inw = rng.randint(5, w - 2)
    bgc, fgc = rng.sample(cols, 2)
    num = unifint(rng, diff_lb, diff_ub, (1, min(inh, inw)))
    mat = canvas(bgc, (inh - 2, inw - 2))
    tol = lambda g: list(list(e) for e in g)
    tot = lambda g: tuple(tuple(e) for e in g)
    mat = fill(mat, fgc, connect((0, 0), (num - 1, num - 1)))
    mat = tol(mat)
    rng.shuffle(mat)
    mat = tol(dmirror(tot(mat)))
    rng.shuffle(mat)
    mat = dmirror(tot(mat))
    sgi = paint(canvas(bgc, (inh, inw)), shift(asobject(mat), (1, 1)))
    inds = ofcolor(sgi, fgc)
    lins = mapply(fork(combine, vfrontier, hfrontier), inds)
    go = fill(sgi, fgc, lins)
    numci = unifint(rng, diff_lb, diff_ub, (3, 10))
    numc = 13 - numci
    ccols = rng.sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = asindices(c)
    obj = {(rng.choice(ccols), ij) for ij in inds}
    gi = paint(c, obj)
    loci = rng.randint(1, h - inh - 1)
    locj = rng.randint(1, w - inw - 1)
    loc = (loci, locj)
    plcd = shift(asobject(sgi), loc)
    gi = paint(gi, plcd)
    a, b = ulcorner(plcd)
    c, d = lrcorner(plcd)
    p1 = rng.choice(totuple(connect((a - 1, b), (a - 1, d))))
    p2 = rng.choice(totuple(connect((a, b - 1), (c, b - 1))))
    p3 = rng.choice(totuple(connect((c + 1, b), (c + 1, d))))
    p4 = rng.choice(totuple(connect((a, d + 1), (c, d + 1))))
    remcols = remove(bgc, ccols)
    fixobj = {
        (rng.choice(remcols), p1),
        (rng.choice(remcols), p2),
        (rng.choice(remcols), p3),
        (rng.choice(remcols), p4),
    }
    gi = paint(gi, fixobj)
    return {"input": gi, "output": go}


def generate_cce03e0d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 8))
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    nred = unifint(rng, diff_lb, diff_ub, (1, h * w - 1))
    ncols = unifint(rng, diff_lb, diff_ub, (1, min(8, nred)))
    ncells = unifint(rng, diff_lb, diff_ub, (1, h * w - nred))
    ccols = rng.sample(cols, ncols)
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    reds = rng.sample(totuple(inds), nred)
    reminds = difference(inds, reds)
    gi = fill(gi, 2, reds)
    rest = rng.sample(totuple(reminds), ncells)
    rest = {(rng.choice(ccols), ij) for ij in rest}
    gi = paint(gi, rest)
    go = canvas(0, (h**2, w**2))
    locs = apply(rbind(multiply, (h, w)), reds)
    res = mapply(lbind(shift, asobject(gi)), locs)
    go = paint(go, res)
    return {"input": gi, "output": go}


def generate_f9012d9b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    hp = unifint(rng, diff_lb, diff_ub, (2, 10))
    wp = unifint(rng, diff_lb, diff_ub, (2, 10))
    srco = canvas(0, (hp, wp))
    inds = asindices(srco)
    nc = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(cols, nc)
    obj = {(rng.choice(ccols), ij) for ij in inds}
    srco = paint(srco, obj)
    gi = paint(srco, obj)
    numhp = unifint(rng, diff_lb, diff_ub, (3, 30 // hp))
    numwp = unifint(rng, diff_lb, diff_ub, (3, 30 // wp))
    for k in range(numhp - 1):
        gi = vconcat(gi, srco)
    srco = tuple(e for e in gi)
    for k in range(numwp - 1):
        gi = hconcat(gi, srco)
    hcropfac = rng.randint(0, hp)
    for k in range(hcropfac):
        gi = gi[:-1]
    gi = dmirror(gi)
    wcropfac = rng.randint(0, wp)
    for k in range(wcropfac):
        gi = gi[:-1]
    gi = dmirror(gi)
    h, w = shape(gi)
    sgh = unifint(rng, diff_lb, diff_ub, (1, h - hp - 1))
    sgw = unifint(rng, diff_lb, diff_ub, (1, w - wp - 1))
    loci = rng.randint(0, h - sgh)
    locj = rng.randint(0, w - sgw)
    loc = (loci, locj)
    shp = (sgh, sgw)
    obj = {loc, decrement(add(loc, shp))}
    obj = backdrop(obj)
    go = subgrid(obj, gi)
    gi = fill(gi, 0, obj)
    mf = rng.choice((identity, rot90, rot180, rot270, dmirror, vmirror, hmirror, cmirror))
    gi = mf(gi)
    go = mf(go)
    return {"input": gi, "output": go}


def generate_f8ff0b80(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, min(30, (h * w) // 25)))
    gi = canvas(bgc, (h, w))
    numcells = unifint(rng, diff_lb, diff_ub, (nobjs + 1, 36))
    base = asindices(canvas(-1, (6, 6)))
    maxtr = 10
    inds = asindices(gi)
    go = []
    for k in range(nobjs):
        if len(inds) == 0 or numcells < 2:
            break
        numcells = unifint(rng, diff_lb, diff_ub, (nobjs - k, numcells - 1))
        if numcells == 0:
            break
        sp = rng.choice(totuple(base))
        shp = {sp}
        reminds = remove(sp, base)
        for kk in range(numcells - 1):
            shp.add(rng.choice(totuple((reminds - shp) & mapply(neighbors, shp))))
        shp = normalize(shp)
        validloc = False
        rems = sfilter(inds, lambda ij: ij[0] <= h - height(shp) and ij[1] <= w - width(shp))
        if len(rems) == 0:
            break
        loc = rng.choice(totuple(rems))
        tr = 0
        while not validloc and tr < maxtr:
            loc = rng.choice(totuple(inds))
            validloc = shift(shp, loc).issubset(inds)
            tr += 1
        if validloc:
            plcd = shift(shp, loc)
            col = rng.choice(remcols)
            go.append(col)
            inds = (inds - plcd) - mapply(neighbors, plcd)
            gi = fill(gi, col, plcd)
    go = dmirror((tuple(go),))
    return {"input": gi, "output": go}


def generate_e21d9049(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    ph = unifint(rng, diff_lb, diff_ub, (2, 9))
    pw = unifint(rng, diff_lb, diff_ub, (2, 9))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    hbar = frozenset({(rng.choice(remcols), (k, 0)) for k in range(ph)})
    wbar = frozenset({(rng.choice(remcols), (0, k)) for k in range(pw)})
    locih = rng.randint(0, h - ph)
    locjh = rng.randint(0, w - 1)
    loch = (locih, locjh)
    locjw = rng.randint(0, w - pw)
    lociw = rng.randint(0, h - 1)
    locw = (lociw, locjw)
    canv = canvas(bgc, (h, w))
    hbar = shift(hbar, loch)
    wbar = shift(wbar, locw)
    cp = (lociw, locjh)
    col = rng.choice(remcols)
    hbard = extract(hbar, lambda cij: abs(cij[1][0] - lociw) % ph == 0)[1]
    hbar = sfilter(hbar, lambda cij: abs(cij[1][0] - lociw) % ph != 0) | {(col, hbard)}
    wbard = extract(wbar, lambda cij: abs(cij[1][1] - locjh) % pw == 0)[1]
    wbar = sfilter(wbar, lambda cij: abs(cij[1][1] - locjh) % pw != 0) | {(col, wbard)}
    gi = paint(canv, hbar | wbar)
    go = paint(canv, hbar | wbar)
    for k in range(h // ph + 1):
        go = paint(go, shift(hbar, (k * ph, 0)))
        go = paint(go, shift(hbar, (-k * ph, 0)))
    for k in range(w // pw + 1):
        go = paint(go, shift(wbar, (0, k * pw)))
        go = paint(go, shift(wbar, (0, -k * pw)))
    return {"input": gi, "output": go}


def generate_d4f3cd78(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    ih = unifint(rng, diff_lb, diff_ub, (3, h // 3 * 2))
    iw = unifint(rng, diff_lb, diff_ub, (3, w // 3 * 2))
    loci = rng.randint(1, h - ih - 1)
    locj = rng.randint(1, w - iw - 1)
    crns = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    fullcrns = corners(crns)
    bx = box(crns)
    opts = bx - fullcrns
    bgc, fgc = rng.sample(cols, 2)
    c = canvas(bgc, (h, w))
    nholes = unifint(rng, diff_lb, diff_ub, (1, len(opts)))
    holes = rng.sample(totuple(opts), nholes)
    gi = fill(c, fgc, bx - set(holes))
    bib = backdrop(inbox(bx))
    go = fill(gi, 8, bib)
    A, B = ulcorner(bib)
    C, D = lrcorner(bib)
    f1 = lambda idx: 1 if idx > C else (-1 if idx < A else 0)
    f2 = lambda idx: 1 if idx > D else (-1 if idx < B else 0)
    f = lambda d: shoot(d, (f1(d[0]), f2(d[1])))
    res = mapply(f, set(holes))
    go = fill(go, 8, res)
    return {"input": gi, "output": go}


def generate_9d9215db(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 14))
    w = unifint(rng, diff_lb, diff_ub, (5, 14))
    h = h * 2 + 1
    w = w * 2 + 1
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ub = min(h, w) // 4
    nrings = unifint(rng, diff_lb, diff_ub, (1, ub))
    onlinesbase = tuple([(2 * k + 1, 2 * k + 1) for k in range(ub)])
    onlines = rng.sample(onlinesbase, nrings)
    onlines = {(rng.choice(remcols), ij) for ij in onlines}
    gi = canvas(bgc, (h, w))
    gi = paint(gi, onlines)
    linsbase = apply(rbind(add, (0, 2)), onlinesbase[:-1])
    nlines = unifint(rng, diff_lb, diff_ub, (1, len(linsbase)))
    linesps = rng.sample(linsbase, nlines)
    colors = [rng.choice(remcols) for k in range(nlines)]
    dots = {(col, ij) for col, ij in zip(colors, linesps)}
    dots2 = {(col, ij[::-1]) for col, ij in zip(colors, linesps)}
    gi = paint(gi, dots | dots2)
    ff = lambda ij: ij[1] % 2 == 1
    ff2 = lambda ij: ij[0] % 2 == 1
    linesps2 = tuple(x[::-1] for x in linesps)
    lines = tuple(sfilter(connect(ij, (ij[0], w - ij[1] - 1)), ff) for ij in linesps)
    lines2 = tuple(sfilter(connect(ij, (h - ij[0] - 1, ij[1])), ff2) for ij in linesps2)
    lines = merge({recolor(col, l1 | l2) for col, (l1, l2) in zip(colors, zip(lines, lines2))})
    gobase = paint(gi, lines)
    go = paint(gobase, merge(fgpartition(vmirror(gobase))))
    go = paint(go, merge(fgpartition(hmirror(gobase))))
    go = paint(go, merge(fgpartition(vmirror(hmirror(gobase)))))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_0ca9ddb6(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 4, 6, 7, 8))
    xi = {(8, (0, 0))}
    xo = {(8, (0, 0))}
    ai = {(6, (0, 0))}
    ao = {(6, (0, 0))}
    bi = {(2, (1, 1))}
    bo = {(2, (1, 1))} | recolor(4, ineighbors((1, 1)))
    ci = {(1, (1, 1))}
    co = {(1, (1, 1))} | recolor(7, dneighbors((1, 1)))
    arr = ((ai, ao), (bi, bo), (ci, co), (xi, xo))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 4))
    maxtr = 5 * nobjs
    tr = 0
    succ = 0
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        ino, outo = rng.choice(arr)
        loc = rng.choice(totuple(inds))
        oplcd = shift(outo, loc)
        oplcdi = toindices(oplcd)
        if oplcdi.issubset(inds):
            succ += 1
            gi = paint(gi, shift(ino, loc))
            go = paint(go, oplcd)
            inds = inds - oplcdi
        tr += 1
    return {"input": gi, "output": go}


def generate_5521c0d9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    inds = interval(0, w, 1)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, w // 3))
    speps = rng.sample(inds, nobjs * 2)
    while 0 in speps or w - 1 in speps:
        nobjs = unifint(rng, diff_lb, diff_ub, (1, w // 3))
        speps = rng.sample(inds, nobjs * 2)
    speps = sorted(speps)
    starts = speps[::2]
    ends = speps[1::2]
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, ncols)
    forb = -1
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    forb = -1
    for sp, ep in zip(starts, ends):
        col = rng.choice(remove(forb, ccols))
        forb = col
        hdev = unifint(rng, diff_lb, diff_ub, (0, h // 2))
        hei = rng.choice((hdev, h - hdev))
        hei = min(max(1, hei), h - 1)
        ulc = (h - hei, sp)
        lrc = (h - 1, ep)
        obj = backdrop(frozenset({ulc, lrc}))
        gi = fill(gi, col, obj)
        go = fill(go, col, shift(obj, (-hei, 0)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_e3497940(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 14))
    bgc, barc = rng.sample(cols, 2)
    remcols = remove(barc, remove(bgc, cols))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, ncols)
    nlinesocc = unifint(rng, diff_lb, diff_ub, (1, h))
    lopts = interval(0, h, 1)
    linesocc = rng.sample(lopts, nlinesocc)
    rs = canvas(bgc, (h, w))
    ls = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for idx in linesocc:
        j = unifint(rng, diff_lb, diff_ub, (1, w - 1))
        obj = [(rng.choice(ccols), (idx, jj)) for jj in range(j)]
        go = paint(go, obj)
        slen = rng.randint(1, j)
        obj2 = obj[:slen]
        if rng.choice((True, False)):
            obj, obj2 = obj2, obj
        rs = paint(rs, obj)
        ls = paint(ls, obj2)
    gi = hconcat(hconcat(vmirror(ls), canvas(barc, (h, 1))), rs)
    go = vmirror(go)
    return {"input": gi, "output": go}


def generate_6cdd2623(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    remcols = remove(linc, remcols)
    nnoisecols = unifint(rng, diff_lb, diff_ub, (1, 7))
    noisecols = rng.sample(remcols, nnoisecols)
    c = canvas(bgc, (h, w))
    ininds = totuple(shift(asindices(canvas(-1, (h - 2, w - 1))), (1, 1)))
    fixinds = rng.sample(ininds, nnoisecols)
    fixobj = {(col, ij) for col, ij in zip(list(noisecols), fixinds)}
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = paint(gi, fixobj)
    nnoise = unifint(rng, diff_lb, diff_ub, (1, (h * w - nnoisecols) // 3))
    noise = rng.sample(totuple(asindices(c) - set(fixinds)), nnoise)
    noise = {(rng.choice(remcols), ij) for ij in noise}
    gi = paint(gi, noise)
    ilocs = interval(1, h - 1, 1)
    jlocs = interval(1, w - 1, 1)
    aa, bb = rng.sample((0, 1), 2)
    nilocs = unifint(rng, diff_lb, diff_ub, (aa, (h - 2) // 2))
    njlocs = unifint(rng, diff_lb, diff_ub, (bb, (w - 2) // 2))
    ilocs = rng.sample(ilocs, nilocs)
    jlocs = rng.sample(jlocs, njlocs)
    for ii in ilocs:
        gi = fill(gi, linc, {(ii, 0)})
        gi = fill(gi, linc, {(ii, w - 1)})
        go = fill(go, linc, connect((ii, 0), (ii, w - 1)))
    for jj in jlocs:
        gi = fill(gi, linc, {(0, jj)})
        gi = fill(gi, linc, {(h - 1, jj)})
        go = fill(go, linc, connect((0, jj), (h - 1, jj)))
    return {"input": gi, "output": go}


def generate_dc433765(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc, src = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    if rng.choice((True, False)):
        opts = {(ii, 0) for ii in range(h - 2)} | {(0, jj) for jj in range(1, w - 2, 1)}
        opts = tuple([inds & shoot(src, (1, 1)) for src in opts])
        opts = order(opts, size)
        k = len(opts)
        opt = unifint(rng, diff_lb, diff_ub, (0, k - 1))
        ln = order(opts[opt], first)
        epi = unifint(rng, diff_lb, diff_ub, (2, len(ln) - 1))
        ep = ln[epi]
        ln = ln[: epi - 1][::-1]
        spi = unifint(rng, diff_lb, diff_ub, (0, len(ln) - 1))
        sp = ln[spi]
        gi = fill(gi, src, {sp})
        gi = fill(gi, 4, {ep})
        go = fill(go, src, {add(sp, (1, 1))})
        go = fill(go, 4, {ep})
    else:
        loci = rng.randint(0, h - 1)
        objw = unifint(rng, diff_lb, diff_ub, (3, w))
        locj1 = rng.randint(0, w - objw)
        locj2 = locj1 + objw - 1
        sp = (loci, locj1)
        ep = (loci, locj2)
        gi = fill(gi, src, {sp})
        gi = fill(gi, 4, {ep})
        go = fill(go, src, {add(sp, (0, 1))})
        go = fill(go, 4, {ep})
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_d2abd087(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    bgc = rng.choice(difference(cols, (1, 2)))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 10))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if len(inds) == 0:
            break
        opts = asindices(canvas(-1, (5, 5)))
        sp = rng.choice(totuple(opts))
        opts = remove(sp, opts)
        lb = unifint(rng, diff_lb, diff_ub, (1, 5))
        lopts = interval(lb, 6, 1)
        ubi = unifint(rng, diff_lb, diff_ub, (1, 5))
        ub = 12 - ubi
        uopts = interval(7, ub + 1, 1)
        if rng.choice((True, False)):
            numcells = 6
        else:
            numcells = rng.choice(lopts + uopts)
        obj = {sp}
        for k in range(numcells - 1):
            obj.add(rng.choice(totuple((opts - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        loc = rng.choice(totuple(inds))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            gi = fill(gi, rng.choice(remcols), plcd)
            go = fill(go, 1 + (len(obj) == 6), plcd)
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
        tr += 1
    return {"input": gi, "output": go}


def generate_88a10436(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    objh = unifint(rng, diff_lb, diff_ub, (0, 2))
    objw = unifint(rng, diff_lb, diff_ub, (0 if objh > 0 else 1, 2))
    objh = objh * 2 + 1
    objw = objw * 2 + 1
    bb = asindices(canvas(-1, (objh, objw)))
    sp = (objh // 2, objw // 2)
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(rng, diff_lb, diff_ub, (max(objh, objw), objh * objw))
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    while height(obj) != objh or width(obj) != objw:
        obj.add(rng.choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    bgc, fgc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(fgc, cols))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, ncols)
    obj = {(rng.choice(ccols), ij) for ij in obj}
    obj = normalize(obj)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    loci = rng.randint(0, h - objh)
    locj = rng.randint(0, w - objw)
    loc = (loci, locj)
    plcd = shift(obj, loc)
    gi = paint(gi, plcd)
    go = paint(go, plcd)
    inds = (asindices(gi) - toindices(plcd)) - mapply(neighbors, toindices(plcd))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // (2 * ncells)))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if len(inds) == 0:
            break
        loc = rng.choice(totuple(inds))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            go = paint(go, plcd)
            gi = fill(gi, fgc, {center(plcdi)})
            succ += 1
            inds = (inds - plcdi) - mapply(dneighbors, plcdi)
        tr += 1
    return {"input": gi, "output": go}


def generate_05f2a901(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    objh = unifint(rng, diff_lb, diff_ub, (2, min(w // 2, h // 2)))
    objw = unifint(rng, diff_lb, diff_ub, (objh, w // 2))
    bb = asindices(canvas(-1, (objh, objw)))
    sp = rng.choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(rng, diff_lb, diff_ub, (objh + objw, objh * objw))
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    if height(obj) * width(obj) == len(obj):
        obj = remove(rng.choice(totuple(obj)), obj)
    obj = normalize(obj)
    objh, objw = shape(obj)
    loci = unifint(rng, diff_lb, diff_ub, (3, h - objh))
    locj = unifint(rng, diff_lb, diff_ub, (0, w - objw))
    loc = (loci, locj)
    bgc, fgc, destc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    obj = shift(obj, loc)
    gi = fill(gi, fgc, obj)
    sqd = rng.randint(1, min(w, loci - 1))
    locisq = rng.randint(0, loci - sqd - 1)
    locjsq = rng.randint(locj - sqd + 1, locj + objw - 1)
    locsq = (locisq, locjsq)
    sq = backdrop({(locisq, locjsq), (locisq + sqd - 1, locjsq + sqd - 1)})
    gi = fill(gi, destc, sq)
    go = fill(go, destc, sq)
    while len(obj & sq) == 0:
        obj = shift(obj, (-1, 0))
    obj = shift(obj, (1, 0))
    go = fill(go, fgc, obj)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_928ad970(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    ih = unifint(rng, diff_lb, diff_ub, (9, h))
    iw = unifint(rng, diff_lb, diff_ub, (9, w))
    bgc, linc, dotc = rng.sample(cols, 3)
    loci = rng.randint(0, h - ih)
    locj = rng.randint(0, w - iw)
    ulc = (loci, locj)
    lrc = (loci + ih - 1, locj + iw - 1)
    dot1 = rng.choice(totuple(connect(ulc, (loci + ih - 1, locj)) - {ulc, (loci + ih - 1, locj)}))
    dot2 = rng.choice(totuple(connect(ulc, (loci, locj + iw - 1)) - {ulc, (loci, locj + iw - 1)}))
    dot3 = rng.choice(totuple(connect(lrc, (loci + ih - 1, locj)) - {lrc, (loci + ih - 1, locj)}))
    dot4 = rng.choice(totuple(connect(lrc, (loci, locj + iw - 1)) - {lrc, (loci, locj + iw - 1)}))
    a, b = sorted(rng.sample(interval(loci + 2, loci + ih - 2, 1), 2))
    while a + 1 == b:
        a, b = sorted(rng.sample(interval(loci + 2, loci + ih - 2, 1), 2))
    c, d = sorted(rng.sample(interval(locj + 2, locj + iw - 2, 1), 2))
    while c + 1 == d:
        c, d = sorted(rng.sample(interval(locj + 2, locj + iw - 2, 1), 2))
    sp = box(frozenset({(a, c), (b, d)}))
    bx = {dot1, dot2, dot3, dot4}
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, dotc, bx)
    gi = fill(gi, linc, sp)
    go = fill(gi, linc, inbox(bx))
    return {"input": gi, "output": go}


def generate_f8b3ba0a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 5))
    w = unifint(rng, diff_lb, diff_ub, (1, 5))
    nh = unifint(rng, diff_lb, diff_ub, (3, 29 // (h + 1)))
    nw = unifint(rng, diff_lb, diff_ub, (3, 29 // (w + 1)))
    fullh = (h + 1) * nh + 1
    fullw = (w + 1) * nw + 1
    fullbgc, bgc = rng.sample(cols, 2)
    remcols = remove(fullbgc, remove(bgc, cols))
    shp = shift(asindices(canvas(-1, (h, w))), (1, 1))
    gi = canvas(fullbgc, (fullh, fullw))
    locs = set()
    for a in range(nh):
        for b in range(nw):
            loc = (a * (h + 1), b * (w + 1))
            locs.add(loc)
            gi = fill(gi, bgc, shift(shp, loc))
    numc = unifint(rng, diff_lb, diff_ub, (1, (nh * nw) // 2 - 1))
    stack = []
    nn = numc + 1
    ncols = 0
    while nn > 1 and numc > 0 and len(remcols) > 0:
        nn3 = int(0.5 * (8 * numc + 1) ** 0.5 - 1)
        nn = min(max(1, nn3), nn - 1)
        col = rng.choice(remcols)
        remcols = remove(col, remcols)
        numc -= nn
        stack.append((col, nn))
    go = dmirror((tuple(c for c, nn in stack),))
    for col, nn in stack:
        slocs = rng.sample(totuple(locs), nn)
        gi = fill(gi, col, mapply(lbind(shift, shp), slocs))
        locs = locs - set(slocs)
    return {"input": gi, "output": go}


def generate_fcb5c309(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc, dotc, sqc = rng.sample(cols, 3)
    numsq = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 25))
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    maxtr = 4 * numsq
    tr = 0
    succ = 0
    numcells = None
    take = False
    while tr < maxtr and succ < numsq:
        oh = rng.randint(3, 7)
        ow = rng.randint(3, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        sq = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bd = backdrop(sq)
        if bd.issubset(inds):
            gi = fill(gi, sqc, sq)
            ib = backdrop(inbox(sq))
            if numcells is None:
                numcells = unifint(rng, diff_lb, diff_ub, (1, len(ib)))
                cells = rng.sample(totuple(ib), numcells)
                take = True
            else:
                nc = unifint(rng, diff_lb, diff_ub, (0, min(max(0, numcells - 1), len(ib))))
                cells = rng.sample(totuple(ib), nc)
            gi = fill(gi, dotc, cells)
            if take:
                go = replace(subgrid(sq, gi), sqc, dotc)
                take = False
            inds = (inds - bd) - outbox(bd)
            succ += 1
        tr += 1
    nnoise = unifint(rng, diff_lb, diff_ub, (0, max(0, len(inds) // 2 - 1)))
    noise = rng.sample(totuple(inds), nnoise)
    gi = fill(gi, dotc, noise)
    return {"input": gi, "output": go}


def generate_54d9e175(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = (0, 5)
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    nh = unifint(rng, diff_lb, diff_ub, (1, 31 // (h + 1)))
    nw = unifint(rng, diff_lb, diff_ub, (1 if nh > 1 else 2, 31 // (w + 1)))
    fullh = (h + 1) * nh - 1
    fullw = (w + 1) * nw - 1
    linc, bgc = rng.sample(cols, 2)
    gi = canvas(linc, (fullh, fullw))
    go = canvas(linc, (fullh, fullw))
    obj = asindices(canvas(bgc, (h, w)))
    for a in range(nh):
        for b in range(nw):
            plcd = shift(obj, (a * (h + 1), b * (w + 1)))
            icol = rng.randint(1, 4)
            ocol = icol + 5
            gi = fill(gi, bgc, plcd)
            go = fill(go, ocol, plcd)
            dot = rng.choice(totuple(plcd))
            gi = fill(gi, icol, {dot})
    return {"input": gi, "output": go}


def generate_7f4411dc(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc, fgc = rng.sample(cols, 2)
    nsq = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 15))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < nsq:
        oh = rng.randint(2, 6)
        ow = rng.randint(2, 6)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        obj = shift(obj, loc)
        if obj.issubset(inds):
            go = fill(go, fgc, obj)
            succ += 1
            inds = (inds - obj) - outbox(obj)
        tr += 1
    inds = ofcolor(go, bgc)
    nnoise = unifint(rng, diff_lb, diff_ub, (0, len(inds) // 2 - 1))
    gi = tuple(e for e in go)
    for k in range(nnoise):
        loc = rng.choice(totuple(inds))
        inds = inds - dneighbors(loc)
        gi = fill(gi, fgc, {loc})
    return {"input": gi, "output": go}


def generate_67385a82(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, remove(8, interval(0, 10, 1)))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    col = rng.choice(cols)
    gi = canvas(0, (h, w))
    inds = totuple(asindices(gi))
    ncd = unifint(rng, diff_lb, diff_ub, (0, len(inds) // 2))
    nc = rng.choice((ncd, len(inds) - ncd))
    nc = min(max(1, nc), len(inds) - 1)
    locs = rng.sample(inds, nc)
    gi = fill(gi, col, locs)
    objs = objects(gi, T, F, F)
    rems = toindices(merge(sizefilter(colorfilter(objs, col), 1)))
    blues = difference(ofcolor(gi, col), rems)
    go = fill(gi, 8, blues)
    return {"input": gi, "output": go}


def generate_d6ad076f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    inh = unifint(rng, diff_lb, diff_ub, (3, h))
    inw = unifint(rng, diff_lb, diff_ub, (3, w))
    bgc, c1, c2 = rng.sample(cols, 3)
    itv = interval(0, inh, 1)
    loci2i = unifint(rng, diff_lb, diff_ub, (2, inh - 1))
    loci2 = itv[loci2i]
    itv = itv[: loci2i - 1][::-1]
    loci1i = unifint(rng, diff_lb, diff_ub, (0, len(itv) - 1))
    loci1 = itv[loci1i]
    cp = rng.randint(1, inw - 2)
    ajs = rng.randint(0, cp - 1)
    aje = rng.randint(cp + 1, inw - 1)
    bjs = rng.randint(0, cp - 1)
    bje = rng.randint(cp + 1, inw - 1)
    obja = backdrop(frozenset({(0, ajs), (loci1, aje)}))
    objb = backdrop(frozenset({(loci2, bjs), (inh - 1, bje)}))
    c = canvas(bgc, (inh, inw))
    c = fill(c, c1, obja)
    c = fill(c, c2, objb)
    obj = asobject(c)
    loci = rng.randint(0, h - inh)
    locj = rng.randint(0, w - inw)
    loc = (loci, locj)
    obj = shift(obj, loc)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, obj)
    midobj = backdrop(frozenset({(loci1 + 1, max(ajs, bjs) + 1), (loci2 - 1, min(aje, bje) - 1)}))
    go = fill(gi, 8, shift(midobj, loc))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_e48d4e1a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    loci = rng.randint(1, h - 2)
    locj = rng.randint(1, w - 2)
    inds = asindices(canvas(-1, (loci, locj)))
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = unifint(rng, diff_lb, diff_ub, (1, maxn))
    ss = rng.sample(totuple(inds), nn)
    bgc, fgc, dotc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, fgc, hfrontier((loci, 0)) | vfrontier((0, locj)))
    gi = fill(gi, dotc, ss)
    go = fill(go, fgc, hfrontier((loci + nn, 0)) | vfrontier((0, locj + nn)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_a48eeaf7(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    ih = unifint(rng, diff_lb, diff_ub, (2, h // 2))
    iw = unifint(rng, diff_lb, diff_ub, (2, w // 2))
    loci = rng.randint(2, h - ih - 2)
    locj = rng.randint(2, w - iw - 2)
    bgc, sqc, dotc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    sq = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    A = [(x, locj - 1) for x in interval(loci, loci + ih, 1)]
    Ap = [(x, rng.randint(0, locj - 2)) for x in interval(loci, loci + ih, 1)]
    B = [(x, locj + iw) for x in interval(loci, loci + ih, 1)]
    Bp = [(x, rng.randint(locj + iw + 1, w - 1)) for x in interval(loci, loci + ih, 1)]
    C = [(loci - 1, x) for x in interval(locj, locj + iw, 1)]
    Cp = [(rng.randint(0, loci - 2), x) for x in interval(locj, locj + iw, 1)]
    D = [(loci + ih, x) for x in interval(locj, locj + iw, 1)]
    Dp = [(rng.randint(loci + ih + 1, h - 1), x) for x in interval(locj, locj + iw, 1)]
    srarr = Ap + Bp + Cp + Dp
    dearr = A + B + C + D
    inds = interval(0, len(srarr), 1)
    num = unifint(rng, diff_lb, diff_ub, (1, len(srarr)))
    locs = rng.sample(inds, num)
    srarr = [e for j, e in enumerate(srarr) if j in locs]
    dearr = [e for j, e in enumerate(dearr) if j in locs]
    gi = fill(gi, sqc, sq)
    go = fill(go, sqc, sq)
    for s, d in zip(srarr, dearr):
        gi = fill(gi, dotc, {s})
        go = fill(go, dotc, {d})
    ncorn = unifint(rng, diff_lb, diff_ub, (0, 4))
    fullinds = asindices(gi)
    if ncorn > 0:
        go = fill(go, dotc, {(loci - 1, locj - 1)})
        cands = shoot((loci - 2, locj - 2), (-1, -1)) & fullinds
        locc = rng.choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 1:
        go = fill(go, dotc, {(loci - 1, locj + iw)})
        cands = shoot((loci - 2, locj + iw + 1), (-1, 1)) & fullinds
        locc = rng.choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 2:
        go = fill(go, dotc, {(loci + ih, locj - 1)})
        cands = shoot((loci + ih + 1, locj - 2), (1, -1)) & fullinds
        locc = rng.choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 3:
        go = fill(go, dotc, {(loci + ih, locj + iw)})
        cands = shoot((loci + ih + 1, locj + iw + 1), (1, 1)) & fullinds
        locc = rng.choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_56dc2b01(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 8))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    oh = unifint(rng, diff_lb, diff_ub, (1, h))
    ow = unifint(rng, diff_lb, diff_ub, (1, (w - 1) // 2 - 1))
    bb = asindices(canvas(-1, (oh, ow)))
    sp = rng.choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncellsd = unifint(rng, diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = rng.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for k in range(ncells):
        obj.add(rng.choice(totuple((bb - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = rng.randint(0, h - oh)
    locj = unifint(rng, diff_lb, diff_ub, (1, w - ow))
    bgc, objc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    barlocji = unifint(rng, diff_lb, diff_ub, (0, locj))
    barlocj = locj - barlocji
    barlocj = min(max(0, barlocj), locj - 1)
    gi = fill(gi, 2, connect((0, barlocj), (h - 1, barlocj)))
    go = fill(gi, objc, shift(obj, (loci, barlocj + 1)))
    go = fill(go, 8, connect((0, barlocj + ow + 1), (h - 1, barlocj + ow + 1)))
    gi = fill(gi, objc, shift(obj, (loci, locj)))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_1caeab9d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1,))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    oh = unifint(rng, diff_lb, diff_ub, (1, h // 2))
    ow = unifint(rng, diff_lb, diff_ub, (1, w // 3))
    bb = asindices(canvas(-1, (oh, ow)))
    sp = rng.choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncellsd = unifint(rng, diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = rng.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for k in range(ncells):
        obj.add(rng.choice(totuple((bb - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = rng.randint(0, h - oh)
    numo = unifint(rng, diff_lb, diff_ub, (2, min(8, w // ow))) - 1
    itv = interval(0, w, 1)
    locj = rng.randint(0, w - ow)
    objp = shift(obj, (loci, locj))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    gi = fill(c, 1, objp)
    go = fill(c, 1, objp)
    itv = difference(itv, interval(locj, locj + ow, 1))
    for k in range(numo):
        cands = sfilter(itv, lambda j: set(interval(j, j + ow, 1)).issubset(set(itv)))
        if len(cands) == 0:
            break
        locj = rng.choice(cands)
        col = rng.choice(remcols)
        remcols = remove(col, remcols)
        gi = fill(gi, col, shift(obj, (rng.randint(0, h - oh), locj)))
        go = fill(go, col, shift(obj, (loci, locj)))
        itv = difference(itv, interval(locj, locj + ow, 1))
    return {"input": gi, "output": go}


def generate_b91ae062(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    numc = unifint(rng, diff_lb, diff_ub, (3, min(h * w, min(10, 30 // max(h, w)))))
    ccols = rng.sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    fixinds = rng.sample(inds, numc)
    obj = {(cc, ij) for cc, ij in zip(ccols, fixinds)}
    for ij in difference(inds, fixinds):
        obj.add((rng.choice(ccols), ij))
    gi = paint(c, obj)
    go = upscale(gi, numc - 1)
    return {"input": gi, "output": go}


def generate_834ec97d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    loci = unifint(rng, diff_lb, diff_ub, (0, h - 2))
    locjd = unifint(rng, diff_lb, diff_ub, (0, w // 2))
    locj = rng.choice((locjd, w - locjd))
    locj = min(max(0, locj), w - 1)
    loc = (loci, locj)
    bgc, fgc = rng.sample(cols, 2)
    c = canvas(bgc, (h, w))
    gi = fill(c, fgc, {loc})
    go = fill(c, fgc, {add(loc, (1, 0))})
    for jj in range(w // 2 + 1):
        go = fill(go, 4, connect((0, locj + 2 * jj), (loci, locj + 2 * jj)))
        go = fill(go, 4, connect((0, locj - 2 * jj), (loci, locj - 2 * jj)))
    return {"input": gi, "output": go}


def generate_a699fb00(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    numls = unifint(rng, diff_lb, diff_ub, (1, h - 1))
    opts = interval(0, h, 1)
    locs = rng.sample(opts, numls)
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for ii in locs:
        endidx = unifint(rng, diff_lb, diff_ub, (2, w - 2))
        ofs = unifint(rng, diff_lb, diff_ub, (1, endidx // 2)) * 2
        ofs = min(max(2, ofs), endidx)
        startidx = endidx - ofs
        ln = connect((ii, startidx), (ii, endidx))
        go = fill(go, 2, ln)
        sparseln = {(ii, jj) for jj in range(startidx, endidx + 1, 2)}
        go = fill(go, fgc, sparseln)
        gi = fill(gi, fgc, sparseln)
    return {"input": gi, "output": go}


def generate_91413438(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    maxnb = min(h * w - 1, min(30 // h, 30 // w))
    minnb = int(0.5 * ((4 * h * w + 1) ** 0.5 - 1)) + 1
    nbi = unifint(rng, diff_lb, diff_ub, (0, maxnb - minnb))
    nb = min(max(minnb, maxnb - nbi), maxnb)
    fgc = rng.choice(cols)
    c = canvas(0, (h, w))
    obj = rng.sample(totuple(asindices(c)), h * w - nb)
    gi = fill(c, fgc, obj)
    go = canvas(0, (h * nb, w * nb))
    for j in range(h * w - nb):
        loc = (j // nb, j % nb)
        go = fill(go, fgc, shift(obj, multiply((h, w), loc)))
    return {"input": gi, "output": go}


def generate_99fa7670(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    num = unifint(rng, diff_lb, diff_ub, (1, h // 2))
    inds = interval(0, h, 1)
    starts = sorted(rng.sample(inds, num))
    ends = [x - 1 for x in starts[1:]] + [h - 1]
    nc = unifint(rng, diff_lb, diff_ub, (1, 9))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ccols = rng.sample(remcols, nc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for s, e in zip(starts, ends):
        col = rng.choice(ccols)
        locj = rng.randint(0, w - 2)
        l1 = connect((s, locj), (s, w - 1))
        l2 = connect((s, w - 1), (e, w - 1))
        gi = fill(gi, col, {(s, locj)})
        go = fill(go, col, l1 | l2)
    return {"input": gi, "output": go}


def generate_d13f3404(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 15))
    w = unifint(rng, diff_lb, diff_ub, (3, 15))
    vopts = {(ii, 0) for ii in interval(0, h, 1)}
    hopts = {(0, jj) for jj in interval(1, w, 1)}
    opts = tuple(vopts | hopts)
    num = unifint(rng, diff_lb, diff_ub, (1, len(opts)))
    locs = rng.sample(opts, num)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h * 2, w * 2))
    inds = asindices(gi)
    for loc in locs:
        ln = tuple(shoot(loc, (1, 1)) & inds)
        locc = rng.choice(ln)
        col = rng.choice(remcols)
        gi = fill(gi, col, {locc})
        go = fill(go, col, shoot(locc, (1, 1)))
    return {"input": gi, "output": go}


def generate_c3f564a4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (7, 30))
    p = unifint(rng, diff_lb, diff_ub, (2, min(9, min(h // 3, w // 3))))
    fixc = rng.choice(cols)
    remcols = remove(fixc, cols)
    ccols = list(rng.sample(remcols, p))
    rng.shuffle(ccols)
    c = canvas(-1, (h, w))
    baseobj = {(cc, (0, jj)) for cc, jj in zip(ccols, range(p))}
    obj = {c for c in baseobj}
    while rightmost(obj) < 2 * max(w, h):
        obj = obj | shift(obj, (0, p))
    if rng.choice((True, False)):
        obj = mapply(lbind(shift, obj), {(jj, 0) for jj in interval(0, h, 1)})
    else:
        obj = mapply(lbind(shift, obj), {(jj, -jj) for jj in interval(0, h, 1)})
    go = paint(c, obj)
    gi = tuple(e for e in go)
    nsq = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    while succ < nsq and tr < maxtr:
        oh = unifint(rng, diff_lb, diff_ub, (2, 5))
        ow = unifint(rng, diff_lb, diff_ub, (2, 5))
        loci = rng.randint(0, h - oh)
        locj = rng.randint(0, w - ow)
        tmpg = fill(gi, fixc, backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})))
        if (
            len(occurrences(tmpg, baseobj)) > 1
            and len([r for r in tmpg if fixc not in r]) > 0
            and len([r for r in dmirror(tmpg) if fixc not in r]) > 0
        ):
            gi = tmpg
            succ += 1
        tr += 1
    if rng.choice((True, False)):
        gi = rot90(gi)
        go = rot90(go)
    return {"input": gi, "output": go}


def generate_ecdecbb3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc, dotc, linc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    nl = unifint(rng, diff_lb, diff_ub, (1, h // 4))
    inds = interval(0, h, 1)
    locs = []
    for k in range(nl):
        if len(inds) == 0:
            break
        idx = rng.choice(inds)
        locs.append(idx)
        inds = remove(idx, inds)
        inds = remove(idx - 1, inds)
        inds = remove(idx + 1, inds)
        inds = remove(idx - 2, inds)
        inds = remove(idx + 2, inds)
    locs = sorted(locs)
    for ii in locs:
        gi = fill(gi, linc, hfrontier((ii, 0)))
    iopts = difference(difference(difference(interval(0, h, 1), locs), apply(increment, locs)), apply(decrement, locs))
    jopts = interval(0, w, 1)
    ndots = unifint(rng, diff_lb, diff_ub, (1, min(len(iopts), w // 2)))
    dlocs = []
    for k in range(ndots):
        if len(iopts) == 0 or len(jopts) == 0:
            break
        loci = rng.choice(iopts)
        locj = rng.choice(jopts)
        dlocs.append((loci, locj))
        jopts = remove(locj, jopts)
        jopts = remove(locj + 1, jopts)
        jopts = remove(locj - 1, jopts)
    go = gi
    for d in dlocs:
        loci, locj = d
        if loci < min(locs):
            go = fill(go, dotc, connect(d, (min(locs), locj)))
            go = fill(go, linc, neighbors((min(locs), locj)))
        elif loci > max(locs):
            go = fill(go, dotc, connect(d, (max(locs), locj)))
            go = fill(go, linc, neighbors((max(locs), locj)))
        else:
            sp = [e for e in locs if e < loci][-1]
            ep = [e for e in locs if e > loci][0]
            go = fill(go, dotc, connect((sp, locj), (ep, locj)))
            go = fill(go, linc, neighbors((sp, locj)))
            go = fill(go, linc, neighbors((ep, locj)))
        gi = fill(gi, dotc, {d})
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_ac0a08a4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    num = unifint(rng, diff_lb, diff_ub, (1, min(min(9, h * w - 2), min(30 // h, 30 // w))))
    bgc = rng.choice(cols)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = rng.sample(totuple(inds), num)
    remcols = remove(bgc, cols)
    obj = {(col, loc) for col, loc in zip(rng.sample(remcols, num), locs)}
    gi = paint(c, obj)
    go = upscale(gi, num)
    return {"input": gi, "output": go}


def generate_22168020(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    num = unifint(rng, diff_lb, diff_ub, (1, min(9, (h * w) // 10)))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    succ = 0
    tr = 0
    maxtr = 6 * num
    inds = asindices(gi)
    while tr < maxtr and succ < num:
        d = unifint(rng, diff_lb, diff_ub, (2, 5))
        oh = d + 1
        ow = 2 * d
        if len(inds) == 0:
            tr += 1
            continue
        loc = rng.choice(totuple(inds))
        loci, locj = loc
        io1 = connect(loc, (loci + d - 1, locj + d - 1))
        io2 = connect((loci, locj + ow - 1), (loci + d - 1, locj + d))
        io = io1 | io2 | {(loci + d, locj + d - 1), (loci + d, locj + d)}
        oo = merge(sfilter(prapply(connect, io, io), hline))
        mf = rng.choice((identity, dmirror, cmirror, hmirror, vmirror))
        io = mf(io)
        oo = mf(oo)
        col = rng.choice(remcols)
        if oo.issubset(inds):
            gi = fill(gi, col, io)
            go = fill(go, col, oo)
            succ += 1
            inds = inds - oo
            remcols = remove(col, remcols)
        tr += 1
    return {"input": gi, "output": go}


def generate_ff805c23(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
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
    gi = fill(gi, 0, bd)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_4093f84a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (7, 30))
    loci1, loci2 = sorted(rng.sample(interval(2, h - 2, 1), 2))
    bgc, barc, dotc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    for ii in range(loci1, loci2 + 1, 1):
        gi = fill(gi, barc, connect((ii, 0), (ii, w - 1)))
    go = tuple(e for e in gi)
    opts = interval(0, w, 1)
    num1 = unifint(rng, diff_lb, diff_ub, (1, w // 2))
    num2 = unifint(rng, diff_lb, diff_ub, (1, w // 2))
    locs1 = rng.sample(opts, num1)
    locs2 = rng.sample(opts, num2)
    for l1 in locs1:
        k = unifint(rng, diff_lb, diff_ub, (1, loci1 - 1))
        locsx = rng.sample(interval(0, loci1, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l1), locsx))
        go = fill(go, barc, connect((loci1 - 1, l1), (loci1 - k, l1)))
    for l2 in locs2:
        k = unifint(rng, diff_lb, diff_ub, (1, h - loci2 - 2))
        locsx = rng.sample(interval(loci2 + 1, h, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l2), locsx))
        go = fill(go, barc, connect((loci2 + 1, l2), (loci2 + k, l2)))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_760b3cac(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    objL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
    objR = vmirror(objL)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 14))
    w = 2 * w + 1
    bgc, objc, indc = rng.sample(cols, 3)
    objh = unifint(rng, diff_lb, diff_ub, (1, h - 3))
    objw = unifint(rng, diff_lb, diff_ub, (1, w // 6))
    objw = 2 * objw + 1
    c = canvas(-1, (objh, objw))
    gi = canvas(bgc, (h, w))
    if rng.choice((True, False)):
        obj = objL
        sgn = -1
    else:
        obj = objR
        sgn = 1
    gi = fill(gi, indc, shift(obj, (h - 3, w // 2 - 1)))
    inds = asindices(c)
    sp = rng.choice(totuple(inds))
    objx = {sp}
    numcd = unifint(rng, diff_lb, diff_ub, (0, (objh * objw) // 2))
    numc = rng.choice((numcd, objh * objw - numcd))
    numc = min(max(1, numc), objh * objw)
    for k in range(numc - 1):
        objx.add(rng.choice(totuple((inds - objx) & mapply(neighbors, objx))))
    while width(objx) != objw:
        objx.add(rng.choice(totuple((inds - objx) & mapply(neighbors, objx))))
    objx = normalize(objx)
    objh, objw = shape(objx)
    loci = rng.randint(0, h - 3 - objh)
    locj = w // 2 - objw // 2
    loc = (loci, locj)
    plcd = shift(objx, loc)
    gi = fill(gi, objc, plcd)
    objx2 = vmirror(plcd)
    plcd2 = shift(objx2, (0, objw * sgn))
    go = fill(gi, objc, plcd2)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_8efcae92(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc, sqc, dotc = rng.sample(cols, 3)
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    maxtr = 4 * num
    tr = 0
    gi = canvas(bgc, (h, w))
    go = None
    inds = asindices(gi)
    oho, owo = None, None
    while succ < num and tr < maxtr:
        if oho is None and owo is None:
            oh = rng.randint(2, h - 1)
            ow = rng.randint(2, w - 1)
            oho = oh
            owo = ow
        else:
            ohd = unifint(rng, diff_lb, diff_ub, (0, min(oho, h - 1 - oho)))
            owd = unifint(rng, diff_lb, diff_ub, (0, min(owo, w - 1 - owo)))
            ohd = min(oho, h - 1 - oho) - ohd
            owd = min(owo, w - 1 - owo) - owd
            oh = rng.choice((oho - ohd, oho + ohd))
            ow = rng.choice((owo - owd, owo + owd))
            oh = min(max(2, oh), h - 1)
            ow = min(max(2, ow), w - 1)
        minig = canvas(sqc, (oh, ow))
        mini = asindices(minig)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        tr += 1
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        if not shift(mini, loc).issubset(inds):
            continue
        succ += 1
        if go is None:
            numdots = unifint(rng, diff_lb, diff_ub, (1, (oh * ow) // 2 - 1))
            nd = numdots
        else:
            nd = unifint(rng, diff_lb, diff_ub, (0, min((oh * ow) // 2 - 1, numdots - 1)))
        locs = rng.sample(totuple(mini), nd)
        minig = fill(minig, dotc, locs)
        if go is None:
            go = minig
        obj = asobject(minig)
        plcd = shift(obj, loc)
        gi = paint(gi, plcd)
        inds = (inds - toindices(plcd)) - mapply(dneighbors, toindices(plcd))
    return {"input": gi, "output": go}


def generate_48d8fb45(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (2, (h * w) // 15))
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    while tr < maxtr and succ < nobjs:
        oh = rng.randint(2, 6)
        ow = rng.randint(2, 6)
        bx = asindices(canvas(-1, (oh, ow)))
        nc = rng.randint(3, oh * ow)
        sp = rng.choice(totuple(bx))
        bx = remove(sp, bx)
        obj = {sp}
        for k in range(nc - 1):
            obj.add(rng.choice(totuple((bx - obj) & mapply(neighbors, obj))))
        if not done:
            done = True
            idx = rng.choice(totuple(obj))
            coll = rng.choice(remcols)
            obj2 = {(coll, idx)}
            obj3 = recolor(rng.choice(remove(coll, remcols)), remove(idx, obj))
            obj = obj2 | obj3
            go = paint(canvas(bgc, shape(obj3)), normalize(obj3))
        else:
            obj = recolor(rng.choice(remcols), obj)
        locopts = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        tr += 1
        if len(locopts) == 0:
            continue
        loc = rng.choice(totuple(locopts))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            gi = paint(gi, plcd)
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
    return {"input": gi, "output": go}


def generate_8e1813be(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    bgc, sqc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    nbars = unifint(rng, diff_lb, diff_ub, (3, 8))
    ccols = rng.sample(remcols, nbars)
    w = unifint(rng, diff_lb, diff_ub, (nbars + 3, 30))
    hmarg = unifint(rng, diff_lb, diff_ub, (2 * nbars, 30 - nbars))
    ccols = list(ccols)
    go = tuple(repeat(col, nbars) for col in ccols)
    gi = tuple(repeat(col, w) for col in ccols)
    r = repeat(bgc, w)
    for k in range(hmarg):
        idx = rng.randint(0, len(go) - 1)
        gi = gi[:idx] + (r,) + gi[idx:]
    h2 = nbars + hmarg
    oh, ow = nbars, nbars
    loci = rng.randint(1, h2 - oh - 2)
    locj = rng.randint(1, w - ow - 2)
    sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, sqc, sq)
    gi = fill(gi, bgc, outbox(sq))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_5117e062(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (2, (h * w) // 15))
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    while tr < maxtr and succ < nobjs:
        oh = rng.randint(2, 6)
        ow = rng.randint(2, 6)
        bx = asindices(canvas(-1, (oh, ow)))
        nc = rng.randint(3, oh * ow)
        sp = rng.choice(totuple(bx))
        bx = remove(sp, bx)
        obj = {sp}
        for k in range(nc - 1):
            obj.add(rng.choice(totuple((bx - obj) & mapply(neighbors, obj))))
        if not done:
            done = True
            idx = rng.choice(totuple(obj))
            coll = rng.choice(remcols)
            obj2 = {(coll, idx)}
            coll2 = rng.choice(remove(coll, remcols))
            obj3 = recolor(coll2, remove(idx, obj))
            obj = obj2 | obj3
            go = fill(canvas(bgc, shape(obj)), coll2, normalize(obj))
        else:
            obj = recolor(rng.choice(remcols), obj)
        locopts = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        tr += 1
        if len(locopts) == 0:
            continue
        loc = rng.choice(totuple(locopts))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            gi = paint(gi, plcd)
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
    return {"input": gi, "output": go}


def generate_f15e1fac(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    nsps = unifint(rng, diff_lb, diff_ub, (1, (w - 1) // 2))
    ngps = unifint(rng, diff_lb, diff_ub, (1, (h - 1) // 2))
    spsj = sorted(rng.sample(interval(1, w - 1, 1), nsps))
    gpsi = sorted(rng.sample(interval(1, h - 1, 1), ngps))
    ofs = 0
    bgc, linc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, linc, {(0, jj) for jj in spsj})
    gi = fill(gi, 2, {(ii, 0) for ii in gpsi})
    go = tuple(e for e in gi)
    for a, b in zip([0] + gpsi, [x - 1 for x in gpsi] + [h - 1]):
        for jj in spsj:
            go = fill(go, linc, connect((a, jj + ofs), (b, jj + ofs)))
        ofs += 1
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_3906de3d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    oh = unifint(rng, diff_lb, diff_ub, (2, h // 2))
    ow = unifint(rng, diff_lb, diff_ub, (3, w - 2))
    bgc, boxc, linc = rng.sample(cols, 3)
    locj = rng.randint(1, w - ow - 1)
    bx = backdrop(frozenset({(0, locj), (oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, boxc, bx)
    columns_range = range(locj, locj + ow)

    cutoffs = [rng.randint(1, oh - 1) for j in columns_range]
    for jj, co in zip(columns_range, cutoffs):
        gi = fill(gi, bgc, connect((co, jj), (oh - 1, jj)))

    numlns = unifint(rng, diff_lb, diff_ub, (1, ow - 1))
    lnlocs = rng.sample(list(columns_range), numlns)
    go = tuple(e for e in gi)

    for jj, co in zip(columns_range, cutoffs):
        if jj in lnlocs:
            lineh = rng.randint(1, h - co - 1)
            linei = connect((h - lineh, jj), (h - 1, jj))
            lineo = connect((co, jj), (co + lineh - 1, jj))
            gi = fill(gi, linc, linei)
            go = fill(go, linc, lineo)

    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_77fdfe62(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 13))
    w = unifint(rng, diff_lb, diff_ub, (1, 13))
    c1, c2, c3, c4, barc, bgc, inc = rng.sample(cols, 7)
    qd = canvas(bgc, (h, w))
    inds = totuple(asindices(qd))
    fullh = 2 * h + 4
    fullw = 2 * w + 4
    n1 = unifint(rng, diff_lb, diff_ub, (1, h * w))
    n2 = unifint(rng, diff_lb, diff_ub, (1, h * w))
    n3 = unifint(rng, diff_lb, diff_ub, (1, h * w))
    n4 = unifint(rng, diff_lb, diff_ub, (1, h * w))
    i1 = rng.sample(inds, n1)
    i2 = rng.sample(inds, n2)
    i3 = rng.sample(inds, n3)
    i4 = rng.sample(inds, n4)
    gi = canvas(bgc, (2 * h + 4, 2 * w + 4))
    gi = fill(gi, barc, connect((1, 0), (1, fullw - 1)))
    gi = fill(gi, barc, connect((fullh - 2, 0), (fullh - 2, fullw - 1)))
    gi = fill(gi, barc, connect((0, 1), (fullh - 1, 1)))
    gi = fill(gi, barc, connect((0, fullw - 2), (fullh - 1, fullw - 2)))
    gi = fill(gi, c1, {(0, 0)})
    gi = fill(gi, c2, {(0, fullw - 1)})
    gi = fill(gi, c3, {(fullh - 1, 0)})
    gi = fill(gi, c4, {(fullh - 1, fullw - 1)})
    gi = fill(gi, inc, shift(i1, (2, 2)))
    gi = fill(gi, inc, shift(i2, (2, 2 + w)))
    gi = fill(gi, inc, shift(i3, (2 + h, 2)))
    gi = fill(gi, inc, shift(i4, (2 + h, 2 + w)))
    go = canvas(bgc, (2 * h, 2 * w))
    go = fill(go, c1, shift(i1, (0, 0)))
    go = fill(go, c2, shift(i2, (0, w)))
    go = fill(go, c3, shift(i3, (h, 0)))
    go = fill(go, c4, shift(i4, (h, w)))
    return {"input": gi, "output": go}


def generate_d406998b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    bgc, dotc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    itv = interval(0, h, 1)
    for j in range(w):
        nilocs = unifint(rng, diff_lb, diff_ub, (1, h // 2 - 1 if h % 2 == 0 else h // 2))
        ilocs = rng.sample(itv, nilocs)
        locs = {(ii, j) for ii in ilocs}
        gi = fill(gi, dotc, locs)
        go = fill(go, dotc if (j - w) % 2 == 0 else 3, locs)
    return {"input": gi, "output": go}


def generate_694f12f3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(rng, diff_lb, diff_ub, (9, 30))
    w = unifint(rng, diff_lb, diff_ub, (9, 30))
    seploc = rng.randint(4, h - 5)
    bigh = unifint(rng, diff_lb, diff_ub, (4, seploc))
    bigw = unifint(rng, diff_lb, diff_ub, (3, w - 1))
    bigloci = rng.randint(0, seploc - bigh)
    biglocj = rng.randint(0, w - bigw)
    smallmaxh = h - seploc - 1
    smallmaxw = w - 1
    cands = []
    bigsize = bigh * bigw
    for a in range(3, smallmaxh + 1):
        for b in range(3, smallmaxw + 1):
            if a * b < bigsize:
                cands.append((a, b))
    cands = sorted(cands, key=lambda ab: ab[0] * ab[1])
    num = len(cands)
    idx = unifint(rng, diff_lb, diff_ub, (0, num - 1))
    smallh, smallw = cands[idx]
    smallloci = rng.randint(seploc + 1, h - smallh)
    smalllocj = rng.randint(0, w - smallw)
    bgc, sqc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    bigsq = backdrop(frozenset({(bigloci, biglocj), (bigloci + bigh - 1, biglocj + bigw - 1)}))
    smallsq = backdrop(frozenset({(smallloci, smalllocj), (smallloci + smallh - 1, smalllocj + smallw - 1)}))
    gi = fill(gi, sqc, bigsq | smallsq)
    go = fill(gi, 2, backdrop(inbox(bigsq)))
    go = fill(go, 1, backdrop(inbox(smallsq)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_3befdf3e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, numcols)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, ((h * w) // 40)))
    succ = 0
    maxtr = 5 * nobjs
    tr = 0
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        if len(inds) == 0:
            break
        rh = rng.choice((1, 2))
        rw = rng.choice((1, 2))
        fullh = 2 + 3 * rh
        fullw = 2 + 3 * rw
        cands = sfilter(inds, lambda ij: ij[0] <= h - fullh and ij[1] <= w - fullw)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        fullobj = backdrop(frozenset({loc, (loci + fullh - 1, locj + fullw - 1)}))
        if fullobj.issubset(inds):
            succ += 1
            inds = inds - fullobj
            incol, outcol = rng.sample(ccols, 2)
            ofincol = backdrop(frozenset({(loci + rh + 1, locj + rw + 1), (loci + 2 * rh, locj + 2 * rw)}))
            ofoutcol = outbox(ofincol)
            gi = fill(gi, incol, ofincol)
            gi = fill(gi, outcol, ofoutcol)
            go = fill(go, outcol, ofincol)
            go = fill(go, incol, ofoutcol)
            ilocs = apply(first, ofoutcol)
            jlocs = apply(last, ofoutcol)
            ff = lambda ij: ij[0] in ilocs or ij[1] in jlocs
            addon = sfilter(fullobj - (ofincol | ofoutcol), ff)
            go = fill(go, outcol, addon)
    return {"input": gi, "output": go}


def generate_9f236235(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    numh = unifint(rng, diff_lb, diff_ub, (2, 14))
    numw = unifint(rng, diff_lb, diff_ub, (2, 14))
    h = unifint(rng, diff_lb, diff_ub, (1, 31 // numh - 1))
    w = unifint(rng, diff_lb, diff_ub, (1, 31 // numw - 1))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    frontcol = rng.choice(remcols)
    remcols = remove(frontcol, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, min(9, numh * numw)))
    ccols = rng.sample(remcols, numcols)
    numcells = unifint(rng, diff_lb, diff_ub, (1, numh * numw))
    cands = asindices(canvas(-1, (numh, numw)))
    inds = asindices(canvas(-1, (h, w)))
    locs = rng.sample(totuple(cands), numcells)
    gi = canvas(frontcol, (h * numh + numh - 1, w * numw + numw - 1))
    go = canvas(bgc, (numh, numw))
    for cand in cands:
        a, b = cand
        plcd = shift(inds, (a * (h + 1), b * (w + 1)))
        col = rng.choice(remcols) if cand in locs else bgc
        gi = fill(gi, col, plcd)
        go = fill(go, col, {cand})
    go = vmirror(go)
    return {"input": gi, "output": go}


def generate_d8c310e9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    p = unifint(rng, diff_lb, diff_ub, (2, (w - 1) // 3))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, numc)
    obj = set()
    for j in range(p):
        numcells = unifint(rng, diff_lb, diff_ub, (1, h - 1))
        for ii in range(h - 1, h - numcells - 1, -1):
            loc = (ii, j)
            col = rng.choice(ccols)
            cell = (col, loc)
            obj.add(cell)
    gi = canvas(bgc, (h, w))
    minobj = obj | shift(obj, (0, p))
    addonw = rng.randint(0, p)
    addon = sfilter(obj, lambda cij: cij[1][1] < addonw)
    fullobj = minobj | addon
    leftshift = rng.randint(0, addonw)
    fullobj = shift(fullobj, (0, -leftshift))
    gi = paint(gi, fullobj)
    go = tuple(e for e in gi)
    for j in range(w // (2 * p) + 2):
        go = paint(go, shift(fullobj, (0, j * 2 * p)))
    mfs = (identity, rot90, rot180, rot270)
    fn = rng.choice(mfs)
    gi = fn(gi)
    go = fn(go)
    return {"input": gi, "output": go}


def generate_7e0986d6(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    nsqcols = unifint(rng, diff_lb, diff_ub, (1, 5))
    sqcols = rng.sample(remcols, nsqcols)
    remcols = difference(remcols, sqcols)
    nnoisecols = unifint(rng, diff_lb, diff_ub, (1, len(remcols)))
    noisecols = rng.sample(remcols, nnoisecols)
    numsq = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    tr = 0
    maxtr = 5 * numsq
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = rng.randint(2, 7)
        ow = rng.randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - outbox(sq)
            col = rng.choice(sqcols)
            go = fill(go, col, sq)
    gi = tuple(e for e in go)
    namt = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 9))
    cands = asindices(gi)
    for k in range(namt):
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        col = gi[loc[0]][loc[1]]
        torem = neighbors(loc) & ofcolor(gi, col)
        cands = cands - torem
        noisec = rng.choice(noisecols)
        gi = fill(gi, noisec, {loc})
    return {"input": gi, "output": go}


def generate_a64e4611(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (18, 30))
    w = unifint(rng, diff_lb, diff_ub, (18, 30))
    bgc, noisec = rng.sample(cols, 2)
    lb = int(0.4 * h * w)
    ub = int(0.5 * h * w)
    nbgc = unifint(rng, diff_lb, diff_ub, (lb, ub))
    gi = canvas(noisec, (h, w))
    inds = totuple(asindices(gi))
    bgcinds = rng.sample(inds, nbgc)
    gi = fill(gi, bgc, bgcinds)
    sinds = asindices(canvas(-1, (3, 3)))
    bgcf = recolor(bgc, sinds)
    noisecf = recolor(noisec, sinds)
    addn = set()
    addb = set()
    for occ in occurrences(gi, bgcf):
        occi, occj = occ
        addn.add((rng.randint(0, 2) + occi, rng.randint(0, 2) + occj))
    for occ in occurrences(gi, noisecf):
        occi, occj = occ
        addb.add((rng.randint(0, 2) + occi, rng.randint(0, 2) + occj))
    gi = fill(gi, noisec, addn)
    gi = fill(gi, bgc, addb)
    go = tuple(e for e in gi)
    dim = rng.randint(rng.randint(3, 8), 8)
    locj = rng.randint(3, h - dim - 4)
    spi = rng.choice((0, rng.randint(3, h // 2)))
    for j in range(locj, locj + dim):
        ln = connect((spi, j), (h - 1, j))
        gi = fill(gi, bgc, ln)
        go = fill(go, bgc, ln)
    for j in range(locj + 1, locj + dim - 1):
        ln = connect((spi + 1 if spi > 0 else spi, j), (h - 1, j))
        go = fill(go, 3, ln)
    sgns = rng.choice(((-1,), (1,), (-1, 1)))
    startloc = rng.choice((spi, rng.randint(spi + 3, h - 6)))
    hh = rng.randint(3, min(8, h - startloc - 3))
    for sgn in sgns:
        for ii in range(startloc, startloc + hh, 1):
            ln = shoot((ii, locj), (0, sgn))
            gi = fill(gi, bgc, ln)
            go = fill(go, bgc, ln - ofcolor(go, 3))
    for sgn in sgns:
        for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1, 1):
            ln = shoot((ii, locj + dim - 2 if sgn == -1 else locj + 1), (0, sgn))
            go = fill(go, 3, ln)
    if len(sgns) == 1 and unifint(rng, diff_lb, diff_ub, (0, 1)) == 1:
        sgns = (-sgns[0],)
        startloc = rng.choice((spi, rng.randint(spi + 3, h - 6)))
        hh = rng.randint(3, min(8, h - startloc - 3))
        for sgn in sgns:
            for ii in range(startloc, startloc + hh, 1):
                ln = shoot((ii, locj), (0, sgn))
                gi = fill(gi, bgc, ln)
                go = fill(go, bgc, ln - ofcolor(go, 3))
        for sgn in sgns:
            for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1, 1):
                ln = shoot((ii, locj + dim - 2 if sgn == -1 else locj + 1), (0, sgn))
                go = fill(go, 3, ln)
    return {"input": gi, "output": go}


def generate_b782dc8a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    wall_pairs = {"N": "S", "S": "N", "E": "W", "W": "E"}
    dlt = [("W", (-1, 0)), ("E", (1, 0)), ("S", (0, 1)), ("N", (0, -1))]
    walls = {"N": True, "S": True, "E": True, "W": True}
    fullsucc = False
    while True:
        h = unifint(rng, diff_lb, diff_ub, (3, 15))
        w = unifint(rng, diff_lb, diff_ub, (3, 15))
        maze = [[{"x": x, "y": y, "walls": {**walls}} for y in range(h)] for x in range(w)]
        kk = h * w
        stck = []
        cc = maze[0][0]
        nv = 1
        while nv < kk:
            nbhs = []
            for direc, (dx, dy) in dlt:
                x2, y2 = cc["x"] + dx, cc["y"] + dy
                if 0 <= x2 < w and 0 <= y2 < h:
                    neighbour = maze[x2][y2]
                    if all(neighbour["walls"].values()):
                        nbhs.append((direc, neighbour))
            if not nbhs:
                cc = stck.pop()
                continue
            direc, next_cell = rng.choice(nbhs)
            cc["walls"][direc] = False
            next_cell["walls"][wall_pairs[direc]] = False
            stck.append(cc)
            cc = next_cell
            nv += 1
        pathcol, wallcol, dotcol, ncol = rng.sample(cols, 4)
        grid = [[pathcol for x in range(w * 2)]]
        for y in range(h):
            row = [pathcol]
            for x in range(w):
                row.append(wallcol)
                row.append(pathcol if maze[x][y]["walls"]["E"] else wallcol)
            grid.append(row)
            row = [pathcol]
            for x in range(w):
                row.append(pathcol if maze[x][y]["walls"]["S"] else wallcol)
                row.append(pathcol)
            grid.append(row)
        gi = tuple(tuple(r[1:-1]) for r in grid[1:-1])
        objs = objects(gi, T, F, F)
        objs = colorfilter(objs, pathcol)
        objs = sfilter(objs, lambda obj: size(obj) > 4)
        if len(objs) == 0:
            continue
        objs = order(objs, size)
        nobjs = len(objs)
        idx = unifint(rng, diff_lb, diff_ub, (0, nobjs - 1))
        obj = toindices(objs[idx])
        cell = rng.choice(totuple(obj))
        gi = fill(gi, dotcol, {cell})
        nbhs = dneighbors(cell) & ofcolor(gi, pathcol)
        gi = fill(gi, ncol, nbhs)
        obj1 = sfilter(obj, lambda ij: even(manhattan({ij}, {cell})))
        obj2 = obj - obj1
        go = fill(gi, dotcol, obj1)
        go = fill(go, ncol, obj2)
        break
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_af902bf9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numcols)
    numsq = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    maxtr = 5 * numsq
    tr = 0
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = rng.randint(3, 5)
        ow = rng.randint(3, 5)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            inds = inds - sq
            succ += 1
            col = rng.choice(ccols)
            crns = corners(sq)
            gi = fill(gi, col, crns)
            go = fill(go, col, crns)
            ins = backdrop(inbox(crns))
            go = fill(go, 2, ins)
    return {"input": gi, "output": go}


def generate_a87f7484(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    num = unifint(rng, diff_lb, diff_ub, (3, min(30 // h, 9)))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ccols = rng.sample(remcols, num)
    ncd = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    nc = rng.choice((ncd, h * w - ncd))
    nc = min(max(1, nc), h * w - 1)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    origlocs = rng.sample(totuple(inds), nc)
    canbrem = {l for l in origlocs}
    canbeadd = inds - set(origlocs)
    otherlocs = {l for l in origlocs}
    nchangesinv = unifint(rng, diff_lb, diff_ub, (0, h * w - 1))
    nchanges = h * w - nchangesinv
    for k in range(nchanges):
        if rng.choice((True, False)):
            if len(canbrem) > 1:
                ch = rng.choice(totuple(canbrem))
                otherlocs = remove(ch, otherlocs)
                canbrem = remove(ch, canbrem)
            elif len(canbeadd) > 1:
                ch = rng.choice(totuple(canbeadd))
                otherlocs = insert(ch, otherlocs)
                canbeadd = remove(ch, canbeadd)
        else:
            if len(canbeadd) > 1:
                ch = rng.choice(totuple(canbeadd))
                otherlocs = insert(ch, otherlocs)
                canbeadd = remove(ch, canbeadd)
            elif len(canbrem) > 1:
                ch = rng.choice(totuple(canbrem))
                otherlocs = remove(ch, otherlocs)
                canbrem = remove(ch, canbrem)
    go = fill(c, ccols[0], origlocs)
    grids = [go]
    for cc in ccols[1:]:
        grids.append(fill(c, cc, otherlocs))
    rng.shuffle(grids)
    grids = tuple(grids)
    gi = merge(grids)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_fcc82909(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (7, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, w // 3))
    opts = interval(0, w, 1)
    tr = 0
    maxtr = 4 * nobjs
    succ = 0
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    while succ < nobjs and tr < maxtr:
        tr += 1
        sopts = sfilter(opts, lambda j: set(interval(j, j + 2, 1)).issubset(opts))
        if len(sopts) == 0:
            break
        numc = unifint(rng, diff_lb, diff_ub, (1, 4))
        jstart = rng.choice(sopts)
        opts = remove(jstart, opts)
        opts = remove(jstart + 1, opts)
        options = interval(0, h - 2 - numc + 1, 1)
        if len(options) == 0:
            break
        iloc = rng.choice(options)
        ccols = rng.sample(remcols, numc)
        bd = backdrop(frozenset({(iloc, jstart), (iloc + 1, jstart + 1)}))
        bd = list(bd)
        rng.shuffle(bd)
        obj = {(c, ij) for c, ij in zip(ccols, bd[:numc])} | {(rng.choice(ccols), ij) for ij in bd[numc:]}
        if not mapply(dneighbors, toindices(obj)).issubset(ofcolor(gi, bgc)):
            continue
        gi = paint(gi, obj)
        go = paint(go, obj)
        for k in range(numc):
            go = fill(go, 3, {(iloc + k + 2, jstart), (iloc + k + 2, jstart + 1)})
    return {"input": gi, "output": go}


def generate_d9fac9be(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    bgc, noisec, ringc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))

    # Generate noise patterns
    nnoise1 = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 3 - 1))
    nnoise2 = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // 3 - 9)))
    inds = asindices(gi)
    noise1 = rng.sample(totuple(inds), nnoise1)
    noise2 = rng.sample(difference(totuple(inds), noise1), nnoise2)
    gi = fill(gi, noisec, noise1)
    gi = fill(gi, ringc, noise2)

    neighbor_pattern = neighbors((1, 1))
    fp1 = recolor(noisec, neighbor_pattern)
    fp2 = recolor(ringc, neighbor_pattern)

    fp1occ = occurrences(gi, fp1)
    fp2occ = occurrences(gi, fp2)

    for occ1 in fp1occ:
        loc = rng.choice(totuple(shift(neighbor_pattern, occ1)))
        gi = fill(gi, rng.choice((bgc, ringc)), {loc})

    for occ2 in fp2occ:
        loc = rng.choice(totuple(shift(neighbor_pattern, occ2)))
        gi = fill(gi, rng.choice((bgc, noisec)), {loc})

    loci = rng.randint(0, h - 3)
    locj = rng.randint(0, w - 3)
    ringp = shift(neighbor_pattern, (loci, locj))
    gi = fill(gi, ringc, ringp)
    gi = fill(gi, noisec, {(loci + 1, locj + 1)})

    go = canvas(noisec, (1, 1))
    return {"input": gi, "output": go}


def generate_eb281b96(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 8))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, numc)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    ncells = unifint(rng, diff_lb, diff_ub, (1, h * w))
    locs = rng.sample(totuple(inds), ncells)
    obj = {(rng.choice(ccols), ij) for ij in locs}
    gi = paint(c, obj)
    go = vconcat(gi, hmirror(gi[:-1]))
    go = vconcat(go, hmirror(go[:-1]))
    return {"input": gi, "output": go}


def generate_d43fd935(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    boxh = unifint(rng, diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(rng, diff_lb, diff_ub, (2, w // 2))
    loci = rng.randint(0, h - boxh)
    locj = rng.randint(0, w - boxw)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ccol = rng.choice(remcols)
    remcols = remove(ccol, remcols)
    ndcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    dcols = rng.sample(remcols, ndcols)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - bd)
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(rng, diff_lb, diff_ub, (0, noiseb))
    noise = rng.sample(reminds, nnoise)
    truenoise = sfilter(
        noise, lambda ij: (ij[0] < loci or ij[0] > loci + boxh - 1) and (ij[1] < locj or ij[1] > locj + boxw - 1)
    )
    rem = difference(noise, truenoise)
    top = sfilter(rem, lambda ij: ij[0] < loci)
    bottom = sfilter(rem, lambda ij: ij[0] > loci + boxh - 1)
    left = sfilter(rem, lambda ij: ij[1] < locj)
    right = sfilter(rem, lambda ij: ij[1] > locj + boxw - 1)
    truenoiseobj = {(rng.choice(dcols), ij) for ij in truenoise}
    gi = paint(gi, truenoiseobj)
    go = tuple(e for e in gi)
    for jj in apply(last, top):
        col = rng.choice(dcols)
        mf = matcher(last, jj)
        subs = sfilter(top, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmin(subs, first), jj), (loci - 1, jj)))
    for jj in apply(last, bottom):
        col = rng.choice(dcols)
        mf = matcher(last, jj)
        subs = sfilter(bottom, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmax(subs, first), jj), (loci + boxh, jj)))
    for ii in apply(first, left):
        col = rng.choice(dcols)
        mf = matcher(first, ii)
        subs = sfilter(left, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmin(subs, last)), (ii, locj - 1)))
    for ii in apply(first, right):
        col = rng.choice(dcols)
        mf = matcher(first, ii)
        subs = sfilter(right, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmax(subs, last)), (ii, locj + boxw)))
    return {"input": gi, "output": go}


def generate_44f52bb0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    numcells = unifint(rng, diff_lb, diff_ub, (1, h * w - 1))
    inds = asindices(gi)
    while gi == hmirror(gi):
        cells = rng.sample(totuple(inds), numcells)
        gi = canvas(bgc, (h, w))
        for ij in cells:
            a, b = ij
            col = rng.choice(ccols)
            gi = fill(gi, col, {ij})
            gi = fill(gi, col, {(a, w - 1 - b)})
    issymm = rng.choice((True, False))
    if not issymm:
        numpert = unifint(rng, diff_lb, diff_ub, (1, h * (w // 2)))
        cands = asindices(canvas(-1, (h, w // 2)))
        locs = rng.sample(totuple(cands), numpert)
        for a, b in locs:
            col = gi[a][b]
            newcol = rng.choice(totuple(remove(col, insert(bgc, set(ccols)))))
            gi = fill(gi, newcol, {(a, b)})
        go = canvas(7, (1, 1))
    else:
        go = canvas(1, (1, 1))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_d22278a0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    crns = corners(inds)
    ncorns = unifint(rng, diff_lb, diff_ub, (1, 4))
    crns = rng.sample(totuple(crns), ncorns)
    ccols = rng.sample(remcols, ncorns)
    for col, crn in zip(ccols, crns):
        gi = fill(gi, col, {crn})
        go = fill(go, col, {crn})
        rings = {crn}
        for k in range(1, max(h, w) // 2 + 2, 1):
            rings = rings | outbox(outbox(rings))
        if len(crns) > 1:
            ff = lambda ij: manhattan({ij}, {crn}) < min(
                apply(rbind(manhattan, {ij}), apply(initset, remove(crn, crns)))
            )
        else:
            ff = lambda ij: True
        locs = sfilter(inds, ff) & rings
        go = fill(go, col, locs)
    return {"input": gi, "output": go}


def generate_272f95fa(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4, 6))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc, linc = rng.sample(cols, 2)
    c = canvas(bgc, (5, 5))
    l1 = connect((1, 0), (1, 4))
    l2 = connect((3, 0), (3, 4))
    lns = l1 | l2
    gi = fill(dmirror(fill(c, linc, lns)), linc, lns)
    hdist = [0, 0, 0]
    wdist = [0, 0, 0]
    idx = 0
    for k in range(h - 2):
        hdist[idx] += 1
        idx = (idx + 1) % 3
    for k in range(w - 2):
        wdist[idx] += 1
        idx = (idx + 1) % 3
    rng.shuffle(hdist)
    rng.shuffle(wdist)
    hdelt1 = unifint(rng, diff_lb, diff_ub, (0, hdist[0] - 1))
    hdist[0] -= hdelt1
    hdist[1] += hdelt1
    hdelt2 = unifint(rng, diff_lb, diff_ub, (0, min(hdist[1], hdist[2]) - 1))
    hdelt2 = rng.choice((+hdelt2, -hdelt2))
    hdist[1] += hdelt2
    hdist[2] -= hdelt2
    wdelt1 = unifint(rng, diff_lb, diff_ub, (0, wdist[0] - 1))
    wdist[0] -= wdelt1
    wdist[1] += wdelt1
    wdelt2 = unifint(rng, diff_lb, diff_ub, (0, min(wdist[1], wdist[2]) - 1))
    wdelt2 = rng.choice((+wdelt2, -wdelt2))
    wdist[1] += wdelt2
    wdist[2] -= wdelt2
    gi = gi[:1] * hdist[0] + gi[1:2] + gi[2:3] * hdist[1] + gi[3:4] + gi[4:5] * hdist[2]
    gi = dmirror(gi)
    gi = gi[:1] * wdist[0] + gi[1:2] + gi[2:3] * wdist[1] + gi[3:4] + gi[4:5] * wdist[2]
    gi = dmirror(gi)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
    objs = objects(gi, T, T, F)
    bgobjs = colorfilter(objs, bgc)
    cnrs = corners(asindices(gi))
    bgobjs = sfilter(bgobjs, lambda o: len(toindices(o) & cnrs) == 0)
    pinkobj = extract(bgobjs, lambda o: not bordering(o, gi))
    yellobj = argmin(bgobjs, leftmost)
    greenobj = argmax(bgobjs, rightmost)
    redobj = argmin(bgobjs, uppermost)
    blueobj = argmax(bgobjs, lowermost)
    go = fill(gi, 6, pinkobj)
    go = fill(go, 4, yellobj)
    go = fill(go, 3, greenobj)
    go = fill(go, 2, redobj)
    go = fill(go, 1, blueobj)
    return {"input": gi, "output": go}


def generate_5c0a986e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    nobjs = unifint(rng, diff_lb, diff_ub, (2, (h * w) // 10))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    tr = 0
    maxtr = 5 * nobjs
    succ = 0
    inds = asindices(gi)
    fullinds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: 0 < ij[0] <= h - 3 and 0 < ij[1] <= w - 3)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        col = rng.choice((1, 2))
        sq = {(loc), add(loc, (0, 1)), add(loc, (1, 0)), add(loc, (1, 1))}
        if col == 1:
            obj = sq | (shoot(loc, (-1, -1)) & fullinds)
        else:
            obj = sq | (shoot(loc, (1, 1)) & fullinds)
        if obj.issubset(inds):
            succ += 1
            inds = (inds - obj) - mapply(dneighbors, sq)
            gi = fill(gi, col, sq)
            go = fill(go, col, obj)
    return {"input": gi, "output": go}


def generate_9af7a82c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    prods = dict()
    for a in range(1, 31, 1):
        for b in range(1, 31, 1):
            prd = a * b
            if prd in prods:
                prods[prd].append((a, b))
            else:
                prods[prd] = [(a, b)]
    ncols = unifint(rng, diff_lb, diff_ub, (2, 9))
    leastnc = sum(range(1, ncols + 1, 1))
    maxnc = sum(range(30, 30 - ncols, -1))
    cands = {k: v for k, v in prods.items() if leastnc <= k <= maxnc}
    options = set()
    for v in cands.values():
        for opt in v:
            options.add(opt)
    options = sorted(options, key=lambda ij: ij[0] * ij[1])
    idx = unifint(rng, diff_lb, diff_ub, (0, len(options) - 1))
    h, w = options[idx]
    ccols = rng.sample(cols, ncols)
    counts = list(range(1, ncols + 1, 1))
    eliginds = {ncols - 1}
    while sum(counts) < h * w:
        eligindss = sorted(eliginds, reverse=True)
        idx = unifint(rng, diff_lb, diff_ub, (0, len(eligindss) - 1))
        idx = eligindss[idx]
        counts[idx] += 1
        if idx > 0:
            eliginds.add(idx - 1)
        if idx < ncols - 1:
            if counts[idx] == counts[idx + 1] - 1:
                eliginds = eliginds - {idx}
        if counts[idx] == 30:
            eliginds = eliginds - {idx}
    gi = canvas(-1, (h, w))
    go = canvas(0, (max(counts), ncols))
    inds = asindices(gi)
    counts = counts[::-1]
    for j, (col, cnt) in enumerate(zip(ccols, counts)):
        locs = rng.sample(totuple(inds), cnt)
        gi = fill(gi, col, locs)
        inds = inds - set(locs)
        go = fill(go, col, connect((0, j), (cnt - 1, j)))
    return {"input": gi, "output": go}


def generate_d4469b4b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    canv = canvas(5, (3, 3))
    A = fill(canv, 0, {(1, 0), (2, 0), (1, 2), (2, 2)})
    B = fill(canv, 0, corners(asindices(canv)))
    C = fill(canv, 0, {(0, 0), (0, 1), (1, 0), (1, 1)})
    colabc = ((2, A), (1, B), (3, C))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    col, go = rng.choice(colabc)
    gi = canvas(col, (h, w))
    inds = asindices(gi)
    numc = unifint(rng, diff_lb, diff_ub, (1, 7))
    ccols = rng.sample(cols, numc)
    numcells = unifint(rng, diff_lb, diff_ub, (0, h * w - 1))
    locs = rng.sample(totuple(inds), numcells)
    otherobj = {(rng.choice(ccols), ij) for ij in locs}
    gi = paint(gi, otherobj)
    return {"input": gi, "output": go}


def generate_bdad9b1f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    numh = unifint(rng, diff_lb, diff_ub, (1, h // 2 - 1))
    numw = unifint(rng, diff_lb, diff_ub, (1, w // 2 - 1))
    hlocs = rng.sample(interval(2, h - 1, 1), numh)
    wlocs = rng.sample(interval(2, w - 1, 1), numw)
    numcols = unifint(rng, diff_lb, diff_ub, (2, 8))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ccols = rng.sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    fc = -1
    for ii in sorted(hlocs):
        col = rng.choice(remove(fc, ccols))
        fc = col
        objw = rng.randint(2, ii)
        gi = fill(gi, col, connect((ii, 0), (ii, objw - 1)))
        go = fill(go, col, connect((ii, 0), (ii, w - 1)))
    fc = -1
    for jj in sorted(wlocs):
        col = rng.choice(remove(fc, ccols))
        fc = col
        objh = rng.randint(2, jj)
        gi = fill(gi, col, connect((0, jj), (objh - 1, jj)))
        go = fill(go, col, connect((0, jj), (h - 1, jj)))
    yells = product(set(hlocs), set(wlocs))
    go = fill(go, 4, yells)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_3345333e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    oh = unifint(rng, diff_lb, diff_ub, (4, h - 2))
    ow = unifint(rng, diff_lb, diff_ub, (4, (w - 2) // 2))
    nc = unifint(rng, diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))
    shp = {(0, 0)}
    bounds = asindices(canvas(-1, (oh, ow)))
    for j in range(nc):
        ij = rng.choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    while height(shp) < 3 or width(shp) < 3:
        ij = rng.choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    vmshp = vmirror(shp)
    if rng.choice((True, False)):
        vmshp = sfilter(vmshp, lambda ij: ij[1] != width(shp) - 1)
    shp = normalize(combine(shp, shift(vmshp, (0, -width(vmshp)))))
    oh, ow = shape(shp)
    bgc, objc, occcol = rng.sample(cols, 3)
    loci = rng.randint(1, h - oh - 1)
    locj = rng.randint(1, w - ow - 1)
    loc = (loci, locj)
    shp = shift(shp, loc)
    c = canvas(bgc, (h, w))
    go = fill(c, objc, shp)
    boxh = unifint(rng, diff_lb, diff_ub, (2, oh - 1))
    boxw = unifint(rng, diff_lb, diff_ub, (2, ow // 2))
    ulci = rng.randint(loci - 1, loci + oh - boxh + 1)
    ulcj = rng.randint(locj + ow // 2 + 1, locj + ow - boxw + 1)
    bx = backdrop(frozenset({(ulci, ulcj), (ulci + boxh - 1, ulcj + boxw - 1)}))
    gi = fill(go, occcol, bx)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_253bf280(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, dim_bounds)
    w = unifint(rng, diff_lb, diff_ub, dim_bounds)
    bgc = rng.choice(colopts)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    card_bounds = (0, max(1, (h * w) // 4))
    num = unifint(rng, diff_lb, diff_ub, card_bounds)
    s = rng.sample(inds, num)
    fgcol = rng.choice(remove(bgc, colopts))
    gi = fill(c, fgcol, s)
    resh = frozenset()
    for x, r in enumerate(gi):
        if r.count(fgcol) > 1:
            resh = combine(resh, connect((x, r.index(fgcol)), (x, -1 + w - r[::-1].index(fgcol))))
    go = fill(c, 3, resh)
    resv = frozenset()
    for x, r in enumerate(dmirror(gi)):
        if r.count(fgcol) > 1:
            resv = combine(resv, connect((x, r.index(fgcol)), (x, -1 + h - r[::-1].index(fgcol))))
    go = dmirror(fill(dmirror(go), 3, resv))
    go = fill(go, fgcol, s)
    return {"input": gi, "output": go}


def generate_5582e5ca(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    numc = unifint(rng, diff_lb, diff_ub, (2, min(10, h * w - 1)))
    ccols = rng.sample(colopts, numc)
    mostc = ccols[0]
    remcols = ccols[1:]
    leastnummostcol = (h * w) // numc + 1
    maxnummostcol = h * w - numc + 1
    nummostcold = unifint(rng, diff_lb, diff_ub, (0, maxnummostcol - leastnummostcol))
    nummostcol = min(max(leastnummostcol, maxnummostcol - nummostcold), maxnummostcol)
    kk = len(remcols)
    remcount = h * w - nummostcol - kk
    remcounts = [1 for k in range(kk)]
    for j in range(remcount):
        cands = [idx for idx, c in enumerate(remcounts) if c < nummostcol - 1]
        if len(cands) == 0:
            break
        idx = rng.choice(cands)
        remcounts[idx] += 1
    nummostcol = h * w - sum(remcounts)
    gi = canvas(-1, (h, w))
    inds = asindices(gi)
    mclocs = rng.sample(totuple(inds), nummostcol)
    gi = fill(gi, mostc, mclocs)
    go = canvas(mostc, (h, w))
    inds = inds - set(mclocs)
    for col, count in zip(remcols, remcounts):
        locs = rng.sample(totuple(inds), count)
        inds = inds - set(locs)
        gi = fill(gi, col, locs)
    return {"input": gi, "output": go}


def generate_a1570a43(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    oh = unifint(rng, diff_lb, diff_ub, (3, h))
    ow = unifint(rng, diff_lb, diff_ub, (3, w))
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    crns = {(loci, locj), (loci + oh - 1, locj), (loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1)}
    cands = shift(asindices(canvas(-1, (oh - 2, ow - 2))), (loci + 1, locj + 1))
    bgc, dotc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(dotc, cols))
    numc = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numc)
    gipro = canvas(bgc, (h, w))
    gipro = fill(gipro, dotc, crns)
    sp = rng.choice(totuple(cands))
    obj = {sp}
    cands = remove(sp, cands)
    ncells = unifint(rng, diff_lb, diff_ub, (oh + ow - 5, max(oh + ow - 5, ((oh - 2) * (ow - 2)) // 2)))
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((cands - obj) & mapply(neighbors, obj))))
    while shape(obj) != (oh - 2, ow - 2):
        obj.add(rng.choice(totuple((cands - obj) & mapply(neighbors, obj))))
    obj = {(rng.choice(ccols), ij) for ij in obj}
    go = paint(gipro, obj)
    nperts = unifint(rng, diff_lb, diff_ub, (1, max(h, w)))
    k = 0
    fullinds = asindices(go)
    while ulcorner(obj) == (loci + 1, locj + 1) or k < nperts:
        k += 1
        options = sfilter(
            neighbors((0, 0)),
            lambda ij: len(crns & shift(toindices(obj), ij)) == 0 and shift(toindices(obj), ij).issubset(fullinds),
        )
        direc = rng.choice(totuple(options))
        obj = shift(obj, direc)
    gi = paint(gipro, obj)
    return {"input": gi, "output": go}


def generate_f5b8619d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 15))
    w = unifint(rng, diff_lb, diff_ub, (2, 15))
    ncells = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    locs = rng.sample(totuple(inds), ncells)
    blockcol = rng.randint(0, w - 1)
    locs = sfilter(locs, lambda ij: ij[1] != blockcol)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numcols)
    obj = frozenset({(rng.choice(ccols), ij) for ij in locs})
    gi = paint(gi, obj)
    go = fill(gi, 8, mapply(vfrontier, set(locs)) & (inds - set(locs)))
    go = hconcat(go, go)
    go = vconcat(go, go)
    return {"input": gi, "output": go}


def generate_444801d8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 25))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (2, 9))
    ccols = rng.sample(remcols, numcols)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = rng.randint(4, 6)
        ow = 5
        bx = box({(1, 0), (oh - 1, 4)}) - {(1, 2)}
        fullobj = backdrop({(0, 0), (oh - 1, 4)})
        cands = backdrop(bx) - bx
        dot = rng.choice(totuple(cands))
        dcol, bxcol = rng.sample(ccols, 2)
        inobj = recolor(bxcol, bx) | recolor(dcol, {dot})
        outobj = recolor(bxcol, bx) | recolor(dcol, fullobj - bx)
        if rng.choice((True, False)):
            inobj = shift(hmirror(inobj), UP)
            outobj = hmirror(outobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        outplcd = shift(outobj, loc)
        outplcdi = toindices(outplcd)
        if outplcdi.issubset(inds):
            succ += 1
            inplcd = shift(inobj, loc)
            inds = (inds - outplcdi) - outbox(inplcd)
            gi = paint(gi, inplcd)
            go = paint(go, outplcd)
    return {"input": gi, "output": go}


def generate_00d62c1b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc, fgc = rng.sample(cols, 2)
    gi = canvas(bgc, (h, w))
    nblocks = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nblocks
    inds = asindices(gi)
    while succ < nblocks and tr < maxtr:
        tr += 1
        oh = rng.randint(3, 8)
        ow = rng.randint(3, 8)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bx = bx - set(rng.sample(totuple(corners(bx)), rng.randint(0, 4)))
        if bx.issubset(inds) and len(inds - bx) > (h * w) // 2 + 1:
            gi = fill(gi, fgc, bx)
            succ += 1
            inds = inds - bx
    maxnnoise = max(0, (h * w) // 2 - 1 - colorcount(gi, fgc))
    namt = unifint(rng, diff_lb, diff_ub, (0, maxnnoise))
    noise = rng.sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    objs = objects(gi, T, F, F)
    cands = colorfilter(objs, bgc)
    res = mfilter(cands, compose(flip, rbind(bordering, gi)))
    go = fill(gi, 4, res)
    return {"input": gi, "output": go}


def generate_10fcaaa3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 15))
    w = unifint(rng, diff_lb, diff_ub, (2, 15))
    ncells = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // 6)))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ccols = rng.sample(remcols, ncols)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = frozenset(rng.sample(totuple(inds), ncells))
    obj = frozenset({(rng.choice(ccols), ij) for ij in locs})
    gi = paint(c, obj)
    go = hconcat(gi, gi)
    go = vconcat(go, go)
    fullocs = locs | shift(locs, (0, w)) | shift(locs, (h, 0)) | shift(locs, (h, w))
    nbhs = mapply(ineighbors, fullocs)
    topaint = nbhs & ofcolor(go, bgc)
    go = fill(go, 8, topaint)
    return {"input": gi, "output": go}


def generate_1a07d186(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    nlines = unifint(rng, diff_lb, diff_ub, (1, w // 5))
    linecols = rng.sample(remcols, nlines)
    remcols = difference(remcols, linecols)
    nnoisecols = unifint(rng, diff_lb, diff_ub, (0, len(remcols)))
    noisecols = rng.sample(remcols, nnoisecols)
    locopts = interval(0, w, 1)
    locs = []
    for k in range(nlines):
        if len(locopts) == 0:
            break
        loc = rng.choice(locopts)
        locopts = difference(locopts, interval(loc - 2, loc + 3, 1))
        locs.append(loc)
    locs = sorted(locs)
    nlines = len(locs)
    linecols = linecols[:nlines]
    gi = canvas(bgc, (h, w))
    for loc, col in zip(locs, linecols):
        gi = fill(gi, col, connect((0, loc), (h - 1, loc)))
    go = tuple(e for e in gi)
    nilocs = unifint(rng, diff_lb, diff_ub, (1, h))
    ilocs = rng.sample(interval(0, h, 1), nilocs)
    dotlocopts = difference(interval(0, w, 1), locs)
    for ii in ilocs:
        ndots = unifint(rng, diff_lb, diff_ub, (1, min(nlines + nnoisecols, (w - nlines) // 2 - 1)))
        dotlocs = rng.sample(dotlocopts, ndots)
        dotcols = rng.sample(totuple(set(linecols) | set(noisecols)), ndots)
        for dotlocj, col in zip(dotlocs, dotcols):
            gi = fill(gi, col, {(ii, dotlocj)})
            if col in linecols:
                idx = linecols.index(col)
                linelocj = locs[idx]
                if dotlocj > linelocj:
                    go = fill(go, col, {(ii, linelocj + 1)})
                else:
                    go = fill(go, col, {(ii, linelocj - 1)})
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_83302e8f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3, 4))
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    nh = unifint(rng, diff_lb, diff_ub, (3, 30 // (h + 1)))
    nw = unifint(rng, diff_lb, diff_ub, (3, 30 // (w + 1)))
    bgc, linc = rng.sample(cols, 2)
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h + 1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w + 1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    ofc = ofcolor(gi, linc)
    dots = sfilter(ofc, lambda ij: dneighbors(ij).issubset(ofc))
    tmp = fill(gi, bgc, dots)
    lns = apply(toindices, colorfilter(objects(tmp, T, F, F), linc))
    dts = apply(initset, dots)
    cands = lns | dts
    nbreaks = unifint(rng, diff_lb, diff_ub, (0, len(cands) // 2))
    breaklocs = set()
    breakobjs = rng.sample(totuple(cands), nbreaks)
    for breakobj in breakobjs:
        loc = rng.choice(totuple(breakobj))
        breaklocs.add(loc)
    gi = fill(gi, bgc, breaklocs)
    objs = objects(gi, T, F, F)
    objs = colorfilter(objs, bgc)
    objs = sfilter(objs, lambda o: len(o) == h * w)
    res = toindices(merge(objs))
    go = fill(gi, 3, res)
    go = replace(go, bgc, 4)
    return {"input": gi, "output": go}


def generate_98cf29f8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    objh = unifint(rng, diff_lb, diff_ub, (2, h - 5))
    objw = unifint(rng, diff_lb, diff_ub, (2, w - 5))
    loci = rng.randint(0, h - objh)
    locj = rng.randint(0, w - objw)
    loc = (loci, locj)
    obj = backdrop(frozenset({(loci, locj), (loci + objh - 1, locj + objw - 1)}))
    bgc, objc, otherc = rng.sample(cols, 3)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    bmarg = h - (loci + objh)
    rmarg = w - (locj + objw)
    tmarg = loci
    lmarg = locj
    margs = (bmarg, rmarg, tmarg, lmarg)
    options = [idx for idx, marg in enumerate(margs) if marg > 2]
    pos = rng.choice(options)
    for k in range(pos):
        gi = rot90(gi)
    h, w = shape(gi)
    ofc = ofcolor(gi, objc)
    locis = rng.randint(lowermost(ofc) + 2, h - 2)
    locie = rng.randint(locis + 1, h - 1)
    locjs = rng.randint(0, min(w - 2, rightmost(ofc)))
    locje = rng.randint(max(locjs + 1, leftmost(ofc)), w - 1)
    otherobj = backdrop(frozenset({(locis, locjs), (locie, locje)}))
    ub = min(rightmost(ofc), rightmost(otherobj))
    lb = max(leftmost(ofc), leftmost(otherobj))
    jloc = rng.randint(lb, ub)
    ln = connect((lowermost(ofc) + 1, jloc), (uppermost(otherobj) - 1, jloc))
    gib = tuple(e for e in gi)
    gi = fill(gi, otherc, otherobj)
    gi = fill(gi, otherc, ln)
    go = fill(gib, otherc, shift(otherobj, (-len(ln), 0)))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_1f85a75f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    oh = rng.randint(3, min(8, h // 2))
    ow = rng.randint(3, min(8, w // 2))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = rng.randint(max(oh, ow), oh * ow)
    sp = rng.choice(totuple(bounds))
    obj = {sp}
    cands = remove(sp, bounds)
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    bgc, objc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(objc, cols))
    numc = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, numc)
    nnoise = unifint(rng, diff_lb, diff_ub, (0, max(0, ((h * w) - len(backdrop(obj))) // 4)))
    gi = canvas(bgc, (h, w))
    obj = shift(obj, (loci, locj))
    gi = fill(gi, objc, obj)
    inds = asindices(gi)
    noisecells = rng.sample(totuple(inds - backdrop(obj)), nnoise)
    noiseobj = frozenset({(rng.choice(ccols), ij) for ij in noisecells})
    gi = paint(gi, noiseobj)
    go = fill(canvas(bgc, (oh, ow)), objc, normalize(obj))
    return {"input": gi, "output": go}


def generate_8eb1be9a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (8, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    oh = unifint(rng, diff_lb, diff_ub, (2, h // 3))
    ow = unifint(rng, diff_lb, diff_ub, (2, w))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = unifint(rng, diff_lb, diff_ub, (2, (oh * ow) // 3 * 2))
    obj = normalize(frozenset(rng.sample(totuple(bounds), ncells)))
    oh, ow = shape(obj)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(remcols, ncols)
    obj = frozenset({(rng.choice(ccols), ij) for ij in obj})
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    obj = shift(obj, (loci, locj))
    c = canvas(bgc, (h, w))
    gi = paint(c, obj)
    go = paint(c, obj)
    for k in range(h // oh + 1):
        go = paint(go, shift(obj, (-oh * k, 0)))
        go = paint(go, shift(obj, (oh * k, 0)))
    return {"input": gi, "output": go}


def generate_ba26e723(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 6))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    gi = canvas(0, (h, w))
    go = canvas(0, (h, w))
    opts = interval(0, h, 1)
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(cols, ncols)
    for j in range(w):
        nc = unifint(rng, diff_lb, diff_ub, (1, h - 1))
        locs = rng.sample(opts, nc)
        obj = frozenset({(rng.choice(ccols), (ii, j)) for ii in locs})
        gi = paint(gi, obj)
        if j % 3 == 0:
            obj = recolor(6, obj)
        go = paint(go, obj)
    return {"input": gi, "output": go}


def generate_25d487eb(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (2, 8))
    ccols = rng.sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 30))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    inds = asindices(go)
    while tr < maxtr and succ < nobjs:
        if len(inds) == 0:
            break
        tr += 1
        dim = rng.randint(1, 3)
        obj = backdrop(frozenset({(0, 0), (dim, dim)}))
        obj = sfilter(obj, lambda ij: ij[0] <= ij[1])
        obj = obj | shift(vmirror(obj), (0, dim))
        mp = {(0, dim)}
        tric, linc = rng.sample(ccols, 2)
        inobj = recolor(tric, obj - mp) | recolor(linc, mp)
        loc = rng.choice(totuple(inds))
        iplcd = shift(inobj, loc)
        loci, locj = loc
        oplcd = iplcd | recolor(linc, connect((loci, locj + dim), (h - 1, locj + dim)) - toindices(iplcd))
        fullinds = asindices(gi)
        oplcdi = toindices(oplcd)
        if oplcdi.issubset(inds):
            succ += 1
            gi = paint(gi, iplcd)
            go = paint(go, oplcd)
        rotf = rng.choice((identity, rot90, rot180, rot270))
        gi = rotf(gi)
        go = rotf(go)
        h, w = shape(gi)
        ofc = ofcolor(go, bgc)
        inds = ofc - mapply(dneighbors, asindices(go) - ofc)
    return {"input": gi, "output": go}


def generate_4be741c5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    numcolors = unifint(rng, diff_lb, diff_ub, (2, w // 3))
    ccols = rng.sample(cols, numcolors)
    go = (tuple(ccols),)
    gi = merge(tuple(repeat(repeat(c, h), 3) for c in ccols))
    while len(gi) < w:
        idx = rng.randint(0, len(gi) - 1)
        gi = gi[:idx] + gi[idx : idx + 1] + gi[idx:]
    gi = dmirror(gi)
    ndisturbances = unifint(rng, diff_lb, diff_ub, (0, 3 * h * numcolors))
    for k in range(ndisturbances):
        options = []
        for a in range(h):
            for b in range(w - 3):
                if gi[a][b] == gi[a][b + 1] and gi[a][b + 2] == gi[a][b + 3]:
                    options.append((a, b, gi[a][b], gi[a][b + 2]))
        if len(options) == 0:
            break
        a, b, c1, c2 = rng.choice(options)
        if rng.choice((True, False)):
            gi = fill(gi, c2, {(a, b + 1)})
        else:
            gi = fill(gi, c1, {(a, b + 2)})
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_e509e548(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 6))
    getL = lambda h, w: connect((0, 0), (h - 1, 0)) | connect((0, 0), (0, w - 1))
    getU = (
        lambda h, w: connect((0, 0), (0, w - 1))
        | connect((0, 0), (rng.randint(1, h - 1), 0))
        | connect((0, w - 1), (rng.randint(1, h - 1), w - 1))
    )
    getH = lambda h, w: connect((0, 0), (0, w - 1)) | shift(
        connect((0, 0), (h - 1, 0)) | connect((h - 1, 0), (h - 1, rng.randint(1, w - 1))), (0, rng.randint(1, w - 2))
    )
    minshp_getter_pairs = ((2, 2, getL), (2, 3, getU), (3, 3, getH))
    colmapper = {getL: 1, getU: 6, getH: 2}
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (1, 6))
    ccols = rng.sample(remcols, ncols)
    nobjs = unifint(rng, diff_lb, diff_ub, (3, (h * w) // 10))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        minh, minw, getter = rng.choice(minshp_getter_pairs)
        oh = rng.randint(minh, 6)
        ow = rng.randint(minw, 6)
        obj = getter(oh, ow)
        mfs = (identity, dmirror, cmirror, vmirror, hmirror)
        nmfs = rng.choice((1, 2))
        for fn in rng.sample(mfs, nmfs):
            obj = fn(obj)
            obj = normalize(obj)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            col = rng.choice(ccols)
            gi = fill(gi, col, plcd)
            go = fill(go, colmapper[getter], plcd)
    return {"input": gi, "output": go}


def generate_810b9b61(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3,))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(rng, diff_lb, diff_ub, (1, 6))
    ccols = rng.sample(remcols, ncols)
    nobjs = unifint(rng, diff_lb, diff_ub, (3, (h * w) // 10))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = rng.randint(3, 5)
        ow = rng.randint(3, 5)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        loci, locj = loc
        obj = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        mfs = (identity, dmirror, cmirror, vmirror, hmirror)
        nmfs = rng.choice((1, 2))
        for fn in rng.sample(mfs, nmfs):
            obj = fn(obj)
            obj = normalize(obj)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        if rng.choice((True, False)):
            ninobjc = unifint(rng, diff_lb, diff_ub, (1, len(plcd) - 1))
            inobj = frozenset(rng.sample(totuple(plcd), ninobjc))
        else:
            inobj = plcd
        if inobj.issubset(inds):
            succ += 1
            inds = (inds - inobj) - mapply(dneighbors, inobj)
            col = rng.choice(ccols)
            gi = fill(gi, col, inobj)
            go = fill(go, 3 if box(inobj) == inobj and min(shape(inobj)) > 2 else col, inobj)
    return {"input": gi, "output": go}


def generate_6d0160f0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (4,))
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    nh, nw = h, w
    bgc, linc = rng.sample(cols, 2)
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h + 1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w + 1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    noccs = unifint(rng, diff_lb, diff_ub, (1, h * w))
    denseinds = asindices(canvas(-1, (h, w)))
    sparseinds = {(a * (h + 1), b * (w + 1)) for a, b in denseinds}
    locs = rng.sample(totuple(sparseinds), noccs)
    trgtl = rng.choice(locs)
    remlocs = remove(trgtl, locs)
    ntrgt = unifint(rng, diff_lb, diff_ub, (1, (h * w - 1)))
    place = rng.choice(totuple(denseinds))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 9))
    ccols = rng.sample(cols, ncols)
    candss = totuple(remove(place, denseinds))
    trgrem = rng.sample(candss, ntrgt)
    trgrem = {(rng.choice(ccols), ij) for ij in trgrem}
    trgtobj = {(4, place)} | trgrem
    go = paint(gi, shift(sfilter(trgtobj, lambda cij: cij[0] != linc), multiply(place, increment((h, w)))))
    gi = paint(gi, shift(trgtobj, trgtl))
    toleaveout = ccols
    for rl in remlocs:
        tlo = rng.choice(totuple(ccols))
        ncells = unifint(rng, diff_lb, diff_ub, (1, h * w - 1))
        inds = rng.sample(totuple(denseinds), ncells)
        obj = {(rng.choice(remove(tlo, ccols) if len(ccols) > 1 else ccols), ij) for ij in inds}
        toleaveout = remove(tlo, toleaveout)
        gi = paint(gi, shift(obj, rl))
    return {"input": gi, "output": go}


def generate_63613498(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc, sepc = rng.sample(cols, 2)
    remcols = remove(bgc, remove(sepc, cols))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(remcols, ncols)
    objh = unifint(rng, diff_lb, diff_ub, (1, h // 3))
    objw = unifint(rng, diff_lb, diff_ub, (1, w // 3))
    bounds = asindices(canvas(-1, (objh, objw)))
    sp = rng.choice(totuple(bounds))
    obj = {sp}
    ncells = unifint(rng, diff_lb, diff_ub, (1, (objh * objw)))
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    gi = canvas(bgc, (h, w))
    objc = rng.choice(ccols)
    gi = fill(gi, objc, obj)
    sep = connect((objh + 1, 0), (objh + 1, objw + 1)) | connect((0, objw + 1), (objh + 1, objw + 1))
    gi = fill(gi, sepc, sep)
    inds = asindices(gi)
    inds -= backdrop(sep)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    baseobj = normalize(obj)
    obj = normalize(obj)
    go = tuple(e for e in gi)
    while (succ < nobjs and tr < maxtr) or succ == 0:
        tr += 1
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = rng.choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            col = rng.choice(ccols)
            gi = fill(gi, col, plcd)
            go = fill(go, sepc if succ == 0 else col, plcd)
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
        objh = rng.randint(1, h // 3)
        objw = rng.randint(2 if objh == 1 else 1, w // 3)
        if rng.choice((True, False)):
            objh, objw = objw, objh
        bounds = asindices(canvas(-1, (objh, objw)))
        sp = rng.choice(totuple(bounds))
        obj = {sp}
        ncells = unifint(rng, diff_lb, diff_ub, (1, (objh * objw)))
        for k in range(ncells - 1):
            obj.add(rng.choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        obj = set(obj)
        if obj == baseobj:
            if len(obj) < objh * objw:
                obj.add(rng.choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
            else:
                obj = remove(rng.choice(totuple(corners(obj))), obj)
        obj = normalize(obj)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_e5062a87(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    eligcol, objc = rng.sample(cols, 2)
    gi = canvas(eligcol, (h, w))
    inds = asindices(gi)
    sp = rng.choice(totuple(inds))
    obj = {sp}
    ncells = unifint(rng, diff_lb, diff_ub, (3, 9))
    for k in range(ncells - 1):
        obj.add(rng.choice(totuple((inds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    nnoise = unifint(rng, diff_lb, diff_ub, (int(0.2 * h * w), int(0.5 * h * w)))
    locs = rng.sample(totuple(inds), nnoise)
    gi = fill(gi, 0, locs)
    noccs = unifint(rng, diff_lb, diff_ub, (2, max(2, (h * w) // (len(obj) * 3))))
    oh, ow = shape(obj)
    for k in range(noccs):
        loci = rng.randint(0, h - oh)
        locj = rng.randint(0, w - ow)
        loc = (loci, locj)
        gi = fill(gi, objc if k == noccs - 1 else 0, shift(obj, loc))
    occs = occurrences(gi, recolor(0, obj))
    res = mapply(lbind(shift, obj), occs)
    go = fill(gi, objc, res)
    return {"input": gi, "output": go}


def generate_bc1d5164(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 15))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    fullh = 2 * h - 1
    fullw = 2 * w + 1
    bgc, objc = rng.sample(cols, 2)
    inds = asindices(canvas(-1, (h, w)))
    nA = rng.randint(1, (h - 1) * (w - 1) - 1)
    nB = rng.randint(1, (h - 1) * (w - 1) - 1)
    nC = rng.randint(1, (h - 1) * (w - 1) - 1)
    nD = rng.randint(1, (h - 1) * (w - 1) - 1)
    A = rng.sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] < w - 1)), nA)
    B = rng.sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] > 0)), nB)
    C = rng.sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] < w - 1)), nC)
    D = rng.sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] > 0)), nD)
    gi = canvas(bgc, (fullh, fullw))
    gi = fill(gi, objc, A)
    gi = fill(gi, objc, shift(B, (0, fullw - w)))
    gi = fill(gi, objc, shift(C, (fullh - h, 0)))
    gi = fill(gi, objc, shift(D, (fullh - h, fullw - w)))
    go = canvas(bgc, (h, w))
    go = fill(go, objc, set(A) | set(B) | set(C) | set(D))
    return {"input": gi, "output": go}
