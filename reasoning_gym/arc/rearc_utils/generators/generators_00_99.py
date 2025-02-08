from ..dsl import *
from ..utils import *


def generate_dbc1a6ce(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = remove(8, interval(0, 10, 1))
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
    go = fill(c, 8, resh)
    resv = frozenset()
    for x, r in enumerate(dmirror(gi)):
        if r.count(fgcol) > 1:
            resv = combine(resv, connect((x, r.index(fgcol)), (x, -1 + h - r[::-1].index(fgcol))))
    go = dmirror(fill(dmirror(go), 8, resv))
    go = fill(go, fgcol, s)
    return {"input": gi, "output": go}


def generate_2281f1f4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, dim_bounds)
    w = unifint(rng, diff_lb, diff_ub, dim_bounds)
    card_h_bounds = (1, h // 2 + 1)
    card_w_bounds = (1, w // 2 + 1)
    numtop = unifint(rng, diff_lb, diff_ub, card_w_bounds)
    numright = unifint(rng, diff_lb, diff_ub, card_h_bounds)
    if numtop == numright == 1:
        numtop, numright = rng.sample([1, 2], 2)
    tp = rng.sample(interval(0, w - 1, 1), numtop)
    rp = rng.sample(interval(1, h, 1), numright)
    res = combine(apply(lbind(astuple, 0), tp), apply(rbind(astuple, w - 1), rp))
    bgc = rng.choice(colopts)
    dc = rng.choice(remove(bgc, colopts))
    gi = fill(canvas(bgc, (h, w)), dc, res)
    go = fill(gi, 2, product(rp, tp))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_c1d99e64(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (4, 30)
    colopts = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, dim_bounds)
    w = unifint(rng, diff_lb, diff_ub, dim_bounds)
    nofrontcol = rng.choice(colopts)
    noisefrontcol = rng.choice(remove(nofrontcol, colopts))
    gi = canvas(nofrontcol, (h, w))
    cands = totuple(asindices(gi))
    horifront_bounds = (1, h // 4)
    vertifront_bounds = (1, w // 4)
    nhf = unifint(rng, diff_lb, diff_ub, horifront_bounds)
    nvf = unifint(rng, diff_lb, diff_ub, vertifront_bounds)
    vfs = mapply(compose(vfrontier, tojvec), rng.sample(interval(0, w, 1), nvf))
    hfs = mapply(compose(hfrontier, toivec), rng.sample(interval(0, h, 1), nhf))
    gi = fill(gi, noisefrontcol, combine(vfs, hfs))
    cands = totuple(ofcolor(gi, nofrontcol))
    kk = size(cands)
    midp = (h * w) // 2
    noise_bounds = (0, max(0, kk - midp - 1))
    num_noise = unifint(rng, diff_lb, diff_ub, noise_bounds)
    noise = rng.sample(cands, num_noise)
    gi = fill(gi, noisefrontcol, noise)
    go = fill(gi, 2, merge(colorfilter(frontiers(gi), noisefrontcol)))
    return {"input": gi, "output": go}


def generate_623ea044(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, dim_bounds)
    w = unifint(rng, diff_lb, diff_ub, dim_bounds)
    bgc = rng.choice(colopts)
    g = canvas(bgc, (h, w))
    fullinds = asindices(g)
    inds = totuple(asindices(g))
    card_bounds = (0, max(int(h * w * 0.1), 1))
    numdots = unifint(rng, diff_lb, diff_ub, card_bounds)
    dots = rng.sample(inds, numdots)
    gi = canvas(bgc, (h, w))
    fgc = rng.choice(remove(bgc, colopts))
    gi = fill(gi, fgc, dots)
    go = fill(gi, fgc, mapply(rbind(shoot, UP_RIGHT), dots))
    go = fill(go, fgc, mapply(rbind(shoot, DOWN_LEFT), dots))
    go = fill(go, fgc, mapply(rbind(shoot, UNITY), dots))
    go = fill(go, fgc, mapply(rbind(shoot, NEG_UNITY), dots))
    return {"input": gi, "output": go}


def generate_1190e5a7(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, dim_bounds)
    w = unifint(rng, diff_lb, diff_ub, dim_bounds)
    bgc = rng.choice(colopts)
    c = canvas(bgc, (h, w))
    nhf_bounds = (1, h // 3)
    nvf_bounds = (1, w // 3)
    nhf = unifint(rng, diff_lb, diff_ub, nhf_bounds)
    nvf = unifint(rng, diff_lb, diff_ub, nvf_bounds)
    hf_options = interval(1, h - 1, 1)
    vf_options = interval(1, w - 1, 1)
    hf_selection = []
    for k in range(nhf):
        hf = rng.choice(hf_options)
        hf_selection.append(hf)
        hf_options = difference(hf_options, (hf - 1, hf, hf + 1))
    vf_selection = []
    for k in range(nvf):
        vf = rng.choice(vf_options)
        vf_selection.append(vf)
        vf_options = difference(vf_options, (vf - 1, vf, vf + 1))
    remcols = remove(bgc, colopts)
    rcf = lambda x: recolor(rng.choice(remcols), x)
    hfs = mapply(chain(rcf, hfrontier, toivec), tuple(hf_selection))
    vfs = mapply(chain(rcf, vfrontier, tojvec), tuple(vf_selection))
    gi = paint(c, combine(hfs, vfs))
    go = canvas(bgc, (nhf + 1, nvf + 1))
    return {"input": gi, "output": go}


def generate_5614dbcf(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (2, 10)
    col_card_bounds = (1, 8)
    noise_card_bounds = (0, 8)
    colopts = remove(5, interval(1, 10, 1))
    noisedindscands = totuple(asindices(canvas(0, (3, 3))))
    d = unifint(rng, diff_lb, diff_ub, dim_bounds)
    cells_card_bounds = (1, d * d)
    go = canvas(0, (d, d))
    inds = totuple(asindices(go))
    numocc = unifint(rng, diff_lb, diff_ub, cells_card_bounds)
    numcol = unifint(rng, diff_lb, diff_ub, col_card_bounds)
    occs = rng.sample(inds, numocc)
    colset = rng.sample(colopts, numcol)
    gi = upscale(go, THREE)
    for occ in inds:
        offset = multiply(3, occ)
        numnoise = unifint(rng, diff_lb, diff_ub, noise_card_bounds)
        noise = rng.sample(noisedindscands, numnoise)
        if occ in occs:
            col = rng.choice(colset)
            go = fill(go, col, initset(occ))
            gi = fill(gi, col, shift(noisedindscands, offset))
        gi = fill(gi, 5, shift(noise, offset))
    return {"input": gi, "output": go}


def generate_05269061(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (2, 30)
    colopts = interval(1, 10, 1)
    d = unifint(rng, diff_lb, diff_ub, dim_bounds)
    go = canvas(0, (d, d))
    gi = canvas(0, (d, d))
    if rng.choice((True, False)):
        period_bounds = (2, min(2 * d - 2, 9))
        num = unifint(rng, diff_lb, diff_ub, period_bounds)
        cols = tuple(rng.choice(colopts) for k in range(num))
        keeps = [rng.choice(interval(j, 2 * d - 1, num)) for j in range(num)]
        for k, col in enumerate((cols * 30)[: 2 * d - 1]):
            lin = shoot(toivec(k), UP_RIGHT)
            go = fill(go, col, lin)
            if keeps[k % num] == k:
                gi = fill(gi, col, lin)
    else:
        period_bounds = (2, min(d, 9))
        num = unifint(rng, diff_lb, diff_ub, period_bounds)
        cols = tuple(rng.choice(colopts) for k in range(num))
        keeps = [rng.choice(interval(j, d, num)) for j in range(num)]
        for k, col in enumerate((cols * 30)[:d]):
            lin = hfrontier(toivec(k))
            go = fill(go, col, lin)
            if keeps[k % num] == k:
                gi = fill(gi, col, lin)
    if rng.choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    return {"input": gi, "output": go}


def generate_1c786137(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    num_cols_card_bounds = (1, 8)
    colopts = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, dim_bounds)
    w = unifint(rng, diff_lb, diff_ub, dim_bounds)
    noise_card_bounds = (0, h * w)
    c = canvas(0, (h, w))
    inds = totuple(asindices(c))
    num_noise = unifint(rng, diff_lb, diff_ub, noise_card_bounds)
    num_cols = unifint(rng, diff_lb, diff_ub, num_cols_card_bounds)
    noiseinds = rng.sample(inds, num_noise)
    colset = rng.sample(colopts, num_cols)
    trgcol = rng.choice(difference(colopts, colset))
    noise = frozenset((rng.choice(colset), ij) for ij in noiseinds)
    gi = paint(c, noise)
    boxhrng = (3, max(3, h // 2))
    boxwrng = (3, max(3, w // 2))
    boxh = unifint(rng, diff_lb, diff_ub, boxhrng)
    boxw = unifint(rng, diff_lb, diff_ub, boxwrng)
    boxi = rng.choice(interval(0, h - boxh + 1, 1))
    boxj = rng.choice(interval(0, w - boxw + 1, 1))
    loc = (boxi, boxj)
    llc = add(loc, toivec(boxh - 1))
    urc = add(loc, tojvec(boxw - 1))
    lrc = add(loc, (boxh - 1, boxw - 1))
    l1 = connect(loc, llc)
    l2 = connect(loc, urc)
    l3 = connect(urc, lrc)
    l4 = connect(llc, lrc)
    l = l1 | l2 | l3 | l4
    gi = fill(gi, trgcol, l)
    go = crop(gi, increment(loc), (boxh - 2, boxw - 2))
    return {"input": gi, "output": go}


def generate_2204b7a8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (4, 30)
    colopts = interval(0, 10, 1)
    while True:
        h = unifint(rng, diff_lb, diff_ub, dim_bounds)
        w = unifint(rng, diff_lb, diff_ub, dim_bounds)
        bgc = rng.choice(colopts)
        remcols = remove(bgc, colopts)
        c = canvas(bgc, (h, w))
        inds = totuple(shift(asindices(canvas(0, (h, w - 2))), RIGHT))
        ccol = rng.choice(remcols)
        remcols2 = remove(ccol, remcols)
        c1 = rng.choice(remcols2)
        c2 = rng.choice(remove(c1, remcols2))
        nc_bounds = (1, (h * (w - 2)) // 2 - 1)
        nc = unifint(rng, diff_lb, diff_ub, nc_bounds)
        locs = rng.sample(inds, nc)
        if w % 2 == 1:
            locs = difference(locs, vfrontier(tojvec(w // 2)))
        gi = fill(c, c1, vfrontier(ORIGIN))
        gi = fill(gi, c2, vfrontier(tojvec(w - 1)))
        gi = fill(gi, ccol, locs)
        a = sfilter(locs, lambda ij: last(ij) < w // 2)
        b = difference(locs, a)
        go = fill(gi, c1, a)
        go = fill(go, c2, b)
        if len(palette(gi)) == 4:
            break
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_23581191(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = remove(2, interval(0, 10, 1))
    f = fork(combine, hfrontier, vfrontier)
    h = unifint(rng, diff_lb, diff_ub, dim_bounds)
    w = unifint(rng, diff_lb, diff_ub, dim_bounds)
    bgcol = rng.choice(colopts)
    remcols = remove(bgcol, colopts)
    c = canvas(bgcol, (h, w))
    inds = totuple(asindices(c))
    acol = rng.choice(remcols)
    bcol = rng.choice(remove(acol, remcols))
    card_bounds = (1, (h * w) // 4)
    na = unifint(rng, diff_lb, diff_ub, card_bounds)
    nb = unifint(rng, diff_lb, diff_ub, card_bounds)
    a = rng.sample(inds, na)
    b = rng.sample(difference(inds, a), nb)
    gi = fill(c, acol, a)
    gi = fill(gi, bcol, b)
    fa = apply(first, a)
    la = apply(last, a)
    fb = apply(first, b)
    lb = apply(last, b)
    alins = sfilter(inds, lambda ij: first(ij) in fa or last(ij) in la)
    blins = sfilter(inds, lambda ij: first(ij) in fb or last(ij) in lb)
    go = fill(c, acol, alins)
    go = fill(go, bcol, blins)
    go = fill(go, 2, intersection(set(alins), set(blins)))
    go = fill(go, acol, a)
    go = fill(go, bcol, b)
    return {"input": gi, "output": go}


def generate_8be77c9e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 15))
    w = unifint(rng, diff_lb, diff_ub, (1, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(cols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = vconcat(gi, hmirror(gi))
    return {"input": gi, "output": go}


def generate_6d0aefbc(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
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


def generate_74dd1130(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
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
    go = dmirror(gi)
    return {"input": gi, "output": go}


def generate_62c24649(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
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
    go = vconcat(hconcat(gi, vmirror(gi)), hconcat(hmirror(gi), hmirror(vmirror(gi))))
    return {"input": gi, "output": go}


def generate_6150a2bd(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
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


def generate_6fa7a44f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 15))
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
    go = vconcat(gi, hmirror(gi))
    return {"input": gi, "output": go}


def generate_8d5021e8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 10))
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
    go1 = hconcat(vmirror(gi), gi)
    go2 = vconcat(go1, hmirror(go1))
    go = vconcat(hmirror(go1), go2)
    return {"input": gi, "output": go}


def generate_0520fde7(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = rng.choice(remcols)
    remcols = remove(barcol, remcols)
    cola = rng.choice(remcols)
    colb = rng.choice(remcols)
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
    go = fill(canv, 2, set(a) & set(b))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_46442a0e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 15))
    w = h
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
    go1 = hconcat(gi, rot90(gi))
    go2 = hconcat(rot270(gi), rot180(gi))
    go = vconcat(go1, go2)
    return {"input": gi, "output": go}


def generate_1b2d62fb(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = rng.choice(remcols)
    remcols = remove(barcol, remcols)
    cola = rng.choice(remcols)
    colb = rng.choice(remcols)
    canv = canvas(0, (h, w))
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
    go = fill(canv, 8, ofcolor(gia, 0) & ofcolor(gib, 0))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_3428a4f5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = rng.choice(remcols)
    remcols = remove(barcol, remcols)
    cola = rng.choice(remcols)
    colb = rng.choice(remcols)
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
    go = fill(canv, 3, (set(a) | set(b)) - (set(a) & set(b)))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_42a50994(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 30))
    bgc = rng.choice(colopts)
    remcols = remove(bgc, colopts)
    c = canvas(bgc, (h, w))
    card_bounds = (0, max(0, (h * w) // 2 - 1))
    num = unifint(rng, diff_lb, diff_ub, card_bounds)
    numcols = unifint(rng, diff_lb, diff_ub, (0, min(9, num)))
    inds = totuple(asindices(c))
    chosinds = rng.sample(inds, num)
    choscols = rng.sample(remcols, numcols)
    locs = interval(0, len(chosinds), 1)
    choslocs = rng.sample(locs, numcols)
    gi = canvas(bgc, (h, w))
    for col, endidx in zip(choscols, sorted(choslocs)[::-1]):
        gi = fill(gi, col, chosinds[:endidx])
    objs = objects(gi, F, T, T)
    res = merge(sizefilter(objs, 1))
    go = fill(gi, bgc, res)
    return {"input": gi, "output": go}


def generate_08ed6ac7(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc = rng.choice(difference(colopts, (1, 2, 3, 4)))
    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    barrange = (4, w)
    locopts = interval(0, w, 1)
    nbars = unifint(rng, diff_lb, diff_ub, barrange)
    barlocs = rng.sample(locopts, nbars)
    barhopts = interval(0, h, 1)
    barhs = rng.sample(barhopts, 4)
    barcols = [rng.choice(remcols) for j in range(nbars)]
    barhsfx = [rng.choice(barhs) for j in range(nbars - 4)] + list(barhs)
    rng.shuffle(barhsfx)
    ordered = sorted(barhs)
    colord = interval(1, 5, 1)
    for col, (loci, locj) in zip(barcols, list(zip(barhsfx, barlocs))):
        bar = connect((loci, locj), (h - 1, locj))
        gi = fill(gi, col, bar)
        go = fill(go, colord[ordered.index(loci)], bar)
    return {"input": gi, "output": go}


def generate_8f2ea7aa(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    d = unifint(rng, diff_lb, diff_ub, (2, 5))
    bgc = rng.choice(colopts)
    remcols = remove(bgc, colopts)
    d2 = d**2
    gi = canvas(bgc, (d2, d2))
    go = canvas(bgc, (d2, d2))
    minig = canvas(bgc, (d, d))
    inds = totuple(asindices(minig))
    mp = d2 // 2
    devrng = (0, mp)
    dev = unifint(rng, diff_lb, diff_ub, devrng)
    devs = rng.choice((+1, -1))
    num = mp + devs * dev
    num = max(min(num, d2), 0)
    locs = set(rng.sample(inds, num))
    while shape(locs) != (d, d):
        locs.add(rng.choice(totuple(set(inds) - locs)))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 9))
    cols = rng.sample(remcols, ncols)
    for ij in locs:
        minig = fill(minig, rng.choice(cols), {ij})
    itv = interval(0, d2, d)
    plcopts = totuple(product(itv, itv))
    plc = rng.choice(plcopts)
    minigo = asobject(minig)
    gi = paint(gi, shift(minigo, plc))
    for ij in locs:
        go = paint(go, shift(minigo, multiply(ij, d)))
    return {"input": gi, "output": go}


def generate_7fe24cdd(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 15))
    w = h
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
    go1 = hconcat(gi, rot90(gi))
    go2 = hconcat(rot270(gi), rot180(gi))
    go = vconcat(go1, go2)
    return {"input": gi, "output": go}


def generate_85c4e7cd(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 15))
    w = unifint(rng, diff_lb, diff_ub, (1, 15))
    ncols = unifint(rng, diff_lb, diff_ub, (1, 10))
    cols = rng.sample(colopts, ncols)
    colord = [rng.choice(cols) for j in range(min(h, w))]
    shp = (h * 2, w * 2)
    gi = canvas(0, shp)
    go = canvas(0, shp)
    for idx, (ci, co) in enumerate(zip(colord, colord[::-1])):
        ulc = (idx, idx)
        lrc = (h * 2 - 1 - idx, w * 2 - 1 - idx)
        bx = box(frozenset({ulc, lrc}))
        gi = fill(gi, ci, bx)
        go = fill(go, co, bx)
    return {"input": gi, "output": go}


def generate_8e5a5113(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    d = unifint(rng, diff_lb, diff_ub, (2, 9))
    bgc = rng.choice(colopts)
    remcols = remove(bgc, colopts)
    k = 4 if d < 7 else 3
    nbound = (2, k)
    num = unifint(rng, diff_lb, diff_ub, nbound)
    rotfs = (identity, rot90, rot180, rot270)
    barc = rng.choice(remcols)
    remcols = remove(barc, remcols)
    colbnds = (1, 8)
    ncols = unifint(rng, diff_lb, diff_ub, colbnds)
    patcols = rng.sample(remcols, ncols)
    bgcanv = canvas(bgc, (d, d))
    c = canvas(bgc, (d, d))
    inds = totuple(asindices(c))
    ncolbnds = (1, d**2 - 1)
    ncells = unifint(rng, diff_lb, diff_ub, ncolbnds)
    indsss = rng.sample(inds, ncells)
    for ij in indsss:
        c = fill(c, rng.choice(patcols), {ij})
    barr = canvas(barc, (d, 1))
    fillinidx = rng.choice(interval(0, num, 1))
    gi = rot90(rot270(c if fillinidx == 0 else bgcanv))
    go = rot90(rot270(c))
    for j in range(num - 1):
        c = rot90(c)
        gi = hconcat(hconcat(gi, barr), c if j + 1 == fillinidx else bgcanv)
        go = hconcat(hconcat(go, barr), c)
    if rng.choice((True, False)):
        gi = rot90(gi)
        go = rot90(go)
    return {"input": gi, "output": go}


def generate_4c4377d9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 15))
    w = unifint(rng, diff_lb, diff_ub, (1, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = rng.sample(cols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = vconcat(hmirror(gi), gi)
    return {"input": gi, "output": go}


def generate_a65b410d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = difference(interval(0, 10, 1), (1, 3))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    mpi = h // 2
    mpj = w // 2
    devi = unifint(rng, diff_lb, diff_ub, (0, mpi))
    devj = unifint(rng, diff_lb, diff_ub, (0, mpj))
    if rng.choice((True, False)):
        locj = devj
        loci = devi
    else:
        loci = h - devi
        locj = w - devj
    loci = max(min(h - 2, loci), 1)
    locj = max(min(w - 2, locj), 1)
    loc = (loci, locj)
    bgc = rng.choice(colopts)
    linc = rng.choice(remove(bgc, colopts))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, linc, connect((loci, 0), (loci, locj)))
    blues = shoot((loci + 1, locj - 1), (1, -1))
    f = lambda ij: connect(ij, (ij[0], 0)) if ij[1] >= 0 else frozenset({})
    blues = mapply(f, blues)
    greens = shoot((loci - 1, locj + 1), (-1, 1))
    greens = mapply(f, greens)
    go = fill(gi, 1, blues)
    go = fill(go, 3, greens)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_5168d44c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (7, 30))
    doth = unifint(rng, diff_lb, diff_ub, (1, h // 3))
    dotw = unifint(rng, diff_lb, diff_ub, (1, w // 3))
    borderh = unifint(rng, diff_lb, diff_ub, (1, h // 4))
    borderw = unifint(rng, diff_lb, diff_ub, (1, w // 4))
    direc = rng.choice((DOWN, RIGHT, UNITY))
    dotloci = rng.randint(0, h - doth - 1 if direc == RIGHT else h - doth - borderh - 1)
    dotlocj = rng.randint(0, w - dotw - 1 if direc == DOWN else w - dotw - borderw - 1)
    dotloc = (dotloci, dotlocj)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    dotcol = rng.choice(remcols)
    remcols = remove(dotcol, remcols)
    boxcol = rng.choice(remcols)
    gi = canvas(bgc, (h, w))
    dotshap = (doth, dotw)
    starterdot = backdrop(frozenset({dotloc, add(dotloc, decrement(dotshap))}))
    bordershap = (borderh, borderw)
    offset = add(multiply(direc, dotshap), multiply(direc, bordershap))
    itv = interval(-15, 16, 1)
    itv = apply(lbind(multiply, offset), itv)
    dots = mapply(lbind(shift, starterdot), itv)
    gi = fill(gi, dotcol, dots)
    protobx = backdrop(
        frozenset(
            {
                (dotloci - borderh, dotlocj - borderw),
                (dotloci + doth + borderh - 1, dotlocj + dotw + borderw - 1),
            }
        )
    )
    bx = protobx - starterdot
    bxshifted = shift(bx, offset)
    go = fill(gi, boxcol, bxshifted)
    gi = fill(gi, boxcol, bx)
    return {"input": gi, "output": go}


def generate_a9f96cdd(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3, 6, 7, 8))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    fgc = rng.choice(remove(bgc, cols))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    locs = asindices(gi)
    noccs = unifint(rng, diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    for k in range(noccs):
        if len(locs) == 0:
            break
        loc = rng.choice(totuple(locs))
        locs = locs - mapply(neighbors, neighbors(loc))
        plcd = {loc}
        gi = fill(gi, fgc, plcd)
        go = fill(go, 3, shift(plcd, (-1, -1)))
        go = fill(go, 7, shift(plcd, (1, 1)))
        go = fill(go, 8, shift(plcd, (1, -1)))
        go = fill(go, 6, shift(plcd, (-1, 1)))
    return {"input": gi, "output": go}


def generate_9172f3a0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 10))
    w = unifint(rng, diff_lb, diff_ub, (1, 10))
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
    go = upscale(gi, 3)
    return {"input": gi, "output": go}


def generate_67a423a3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    lineh = unifint(rng, diff_lb, diff_ub, (1, h // 3))
    linew = unifint(rng, diff_lb, diff_ub, (1, w // 3))
    loci = rng.randint(1, h - lineh - 1)
    locj = rng.randint(1, w - linew - 1)
    acol = rng.choice(remcols)
    bcol = rng.choice(remove(acol, remcols))
    for a in range(lineh):
        gi = fill(gi, acol, connect((loci + a, 0), (loci + a, w - 1)))
    for b in range(linew):
        gi = fill(gi, bcol, connect((0, locj + b), (h - 1, locj + b)))
    bx = outbox(frozenset({(loci, locj), (loci + lineh - 1, locj + linew - 1)}))
    go = fill(gi, 4, bx)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_db3e9e38(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    barth = unifint(rng, diff_lb, diff_ub, (1, max(1, w // 5)))
    loci = unifint(rng, diff_lb, diff_ub, (1, h - 2))
    locj = rng.randint(1, w - barth - 1)
    bar = backdrop(frozenset({(loci, locj), (0, locj + barth - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, fgc, bar)
    go = canvas(bgc, (h, w))
    for k in range(16):
        rsh = multiply(2 * k, (-1, barth))
        go = fill(go, fgc, shift(bar, rsh))
        lsh = multiply(2 * k, (-1, -barth))
        go = fill(go, fgc, shift(bar, lsh))
        rsh = multiply(2 * k + 1, (-1, barth))
        go = fill(go, 8, shift(bar, rsh))
        lsh = multiply(2 * k + 1, (-1, -barth))
        go = fill(go, 8, shift(bar, lsh))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_9dfd6313(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    dh = unifint(rng, diff_lb, diff_ub, (1, 14))
    d = 2 * dh + 1
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    remcols = remove(linc, remcols)
    gi = canvas(bgc, (d, d))
    inds = asindices(gi)
    lni = rng.randint(1, 4)
    if lni == 1:
        ln = connect((dh, 0), (dh, d - 1))
        mirrf = hmirror
        cands = sfilter(inds, lambda ij: ij[0] > dh)
    elif lni == 2:
        ln = connect((0, dh), (d - 1, dh))
        mirrf = vmirror
        cands = sfilter(inds, lambda ij: ij[1] > dh)
    elif lni == 3:
        ln = connect((0, 0), (d - 1, d - 1))
        mirrf = dmirror
        cands = sfilter(inds, lambda ij: ij[0] > ij[1])
    elif lni == 4:
        ln = connect((d - 1, 0), (0, d - 1))
        mirrf = cmirror
        cands = sfilter(inds, lambda ij: (ij[0] + ij[1]) > d)
    gi = fill(gi, linc, ln)
    mp = (d * (d - 1)) // 2
    numcols = unifint(rng, diff_lb, diff_ub, (1, min(7, mp)))
    colsch = rng.sample(remcols, numcols)
    numpix = unifint(rng, diff_lb, diff_ub, (1, len(cands)))
    pixs = rng.sample(totuple(cands), numpix)
    for pix in pixs:
        gi = fill(gi, rng.choice(colsch), {pix})
    go = mirrf(gi)
    if rng.choice((True, False)):
        gi, go = go, gi
    return {"input": gi, "output": go}


def generate_746b3537(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    fullcols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 15))
    w = unifint(rng, diff_lb, diff_ub, (1, 30))
    cols = []
    lastc = -1
    for k in range(h):
        c = rng.choice(remove(lastc, fullcols))
        cols.append(c)
        lastc = c
    go = tuple((c,) for c in cols)
    gi = tuple(repeat(c, w) for c in cols)
    numinserts = unifint(rng, diff_lb, diff_ub, (1, 30 - h))
    for k in range(numinserts):
        loc = rng.randint(0, len(gi) - 1)
        gi = gi[: loc + 1] + gi[loc:]
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_75b8110e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 15))
    w = unifint(rng, diff_lb, diff_ub, (2, 15))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    c1, c2, c3, c4 = rng.sample(remcols, 4)
    canv = canvas(bgc, (h, w))
    cels = totuple(asindices(canv))
    mp = (h * w) // 2
    nums = []
    for k in range(4):
        dev = unifint(rng, diff_lb, diff_ub, (0, mp))
        if rng.choice((True, False)):
            num = h * w - dev
        else:
            num = dev
        num = min(max(0, num), h * w - 1)
        nums.append(num)
    s1, s2, s3, s4 = [rng.sample(cels, num) for num in nums]
    gi1 = fill(canv, c1, s1)
    gi2 = fill(canv, c2, s2)
    gi3 = fill(canv, c3, s3)
    gi4 = fill(canv, c4, s4)
    gi = vconcat(hconcat(gi1, gi2), hconcat(gi3, gi4))
    go = fill(gi1, c4, s4)
    go = fill(go, c3, s3)
    go = fill(go, c2, s2)
    return {"input": gi, "output": go}


def generate_1cf80156(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(colopts)
    fgc = rng.choice(remove(bgc, colopts))
    gi = canvas(bgc, (h, w))
    hb = unifint(rng, diff_lb, diff_ub, (1, min(15, h - 1)))
    wb = unifint(rng, diff_lb, diff_ub, (1, min(15, w - 1)))
    bounds = asindices(canvas(0, (hb, wb)))
    shp = {rng.choice(totuple(corners(bounds)))}
    mp = (hb * wb) // 2
    dev = unifint(rng, diff_lb, diff_ub, (0, mp))
    nc = rng.choice((dev, hb * wb - dev))
    nc = max(0, min(hb * wb - 1, nc))
    for j in range(nc):
        shp.add(rng.choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    di = rng.randint(0, h - height(shp))
    dj = rng.randint(0, w - width(shp))
    shpp = shift(shp, (di, dj))
    gi = fill(gi, fgc, shpp)
    go = fill(canvas(bgc, shape(shp)), fgc, shp)
    return {"input": gi, "output": go}


def generate_28bf18c6(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(colopts)
    fgc = rng.choice(remove(bgc, colopts))
    gi = canvas(bgc, (h, w))
    hb = unifint(rng, diff_lb, diff_ub, (1, min(14, h - 1)))
    wb = unifint(rng, diff_lb, diff_ub, (1, min(14, w - 1)))
    bounds = asindices(canvas(0, (hb, wb)))
    shp = {rng.choice(totuple(corners(bounds)))}
    mp = (hb * wb) // 2
    dev = unifint(rng, diff_lb, diff_ub, (0, mp))
    nc = rng.choice((dev, hb * wb - dev))
    nc = max(0, min(hb * wb - 1, nc))
    for j in range(nc):
        shp.add(rng.choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    di = rng.randint(0, h - height(shp))
    dj = rng.randint(0, w - width(shp))
    shpp = shift(shp, (di, dj))
    gi = fill(gi, fgc, shpp)
    go = fill(canvas(bgc, shape(shp)), fgc, shp)
    go = hconcat(go, go)
    return {"input": gi, "output": go}


def generate_22eb0ac0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    gi = canvas(0, (1, 1))
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    bgc = rng.choice(colopts)
    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nlocs = unifint(rng, diff_lb, diff_ub, (1, h))
    locs = rng.sample(interval(0, h, 1), nlocs)
    while set(locs).issubset({0, h - 1}):
        locs = rng.sample(interval(0, h, 1), nlocs)
    mp = nlocs // 2
    nbarsdev = unifint(rng, diff_lb, diff_ub, (0, mp))
    nbars = rng.choice((nbarsdev, h - nbarsdev))
    nbars = max(0, min(nbars, nlocs))
    barlocs = rng.sample(locs, nbars)
    nonbarlocs = difference(locs, barlocs)
    barcols = [rng.choice(remcols) for j in range(nbars)]
    acols = [rng.choice(remcols) for j in range(len(nonbarlocs))]
    bcols = [rng.choice(remove(acols[j], remcols)) for j in range(len(nonbarlocs))]
    for bc, bl in zip(barcols, barlocs):
        gi = fill(gi, bc, ((bl, 0), (bl, w - 1)))
        go = fill(go, bc, connect((bl, 0), (bl, w - 1)))
    for (a, b), loc in zip(zip(acols, bcols), nonbarlocs):
        gi = fill(gi, a, {(loc, 0)})
        go = fill(go, a, {(loc, 0)})
        gi = fill(gi, b, {(loc, w - 1)})
        go = fill(go, b, {(loc, w - 1)})
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_4258a5f9(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(1, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    bgc = rng.choice(colopts)
    remcols = remove(bgc, colopts)
    fgc = rng.choice(remcols)
    gi = canvas(bgc, (h, w))
    mp = ((h * w) // 2) if (h * w) % 2 == 1 else ((h * w) // 2 - 1)
    ndots = unifint(rng, diff_lb, diff_ub, (1, mp))
    inds = totuple(asindices(gi))
    dots = rng.sample(inds, ndots)
    go = fill(gi, 1, mapply(neighbors, frozenset(dots)))
    go = fill(go, fgc, dots)
    gi = fill(gi, fgc, dots)
    return {"input": gi, "output": go}


def generate_1e0a9b12(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    ff = chain(dmirror, lbind(apply, rbind(order, identity)), dmirror)
    while True:
        h = unifint(rng, diff_lb, diff_ub, (3, 30))
        w = unifint(rng, diff_lb, diff_ub, (3, 30))
        nc = unifint(rng, diff_lb, diff_ub, (1, w))
        bgc = rng.choice(cols)
        gi = canvas(bgc, (h, w))
        remcols = remove(bgc, cols)
        scols = [rng.choice(remcols) for j in range(nc)]
        slocs = rng.sample(interval(0, w, 1), nc)
        inds = totuple(connect(ORIGIN, (h - 1, 0)))
        for c, l in zip(scols, slocs):
            nc2 = rng.randint(1, h - 1)
            sel = rng.sample(inds, nc2)
            gi = fill(gi, c, shift(sel, tojvec(l)))
        go = replace(ff(replace(gi, bgc, -1)), -1, bgc)
        if colorcount(gi, bgc) > argmax(remove(bgc, palette(gi)), lbind(colorcount, gi)):
            break
    return {"input": gi, "output": go}


def generate_9565186b(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    wg = canvas(5, (h, w))
    numcols = unifint(rng, diff_lb, diff_ub, (2, min(h * w - 1, 8)))
    mostcol = rng.choice(cols)
    nummostcol_lb = (h * w) // numcols + 1
    nummostcol_ub = h * w - numcols + 1
    ubmlb = nummostcol_ub - nummostcol_lb
    nmcdev = unifint(rng, diff_lb, diff_ub, (0, ubmlb))
    nummostcol = nummostcol_ub - nmcdev
    nummostcol = min(max(nummostcol, nummostcol_lb), nummostcol_ub)
    inds = totuple(asindices(wg))
    mostcollocs = rng.sample(inds, nummostcol)
    gi = fill(wg, mostcol, mostcollocs)
    go = fill(wg, mostcol, mostcollocs)
    remcols = remove(mostcol, cols)
    othcols = rng.sample(remcols, numcols - 1)
    reminds = difference(inds, mostcollocs)
    bufferlocs = rng.sample(reminds, numcols - 1)
    for c, l in zip(othcols, bufferlocs):
        gi = fill(gi, c, {l})
    reminds = difference(reminds, bufferlocs)
    colcounts = {c: 1 for c in othcols}
    for ij in reminds:
        if len(othcols) == 0:
            gi = fill(gi, mostcol, {ij})
            go = fill(go, mostcol, {ij})
        else:
            chc = rng.choice(othcols)
            gi = fill(gi, chc, {ij})
            colcounts[chc] += 1
            if colcounts[chc] == nummostcol - 1:
                othcols = remove(chc, othcols)
    return {"input": gi, "output": go}


def generate_6e02f1e3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    d = unifint(rng, diff_lb, diff_ub, (3, 30))
    c = canvas(0, (d, d))
    inds = list(asindices(c))
    rng.shuffle(inds)
    num = d**2
    numcols = rng.choice((1, 2, 3))
    chcols = rng.sample(cols, numcols)
    if len(chcols) == 1:
        gi = canvas(chcols[0], (d, d))
        go = canvas(0, (d, d))
        go = fill(go, 5, connect((0, 0), (0, d - 1)))
    elif len(chcols) == 2:
        c1, c2 = chcols
        mp = (d**2) // 2
        nc1 = unifint(rng, diff_lb, diff_ub, (1, mp))
        a = inds[:nc1]
        b = inds[nc1:]
        gi = fill(c, c1, a)
        gi = fill(gi, c2, b)
        go = fill(canvas(0, (d, d)), 5, connect((0, 0), (d - 1, d - 1)))
    elif len(chcols) == 3:
        c1, c2, c3 = chcols
        kk = d**2
        a = int(1 / 3 * kk)
        b = int(2 / 3 * kk)
        adev = unifint(rng, diff_lb, diff_ub, (0, a - 1))
        bdev = unifint(rng, diff_lb, diff_ub, (0, kk - b - 1))
        a -= adev
        b -= bdev
        x1, x2, x3 = inds[:a], inds[a:b], inds[b:]
        gi = fill(c, c1, x1)
        gi = fill(gi, c2, x2)
        gi = fill(gi, c3, x3)
        go = fill(canvas(0, (d, d)), 5, connect((d - 1, 0), (0, d - 1)))
    return {"input": gi, "output": go}


def generate_2dc579da(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    remcols = remove(linc, remcols)
    dotc = rng.choice(remcols)
    hdev = unifint(rng, diff_lb, diff_ub, (0, (h - 2) // 2))
    lineh = rng.choice((hdev, h - 2 - hdev))
    lineh = max(min(h - 2, lineh), 1)
    wdev = unifint(rng, diff_lb, diff_ub, (0, (w - 2) // 2))
    linew = rng.choice((wdev, w - 2 - wdev))
    linew = max(min(w - 2, linew), 1)
    locidev = unifint(rng, diff_lb, diff_ub, (1, h // 2))
    loci = rng.choice((h // 2 - locidev, h // 2 + locidev))
    loci = min(max(1, loci), h - lineh - 1)
    locjdev = unifint(rng, diff_lb, diff_ub, (1, w // 2))
    locj = rng.choice((w // 2 - locjdev, w // 2 + locjdev))
    locj = min(max(1, locj), w - linew - 1)
    gi = canvas(bgc, (h, w))
    for a in range(loci, loci + lineh):
        gi = fill(gi, linc, connect((a, 0), (a, w - 1)))
    for b in range(locj, locj + linew):
        gi = fill(gi, linc, connect((0, b), (h - 1, b)))
    doth = rng.randint(1, loci)
    dotw = rng.randint(1, locj)
    dotloci = rng.randint(0, loci - doth)
    dotlocj = rng.randint(0, locj - dotw)
    dot = backdrop(frozenset({(dotloci, dotlocj), (dotloci + doth - 1, dotlocj + dotw - 1)}))
    gi = fill(gi, dotc, dot)
    go = crop(gi, (0, 0), (loci, locj))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_2dee498d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 30))
    w = unifint(rng, diff_lb, diff_ub, (1, 10))
    bgc = rng.choice(cols)
    go = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(go))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        go = fill(go, col, chos)
        inds = difference(inds, chos)
    gi = hconcat(go, hconcat(go, go))
    return {"input": gi, "output": go}


def generate_508bd3b6(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (h, 30))
    barh = unifint(rng, diff_lb, diff_ub, (1, h // 2))
    barloci = unifint(rng, diff_lb, diff_ub, (2, h - barh))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    barc = rng.choice(remcols)
    remcols = remove(barc, remcols)
    linc = rng.choice(remcols)
    gi = canvas(bgc, (h, w))
    for j in range(barloci, barloci + barh):
        gi = fill(gi, barc, connect((j, 0), (j, w - 1)))
    dotlociinv = unifint(rng, diff_lb, diff_ub, (0, barloci - 1))
    dotloci = min(max(0, barloci - 2 - dotlociinv), barloci - 1)
    ln1 = shoot((dotloci, 0), (1, 1))
    ofbgc = ofcolor(gi, bgc)
    ln1 = sfilter(ln1 & ofbgc, lambda ij: ij[0] < barloci)
    ln1 = order(ln1, first)
    ln2 = shoot(ln1[-1], (-1, 1))
    ln2 = sfilter(ln2 & ofbgc, lambda ij: ij[0] < barloci)
    ln2 = order(ln2, last)[1:]
    ln = ln1 + ln2
    k = len(ln1)
    lineleninv = unifint(rng, diff_lb, diff_ub, (0, k - 2))
    linelen = k - lineleninv
    givenl = ln[:linelen]
    reml = ln[linelen:]
    gi = fill(gi, linc, givenl)
    go = fill(gi, 3, reml)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_88a62173(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 14))
    w = unifint(rng, diff_lb, diff_ub, (1, 14))
    bgc = rng.choice(cols)
    gib = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(rng, diff_lb, diff_ub, (1, min(9, h * w)))
    colsch = rng.sample(remcols, numc)
    inds = totuple(asindices(gib))
    for col in colsch:
        num = unifint(rng, diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = rng.sample(inds, num)
        gib = fill(gib, col, chos)
        inds = difference(inds, chos)
    numchinv = unifint(rng, diff_lb, diff_ub, (0, h * w - 1))
    numch = h * w - numchinv
    inds2 = totuple(asindices(gib))
    subs = rng.sample(inds2, numch)
    go = hmirror(hmirror(gib))
    for x, y in subs:
        go = fill(go, rng.choice(remove(go[x][y], colsch + [bgc])), {(x, y)})
    gi = canvas(bgc, (h * 2 + 1, w * 2 + 1))
    idxes = ((0, 0), (h + 1, w + 1), (h + 1, 0), (0, w + 1))
    trgloc = rng.choice(idxes)
    remidxes = remove(trgloc, idxes)
    trgobj = asobject(go)
    otherobj = asobject(gib)
    gi = paint(gi, shift(trgobj, trgloc))
    for ij in remidxes:
        gi = paint(gi, shift(otherobj, ij))
    return {"input": gi, "output": go}


def generate_3aa6fb7a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    base = (ORIGIN, RIGHT, DOWN, UNITY)
    cols = remove(1, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    maxnum = ((h * w) // 2) // 3
    num = unifint(rng, diff_lb, diff_ub, (1, maxnum))
    kk, tr = 0, 0
    maxtrials = num * 2
    binds = set()
    while kk < num and tr < maxtrials:
        loc = rng.choice(inds)
        ooo = rng.choice(base)
        oo = remove(ooo, base)
        oop = shift(oo, loc)
        if set(oop).issubset(inds):
            inds = difference(inds, totuple(combine(oop, totuple(mapply(dneighbors, oop)))))
            gi = fill(gi, fgc, oop)
            binds.add(add(ooo, loc))
            kk += 1
        tr += 1
    go = fill(gi, 1, binds)
    return {"input": gi, "output": go}


def generate_3ac3eb23(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    nlocs = unifint(rng, diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = interval(1, w - 1, 1)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for k in range(nlocs):
        if len(locopts) == 0:
            break
        locj = rng.choice(locopts)
        locopts = difference(locopts, interval(locj - 2, locj + 3, 1))
        col = rng.choice(remcols)
        gi = fill(gi, col, {(0, locj)})
        go = fill(go, col, {(p, locj) for p in interval(0, h, 2)})
        go = fill(go, col, {(p, locj - 1) for p in interval(1, h, 2)})
        go = fill(go, col, {(p, locj + 1) for p in interval(1, h, 2)})
    mf = rng.choice((identity, rot90, rot180, rot270))
    gi = mf(gi)
    go = mf(go)
    return {"input": gi, "output": go}


def generate_c3e719e8(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    gob = canvas(-1, (h**2, w**2))
    wg = canvas(-1, (h, w))
    ncols = unifint(rng, diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    nmc = rng.randint(max(1, (h * w) // (ncols + 1) + 1), h * w)
    inds = totuple(asindices(wg))
    mc = rng.choice(cols)
    remcols = remove(mc, cols)
    mcc = rng.sample(inds, nmc)
    inds = difference(inds, mcc)
    gi = fill(wg, mc, mcc)
    ocols = rng.sample(remcols, ncols)
    k = len(inds) // ncols + 1
    for ocol in ocols:
        if len(inds) == 0:
            break
        ub = min(nmc - 1, len(inds))
        ub = min(ub, k)
        ub = max(ub, 1)
        locs = rng.sample(inds, unifint(rng, diff_lb, diff_ub, (1, ub)))
        inds = difference(inds, locs)
        gi = fill(gi, ocol, locs)
    gi = replace(gi, -1, mc)
    o = asobject(gi)
    gob = replace(gob, -1, 0)
    go = paint(gob, mapply(lbind(shift, o), apply(rbind(multiply, (h, w)), ofcolor(gi, mc))))
    return {"input": gi, "output": go}


def generate_29c11459(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(5, interval(0, 10, 1))
    gi = canvas(0, (1, 1))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 29))
    if w % 2 == 0:
        w = rng.choice((max(5, w - 1), min(29, w + 1)))
    bgc = rng.choice(colopts)
    remcols = remove(bgc, colopts)
    ncols = unifint(rng, diff_lb, diff_ub, (2, len(remcols)))
    ccols = rng.sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nlocs = unifint(rng, diff_lb, diff_ub, (1, h))
    locs = rng.sample(interval(0, h, 1), nlocs)
    while set(locs).issubset({0, h - 1}):
        locs = rng.sample(interval(0, h, 1), nlocs)
    acols = []
    bcols = []
    aforb = -1
    bforb = -1
    for k in range(nlocs):
        ac = rng.choice(remove(aforb, ccols))
        acols.append(ac)
        aforb = ac
        bc = rng.choice(remove(bforb, ccols))
        bcols.append(bc)
        bforb = bc
    for (a, b), loc in zip(zip(acols, bcols), sorted(locs)):
        gi = fill(gi, a, {(loc, 0)})
        gi = fill(gi, b, {(loc, w - 1)})
        go = fill(go, a, connect((loc, 0), (loc, w // 2 - 1)))
        go = fill(go, b, connect((loc, w // 2 + 1), (loc, w - 1)))
        go = fill(go, 5, {(loc, w // 2)})
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_23b5c85d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    colopts = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    oh = unifint(rng, diff_lb, diff_ub, (2, h - 1))
    ow = unifint(rng, diff_lb, diff_ub, (2, w - 1))
    num = unifint(rng, diff_lb, diff_ub, (1, 8))
    cnt = 0
    while cnt < num:
        loci = rng.randint(0, h - oh)
        locj = rng.randint(0, w - ow)
        col = rng.choice(colopts)
        colopts = remove(col, colopts)
        obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        gi2 = fill(gi, col, obj)
        if (
            color(
                argmin(
                    sfilter(partition(gi2), fork(equality, size, fork(multiply, height, width))),
                    fork(multiply, height, width),
                )
            )
            != col
        ):
            break
        else:
            gi = gi2
            go = canvas(col, shape(obj))
        oh = unifint(rng, diff_lb, diff_ub, (max(0, oh - 4), oh - 1))
        ow = unifint(rng, diff_lb, diff_ub, (max(0, ow - 4), ow - 1))
        if oh < 1 or ow < 1:
            break
        cnt += 1
    return {"input": gi, "output": go}


def generate_1bfc4729(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (4, 30))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    if h % 2 == 1:
        h = rng.choice((max(4, h - 1), min(30, h + 1)))
    alocj = unifint(rng, diff_lb, diff_ub, (w // 2, w - 1))
    if rng.choice((True, False)):
        alocj = max(min(w // 2, alocj - w // 2), 1)
    aloci = rng.randint(1, h // 2 - 1)
    blocj = unifint(rng, diff_lb, diff_ub, (w // 2, w - 1))
    if rng.choice((True, False)):
        blocj = max(min(w // 2, blocj - w // 2), 1)
    bloci = rng.randint(h // 2, h - 2)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    acol = rng.choice(remcols)
    remcols = remove(acol, remcols)
    bcol = rng.choice(remcols)
    gi = canvas(bgc, (h, w))
    aloc = (aloci, alocj)
    bloc = (bloci, blocj)
    gi = fill(gi, acol, {aloc})
    gi = fill(gi, bcol, {bloc})
    go = fill(gi, acol, hfrontier(aloc))
    go = fill(go, bcol, hfrontier(bloc))
    go = fill(go, acol, connect((0, 0), (0, w - 1)))
    go = fill(go, bcol, connect((h - 1, 0), (h - 1, w - 1)))
    go = fill(go, acol, connect((0, 0), (h // 2 - 1, 0)))
    go = fill(go, acol, connect((0, w - 1), (h // 2 - 1, w - 1)))
    go = fill(go, bcol, connect((h // 2, 0), (h - 1, 0)))
    go = fill(go, bcol, connect((h // 2, w - 1), (h - 1, w - 1)))
    return {"input": gi, "output": go}


def generate_47c1f68c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 14))
    w = unifint(rng, diff_lb, diff_ub, (2, 14))
    bgc, linc = rng.sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    objc = rng.choice(remcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(rng, diff_lb, diff_ub, (1, h * w - 1))
    bx = asindices(canv)
    obj = {rng.choice(totuple(bx))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, obj)
        ch = rng.choice(totuple(bx & dns))
        obj.add(ch)
        bx = bx - {ch}
    obj = recolor(objc, obj)
    gi = paint(canv, obj)
    gi1 = hconcat(hconcat(gi, canvas(linc, (h, 1))), canv)
    gi2 = hconcat(hconcat(canv, canvas(linc, (h, 1))), canv)
    gi = vconcat(vconcat(gi1, canvas(linc, (1, 2 * w + 1))), gi2)
    go = paint(canv, obj)
    go = hconcat(go, vmirror(go))
    go = vconcat(go, hmirror(go))
    go = replace(go, objc, linc)
    scf = rng.choice((identity, hmirror, vmirror, compose(hmirror, vmirror)))
    gi = scf(gi)
    go = scf(go)
    return {"input": gi, "output": go}


def generate_178fcbfb(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    iforb = set()
    jforb = set()
    mp = (h * w) // 3
    for col in (2, 1, 3):
        bnd = unifint(rng, diff_lb, diff_ub, (1, w if col == 2 else h // 2))
        for ndots in range(bnd):
            if col == 2:
                ij = rng.choice(sfilter(inds, lambda ij: last(ij) not in jforb))
                jforb.add(last(ij))
            if col == 1 or col == 3:
                ij = rng.choice(sfilter(inds, lambda ij: first(ij) not in iforb))
                iforb.add(first(ij))
            gi = fill(gi, col, initset(ij))
            go = fill(go, col, (vfrontier if col == 2 else hfrontier)(ij))
            inds = remove(ij, inds)
    return {"input": gi, "output": go}


def generate_ae4f1146(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    h = unifint(rng, diff_lb, diff_ub, (6, 30))
    w = unifint(rng, diff_lb, diff_ub, (6, 30))
    dh = unifint(rng, diff_lb, diff_ub, (2, h // 3))
    dw = unifint(rng, diff_lb, diff_ub, (2, w // 3))
    num = unifint(rng, diff_lb, diff_ub, (1, (h * w) // (2 * dh * dw)))
    cards = interval(0, dh * dw, 1)
    ccards = sorted(rng.sample(cards, min(num, len(cards))))
    sgs = []
    c1 = canvas(fgc, (dh, dw))
    inds = totuple(asindices(c1))
    for card in ccards:
        x = rng.sample(inds, card)
        x1 = fill(c1, 1, x)
        sgs.append(asobject(x1))
    go = paint(c1, sgs[-1])
    gi = canvas(bgc, (h, w))
    inds2 = asindices(canvas(bgc, (h - dh, w - dw)))
    maxtr = 10
    for sg in sgs[::-1]:
        if len(inds2) == 0:
            break
        loc = rng.choice(totuple(inds2))
        plcd = shift(sg, loc)
        tr = 0
        while (not toindices(plcd).issubset(inds2)) and tr < maxtr:
            loc = rng.choice(totuple(inds2))
            plcd = shift(sg, loc)
            tr += 1
        if tr < maxtr:
            inds2 = difference(inds2, toindices(plcd) | outbox(plcd))
            gi = paint(gi, plcd)
    return {"input": gi, "output": go}


def generate_3de23699(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    c = canvas(bgc, (h, w))
    hi = unifint(rng, diff_lb, diff_ub, (4, h))
    wi = unifint(rng, diff_lb, diff_ub, (4, w))
    loci = rng.randint(0, h - hi)
    locj = rng.randint(0, w - wi)
    remcols = remove(bgc, cols)
    ccol = rng.choice(remcols)
    remcols = remove(ccol, remcols)
    ncol = rng.choice(remcols)
    tmpo = frozenset({(loci, locj), (loci + hi - 1, locj + wi - 1)})
    cnds = totuple(backdrop(inbox(tmpo)))
    mp = len(cnds) // 2
    dev = unifint(rng, diff_lb, diff_ub, (0, mp))
    ncnds = rng.choice((dev, len(cnds) - dev))
    ncnds = min(max(0, ncnds), len(cnds))
    ss = rng.sample(cnds, ncnds)
    gi = fill(c, ccol, corners(tmpo))
    gi = fill(gi, ncol, ss)
    go = trim(crop(switch(gi, ccol, ncol), (loci, locj), (hi, wi)))
    return {"input": gi, "output": go}


def generate_7ddcd7ec(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    crns = (((0, 0), (-1, -1)), ((0, 1), (-1, 1)), ((1, 0), (1, -1)), ((1, 1), (1, 1)))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (0, 4))
    chos = rng.sample(crns, num)
    loci = rng.randint(0, h - 2)
    locj = rng.randint(0, w - 2)
    loc = (loci, locj)
    remcols = remove(bgc, cols)
    for sp, dr in crns:
        sp2 = add(loc, sp)
        col = rng.choice(remcols)
        gi = fill(gi, col, {sp2})
        go = fill(go, col, {sp2})
        if (sp, dr) in chos:
            gi = fill(gi, col, {add(sp2, dr)})
            go = fill(go, col, shoot(sp2, dr))
    return {"input": gi, "output": go}


def generate_5c2c9af4(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    boxhd = unifint(rng, diff_lb, diff_ub, (0, h // 2))
    boxwd = unifint(rng, diff_lb, diff_ub, (0, w // 2))
    boxh = rng.choice((boxhd, h - boxhd))
    boxw = rng.choice((boxwd, w - boxwd))
    if boxh % 2 == 0:
        boxh = rng.choice((boxh - 1, boxh + 1))
    if boxw % 2 == 0:
        boxw = rng.choice((boxw - 1, boxw + 1))
    boxh = min(max(1, boxh), h if h % 2 == 1 else h - 1)
    boxw = min(max(1, boxw), w if w % 2 == 1 else w - 1)
    boxshap = (boxh, boxw)
    loci = rng.randint(0, h - boxh)
    locj = rng.randint(0, w - boxw)
    loc = (loci, locj)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    cpi = loci + boxh // 2
    cpj = locj + boxw // 2
    cp = (cpi, cpj)
    A = (loci, locj)
    B = (loci + boxh - 1, locj + boxw - 1)
    gi = fill(c, fgc, {A, B, cp})
    go = fill(c, fgc, {A, B, cp})
    cond = True
    ooo = {A, B, cp}
    if hline(ooo) and len(ooo) == 3:
        go = fill(go, fgc, hfrontier(cp))
        cond = False
    if vline(ooo) and len(ooo) == 3:
        go = fill(go, fgc, vfrontier(cp))
        cond = False
    k = 1
    while cond:
        f1 = k * (boxh // 2)
        f2 = k * (boxw // 2)
        ulci = cpi - f1
        ulcj = cpj - f2
        lrci = cpi + f1
        lrcj = cpj + f2
        ulc = (ulci, ulcj)
        lrc = (lrci, lrcj)
        bx = box(frozenset({ulc, lrc}))
        go2 = fill(go, fgc, bx)
        cond = go != go2
        go = go2
        k += 1
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_0b148d64(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    itv = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (7, 30))
    bgc = rng.choice(itv)
    remitv = remove(bgc, itv)
    g = canvas(bgc, (h, w))
    x = rng.randint(3, h - 3)
    y = rng.randint(3, w - 3)
    di = rng.randint(2, h - x - 1)
    dj = rng.randint(2, w - y - 1)
    A = backdrop(frozenset({(0, 0), (x, y)}))
    B = backdrop(frozenset({(x + di, 0), (h - 1, y)}))
    C = backdrop(frozenset({(0, y + dj), (x, w - 1)}))
    D = backdrop(frozenset({(x + di, y + dj), (h - 1, w - 1)}))
    cola = rng.choice(remitv)
    colb = rng.choice(remove(cola, remitv))
    trg = rng.choice((A, B, C, D))
    rem = remove(trg, (A, B, C, D))
    subf = lambda bx: {
        rng.choice(totuple(connect(ulcorner(bx), urcorner(bx)))),
        rng.choice(totuple(connect(ulcorner(bx), llcorner(bx)))),
        rng.choice(totuple(connect(urcorner(bx), lrcorner(bx)))),
        rng.choice(totuple(connect(llcorner(bx), lrcorner(bx)))),
    }
    sampler = lambda bx: set(rng.sample(totuple(bx), len(bx) - unifint(rng, diff_lb, diff_ub, (0, len(bx) - 1))))
    gi = fill(g, cola, sampler(trg) | subf(trg))
    for r in rem:
        gi = fill(gi, colb, sampler(r) | subf(r))
    go = subgrid(frozenset(trg), gi)
    return {"input": gi, "output": go}


def generate_beb8660c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    h = unifint(rng, diff_lb, diff_ub, (w, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    k = min(8, w - 1)
    k = unifint(rng, diff_lb, diff_ub, (1, k))
    co = rng.sample(remcols, k)
    wds = sorted(rng.sample(interval(1, w, 1), k))
    for j, (c, l) in enumerate(zip(co, wds)):
        j = h - k - 1 + j
        gi = fill(gi, c, connect((j, 0), (j, l - 1)))
    gi = fill(gi, 8, connect((h - 1, 0), (h - 1, w - 1)))
    go = vmirror(gi)
    gi = list(list(r) for r in gi[:-1])
    rng.shuffle(gi)
    gi = tuple(tuple(r) for r in gi)
    gi = gi + go[-1:]
    gif = tuple()
    for r in gi:
        nbc = r.count(bgc)
        ofs = rng.randint(0, nbc)
        gif = gif + (r[-ofs:] + r[:-ofs],)
    gi = vmirror(gif)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_8d510a79(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    barloci = rng.randint(2, h - 3)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    barcol = rng.choice(remcols)
    gi = canvas(bgc, (h, w))
    bar = connect((barloci, 0), (barloci, w - 1))
    gi = fill(gi, barcol, bar)
    go = tuple(e for e in gi)
    jinds = interval(0, w, 1)
    numtop = unifint(rng, diff_lb, diff_ub, (1, w - 1))
    numbot = unifint(rng, diff_lb, diff_ub, (1, w - 1))
    tops = rng.sample(jinds, numtop)
    bots = rng.sample(jinds, numbot)
    for t in tops:
        loci = rng.randint(0, barloci - 2)
        col = rng.choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (0, t)))
        else:
            go = fill(go, col, connect(loc, (barloci - 1, t)))
    for t in bots:
        loci = rng.randint(barloci + 2, h - 1)
        col = rng.choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (h - 1, t)))
        else:
            go = fill(go, col, connect(loc, (barloci + 1, t)))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_7468f01a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    sgc, fgc = rng.sample(remcols, 2)
    oh = unifint(rng, diff_lb, diff_ub, (2, max(2, int(h * (2 / 3)))))
    ow = unifint(rng, diff_lb, diff_ub, (2, max(2, int(w * (2 / 3)))))
    gi = canvas(bgc, (h, w))
    go = canvas(sgc, (oh, ow))
    bounds = asindices(go)
    shp = {ORIGIN}
    nc = unifint(rng, diff_lb, diff_ub, (0, max(1, (oh * ow) // 2)))
    for j in range(nc):
        shp.add(rng.choice(totuple((bounds - shp) & mapply(dneighbors, shp))))
    go = fill(go, fgc, shp)
    objx = asobject(vmirror(go))
    loci = rng.randint(0, h - oh)
    locj = rng.randint(0, w - ow)
    gi = paint(gi, shift(objx, (loci, locj)))
    return {"input": gi, "output": go}


def generate_09629e4f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 5))
    w = unifint(rng, diff_lb, diff_ub, (2, 5))
    nrows, ncolumns = h, w
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    barcol = rng.choice(remcols)
    remcols = remove(barcol, remcols)
    ncols = unifint(rng, diff_lb, diff_ub, (2, min(7, (h * w) - 2)))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    fullh, fullw = h * nrows + nrows - 1, w * ncolumns + ncolumns - 1
    gi = canvas(barcol, (fullh, fullw))
    locs = totuple(product(interval(0, fullh, h + 1), interval(0, fullw, w + 1)))
    trgloc = rng.choice(locs)
    remlocs = remove(trgloc, locs)
    colssf = rng.sample(remcols, ncols)
    colsss = remove(rng.choice(colssf), colssf)
    trgssf = rng.sample(inds, ncols - 1)
    gi = fill(gi, bgc, shift(inds, trgloc))
    for ij, cl in zip(trgssf, colsss):
        gi = fill(gi, cl, {add(trgloc, ij)})
    for rl in remlocs:
        trgss = rng.sample(inds, ncols)
        tmpg = tuple(e for e in c)
        for ij, cl in zip(trgss, colssf):
            tmpg = fill(tmpg, cl, {ij})
        gi = paint(gi, shift(asobject(tmpg), rl))
    go = canvas(bgc, (fullh, fullw))
    go = fill(go, barcol, ofcolor(gi, barcol))
    for ij, cl in zip(trgssf, colsss):
        go = fill(go, cl, shift(inds, multiply(ij, (h + 1, w + 1))))
    return {"input": gi, "output": go}


def generate_4347f46a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
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
            go = fill(go, col, box(obj))
            succ += 1
            indss = indss - bd
        tr += 1
    return {"input": gi, "output": go}


def generate_6d58a25d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    shp = normalize(frozenset({(0, 0), (1, 0), (1, 1), (1, -1), (2, -1), (2, -2), (2, 1), (2, 2), (3, 3), (3, -3)}))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (8, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    c1 = rng.choice(remcols)
    c2 = rng.choice(remove(c1, remcols))
    loci = rng.randint(0, h - 4)
    locj = rng.randint(0, w - 7)
    plcd = shift(shp, (loci, locj))
    rem = difference(inds, plcd)
    nnoise = unifint(rng, diff_lb, diff_ub, (1, max(1, len(rem) // 2 - 1)))
    nois = rng.sample(rem, nnoise)
    gi = fill(c, c2, nois)
    gi = fill(gi, c1, plcd)
    ff = lambda ij: len(intersection(shoot(ij, (-1, 0)), plcd)) > 0
    trg = sfilter(nois, ff)
    gg = lambda ij: valmax(sfilter(plcd, lambda kl: kl[1] == ij[1]), first) + 1
    kk = lambda ij: connect((gg(ij), ij[1]), (h - 1, ij[1]))
    fullres = mapply(kk, trg)
    go = fill(gi, c2, fullres)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_363442ee(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 3))
    w = unifint(rng, diff_lb, diff_ub, (1, 3))
    h = h * 2 + 1
    w = w * 2 + 1
    nremh = unifint(rng, diff_lb, diff_ub, (2, 30 // h))
    nremw = unifint(rng, diff_lb, diff_ub, (2, (30 - w - 1) // w))
    rsh = nremh * h
    rsw = nremw * w
    rss = (rsh, rsw)
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    barcol = rng.choice(remcols)
    remcols = remove(barcol, remcols)
    rsi = canvas(bgc, rss)
    rso = canvas(bgc, rss)
    ls = canvas(bgc, ((nremh - 1) * h, w))
    ulc = canvas(bgc, (h, w))
    bar = canvas(barcol, (nremh * h, 1))
    dotcands = totuple(product(interval(0, rsh, h), interval(0, rsw, w)))
    dotcol = rng.choice(remcols)
    dev = unifint(rng, diff_lb, diff_ub, (1, len(dotcands) // 2))
    ndots = rng.choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = rng.sample(dotcands, ndots)
    nfullremcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    fullremcols = rng.sample(remcols, nfullremcols)
    for ij in asindices(ulc):
        ulc = fill(ulc, rng.choice(fullremcols), {ij})
    ulco = asobject(ulc)
    osf = (h // 2, w // 2)
    for d in dots:
        rsi = fill(rsi, dotcol, {add(osf, d)})
        rso = paint(rso, shift(ulco, d))
    gi = hconcat(hconcat(vconcat(ulc, ls), bar), rsi)
    go = hconcat(hconcat(vconcat(ulc, ls), bar), rso)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_855e0971(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nbarsd = unifint(rng, diff_lb, diff_ub, (1, 4))
    nbars = rng.choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    h = unifint(rng, diff_lb, diff_ub, (nbars, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        j = rng.randint(0, nbars - 1)
        barsizes[j] += 1
    gi = tuple()
    go = tuple()
    locs = interval(0, w, 1)
    dotc = rng.choice(cols)
    remcols = remove(dotc, cols)
    lastcol = -1
    nloclbs = [rng.choice((0, 1)) for k in range(len(barsizes))]
    if sum(nloclbs) < 2:
        loc1, loc2 = rng.sample(interval(0, len(nloclbs), 1), 2)
        nloclbs[loc1] = 1
        nloclbs[loc2] = 1
    for bs, nloclb in zip(barsizes, nloclbs):
        col = rng.choice(remove(lastcol, remcols))
        gim = canvas(col, (bs, w))
        gom = canvas(col, (bs, w))
        nl = unifint(rng, diff_lb, diff_ub, (nloclb, w // 2))
        chlocs = rng.sample(locs, nl)
        for jj in chlocs:
            idx = (rng.randint(0, bs - 1), jj)
            gim = fill(gim, dotc, {idx})
            gom = fill(gom, dotc, vfrontier(idx))
        lastcol = col
        gi = gi + gim
        go = go + gom
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_137eaa0f(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 4))
    w = unifint(rng, diff_lb, diff_ub, (2, 4))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    dotc = rng.choice(remcols)
    remcols = remove(dotc, remcols)
    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = rng.choice(inds)
    reminds = remove(loc, inds)
    nc = unifint(rng, diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    choscols = rng.sample(remcols, nc)
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = rng.choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[rng.choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)
    gih = unifint(rng, diff_lb, diff_ub, (min(h, w) * 2, 30))
    giw = unifint(rng, diff_lb, diff_ub, (min(h, w) * 2, 30))
    objs = tuple(normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]}))) for c in choscols)
    maxtr = min(h, w) * 2
    maxtrtot = 1000
    while True:
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                loci = rng.randint(0, gih - oh)
                locj = rng.randint(0, giw - ow)
                plcd = shift(obj, (loci, locj))
                tr += 1
                if toindices(plcd).issubset(inds):
                    succ2 = True
            if succ2:
                gi = paint(gi, plcd)
                inds = difference(inds, toindices(plcd))
                inds = difference(inds, mapply(neighbors, toindices(plcd)))
            else:
                succ = False
                break
        if succ:
            break
        maxtrtot += 1
        if maxtrtot < 1000:
            break
        maxtr = int(maxtr * 1.5)
        gih = rng.randint(gih, 30)
        giw = rng.randint(giw, 30)
    return {"input": gi, "output": go}


def generate_31aa019c(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(rng, diff_lb, diff_ub, (5, 30))
        w = unifint(rng, diff_lb, diff_ub, (5, 30))
        bgc = rng.choice(cols)
        remcols = remove(bgc, cols)
        canv = canvas(bgc, (h, w))
        inds = totuple(asindices(canv))
        mp = (h * w) // 2 - 1
        ncols = unifint(rng, diff_lb, diff_ub, (2, min(9, mp // 2 - 1)))
        chcols = rng.sample(cols, ncols)
        trgcol = chcols[0]
        chcols = chcols[1:]
        dic = {c: set() for c in chcols}
        nnoise = unifint(rng, diff_lb, diff_ub, (2 * (ncols - 1), mp))
        locc = rng.choice(inds)
        inds = remove(locc, inds)
        noise = rng.sample(inds, nnoise)
        for c in chcols:
            ij = rng.choice(inds)
            dic[c].add(ij)
            inds = remove(ij, inds)
        for c in chcols:
            ij = rng.choice(inds)
            dic[c].add(ij)
            inds = remove(ij, inds)
        for ij in noise:
            c = rng.choice(chcols)
            dic[c].add(ij)
            inds = remove(ij, inds)
        gi = fill(canv, trgcol, {locc})
        for c, ss in dic.items():
            gi = fill(gi, c, ss)
        gi = fill(gi, trgcol, {locc})
        if len(sfilter(palette(gi), lambda c: colorcount(gi, c) == 1)) == 1:
            break
    lc = leastcolor(gi)
    locc = ofcolor(gi, lc)
    go = fill(canv, lc, locc)
    go = fill(go, 2, neighbors(first(locc)))
    return {"input": gi, "output": go}


def generate_2bee17df(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (7, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    indord1 = apply(tojvec, interval(0, w, 1))
    indord2 = apply(rbind(astuple, w - 1), interval(1, h - 1, 1))
    indord3 = apply(lbind(astuple, h - 1), interval(w - 1, 0, -1))
    indord4 = apply(toivec, interval(h - 1, 0, -1))
    indord = indord1 + indord2 + indord3 + indord4
    k = len(indord)
    sp = rng.randint(0, k)
    arr = indord[sp:] + indord[:sp]
    ep = rng.randint(k // 2 - 3, k // 2 + 1)
    a = arr[:ep]
    b = arr[ep:]
    cola = rng.choice(remcols)
    remcols = remove(cola, remcols)
    colb = rng.choice(remcols)
    gi = fill(c, cola, a)
    gi = fill(gi, colb, b)
    nr = unifint(rng, diff_lb, diff_ub, (1, min(4, min(h, w) // 2)))
    for kk in range(nr):
        ring = box(frozenset({(1 + kk, 1 + kk), (h - 1 - kk, w - 1 - kk)}))
        for br in (cola, colb):
            blacks = ofcolor(gi, br)
            bcands = totuple(ring & ofcolor(gi, bgc) & mapply(dneighbors, ofcolor(gi, br)))
            jj = len(bcands)
            jj2 = rng.randint(max(0, jj // 2 - 2), min(jj, jj // 2 + 1))
            ss = rng.sample(bcands, jj2)
            gi = fill(gi, br, ss)
    res = shift(merge(frontiers(trim(gi))), (1, 1))
    go = fill(gi, 3, res)
    return {"input": gi, "output": go}


def generate_50cb2852(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, 8))
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
            go = fill(go, 8, bd)
            go = fill(go, col, box(obj))
            box(obj)
            succ += 1
            indss = indss - bd
        tr += 1
    return {"input": gi, "output": go}


def generate_662c240a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    d = unifint(rng, diff_lb, diff_ub, (2, 7))
    ng = unifint(rng, diff_lb, diff_ub, (2, 30 // d))
    nc = unifint(rng, diff_lb, diff_ub, (2, min(9, d**2)))
    c = canvas(-1, (d, d))
    inds = totuple(asindices(c))
    tria = sfilter(inds, lambda ij: ij[1] >= ij[0])
    tcolset = rng.sample(cols, nc)
    triaf = frozenset((rng.choice(tcolset), ij) for ij in tria)
    triaf = triaf | dmirror(triaf)
    gik = paint(c, triaf)
    ndistinv = unifint(rng, diff_lb, diff_ub, (0, (d * (d - 1) // 2 - 1)))
    ndist = d * (d - 1) // 2 - ndistinv
    distinds = rng.sample(difference(inds, sfilter(inds, lambda ij: ij[0] == ij[1])), ndist)

    for ij in distinds:
        if gik[ij[0]][ij[1]] == gik[ij[1]][ij[0]]:
            gik = fill(gik, rng.choice(remove(gik[ij[0]][ij[1]], tcolset)), {ij})
        else:
            gik = fill(gik, gik[ij[1]][ij[0]], {ij})
    gi = gik
    go = tuple(e for e in gik)
    concatf = rng.choice((hconcat, vconcat))
    for k in range(ng - 1):
        tria = sfilter(inds, lambda ij: ij[1] >= ij[0])
        tcolset = rng.sample(cols, nc)
        triaf = frozenset((rng.choice(tcolset), ij) for ij in tria)
        triaf = triaf | dmirror(triaf)
        gik = paint(c, triaf)
        if rng.choice((True, False)):
            gi = concatf(gi, gik)
        else:
            gi = concatf(gik, gi)
    return {"input": gi, "output": go}


def generate_e8593010(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    a = frozenset({frozenset({ORIGIN})})
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
    a, b, c = totuple(a), totuple(b), totuple(c)
    prs = [(a, 3), (b, 2), (c, 1)]
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    fgc = rng.choice(remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    reminds = asindices(gi)
    nobjs = unifint(rng, diff_lb, diff_ub, (1, ((h * w) // 2) // 2))
    maxtr = 10
    for k in range(nobjs):
        ntr = 0
        objs, col = rng.choice(prs)
        obj = rng.choice(objs)
        while ntr < maxtr:
            if len(reminds) == 0:
                break
            loc = rng.choice(totuple(reminds))
            olcd = shift(obj, loc)
            if olcd.issubset(reminds):
                gi = fill(gi, fgc, olcd)
                go = fill(go, col, olcd)
                reminds = (reminds - olcd) - mapply(dneighbors, olcd)
                break
            ntr += 1
    return {"input": gi, "output": go}


def generate_d9f24cd1(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    remcols = remove(linc, remcols)
    dotc = rng.choice(remcols)
    locopts = interval(1, w - 1, 1)
    maxnloc = (w - 2) // 2
    nlins = unifint(rng, diff_lb, diff_ub, (1, maxnloc))
    locs = []
    for k in range(nlins):
        if len(locopts) == 0:
            break
        loc = rng.choice(locopts)
        locopts = remove(loc, locopts)
        locopts = remove(loc - 1, locopts)
        locopts = remove(loc + 1, locopts)
        locs.append(loc)
    ndots = unifint(rng, diff_lb, diff_ub, (1, maxnloc))
    locopts = interval(1, w - 1, 1)
    dotlocs = []
    for k in range(ndots):
        if len(locopts) == 0:
            break
        loc = rng.choice(locopts)
        locopts = remove(loc, locopts)
        locopts = remove(loc - 1, locopts)
        locopts = remove(loc + 1, locopts)
        dotlocs.append(loc)
    gi = canvas(bgc, (h, w))
    for l in locs:
        gi = fill(gi, linc, {(h - 1, l)})
    dotlocs2 = []
    for l in dotlocs:
        jj = rng.randint(1, h - 2)
        gi = fill(gi, dotc, {(jj, l)})
        dotlocs2.append(jj)
    go = tuple(e for e in gi)
    for linloc in locs:
        if linloc in dotlocs:
            jj = dotlocs2[dotlocs.index(linloc)]
            go = fill(go, linc, connect((h - 1, linloc), (jj + 1, linloc)))
            go = fill(go, linc, connect((jj + 1, linloc + 1), (0, linloc + 1)))
        else:
            go = fill(go, linc, connect((h - 1, linloc), (0, linloc)))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_90c28cc7(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (2, 10))
    w = unifint(rng, diff_lb, diff_ub, (2, 10))
    nc = unifint(rng, diff_lb, diff_ub, (2, 9))
    gi = canvas(-1, (h, w))
    inds = totuple(asindices(gi))
    colss = rng.sample(cols, nc)
    for ij in inds:
        gi = fill(gi, rng.choice(colss), {ij})
    gi = dmirror(dedupe(dmirror(dedupe(gi))))
    go = tuple(e for e in gi)
    h, w = shape(gi)
    fullh = unifint(rng, diff_lb, diff_ub, (h, 30))
    fullw = unifint(rng, diff_lb, diff_ub, (w, 30))
    inh = unifint(rng, diff_lb, diff_ub, (h, fullh))
    inw = unifint(rng, diff_lb, diff_ub, (w, fullw))
    while h < inh or w < inw:
        opts = []
        if h < inh:
            opts.append((h, identity))
        elif w < inw:
            opts.append((w, dmirror))
        dim, mirrf = rng.choice(opts)
        idx = rng.randint(0, dim - 1)
        gi = mirrf(gi)
        gi = gi[: idx + 1] + gi[idx:]
        gi = mirrf(gi)
        h, w = shape(gi)
    while h < fullh or w < fullw:
        opts = []
        if h < fullh:
            opts.append(identity)
        elif w < fullw:
            opts.append(dmirror)
        mirrf = rng.choice(opts)
        gi = mirrf(gi)
        gi = merge(tuple(rng.sample((((0,) * width(gi),), gi), 2)))
        gi = mirrf(gi)
        h, w = shape(gi)
    return {"input": gi, "output": go}


def generate_321b1fc6(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
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
    go = fill(go, bgc, shpp)
    return {"input": gi, "output": go}


def generate_6455b5f5(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 8))
    while True:
        h = unifint(rng, diff_lb, diff_ub, (6, 30))
        w = unifint(rng, diff_lb, diff_ub, (6, 30))
        bgc = rng.choice(cols)
        fgc = rng.choice(remove(bgc, cols))
        gi = canvas(bgc, (h, w))
        ub = int((h * w) ** 0.5 * 1.5)
        num = unifint(rng, diff_lb, diff_ub, (1, ub))
        for k in range(num):
            objs = colorfilter(objects(gi, T, T, F), bgc)
            eligobjs = sfilter(objs, lambda o: height(o) > 2 or width(o) > 2)
            if len(eligobjs) == 0:
                break
            if rng.choice((True, False)):
                ro = argmax(eligobjs, size)
            else:
                ro = rng.choice(totuple(eligobjs))
            if rng.choice((True, False)):
                vfr = height(ro) < width(ro)
            else:
                vfr = rng.choice((True, False))
            if vfr and width(ro) < 3:
                vfr = False
            if (not vfr) and height(ro) < 3:
                vfr = True
            if vfr:
                j = rng.randint(leftmost(ro) + 1, rightmost(ro) - 1)
                ln = connect((uppermost(ro), j), (lowermost(ro), j))
            else:
                j = rng.randint(uppermost(ro) + 1, lowermost(ro) - 1)
                ln = connect((j, leftmost(ro)), (j, rightmost(ro)))
            gi = fill(gi, fgc, ln)
        objs = colorfilter(objects(gi, T, T, F), bgc)
        if valmin(objs, size) != valmax(objs, size):
            break
    lblues = mfilter(objs, matcher(size, valmin(objs, size)))
    dblues = mfilter(objs, matcher(size, valmax(objs, size)))
    go = fill(gi, 8, lblues)
    go = fill(go, 1, dblues)
    return {"input": gi, "output": go}


def generate_4c5c2cf0(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
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
    while ncells == 5 and shape(cells) == (3, 3):
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
    return {"input": gi, "output": go}


def generate_56ff96f3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
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
        oh = rng.randint(2, 7)
        ow = rng.randint(2, 7)
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
            if rng.choice((True, False)):
                cnrs = ((loci, locj), (loci + oh - 1, locj + ow - 1))
            else:
                cnrs = ((loci + oh - 1, locj), (loci, locj + ow - 1))
            gi = fill(gi, col, cnrs)
            go = fill(go, col, bd)
            succ += 1
            indss = indss - bd
        tr += 1
    return {"input": gi, "output": go}


def generate_2c608aff(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
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
    dcol = rng.choice(remcols)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - backdrop(outbox(bd)))
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(rng, diff_lb, diff_ub, (0, noiseb))
    noise = rng.sample(reminds, nnoise)
    gi = fill(gi, dcol, noise)
    go = tuple(e for e in gi)
    hs = interval(loci, loci + boxh, 1)
    ws = interval(locj, locj + boxw, 1)
    for ij in noise:
        a, b = ij
        if a in hs:
            go = fill(go, dcol, connect(ij, (a, locj)))
        elif b in ws:
            go = fill(go, dcol, connect(ij, (loci, b)))
    go = fill(go, ccol, bd)
    return {"input": gi, "output": go}


def generate_e98196ab(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 14))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    linc = rng.choice(remcols)
    remcols = remove(linc, remcols)
    topc = rng.choice(remcols)
    remcols = remove(topc, remcols)
    botc = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    nocc = unifint(rng, diff_lb, diff_ub, (2, (h * w) // 2))
    subs = rng.sample(inds, nocc)
    numa = rng.randint(1, nocc - 1)
    A = rng.sample(subs, numa)
    B = difference(subs, A)
    topg = fill(c, topc, A)
    botg = fill(c, botc, B)
    go = fill(topg, botc, B)
    br = canvas(linc, (1, w))
    gi = vconcat(vconcat(topg, br), botg)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_c9f8e694(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = 0
    remcols = remove(bgc, cols)
    sqc = rng.choice(remcols)
    remcols = remove(sqc, remcols)
    ncols = unifint(rng, diff_lb, diff_ub, (1, min(h, 8)))
    nsq = unifint(rng, diff_lb, diff_ub, (1, 8))
    gir = canvas(bgc, (h, w - 1))
    gil = tuple((rng.choice(remcols),) for j in range(h))
    inds = asindices(gir)
    succ = 0
    fails = 0
    maxfails = nsq * 5
    while succ < nsq and fails < maxfails:
        loci = rng.randint(0, h - 3)
        locj = rng.randint(0, w - 3)
        lock = rng.randint(loci + 1, min(loci + max(1, 2 * h // 3), h - 1))
        locl = rng.randint(locj + 1, min(locj + max(1, 2 * w // 3), w - 1))
        bd = backdrop(frozenset({(loci, locj), (lock, locl)}))
        if bd.issubset(inds):
            gir = fill(gir, sqc, bd)
            succ += 1
            indss = inds - bd
        else:
            fails += 1
    locs = ofcolor(gir, sqc)
    gil = tuple(e if idx in apply(first, locs) else (bgc,) for idx, e in enumerate(gil))
    fullobj = toobject(locs, hupscale(gil, w))
    gi = hconcat(gil, gir)
    giro = paint(gir, fullobj)
    go = hconcat(gil, giro)
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_eb5a1d5d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    d = unifint(rng, diff_lb, diff_ub, (2, 10))
    go = canvas(-1, (d * 2 - 1, d * 2 - 1))
    colss = rng.sample(cols, d)
    for j, cc in enumerate(colss):
        go = fill(go, cc, box(frozenset({(j, j), (2 * d - 2 - j, 2 * d - 2 - j)})))
    nvenl = unifint(rng, diff_lb, diff_ub, (0, 30 - d))
    nhenl = unifint(rng, diff_lb, diff_ub, (0, 30 - d))
    enl = [nvenl, nhenl]
    gi = tuple(e for e in go)
    while (enl[0] > 0 or enl[1] > 0) and max(shape(gi)) < 30:
        opts = []
        if enl[0] > 0:
            opts.append((identity, 0))
        if enl[1] > 0:
            opts.append((dmirror, 1))
        mirrf, ch = rng.choice(opts)
        gi = mirrf(gi)
        idx = rng.randint(0, len(gi) - 1)
        gi = gi[: idx + 1] + gi[idx:]
        gi = mirrf(gi)
        enl[ch] -= 1
    return {"input": gi, "output": go}


def generate_82819916(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    ass, bss = rng.sample(remcols, 2)
    itv = interval(0, w, 1)
    na = rng.randint(2, w - 2)
    alocs = rng.sample(itv, na)
    blocs = difference(itv, alocs)
    if min(alocs) > min(blocs):
        alocs, blocs = blocs, alocs
    llocs = rng.randint(0, h - 1)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ass, {(llocs, j) for j in alocs})
    gi = fill(gi, bss, {(llocs, j) for j in blocs})
    numl = unifint(rng, diff_lb, diff_ub, (1, max(1, (h - 1) // 2)))
    remlocs = remove(llocs, interval(0, h, 1))
    for k in range(numl):
        lloc = rng.choice(remlocs)
        remlocs = remove(lloc, remlocs)
        a, b = rng.sample(remcols, 2)
        gi = fill(gi, a, {(lloc, j) for j in alocs})
        gi = fill(gi, b, {(lloc, j) for j in blocs})
    cutoff = min(blocs) + 1
    go = tuple(e for e in gi)
    gi = fill(gi, bgc, backdrop(frozenset({(0, cutoff), (h - 1, w - 1)})))
    gi = fill(gi, ass, {(llocs, j) for j in alocs})
    gi = fill(gi, bss, {(llocs, j) for j in blocs})
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_5daaa586(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (7, 30))
    w = unifint(rng, diff_lb, diff_ub, (7, 30))
    loci1 = rng.randint(1, h - 4)
    locj1 = rng.randint(1, w - 4)
    loci1dev = unifint(rng, diff_lb, diff_ub, (0, loci1 - 1))
    locj1dev = unifint(rng, diff_lb, diff_ub, (0, locj1 - 1))
    loci1 -= loci1dev
    locj1 -= locj1dev
    loci2 = unifint(rng, diff_lb, diff_ub, (loci1 + 2, h - 2))
    locj2 = unifint(rng, diff_lb, diff_ub, (locj1 + 2, w - 2))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    c1, c2, c3, c4 = rng.sample(remcols, 4)
    f1 = recolor(c1, hfrontier(toivec(loci1)))
    f2 = recolor(c2, hfrontier(toivec(loci2)))
    f3 = recolor(c3, vfrontier(tojvec(locj1)))
    f4 = recolor(c4, vfrontier(tojvec(locj2)))
    gi = canvas(bgc, (h, w))
    fronts = [f1, f2, f3, f4]
    rng.shuffle(fronts)
    for fr in fronts:
        gi = paint(gi, fr)
    cands = totuple(ofcolor(gi, bgc))
    nn = len(cands)
    nnoise = unifint(rng, diff_lb, diff_ub, (1, max(1, nn // 3)))
    noise = rng.sample(cands, nnoise)
    gi = fill(gi, c1, noise)
    while len(frontiers(gi)) > 4:
        gi = fill(gi, bgc, noise)
        nnoise = unifint(rng, diff_lb, diff_ub, (1, max(1, nn // 3)))
        noise = rng.sample(cands, nnoise)
        if len(set(noise) & ofcolor(gi, c1)) >= len(ofcolor(gi, bgc)):
            break
        gi = fill(gi, c1, noise)
    go = crop(gi, (loci1, locj1), (loci2 - loci1 + 1, locj2 - locj1 + 1))
    ns = ofcolor(go, c1)
    go = fill(go, c1, mapply(rbind(shoot, (-1, 0)), ns))
    rotf = rng.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def generate_68b16354(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
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
    go = hmirror(gi)
    return {"input": gi, "output": go}


def generate_bb43febb(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, 8))
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
            go = fill(go, col, box(obj))
            succ += 1
            indss = indss - bd
        tr += 1
    return {"input": gi, "output": go}


def generate_9ecd008a(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
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


def generate_f25ffba3(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (1, 14))
    h = h * 2 + 1
    w = unifint(rng, diff_lb, diff_ub, (3, 15))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    remcols = rng.sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(rng, diff_lb, diff_ub, (2, h * w - 2))
    bx = asindices(canv)
    obj = {(rng.choice(remcols), rng.choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = rng.choice(totuple(bx & dns))
        obj.add((rng.choice(remcols), ch))
        bx = bx - {ch}
    while uppermost(obj) > h // 2 - 1 or lowermost(obj) < h // 2 + 1:
        dns = mapply(neighbors, toindices(obj))
        ch = rng.choice(totuple(bx & dns))
        obj.add((rng.choice(remcols), ch))
        bx = bx - {ch}
    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)
    gi = hconcat(gix, canv)
    go = hconcat(gix, vmirror(gix))
    if rng.choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    if rng.choice((True, False)):
        gi = hmirror(gi)
        go = hmirror(go)
    return {"input": gi, "output": go}


def generate_3bdb4ada(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (10, 30))
    w = unifint(rng, diff_lb, diff_ub, (10, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(rng, diff_lb, diff_ub, (1, 8))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        if rng.choice((True, False)):
            oh = 3
            ow = unifint(rng, diff_lb, diff_ub, (1, max(1, w // 2 - 1))) * 2 + 1
        else:
            ow = 3
            oh = unifint(rng, diff_lb, diff_ub, (1, max(1, h // 2 - 1))) * 2 + 1
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
            go = fill(go, col, bd)
            if oh == 3:
                ln = {(loci + 1, j) for j in range(locj + 1, locj + ow, 2)}
            else:
                ln = {(j, locj + 1) for j in range(loci + 1, loci + oh, 2)}
            go = fill(go, bgc, ln)
            succ += 1
            indss = indss - bd
        tr += 1
    return {"input": gi, "output": go}


def generate_2013d3e2(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 10))
    w = h
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(rng, diff_lb, diff_ub, (1, 8))
    remcols = rng.sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(rng, diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {(rng.choice(remcols), rng.choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = rng.choice(totuple(bx & dns))
        obj.add((rng.choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    gi1 = hconcat(gi, rot90(gi))
    gi2 = hconcat(rot270(gi), rot180(gi))
    gi = vconcat(gi1, gi2)
    fullh = unifint(rng, diff_lb, diff_ub, (2 * h, 30))
    fullw = unifint(rng, diff_lb, diff_ub, (2 * w, 30))
    gio = asobject(gi)
    gic = canvas(bgc, (fullh, fullw))
    loci = rng.randint(0, fullh - 2 * h)
    locj = rng.randint(0, fullw - 2 * w)
    gi = paint(gic, shift(gio, (loci, locj)))
    reminds = difference(asindices(gi), ofcolor(gi, bgc))
    go = lefthalf(tophalf(subgrid(reminds, gi)))
    return {"input": gi, "output": go}


def generate_aabf363d(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 28))
    w = unifint(rng, diff_lb, diff_ub, (3, 28))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    cola = rng.choice(remcols)
    remcols = remove(cola, remcols)
    colb = rng.choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    sp = rng.choice(totuple(bounds))
    ub = min(h * w - 1, max(1, (2 / 3) * h * w))
    ncells = unifint(rng, diff_lb, diff_ub, (1, ub))
    shp = {sp}
    for k in range(ncells):
        ij = rng.choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = shift(shp, (1, 1))
    c2 = canvas(bgc, (h + 2, w + 2))
    gi = fill(c2, cola, shp)
    go = fill(c2, colb, shp)
    gi = fill(gi, colb, {rng.choice(totuple(ofcolor(gi, bgc)))})
    return {"input": gi, "output": go}


def generate_d037b0a7(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nlocs = unifint(rng, diff_lb, diff_ub, (1, w))
    locs = rng.sample(interval(0, w, 1), nlocs)
    for j in locs:
        col = rng.choice(remcols)
        loci = rng.randint(0, h - 1)
        loc = (loci, j)
        gi = fill(gi, col, {loc})
        go = fill(go, col, connect(loc, (h - 1, j)))
    return {"input": gi, "output": go}


def generate_e26a3af2(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nr = unifint(rng, diff_lb, diff_ub, (1, 10))
    w = unifint(rng, diff_lb, diff_ub, (4, 30))
    scols = rng.sample(cols, nr)
    sgs = [canvas(col, (2, w)) for col in scols]
    numexp = unifint(rng, diff_lb, diff_ub, (0, 30 - nr))
    for k in range(numexp):
        idx = rng.randint(0, nr - 1)
        sgs[idx] = sgs[idx] + sgs[idx][-1:]
    sgs2 = []
    for idx, col in enumerate(scols):
        sg = sgs[idx]
        a, b = shape(sg)
        ub = (a * b) // 2 - 1
        nnoise = unifint(rng, diff_lb, diff_ub, (0, ub))
        inds = totuple(asindices(sg))
        noise = rng.sample(inds, nnoise)
        oc = remove(col, cols)
        noise = frozenset({(rng.choice(oc), ij) for ij in noise})
        sg2 = paint(sg, noise)
        for idxx in [0, -1]:
            while sum([e == col for e in sg2[idxx]]) < w // 2:
                locs = [j for j, e in enumerate(sg2[idxx]) if e != col]
                ch = rng.choice(locs)
                if idxx == 0:
                    sg2 = (sg2[0][:ch] + (col,) + sg2[0][ch + 1 :],) + sg2[1:]
                else:
                    sg2 = sg2[:-1] + (sg2[-1][:ch] + (col,) + sg2[-1][ch + 1 :],)
        sgs2.append(sg2)
    gi = tuple(row for sg in sgs2 for row in sg)
    go = tuple(row for sg in sgs for row in sg)
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_b8825c91(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
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
    go = tuple(e for e in gi)
    for alph in (2, 1):
        locidev = unifint(rng, diff_lb, diff_ub, (1, alph * h))
        locjdev = unifint(rng, diff_lb, diff_ub, (1, w))
        loci = alph * h - locidev
        locj = w - locjdev
        loci2 = unifint(rng, diff_lb, diff_ub, (loci, alph * h - 1))
        locj2 = unifint(rng, diff_lb, diff_ub, (locj, w - 1))
        bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
        gi = fill(gi, 4, bd)
        gi, go = rot180(gi), rot180(go)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = rng.choice((1, 2))
    for fn in rng.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def generate_ba97ae07(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (3, 30))
    w = unifint(rng, diff_lb, diff_ub, (3, 30))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    lineh = unifint(rng, diff_lb, diff_ub, (1, h // 3))
    linew = unifint(rng, diff_lb, diff_ub, (1, w // 3))
    loci = rng.randint(1, h - lineh - 1)
    locj = rng.randint(1, w - linew - 1)
    acol = rng.choice(remcols)
    bcol = rng.choice(remove(acol, remcols))
    for a in range(lineh):
        gi = fill(gi, acol, connect((loci + a, 0), (loci + a, w - 1)))
    for b in range(linew):
        gi = fill(gi, bcol, connect((0, locj + b), (h - 1, locj + b)))
    for b in range(linew):
        go = fill(go, bcol, connect((0, locj + b), (h - 1, locj + b)))
    for a in range(lineh):
        go = fill(go, acol, connect((loci + a, 0), (loci + a, w - 1)))
    if rng.choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def generate_c909285e(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(rng, diff_lb, diff_ub, (5, 30))
    w = unifint(rng, diff_lb, diff_ub, (5, 30))
    nfronts = unifint(rng, diff_lb, diff_ub, (1, (h + w) // 2))
    bgc = rng.choice(cols)
    remcols = remove(bgc, cols)
    boxcol = rng.choice(remcols)
    remcols = remove(boxcol, remcols)
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    for k in range(nfronts):
        ff = rng.choice((hfrontier, vfrontier))
        loc = rng.choice(inds)
        inds = remove(loc, inds)
        col = rng.choice(remcols)
        gi = fill(gi, col, ff(loc))
    oh = unifint(rng, diff_lb, diff_ub, (3, max(3, (h - 2) // 2)))
    ow = unifint(rng, diff_lb, diff_ub, (3, max(3, (w - 2) // 2)))
    loci = rng.randint(1, h - oh - 1)
    locj = rng.randint(1, w - ow - 1)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, boxcol, bx)
    go = subgrid(bx, gi)
    return {"input": gi, "output": go}


def generate_d511f180(rng: random.Random, diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (5, 8))
    h = unifint(rng, diff_lb, diff_ub, (2, 30))
    w = unifint(rng, diff_lb, diff_ub, (2, 30))
    numc = unifint(rng, diff_lb, diff_ub, (1, 8))
    ccols = rng.sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    numbg = unifint(rng, diff_lb, diff_ub, (0, (h * w) // 2))
    bginds = rng.sample(inds, numbg)
    idx = rng.randint(0, numbg)
    blues = bginds[:idx]
    greys = bginds[idx:]
    rem = difference(inds, bginds)
    gi = fill(c, 8, blues)
    gi = fill(gi, 5, greys)
    go = fill(c, 5, blues)
    go = fill(go, 8, greys)
    for ij in rem:
        col = rng.choice(ccols)
        gi = fill(gi, col, {ij})
        go = fill(go, col, {ij})
    return {"input": gi, "output": go}
