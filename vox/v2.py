"""v2: few key scenes with original audio + VO breathers, motivated zooms. Timeline from v2/plan.json."""
import json, math, functools
import numpy as np, cv2
from vox_ep import (grade, rot_rgba, blit_rgb, footage_card, pop, ghost, color_block, highlight, big_text, stamp_text, ellipse_marker, arrow, NAVY, YELLOW, to_page)
from xerox_lib import *
from v2lib import *

GUIDE = json.load(open('v2/guide.json'))
@functools.lru_cache(maxsize=40)
def jpg(p): return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
@functools.lru_cache(maxsize=40)
def msk(p): return cv2.imread(p, 0).astype(np.float32) / 255
def ease(q): q = clamp(q); return q * q * (3 - 2 * q)

def zoom_view(arr, z, fx, fy):
    h, w = arr.shape[:2]; cw, ch = w / z, h / z
    x0 = clamp(fx - cw / 2, 0, w - cw); y0 = clamp(fy - ch / 2, 0, h - ch)
    M = np.float32([[z, 0, -x0 * z], [0, z, -y0 * z]])
    return cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def zoom_params(s, tl, k, prog_in, spec):
    if 'zk' in s and s['kind'] == 'key':
        ks = s['zk']
        for a, b in zip(ks[:-1], ks[1:]):
            if a[0] <= tl <= b[0]:
                q = ease((tl - a[0]) / max(1e-6, b[0] - a[0])) if b[0] - a[0] > .05 else 1.0
                if b[0] - a[0] <= .05: return a[1], a[2], a[3]
                return a[1] + (b[1] - a[1]) * q, a[2] + (b[2] - a[2]) * q, a[3] + (b[3] - a[3]) * q
        l = ks[-1]; return l[1], l[2], l[3]
    z0, z1, fx, fy = spec; q = ease(prog_in)
    return z0 + (z1 - z0) * q, fx, fy

def get_frame(s, t):
    tl = clamp(t - s['t0'], 0, s['dur'] - 1e-3)
    k, n, spec, pr = locate(s, tl)
    z, fx, fy = zoom_params(s, tl, k, pr, spec)
    img = jpg(path(s, k, n))
    m = msk(mpath(s, k, n)) if s['kind'] == 'vo' and os.path.exists(mpath(s, k, n)) else None
    # held (15 fps) frame for cutouts
    kh, nh = k, (n if n % 2 == 1 else n - 1)
    imh = jpg(path(s, kh, max(1, nh))); 
    return img, imh, m, (z, fx, fy), tl

# ---------------- subtitles
def sub_words(P, t, words, t_show0, t_show1, gj=(0, 0), reveal=True, size=42):
    """words: [(text, t_start, key)] absolute times"""
    ts = step_t(t)
    if not (t_show0 <= ts <= t_show1): return
    masks = [text_mask(w[0], 'catalogue_regular', size) for w in words]; gap = 13
    widths = [m.shape[1] - 28 for m in masks]; total = sum(widths) + gap * (len(words) - 1)
    x = W / 2 - total / 2; y = 968; shown = []
    for i, (w, m) in enumerate(zip(words, masks)):
        if (not reveal) or ts >= w[1] - .01: shown.append((i, x, m, w))
        x += widths[i] + gap
    if not shown: return
    halo = np.zeros((90, int(total) + 90), np.float32); ox = W / 2 - total / 2 - 45
    for (i, xx, m, w) in shown:
        hx = int(xx - ox - 14); halo[:, hx:hx + m.shape[1]] = np.maximum(halo[:, hx:hx + m.shape[1]][:, :m.shape[1]], 0)  # placeholder to keep size
    for (i, xx, m, w) in shown:
        hm = cv2.GaussianBlur(cv2.dilate(m, np.ones((7, 7), np.uint8)), (0, 0), 2.5)
        P.fill(np.clip(hm * 1.6, 0, 1), xx - 14, y, TILES[0], s=.88)
    for (i, xx, m, w) in shown:
        if w[2]:
            hl = rect_mask(int(widths[i] + 14), 52, 2, 100 + i, True)
            P.ink(hl, xx - 7 + gj[0], y + 22 + gj[1], .9, YELLOW)
        P.ink(m, xx - 14, y, 1.0, NAVY)

def vo_words(s):
    g = GUIDE[s['id']]; t0 = s['vo_t0']
    return [(w['w'], t0 + w['s'], w['key']) for w in g['words']], t0 + g['words'][0]['s'] - .05, t0 + g['dur'] + .9
def vo_time(s, name):
    g = GUIDE[s['id']]
    for w in g['words']:
        if w['w'].lower().strip('.,') == name: return s['vo_t0'] + w['s']
    return s['vo_t0']
def dlg(s, P, t, gj):
    for (text, a, b, keys) in s['subs']:
        ta = s['t0'] + (a - s['src'][0]); tb = s['t0'] + (b - s['src'][0])
        nxt = [x for x in s['subs'] if x[1] > a]; tend = tb + .45 if not nxt else min(tb + .45, s['t0'] + (nxt[0][1] - s['src'][0]) - .03)
        words = text.split()
        # word times spread across the phrase (dialogue pacing)
        n = len(words); wts = [ta + (tb - ta) * (i / n) for i in range(n)]
        sub_words(P, t, [(w, wt, (w in keys)) for w, wt in zip(words, wts)], ta - .02, tend, gj, reveal=True)

# ---------------- segment layouts
def P_new(i): return Page(i % 2, i % 4, i % 2)
def base(s, t, i, block_col=None, gz=(2.0, .5, .4), dens=.26):
    P = P_new(i); gj = jit(t, i * 5 + 1)
    img, imh, m, zp, tl = get_frame(s, t)
    ghost(P, img, gz[0], gz[1], gz[2], dens)
    return P, gj, img, imh, m, zp, tl

def draw_card_and_pop(P, s, t, cx, cy, w, rot, img, imh, m, zp, border=16, popk=1.16):
    z, fx, fy = zp
    zi = zoom_view(img, z, fx, fy)
    footage_card(P, grade(zi, .78), cx, cy, w, rot, border=border)
    if m is not None and popk and .06 < float(m.mean()) < .9:
        zh = zoom_view(imh, z, fx, fy); zm = zoom_view(m, z, fx, fy)
        pop(P, grade(zh, .88), zm, cx, cy - 4, w * popk, rot)

def seg_vo1(t):
    s = SEGS['vo1']; P, gj, img, imh, m, zp, tl = base(s, t, 0)
    color_block(P, 600, 90, 1250, 700, YELLOW, seed=5, s=.55, gj=gj)
    cp = ease((t - .25) / .5); cx = 1250 + (1 - cp) * 700
    draw_card_and_pop(P, s, t, cx, 560, 1040, -2.0, img, imh, m, zp, popk=1.15)
    ot = vo_time(s, 'own.')
    if t > ot - .1:
        pp = sp(t, ot - .1, .22); pink_block(P, int(600 * pp), 320, 60, 300, seed=2, gj=gj)
        draw_ransom(P, 'OWN', 90 + gj[0], 320 + gj[1], 220, 5, t, ot, stagger=.1)
    lt = vo_time(s, 'lie')
    if t >= lt - .05:
        pp = sp(t, lt - .05, .25); bm = rotate(rect_mask(int(540 * pp) + 10, 180, 4, seed=9, torn=True), -6)
        P.card(bm, 100 + gj[0], 640 + gj[1], TILES[1]); P.pink(bm, 100 + gj[0], 640 + gj[1])
        tm = text_mask('THE LIE', 'vandal_stencil', 140, rot=-6, rough=.4, seed=3); P.ink(tm, 112 + gj[0], 630 + gj[1], min(1, pp * 2), NAVY)
    words, a, b = vo_words(s); sub_words(P, t, words, a, b, gj)
    return P

def seg_vo2(t):
    s = SEGS['vo2']; P, gj, img, imh, m, zp, tl = base(s, t, 1)
    color_block(P, 200, 90, 1500, 700, YELLOW, seed=33, s=.5, gj=gj)
    cx, cy, w, rot = 960, 500, 1280, .8
    draw_card_and_pop(P, s, t, cx, cy, w, rot, img, imh, m, zp, popk=1.06)
    z, fx, fy = zp
    tags = [('decision', 470, 20, '$395'), ('never', 545, 250, '$695'), ('made.', 20, 200, '$1,250'), ('pays', 1090, 215, '$2,400')]
    for (wd, ffx, ffy, price) in tags:
        tt = vo_time(s, wd)
        if t < tt: continue
        # tag position follows the zoom on the footage
        px = (ffx - fx) * z + 640; py = (ffy - fy) * z + 278
        x, y = to_page(cx, cy, w, rot, clamp(px, 40, 1180), clamp(py, 20, 520))
        pp = sp(t, tt, .45, lambda q: q); wc = 100
        P.card(rect_mask(int(wc * 1.95) + 26, wc + 18, 2, int(ffx), True), x - 12 + gj[0], y - 8 + gj[1], TILES[0], s=.4)
        price_tag(P, x + gj[0], y + gj[1], wc, pp, price)
    tt = vo_time(s, 'time')
    if t >= tt - .1: stamp_text(P, 'PAID IN TIME', 700, 800, 96, -3, t=t, t0=tt - .1, gj=gj, seed=11)
    words, a, b = vo_words(s); sub_words(P, t, words, a, b, gj)
    return P

def seg_vo3(t):
    s = SEGS['vo3']; P, gj, img, imh, m, zp, tl = base(s, t, 0, gz=(2.0, .5, .5), dens=.3)
    color_block(P, 90, 110, 1040, 640, PINK, seed=21, s=.8, gj=gj)
    draw_card_and_pop(P, s, t, 600, 450, 1020, -1.0, img, imh, None, zp, popk=0)
    for i, (word_, name, y, rot) in enumerate([('violent,', 'VIOLENT', 130, 3), ('anonymous,', 'ANONYMOUS', 360, -3), ('real.', 'REAL', 590, 3)]):
        tt = vo_time(s, word_)
        if t < tt - .05: continue
        pp = sp(t, tt - .05, .28); tm = text_mask(name, 'vandal_stencil', 150, rot=rot, rough=.3, seed=i + 40)
        bm = rotate(rect_mask(max(12, int(tm.shape[1] * pp)), int(tm.shape[0] * .85), 4, seed=i + 50, torn=True), rot)
        x = 1190 - 40 * i
        P.card(bm, x + gj[0], y + gj[1] + 10, TILES[0], s=.4); P.pink(bm, x + gj[0], y + gj[1] + 10)
        P.ink(tm, x + 12 + gj[0], y + gj[1], min(1, pp * 1.5), NAVY)
    words, a, b = vo_words(s); sub_words(P, t, words, a, b, gj)
    return P

def seg_vo4(t):
    s = SEGS['vo4']; P, gj, img, imh, m, zp, tl = base(s, t, 1, gz=(2.0, .5, .45), dens=.28)
    color_block(P, 120, 90, 1100, 700, YELLOW, seed=12, s=.5, gj=gj)
    draw_card_and_pop(P, s, t, 650, 480, 1040, 1.2, img, imh, None, zp, popk=0)
    bt = vo_time(s, 'brand.')
    if t > bt - .1:
        pp = sp(t, bt - .1, .22); pink_block(P, int(640 * pp), 215, 1230, 70, seed=12, gj=gj)
        draw_ransom(P, 'BRAND', 1250 + gj[0], 90 + gj[1], 125, 8, t, bt, stagger=.1)
    rt = vo_time(s, 'rules,')
    rules = ['1.  WEAR THE UNIFORM', '2.  OBEY THE LEADER', '3.  NEVER ASK WHO PROFITS']
    if t > rt - .1:
        pp = sp(t, rt - .1, .25); cm = rect_mask(640, 270, 2, seed=33)
        P.card(cm, 1230 + gj[0], 340 + gj[1] - (1 - pp) * 30, TILES[0], s=.45)
        if pp > .9:
            P.ink(text_mask('HOUSE RULES', 'catalogue', 30), 1255 + gj[0], 355 + gj[1], 1, NAVY)
            for i, r in enumerate(rules):
                if t >= rt + .2 + i * .4: P.ink(text_mask(r, 'catalogue', 30), 1255 + gj[0], 405 + i * 56 + gj[1], 1, NAVY)
    ut = vo_time(s, 'uniforms,')
    if t > ut - .05:
        ellipse_marker(P, 650, 470, 300, 150, sp(t, ut, .5, lambda q: q), 3, gj, pink=True, width=10)
    lt = vo_time(s, 'leader.')
    if t > lt - .05:
        pp = sp(t, lt - .05, .25); bm = rotate(rect_mask(int(330 * pp) + 10, 110, 4, seed=15, torn=True), -5)
        P.card(bm, 1280 + gj[0], 650 + gj[1], TILES[0], s=.4); P.pink(bm, 1280 + gj[0], 650 + gj[1])
        P.ink(text_mask('LEADER™', 'vandal_stencil', 84, rot=-5, rough=.4, seed=4), 1292 + gj[0], 645 + gj[1], min(1, pp * 2), NAVY)
    words, a, b = vo_words(s); sub_words(P, t, words, a, b, gj)
    return P

def seg_vo5(t):
    s = SEGS['vo5']; P, gj, img, imh, m, zp, tl = base(s, t, 0, gz=(1.8, .5, .5), dens=.3)
    color_block(P, 90, 70, 960, 520, PINK, seed=44, s=.75, gj=gj)
    shrink = ease((t - s['t0'] - .2) / 2.0)
    w = 900 - 100 * shrink
    draw_card_and_pop(P, s, t, 560, 330, w, -1.4, img, imh, None, zp, popk=0)
    ft = vo_time(s, 'free.')
    if t >= ft - .1:
        pp = sp(t, ft - .1, .25); bm = rotate(rect_mask(int(520 * pp) + 10, 230, 4, seed=18, torn=True), 5)
        P.card(bm, 1180 + gj[0], 70 + gj[1], TILES[0], s=.4); P.pink(bm, 1180 + gj[0], 70 + gj[1])
        P.ink(text_mask('FREE?', 'vandal_marker', 190, rot=5, rough=.5, seed=2), 1210 + gj[0], 75 + gj[1], min(1, pp * 2), NAVY)
        if t > ft + .5: marker(P, [(1210, 310), (1660, 285)], sp(t, ft + .5, .4, lambda q: q), 12, seed=5, gj=gj, col=NAVY)
    SZ = 100
    wt = [vo_time(s, 'who'), vo_time(s, 'sold'), vo_time(s, 'you'), vo_time(s, 'your'), vo_time(s, 'freedom.')]
    rows = [[('WHO', wt[0], 3, False), ('SOLD', wt[1], 4, True), ('YOU', wt[2], 5, False)], [('YOUR', wt[3], 6, False), ('FREEDOM', wt[4], 7, True)]]
    for r, row in enumerate(rows):
        x = 90 if r == 0 else 130; y = 640 + r * 150
        for (s_, tt, seed, pk) in row:
            lt_, wtot = ransom_layout(s_, SZ, seed)
            if t >= tt - .15:
                if pk:
                    pp = sp(t, tt - .15, .25); pink_block(P, int((wtot + 30) * pp), 135, x - 18, y - 12, seed=seed + 70, gj=gj)
                draw_ransom(P, s_, x + gj[0], y + gj[1], SZ, seed, t, tt, stagger=.08)
            x += wtot + 50
    et = s['vo_t0'] + GUIDE['vo5']['dur'] + .15
    if t >= et:
        pp = sp(t, et, .3, lambda q: q)
        P.card(rect_mask(560, 200, 3, seed=61, torn=True), 1190 + gj[0], 400 + gj[1], TILES[0], s=.45)
        price_tag(P, 1215 + gj[0], 420 + gj[1], 90, pp, '$0.00')
        P.ink(text_mask('FREEDOM', 'catalogue', 40), 1215 + gj[0], 520 + gj[1], 1, NAVY)
        barcode(P, 1400, 435, 320, 70, sp(t, et + .2, .5, lambda q: q), seed=3, gj=gj)
        stamp_text(P, 'SOLD', 1360, 520, 110, -7, t=t, t0=et + .9, gj=gj, seed=8)
    words, a, b = vo_words(s); sub_words(P, t, words, a, b, gj)
    return P

def seg_key(t, sid, idx):
    s = SEGS[sid]; P = P_new(idx); gj = jit(t, idx * 5 + 3)
    img, imh, m, zp, tl = get_frame(s, t)
    ghost(P, img, 2.0, .5, .5, .22)
    color_block(P, 110, 70, 1700, 800, PINK if idx % 2 else YELLOW, seed=idx * 3, s=.6, gj=gj)
    z, fx, fy = zp
    # entry: card eases open to near full-bleed (the film speaks -> it fills the screen)
    gr = ease(tl / .45); w = 1500 + 260 * gr
    footage_card(P, grade(zoom_view(img, z, fx, fy), .85), 960, 470, w, 0.0, border=14, tape=True, shadow=.55)
    dlg(s, P, t, gj)
    return P

SEGF = {'vo1': seg_vo1, 'vo2': seg_vo2, 'vo3': seg_vo3, 'vo4': seg_vo4, 'vo5': seg_vo5,
        'k1': lambda t: seg_key(t, 'k1', 1), 'k2': lambda t: seg_key(t, 'k2', 2), 'k3': lambda t: seg_key(t, 'k3', 3)}
ROTS = {'vo1': -.3, 'k1': .0, 'vo2': .3, 'k2': .0, 'vo3': -.25, 'k3': .0, 'vo4': .3, 'vo5': -.3}
def seg_frame(s, t):
    P = SEGF[s['id']](t); return page_warp(P.finish(), ROTS[s['id']], 1.02)
