"""Episode 00 (test): 'Fight Club' video essay in the XEROX CATALOGUE style."""
import json, math, functools, sys
import numpy as np, cv2
from PIL import Image, ImageDraw
import skimage.data as skd
from xerox_lib import *

TL = json.load(open('audio/timeline.json'))
LN = TL['lines']
def word(line, w, nth=0):
    hits = [x for x in LN[line]['words'] if re.sub(r"[^a-z']", '', x['w'].lower()) == w]
    return hits[nth]['s']
import re
CUTS = [0.0, 4.75, 10.05, 14.15, 20.0, 30.0]

def stamp(P, m, x, y):
    P.card(m, x, y, TILES[0], s=.4); P.pink(m, x, y)

# ------------------------------------------------------------ assets
def gray_of(a): return (a if a.ndim == 2 else a @ np.array([.299, .587, .114], np.float32)).astype(np.float32)
@functools.lru_cache(None)
def photo(name):
    a = np.asarray(Image.open(f'assets/src/{name}.png').convert('L'), np.float32) / 255
    return a
@functools.lru_cache(None)
def cutout(name):
    a = np.asarray(Image.open(f'assets/src/{name}_cut.png').convert('RGBA'), np.float32) / 255
    if name == 'astronaut': a = a[185:]          # torso/suit only: uniform, no face
    ys, xs = np.where(a[..., 3] > .1); return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def draw_photo_bg(P, name, u, z=(1.0, 1.12), c=(0.5, 0.5), dc=(0.04, -0.03), density=1.0, lo=.1, hi=.85, gamma=1.0, x=0, y=0, w=W, h=H, region_mask=None):
    g = photo(name); sh, sw = g.shape
    zz = z[0] + (z[1] - z[0]) * u
    cx = (c[0] + dc[0] * u) * sw; cy = (c[1] + dc[1] * u) * sh
    v = cover_view(g, zz, cx, cy, w, h)
    d = tone(v, lo, hi, gamma) * density
    m = halftone(d, x, y)
    if region_mask is not None: m = m * region_mask
    P.ink(m, x, y, 1.0)

@functools.lru_cache(maxsize=512)
def cut_prep(name, height, rot, border):
    a = cutout(name); sc = height / a.shape[0]
    a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    pad = border + 6
    a = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    g = gray_of(a[..., :3]); al = a[..., 3]
    if rot:
        g = rotate(g, rot); al = rotate(al, rot)
    d = tone(g, .05, .95, 0.85)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
    sticker = cv2.GaussianBlur(cv2.dilate((al > .5).astype(np.float32), k), (0, 0), 1.2)
    sticker = (sticker > .5).astype(np.float32)
    edge = np.clip(cv2.dilate(al, np.ones((5, 5), np.uint8)) - cv2.erode(al, np.ones((3, 3), np.uint8)), 0, 1) * (al < .98)
    return d, al, sticker, edge

def draw_cutout(P, name, cx, cy, height, rot=0.0, border=14, ink_s=1.0, pink_block=None, shadow=True):
    d, al, st, edge = cut_prep(name, int(height), round(rot * 2) / 2, border)
    h, w = d.shape; x = cx - w / 2; y = cy - h / 2
    if pink_block:
        bw, bh, ox, oy = pink_block; pink_block_m = rect_mask(bw, bh, rough=4, seed=4, torn=True)
        P.pink(pink_block_m, cx - bw / 2 + ox, cy - bh / 2 + oy)
    P.card(st, x, y, TILES[0], s=0.46 if shadow else 0, shadow=shadow)
    ink = halftone(d, int(x), int(y)) * al
    P.ink(np.clip(ink + edge * .9, 0, 1), x, y, ink_s)
    return w, h

def photo_panel(P, name, x, y, w, h, t, u, rot=0, z=(1.0, 1.1), c=(.5, .5), dc=(.03, -.02), lo=.1, hi=.85, seed=1, tape=True):
    frame = rect_mask(w + 40, h + 40, rough=2, seed=seed)
    P.card(frame, x - 20, y - 20, TILES[0], s=.45)
    g = photo(name); v = cover_view(g, z[0] + (z[1] - z[0]) * u, (c[0] + dc[0] * u) * g.shape[1], (c[1] + dc[1] * u) * g.shape[0], w, h)
    m = halftone(tone(v, lo, hi), x, y)
    P.ink(m, x, y)
    P.ink(np.clip(cv2.dilate(np.ones((h, w), np.float32), np.ones((1, 1))) * 0, 0, 1), x, y)
    if tape:
        P.tape(x - 60, y - 30, 28); P.tape(x + w - 150, y + h - 40, 28)

# ---- procedural footage (fallback: drawn, not filmed) ----
def crowd_band(t, w, h, seed=2):
    im = Image.new('L', (w, h), 0); dr = ImageDraw.Draw(im)
    rows = [(0.50, 70, 0.38, 38, 1.1), (0.72, 95, 0.60, 52, 1.5), (0.95, 125, 1.0, 70, 2.0)]  # (y, size, gray, spacing mult, speed)
    for (yf, sz, gray, _, spd) in rows:
        gap = sz * 1.5; off = (t * 55 * spd) % gap; n = int(w / gap) + 3
        for i in range(-1, n):
            x = i * gap - off + sz * 0.6; bob = abs(math.sin(t * 5.5 + i * 1.7)) * sz * 0.06
            by = h * yf - bob; col = int(255 * gray)
            dr.ellipse([x - sz * .17, by - sz * 1.15, x + sz * .17, by - sz * .81], fill=col)          # head, faceless
            dr.polygon([(x - sz * .30, by - sz * .78), (x + sz * .30, by - sz * .78), (x + sz * .36, by), (x - sz * .36, by)], fill=col)  # torso
            dr.rectangle([x - sz * .2, by, x - sz * .05, by + sz * .1], fill=col); dr.rectangle([x + sz * .05, by, x + sz * .2, by + sz * .1], fill=col)
    return np.asarray(im, np.float32) / 255

def conveyor_band(P, t, x, y, w, h, name='astronaut'):
    base = np.zeros((h, w), np.float32)
    cv2.line(base, (0, h - 40), (w, h - 40), 1.0, 8, cv2.LINE_AA)
    for rx in range(0, w + 60, 60):
        cx = (rx - int(t * 140) % 60); cv2.circle(base, (cx, h - 18), 14, 1.0, 4, cv2.LINE_AA)
    P.ink(base, x, y)
    gap = 215; off = (t * 140) % gap
    for i in range(-1, int(w / gap) + 2):
        cx = i * gap - off + 100
        if -120 < cx < w + 120:
            d, al, st, edge = cut_prep(name, 118, 0, 5)
            hh, ww = d.shape; px = x + cx - ww / 2; py = y + h - 46 - hh
            m = halftone(d, int(px), int(py)) * al
            # clip to band
            r = P._sl(m, px, py)
            if r is None: continue
            sl, mm = r; band = np.zeros((H, W), np.float32); 
            P.ink(np.clip(m + edge * .9, 0, 1) * 1.0, px, py)

# ---- line-art catalogue items ----
def art(kind, w=440, h=270):
    S = 3; im = Image.new('L', (w * S, h * S), 0); d = ImageDraw.Draw(im); lw = 5 * S
    def L(*p, **k): d.line([c * S for c in p], fill=255, width=lw, joint='curve')
    def R(x0, y0, x1, y1, r=10): d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S, outline=255, width=lw)
    def E(x0, y0, x1, y1): d.ellipse([x0 * S, y0 * S, x1 * S, y1 * S], outline=255, width=lw)
    def PG(*pts): L(*[v for p in pts for v in p], *pts[0])
    if kind == 'sofa':
        R(60, 55, 380, 125, 22); R(30, 105, 410, 205, 18); R(25, 95, 75, 210, 16); R(365, 95, 415, 210, 16)
        L(190, 125, 190, 200); L(255, 125, 255, 200); L(60, 205, 60, 232); L(380, 205, 380, 232)
    elif kind == 'lamp':
        E(160, 228, 280, 250); L(220, 235, 220, 100); PG((170, 38), (270, 38), (305, 100), (135, 100))
        L(220, 100, 220, 112)
    elif kind == 'chair':
        R(135, 30, 305, 150, 28); R(115, 140, 325, 200, 14); R(90, 115, 135, 212, 12); R(305, 115, 350, 212, 12); L(130, 200, 130, 245); L(310, 200, 310, 245)
    elif kind == 'table':
        R(55, 90, 385, 125, 8); L(95, 125, 95, 245); L(345, 125, 345, 245); L(95, 160, 345, 160)
        E(200, 55, 255, 90)
    elif kind == 'shelf':
        R(115, 15, 325, 252, 6); L(115, 80, 325, 80); L(115, 145, 325, 145); L(115, 205, 325, 205)
        for (x0, y0, x1) in [(135, 35, 160), (170, 25, 190), (200, 40, 230), (140, 100, 175), (260, 95, 285), (150, 165, 200)]: R(x0, y0 + 0, x1, 78 if y0 < 80 else (143 if y0 < 145 else 203), 2)
    elif kind == 'plant':
        PG((165, 170), (275, 170), (255, 250), (185, 250))
        for (cx, cy, rx, ry, a) in [(220, 110, 28, 62, 0), (170, 120, 22, 55, 0), (270, 120, 22, 55, 0)]: E(cx - rx, cy - ry, cx + rx, cy + ry)
        L(220, 170, 220, 100)
    im = im.resize((w, h), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255
ITEMS = [('sofa', 'BRÖNN', '$149.00', '412.07.8'), ('lamp', 'VÄLLA', '$29.99', '208.31.5'), ('chair', 'TJÖRK', '$89.00', '603.44.1'),
         ('table', 'NÅLLE', '$59.50', '119.20.7'), ('shelf', 'SKÄPP', '$119.00', '771.02.4'), ('plant', 'FLÖRM', '$12.50', '905.56.0')]

# ------------------------------------------------------------ captions
CHUNKS = [[(0, 4), (5, 12)], [(0, 4), (5, 12)], [(0, 7)], [(0, 5), (6, 11)], [(0, 6), (7, 13)]]
def caption(P, t, gj, only=None):
    ts = step_t(t)
    for li, chunks in enumerate(CHUNKS):
        if only is not None and li != only: continue
        ws = LN[li]['words']
        for (a, b) in chunks:
            t0 = ws[a]['s'] - 0.04; t1 = ws[b]['e'] + 0.30
            if not (t0 <= ts <= t1): continue
            f = 'vandal_marker'; size = CFG['captions']['size']
            texts = [w['w'] for w in ws[a:b + 1]]
            for size_try in (size, 60, 54):
                masks = [text_mask(s, f, size_try, rough=0.35, seed=i * 7 + li) for i, s in enumerate(texts)]
                sp_ = 20; widths = [m.shape[1] - 28 for m in masks]; total = sum(widths) + sp_ * (len(widths) - 1)
                if total <= 1560: break
            sw, shh = total + 110, 124
            x0 = W / 2 - total / 2; yb = 880
            strip = torn_strip(int(sw), shh, 40 + li * 3 + a)
            P.card(strip, W / 2 - sw / 2 + gj[0], yb + gj[1], TILES[0], s=.42)
            cx = x0
            for i, (wd, m) in enumerate(zip(ws[a:b + 1], masks)):
                if ts >= wd['s'] - 0.01:
                    if wd['key']:
                        pm = rect_mask(int(widths[i] + 26), 84, rough=3, seed=i + 90 + li, torn=True)
                        P.pink(pm, cx - 13 + gj[0], yb + 20 + gj[1])
                    P.ink(m, cx - 14 + gj[0], yb + 62 - m.shape[0] / 2 + gj[1] + (0 if i % 2 else 3))
                cx += widths[i] + sp_
            return

# ------------------------------------------------------------ scenes
def S1(t):
    P = Page(0, 0, 0); gj = jit(t, 1); u = prog(t, 0, 4.9)
    draw_photo_bg(P, 'coins', u, z=(.5, .6), c=(.5, .5), dc=(.04, -.03), density=.7, lo=.25, hi=.9)
    # headline words, catalogue voice, on paper strips
    hl = ['YOU', 'ARE', 'WHAT', 'YOU']; ws = [word(0, 'you'), word(0, 'are'), word(0, 'what'), word(0, 'you', 1)]
    x = 100
    for i, (s, tt) in enumerate(zip(hl, ws)):
        p = sp(t, tt, .15, lambda q: q)
        if p <= 0: continue
        m = text_mask(s, 'catalogue', 84, rough=.1); tile = rect_mask(m.shape[1] - 6, 112, 2, seed=i + 11, torn=True)
        P.card(tile, x + gj[0], 120 + gj[1] - (1 - p) * 18, TILES[0])
        P.ink(m, x - 14 + gj[0], 130 + gj[1] - (1 - p) * 18); x += m.shape[1] + 2
    own_t = word(0, 'own')
    if t > own_t - .1:
        pp = sp(t, own_t - .1, .22); pink_block(P, int(640 * pp), 330, 85, 300, seed=2, gj=gj)
        draw_ransom(P, 'OWN', 120 + gj[0], 330 + gj[1], 235, 5, t, own_t, stagger=.1)
    # catalogue card with cutout + parallax
    cp = sp(t, .15, .35)
    cardm = rect_mask(800, 770, 2, seed=7)
    P.card(cardm, 1010 + gj[0], 60 + gj[1] + (1 - cp) * -700, TILES[0], s=.5)
    if cp > .95:
        tm = text_mask('ITEM Nº 0417', 'catalogue', 38); P.ink(tm, 1050 + gj[0], 90 + gj[1])
        cv2.line  # (rule line)
        ln = np.ones((6, 720), np.float32); P.ink(ln, 1040 + gj[0], 150 + gj[1])
        tm = text_mask('ESPRESSO SET — ONE OF 12', 'catalogue_regular', 30); P.ink(tm, 1050 + gj[0], 165 + gj[1])
        draw_cutout(P, 'coffee', 1410 - 28 * u * 1.8 + gj[0], 430 + 14 * u * 1.8 + gj[1], 440 + 18 * u, rot=-4 + 3 * u, pink_block=(520, 400, 28, 24))
        price_tag(P, 1065 + gj[0], 650 + gj[1], 100, sp(t, 1.5, 1.3, lambda q: q), '$4.99', gj=(0, 0))
        barcode(P, 1290, 655, 450, 90, sp(t, 2.3, 1.3, lambda q: q), seed=3, gj=gj)
        tm = text_mask('0417  5 0 3 1 8 2', 'catalogue_regular', 26); P.ink(tm, 1290 + gj[0], 752 + gj[1])
    lie = word(0, 'lie')
    if t >= lie - .05:
        pp = sp(t, lie - .05, .25)
        bm = rotate(rect_mask(int(560 * pp) + 10, 190, 4, seed=9, torn=True), -6)
        P.card(bm, 105 + gj[0], 650 + gj[1], TILES[1]); P.pink(bm, 105 + gj[0], 650 + gj[1])
        tm = text_mask('THE LIE', 'vandal_stencil', 150, rot=-6, rough=.4, seed=3); P.ink(tm, 115 + gj[0], 640 + gj[1], s=min(1, pp * 2))
    caption(P, t, gj, only=0)
    return P

def S2(t):
    P = Page(1, 1, 1); gj = jit(t, 2); u = prog(t, 4.75, 5.4)
    draw_photo_bg(P, 'brick', u, z=(1.0, 1.1), c=(.5, .4), dc=(.03, .1), density=.85, lo=.15, hi=.8)
    sheet = rect_mask(1640, 770, 2, seed=21)
    sp0 = sp(t, 4.8, .3)
    P.card(sheet, 140 + gj[0], 50 + gj[1] - (1 - sp0) * 500, TILES[0], s=.5)
    if sp0 < .9: caption(P, t, gj, only=1); return P
    pink_block(P, 520, 78, 175, 70, seed=4, gj=gj)
    P.fill(np.ones((1,1),np.float32),0,0,TILES[0],s=0)
    tm = text_mask('THE HOME CATALOGUE', 'catalogue', 46); P.ink(tm, 190 + gj[0], 78 + gj[1])
    tm = text_mask('EVERYTHING YOU NEED. FREE DELIVERY*', 'catalogue_regular', 26); P.ink(tm, 800 + gj[0], 100 + gj[1])
    barcode(P, 1530, 78, 230, 70, sp(t, 5.2, .9, lambda q: q), seed=8, gj=gj)
    cw, ch = 520, 300; t_in = 5.15
    cross_times = [7.3, 7.7, 8.1, 8.5, 8.9, 9.15]
    for i, (kind, name, price, no) in enumerate(ITEMS):
        col, row = i % 3, i // 3
        x = 170 + col * 540; y = 165 + row * 310
        p = sp(t, t_in + i * .13, .25)
        if p <= 0: continue
        yy = y + (1 - p) * 30
        border = np.zeros((ch, cw), np.float32); cv2.rectangle(border, (2, 2), (cw - 3, ch - 3), 1.0, 3); P.ink(border, x + gj[0], yy + gj[1], .9)
        a = art(kind); P.ink(a, x + 40 + gj[0], yy + 6 + gj[1])
        tm = text_mask(name, 'catalogue', 34); P.ink(tm, x + 14 + gj[0], yy + ch - 52 + gj[1])
        tm = text_mask(price, 'catalogue', 36); P.ink(tm, x + cw - tm.shape[1] - 6 + gj[0], yy + ch - 52 + gj[1])
        tm = text_mask('ART. ' + no, 'catalogue_regular', 20); P.ink(tm, x + 14 + gj[0], yy + 8 + gj[1], .8)
        c0 = cross_times[i]
        if t >= c0:
            pr = prog(step_t(t), c0, .3)
            marker(P, [(x + 30, yy + 28), (x + cw - 30, yy + ch - 70)], clamp(pr * 2), 14, seed=i * 5 + 1, gj=gj)
            if pr > .45: marker(P, [(x + cw - 30, yy + 28), (x + 30, yy + ch - 70)], clamp(pr * 2 - .9), 14, seed=i * 5 + 2, gj=gj)
    if t >= 9.3:
        pp = sp(t, 9.3, .25)
        bm = rect_mask(int(620 * pp) + 10, 150, 4, seed=13, torn=True)
        stamp(P, rotate(bm, 5), 640 + gj[0], 380 + gj[1])
        tm = text_mask('NEVER CHOSEN', 'vandal_stencil', 118, rot=5, rough=.4, seed=6); P.ink(tm, 655 + gj[0], 370 + gj[1], s=min(1, pp * 2))
    caption(P, t, gj, only=1)
    return P

def S3(t):
    P = Page(0, 2, 0); gj = jit(t, 3); u = prog(t, 10.05, 4.2)
    draw_photo_bg(P, 'gravel', u, z=(1.0, 1.13), c=(.45, .5), dc=(.1, -.05), density=1.05, lo=.18, hi=.7)
    words = [('VIOLENT', word(2, 'violent'), 190, 110, 4), ('ANONYMOUS', word(2, 'anonymous'), 190, 340, -3), ('REAL', word(2, 'real'), 190, 570, 3)]
    for i, (s, tt, size, y, rot) in enumerate(words):
        if t < tt - .05: continue
        pp = sp(t, tt - .05, .28)
        m = text_mask(s, 'vandal_stencil', size, rot=rot, rough=.3, seed=i + 40)
        bw = int(m.shape[1] * pp); bh = int(m.shape[0] * .85)
        bm = rotate(rect_mask(max(10, bw), bh, 4, seed=i + 50, torn=True), rot)
        px = 110 + i * 18
        stamp(P, bm, px + gj[0], y + gj[1] + 10)
        P.ink(m, px + gj[0] + 12, y + gj[1], s=min(1, pp * 1.5))
    rt = word(2, 'real')
    if t >= rt + .25:
        marker(P, [(120, 790), (560, 800), (900, 782)], sp(t, rt + .25, .5, lambda q: q), 13, seed=7, gj=gj)
    # footage panel: crowd silhouettes (procedural fallback), smooth 30fps
    px_, py_, pw, ph = 1020, 120, 780, 560
    pp = sp(t, 10.4, .3)
    if pp > .9:
        frame = rect_mask(pw + 40, ph + 40, 2, seed=77); P.card(frame, px_ - 20, py_ - 20, TILES[0], s=.5)
        band = crowd_band(t, pw, ph)
        yy_, xx_ = np.mgrid[0:ph, 0:pw].astype(np.float32); sky = (1 - yy_ / ph) * .25
        P.ink(halftone(np.clip(sky + band * 1.0, 0, 1), px_, py_), px_, py_)
        P.tape(px_ - 60, py_ - 30, 25); P.tape(px_ + pw - 150, py_ + ph - 40, 25)
        tm = text_mask('FIG. 3 — NOBODY', 'catalogue_regular', 26); lab = rect_mask(tm.shape[1] + 10, 56, 2, seed=5); P.card(lab, px_ - 6, py_ + ph + 24, TILES[0], s=.4); P.ink(tm, px_ - 2, py_ + ph + 28)
    caption(P, t, gj, only=2)
    return P

def S4(t):
    P = Page(1, 3, 1); gj = jit(t, 4); u = prog(t, 14.15, 5.9)
    draw_photo_bg(P, 'gravel', u, z=(1.12, 1.0), c=(.55, .5), dc=(-.1, .05), density=.35, lo=.2, hi=.75)
    cp = sp(t, 14.2, .3)
    cardm = rect_mask(660, 780, 2, seed=31)
    P.card(cardm, 110 + gj[0], 50 + gj[1] - (1 - cp) * 700, TILES[0], s=.5)
    if cp > .9:
        tm = text_mask('STANDARD ISSUE', 'catalogue', 38); P.ink(tm, 145 + gj[0], 80 + gj[1])
        draw_cutout(P, 'astronaut', 455 - 14 * u * 1.8 + gj[0], 485 + 6 * u + gj[1], 520, rot=-2 + 2 * u, pink_block=(420, 420, 30, 40))
        tm = text_mask('ONE SIZE. ONE COLOUR.', 'catalogue_regular', 28); P.ink(tm, 145 + gj[0], 770 + gj[1])
    bt = word(3, 'brand'); rb = word(3, 'rebellion')
    if t > rb and t < bt + .5:
        price_tag(P, 880 + gj[0], 130 + gj[1], 130, sp(t, rb, 1.3, lambda q: q), '$9.99')
        barcode(P, 1170, 140, 560, 110, sp(t, rb + .6, 1.4, lambda q: q), seed=21, gj=gj)
    if t > bt - .1:
        pp = sp(t, bt - .1, .22); pink_block(P, int(780 * pp), 270, 830, 60, seed=12, gj=gj)
        draw_ransom(P, 'BRAND', 850 + gj[0], 80 + gj[1], 200, 8, t, bt, stagger=.1)
    rt = word(3, 'rules')
    rules = ['1.  WEAR THE UNIFORM', '2.  OBEY THE LEADER', '3.  NEVER ASK WHO PROFITS']
    if t > rt - .1:
        pp = sp(t, rt - .1, .25)
        cm = rect_mask(980, 280, 2, seed=33); P.card(cm, 830 + gj[0], 360 + gj[1] - (1 - pp) * 30, TILES[0], s=.45)
        if pp > .9:
            tm = text_mask('HOUSE RULES', 'catalogue', 36); P.ink(tm, 860 + gj[0], 375 + gj[1])
            for i, r in enumerate(rules):
                if t >= rt + .25 + i * .42:
                    tm = text_mask(r, 'catalogue', 42); P.ink(tm, 860 + gj[0], 430 + i * 62 + gj[1])
    ut = word(3, 'uniforms')
    if t > ut - .1:
        pp = sp(t, ut - .1, .25)
        bm = rect_mask(980, 205, 2, seed=35); P.card(bm, 830 + gj[0], 665 + gj[1] + (1 - pp) * 40, TILES[0], s=.45)
        if pp > .9:
            conveyor_band(P, t, 840, 680, 960, 180)
    lt = word(3, 'leader')
    if t > lt - .05:
        pp = sp(t, lt - .05, .25)
        bm = rotate(rect_mask(int(330 * pp) + 10, 110, 4, seed=15, torn=True), -5)
        stamp(P, bm, 1500 + gj[0], 405 + gj[1])
        tm = text_mask('LEADER™', 'vandal_stencil', 84, rot=-5, rough=.4, seed=4); P.ink(tm, 1512 + gj[0], 402 + gj[1], s=min(1, pp * 2))
    caption(P, t, gj, only=3)
    return P

def S5(t):
    P = Page(0, 0, 0); gj = jit(t, 5); u = prog(t, 20.0, 10)
    draw_photo_bg(P, 'coffee', u, z=(1.0, 1.1), c=(.5, .45), dc=(-.04, .06), density=.32, lo=.1, hi=.9)
    ft = word(4, 'free')
    if t > 20.4:
        pp = sp(t, ft - .15, .25)
        bm = rotate(rect_mask(int(540 * pp) + 10, 250, 4, seed=18, torn=True), 5)
        stamp(P, bm, 1180 + gj[0], 80 + gj[1])
        tm = text_mask('FREE?', 'vandal_marker', 200, rot=5, rough=.5, seed=2); P.ink(tm, 1200 + gj[0], 85 + gj[1], s=min(1, pp * 2))
        if t > ft + .5: marker(P, [(1200, 330), (1680, 300)], sp(t, ft + .5, .4, lambda q: q), 12, seed=5, gj=gj)
    wt = [word(4, 'who'), word(4, 'sold'), word(4, 'you'), word(4, 'your'), word(4, 'freedom')]
    SZ = 118
    rows = [[('WHO', wt[0], 3, False), ('SOLD', wt[1], 4, True), ('YOU', wt[2], 5, False)], [('YOUR', wt[3], 6, False), ('FREEDOM', wt[4], 7, True)]]
    for r, row in enumerate(rows):
        x = 110 if r == 0 else 150; y = 430 + r * 190
        for (s_, tt, seed, pk) in row:
            lt_, wtot = ransom_layout(s_, SZ, seed)
            if t >= tt - .15:
                if pk:
                    pp = sp(t, tt - .15, .25); pink_block(P, int((wtot + 36) * pp), 168, x - 22, y - 16, seed=seed + 70, gj=gj)
                draw_ransom(P, s_, x + gj[0], y + gj[1], SZ, seed, t, tt, stagger=.08)
            x += wtot + 60
    if t >= 28.85:
        pp = sp(t, 28.85, .22); bm = rotate(rect_mask(int(520 * pp) + 10, 200, 4, seed=77, torn=True), -7)
        stamp(P, bm, 1180 + gj[0], 640 + gj[1])
        tm = text_mask('SOLD', 'vandal_stencil', 190, rot=-7, rough=.4, seed=8); P.ink(tm, 1230 + gj[0], 630 + gj[1], s=min(1, pp * 2))
    ts_ = 26.45
    if t >= ts_:
        a = sp(t, ts_, .2)
        sm = rect_mask(1560, 175, 3, seed=61, torn=True); P.card(sm, 180 + gj[0], 835 + gj[1], TILES[0], s=.42)
        tl1 = "Fight Club isn't about rebellion."; tl2 = "It's about a man who can't tell what he owns from who he is."
        n1 = int(len(tl1) * clamp((t - ts_) / .7)); n2 = int(len(tl2) * clamp((t - ts_ - .7) / 1.5))
        if n1: P.ink(text_mask(tl1[:n1], 'catalogue', 46), 230 + gj[0], 855 + gj[1])
        if n2: P.ink(text_mask(tl2[:n2], 'catalogue_regular', 44), 230 + gj[0], 925 + gj[1])
    else:
        caption(P, t, gj, only=4)
    return P

SCENES = [S1, S2, S3, S4, S5]
ROT = [-0.35, 0.45, -0.3, 0.4, -0.25]
def scene_frame(k, t):
    P = SCENES[k](t); img = P.finish()
    return page_warp(img, ROT[k], 1.02)
