"""VOX x XEROX demo: footage from user's trailer -> explainer-style edit with cutouts, motion graphics, navy subtitles."""
import math, functools
import numpy as np, cv2
from xerox_lib import *

NAVY = np.array([0x14, 0x28, 0x5A], np.float32) / 255
YELLOW = np.array([255, 216, 77], np.float32) / 255
FPS = 30
CUTS = [0.0, 1.468, 3.003, 4.171, 5.639, 6.840, 8.842, 9.91]
NF = 297

@functools.lru_cache(maxsize=48)
def frame(i):
    i = int(clamp(i, 1, 299))
    return cv2.cvtColor(cv2.imread(f'build/fr/f{i:04d}.png'), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
@functools.lru_cache(maxsize=48)
def mask(j):
    return cv2.imread(f'build/cut/m{j:04d}.png', 0).astype(np.float32) / 255
def held(t):  # stop-motion index: odd frames -> 15 fps
    i = int(t * FPS) + 1; return i if i % 2 == 1 else i - 1
def grade(img, sat=.72, warm=.0):
    g = img @ np.array([.299, .587, .114], np.float32)
    out = g[..., None] + (img - g[..., None]) * sat
    out = np.clip((out - .5) * 1.06 + .5 + .01, 0, 1)
    return out * np.array([1.0, .985, .94], np.float32)

def rot_rgba(rgb, a, deg):
    h, w = a.shape; r = math.radians(deg)
    nw = int(abs(w * math.cos(r)) + abs(h * math.sin(r))) + 2; nh = int(abs(w * math.sin(r)) + abs(h * math.cos(r))) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0); M[0, 2] += (nw - w) / 2; M[1, 2] += (nh - h) / 2
    return (cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR), cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR))
def blit_rgb(P, rgb, a, x, y):
    r = P._sl(a, x, y)
    if r is None: return
    sl, aa = r; h, w = aa.shape; sub = rgb[(sl[0].start - int(round(y))):(sl[0].start - int(round(y))) + h, (sl[1].start - int(round(x))):(sl[1].start - int(round(x))) + w]
    P.img[sl] = P.img[sl] * (1 - aa[..., None]) + sub * aa[..., None]

def card_geom(cx, cy, w, rot): h = w * 556 / 1280; return h
def to_page(cx, cy, w, rot, fx, fy):
    s = w / 1280; px, py = (fx - 640) * s, (fy - 278) * s; r = math.radians(rot)
    return cx + px * math.cos(r) + py * math.sin(r), cy - px * math.sin(r) + py * math.cos(r)

def footage_card(P, img, cx, cy, w, rot=0.0, border=16, tape=True, shadow=.5):
    h = int(w * 556 / 1280); w = int(w)
    im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    pad = border; full = np.ones((h + 2 * pad, w + 2 * pad), np.float32)
    canvas = np.ones((h + 2 * pad, w + 2 * pad, 3), np.float32) * TILES[0]
    canvas[pad:pad + h, pad:pad + w] = im
    canvas_r, a_r = rot_rgba(canvas, full, rot)
    x = cx - a_r.shape[1] / 2; y = cy - a_r.shape[0] / 2
    P.shadow(a_r, x, y, shadow)
    blit_rgb(P, canvas_r, a_r, x, y)
    if tape:
        P.tape(x + 10, y + 4, 28 + rot * 2); P.tape(x + a_r.shape[1] - 170, y + a_r.shape[0] - 52, 28 + rot * 2)
    return x, y

def pop(P, img, m, cx, cy, w, rot=0.0, border=13, ink=True):
    """subject cutout (sticker border, hard shadow) from a frame + matte, centred at (cx,cy)"""
    h = int(w * 556 / 1280); w = int(w)
    im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA); mm = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    mm = np.clip((mm - .12) * 1.5, 0, 1)
    pad = border + 8
    im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_REPLICATE); mm = cv2.copyMakeBorder(mm, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    imr, mr = rot_rgba(im, mm, rot) if rot else (im, mm)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
    st = (cv2.GaussianBlur(cv2.dilate((mr > .4).astype(np.float32), k), (0, 0), 1.3) > .5).astype(np.float32)
    x = cx - mr.shape[1] / 2; y = cy - mr.shape[0] / 2
    P.card(st, x, y, TILES[0], s=.5)
    blit_rgb(P, imr, np.clip(mr * 1.0, 0, 1), x, y)
    edge = np.clip(cv2.dilate(mr, np.ones((3, 3), np.uint8)) - cv2.erode(mr, np.ones((3, 3), np.uint8)), 0, 1)
    P.ink(edge, x, y, .75, NAVY)

def ghost(P, img, z=2.0, cx=.5, cy=.5, density=.30):
    g = img @ np.array([.299, .587, .114], np.float32)
    v = cover_view(g, z, cx * g.shape[1], cy * g.shape[0], W, H)
    P.ink(halftone(tone(v, .1, .9) * density), 0, 0, 1)

def color_block(P, x, y, w, h, col, seed=0, s=.85, gj=(0, 0)):
    m = rect_mask(w, h, 4, seed, True); P.ink(m, x + MIS['pink_dx'] + gj[0], y + MIS['pink_dy'] + gj[1], s, col)

def highlight(P, x, y, w, h, p, seed=1, gj=(0, 0), col=None):
    if p <= 0: return
    m = rect_mask(max(14, int(w * p)), h, 3, seed, True)
    rg = np.random.default_rng(seed); m *= np.clip(cv2.GaussianBlur(rg.random(m.shape).astype(np.float32), (0, 0), 2) * 1.2 + .55, 0, 1)
    P.ink(m, x + gj[0], y + gj[1], .92, YELLOW if col is None else col)

def big_text(P, s, x, y, size, font='catalogue', col=NAVY, gj=(0, 0), rot=0, rough=0):
    m = text_mask(s, font, size, rot=rot, rough=rough, seed=4); P.ink(m, x + gj[0], y + gj[1], 1.0, col); return m.shape[1] - 28, m.shape[0] - 28

def subtitle(P, t, text, t0, t1, gj=(0, 0)):
    ts = step_t(t)
    if not (t0 <= ts <= t1): return
    words = text.split(); n = max(1, int(len(words) * clamp((ts - t0) / max(.3, (t1 - t0) * .35)) + .999)); s = ' '.join(words[:n])
    full = text_mask(text, 'catalogue_regular', 42); m = text_mask(s, 'catalogue_regular', 42)
    x = W / 2 - (full.shape[1] - 28) / 2 - 14; y = 968
    halo = cv2.dilate(m, np.ones((7, 7), np.uint8)); halo = cv2.GaussianBlur(halo, (0, 0), 2.5)
    P.fill(np.clip(halo * 1.6, 0, 1), x, y, TILES[0], s=.88)
    P.ink(m, x, y, 1.0, NAVY)

def stamp_text(P, s, x, y, size, rot, font='vandal_stencil', w=None, seed=1, t=0, t0=0, gj=(0, 0)):
    pp = sp(t, t0, .22)
    if pp <= 0: return
    tm = text_mask(s, font, size, rot=rot, rough=.4, seed=seed)
    bw = int((tm.shape[1] - 10) * pp) + 10; bm = rotate(rect_mask(bw, tm.shape[0] - 16, 4, seed, True), rot)
    P.card(bm, x + gj[0], y + gj[1], TILES[0], s=.4); P.pink(bm, x + gj[0], y + gj[1])
    P.ink(tm, x + 6 + gj[0], y - 6 + gj[1], min(1, pp * 2), NAVY)

def ellipse_marker(P, cx, cy, rx, ry, p, seed=0, gj=(0, 0), col=None, pink=False, width=9):
    pts = [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in np.linspace(-1.2, 2 * math.pi - .4, 18)]
    marker(P, pts, p, width, seed, gj=gj, col=col, pink=pink)

def arrow(P, a, b, p, seed=0, gj=(0, 0), col=NAVY):
    marker(P, [a, ((a[0] + b[0]) / 2 + 20, (a[1] + b[1]) / 2 - 20), b], p, 8, seed, gj=gj, col=col)
    if p > .95:
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        for d in (2.6, -2.6):
            marker(P, [b, (b[0] + 36 * math.cos(ang + d), b[1] + 36 * math.sin(ang + d))], 1, 8, seed + 5, gj=gj, col=col)

# ------------------------------------------------------------ shots
def L(t, t0, t1, paper, lid, tone_i):
    P = Page(paper, lid, tone_i); gj = jit(t, paper * 3 + lid); i = int(clamp(t * FPS + 1, 1, NF)); img = frame(i)
    return P, gj, img, i

def shot_a(t):
    P, gj, img, i = L(t, 0, 1.468, 0, 0, 0); j = held(t)
    ghost(P, img, 2.3, .3, .4, .30)
    color_block(P, 640, 100, 1180, 640, YELLOW * .0 + PINK, seed=3, s=.9, gj=gj)
    cx, cy, w, rot = 1250 + 10 * prog(t, 0, 1.5), 565, 1020, -2.4
    footage_card(P, grade(img), cx, cy, w, rot)
    pop(P, grade(frame(j), .85), mask(j), cx - 6, cy - 8, w * 1.18, rot)
    p = sp(t, .12, .35)
    if p > 0:
        highlight(P, 70, 455, 640, 200, p, 5, gj)
        big_text(P, '9 TO 5', 90 + gj[0] - (1 - p) * 120, 385, 230, gj=gj)
    stamp_text(P, 'THE JOB', 110, 290, 100, -4, t=t, t0=.35, gj=gj)
    cupx, cupy = to_page(cx, cy, w, rot, 80, 450)
    ellipse_marker(P, cupx, cupy, 72, 80, sp(t, .55, .4, lambda q: q), 3, gj, pink=True)
    arrow(P, (cupx - 20, cupy + 85), (cupx - 150, cupy + 190), sp(t, .8, .25, lambda q: q), 9, gj)
    stamp_text(P, '$4.50 A DAY', cupx - 520, cupy + 170, 78, -3, font='vandal_marker', t=t, t0=1.0, gj=gj, seed=3)
    subtitle(P, t, "He works a job he can't stand…", .15, 2.95, gj)
    return P

def shot_b(t):
    P, gj, img, i = L(t, 1.468, 3.003, 1, 1, 1); j = held(t); u = prog(t, 1.468, 1.535)
    ghost(P, img, 2.0, .6, .4, .28)
    color_block(P, 130, 230, 1040, 560, YELLOW, seed=5, s=.55, gj=gj)
    cx, cy, w, rot = 650 - 14 * u, 590, 1100, 1.6
    footage_card(P, grade(img), cx, cy, w, rot)
    pop(P, grade(frame(j), .85), mask(j), cx + 4, cy - 6, w * 1.2, rot)
    # right column
    p = sp(t, 1.62, .3)
    if p > 0:
        big_text(P, 'A STEADY', 1260 + gj[0], 300 + gj[1] - (1 - p) * 60, 100)
        highlight(P, 1250, 440, 560, 110, sp(t, 1.9, .4, lambda q: q), 7, gj)
        big_text(P, 'PAYCHECK', 1260 + gj[0], 430 + gj[1], 100)
    subtitle(P, t, "He works a job he can't stand…", .15, 2.95, gj)
    return P

def shot_c(t):
    P, gj, img, i = L(t, 3.003, 4.171, 0, 2, 0); j = held(t)
    ghost(P, img, 2.0, .35, .3, .26)
    color_block(P, 740, 120, 1100, 600, PINK, seed=8, s=.85, gj=gj)
    cx, cy, w, rot = 1250, 520, 960, 2.2
    footage_card(P, grade(img), cx, cy, w, rot)
    pop(P, grade(frame(j), .85), mask(j), cx - 4, cy - 10, w * 1.22, rot)
    # ransom + uniform row
    draw_ransom(P, 'SAME', 80 + gj[0], 190 + gj[1], 175, 11, t, 3.1, stagger=.07)
    if t > 3.4:
        big_text(P, 'SHIRT. SAME TIE.', 95, 400, 76, gj=gj)
    for k in range(4):
        tt = 3.45 + k * .13
        if t >= tt:
            pp = sp(t, tt, .2)
            pop(P, grade(frame(j), .85), mask(j), 300 + k * 235 + gj[0], 770 + gj[1] - (1 - pp) * 40, 560, 0, border=9)
    subtitle(P, t, "…to buy things he doesn't need.", 3.1, 5.55, gj)
    return P

def shot_d(t):
    P, gj, img, i = L(t, 4.171, 5.639, 1, 3, 1); j = held(t); u = prog(t, 4.171, 1.5)
    ghost(P, img, 2.0, .55, .4, .26)
    color_block(P, 330, 90, 1300, 600, YELLOW, seed=12, s=.5, gj=gj)
    cx, cy, w, rot = 960, 420, 1160, -1.0
    footage_card(P, grade(img), cx, cy, w, rot)
    pop(P, grade(frame(j), .85), mask(j), cx, cy - 6, w * 1.14, rot)
    days = ['MON', 'TUE', 'WED', 'THU', 'FRI']
    for k, dname in enumerate(days):
        x = 150 + k * 340; y = 745; tt = 4.25 + k * .12
        if t < tt: continue
        pp = sp(t, tt, .2)
        bm = rect_mask(280, 150, 3, k + 40, True); P.card(bm, x + gj[0], y + gj[1] + (1 - pp) * 30, TILES[0], s=.45)
        big_text(P, dname, x + 42, y + 38 + (1 - pp) * 30, 72, gj=gj)
        c0 = 4.5 + k * .17
        if t >= c0:
            marker(P, [(x + 20, y + 30), (x + 262, y + 128)], sp(t, c0, .18, lambda q: q), 13, k * 3 + 1, gj=gj, col=NAVY)
            marker(P, [(x + 262, y + 30), (x + 20, y + 128)], sp(t, c0 + .1, .18, lambda q: q), 13, k * 3 + 2, gj=gj, col=NAVY)
    subtitle(P, t, "…to buy things he doesn't need.", 3.1, 5.55, gj)
    return P

def shot_e(t):
    P, gj, img, i = L(t, 5.639, 6.84, 0, 0, 0); u = prog(t, 5.639, 1.2)
    ghost(P, img, 1.8, .5, .5, .45)
    color_block(P, 120, 120, 1250, 620, PINK, seed=21, s=.8, gj=gj)
    cx, cy, w, rot = 720 + 8 * u, 520, 1240, -1.2
    footage_card(P, grade(img * 1.35 + .02, .6), cx, cy, w, rot)
    # clock
    tm = '3:07' if int(t * 2) % 2 == 0 else '3 07'
    p = sp(t, 5.75, .25)
    if p > 0:
        cm = rect_mask(520, 300, 3, 31, True); P.card(cm, 1360 + gj[0], 130 + gj[1], TILES[0], s=.5)
        highlight(P, 1370, 190, 460, 170, 1, 3, gj)
        big_text(P, tm, 1368 + gj[0], 175 + gj[1], 200)
        big_text(P, 'A.M.', 1640 + gj[0], 345 + gj[1], 64)
    stamp_text(P, 'AWAKE', 1330, 560, 170, 5, t=t, t0=6.1, gj=gj, seed=6)
    subtitle(P, t, "He can't sleep.", 5.72, 6.8, gj)
    return P

def shot_f(t):
    P, gj, img, i = L(t, 6.84, 8.842, 1, 1, 1); j = held(t); u = prog(t, 6.84, 2.0)
    ghost(P, img, 2.0, .5, .5, .24)
    color_block(P, 220, 110, 1500, 700, YELLOW, seed=33, s=.5, gj=gj)
    cx, cy, w, rot = 960, 520, 1300, 0.8
    footage_card(P, grade(img, .8), cx, cy, w, rot)
    pop(P, grade(frame(j), .9), mask(j), cx, cy - 4, w * 1.1, rot)
    tags = [(470, 20, '$395', 7.15), (545, 250, '$695', 7.5), (20, 200, '$1,250', 7.85), (1090, 215, '$2,400', 8.2)]
    for (fx, fy, price, tt) in tags:
        if t < tt: continue
        x, y = to_page(cx, cy, w, rot, fx, fy)
        pp = sp(t, tt, .45, lambda q: q); wcard = 110
        P.card(rect_mask(int(wcard * 1.95) + 26, wcard + 18, 2, int(fx), True), x - 12 + gj[0], y - 8 + gj[1], TILES[0], s=.4)
        price_tag(P, x + gj[0], y + gj[1], wcard, pp, price)
    subtitle(P, t, 'Everything in his life has a price tag.', 6.95, 8.75, gj)
    return P

def shot_g(t):
    P, gj, img, i = L(t, 8.842, 9.91, 0, 2, 0); u = prog(t, 8.842, 1.07)
    ghost(P, img, 1.8, .5, .5, .30)
    z = 1 + .035 * u
    color_block(P, 190, 80, 1560, 720, PINK, seed=44, s=.75, gj=gj)
    cx, cy, w, rot = 960, 470, 1500 * z, -.8
    footage_card(P, grade(img, .9), cx, cy, w, rot, tape=True)
    tt_ = max(t, 8.9); fx1 = np.interp(tt_, [8.9, 9.2, 9.5, 9.83], [793, 793, 797, 797]); fy1 = np.interp(tt_, [8.9, 9.2, 9.5, 9.83], [120, 87, 88, 71])
    fx2 = np.interp(tt_, [8.9, 9.2, 9.5, 9.83], [963, 997, 1020, 1016]); fy2 = np.interp(tt_, [8.9, 9.2, 9.5, 9.83], [163, 140, 151, 137])
    px, py = to_page(cx, cy, w, rot, fx1, fy1); qx, qy = to_page(cx, cy, w, rot, fx2, fy2)
    highlight(P, px - 100 + gj[0], py - 38, 215, 78, sp(t, 9.05, .22, lambda q: q), 3, gj, col=YELLOW)
    highlight(P, qx - 62 + gj[0], qy - 34, 130, 70, sp(t, 9.3, .2, lambda q: q), 5, gj, col=YELLOW)
    ellipse_marker(P, px, py, 120, 66, sp(t, 9.15, .4, lambda q: q), 2, gj, pink=True, width=10)
    stamp_text(P, 'YOU "CHOSE" THIS?', 850, 830, 90, -3, t=t, t0=9.4, gj=gj, seed=9)
    subtitle(P, t, 'Who chose any of it?', 8.95, 9.95, gj)
    return P

SHOTS = [shot_a, shot_b, shot_c, shot_d, shot_e, shot_f, shot_g]
ROT = [-.3, .35, -.25, .3, -.2, .3, -.3]
def shot_frame(k, t):
    P = SHOTS[k](t); return page_warp(P.finish(), ROT[k], 1.02)
