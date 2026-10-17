"""Transitions from the XEROX CATALOGUE style: paper_slap, copier_flash, torn_page_reveal, staple_pop + film damage."""
import math, functools
import numpy as np, cv2
from xerox_lib import *

def _lerp(a, b, q): return a + (b - a) * q
def sprog(t, t0, dur, fps=15):  # stepped progress
    return clamp((math.floor(t * fps) / fps - t0) / dur)

def paper_slap(A, B, t, t0, dur=0.40):
    p = sprog(t, t0, dur); e = out_expo(p * 1.05)
    if A is None: A = np.full((H, W, 3), (0.05, 0.05, 0.05), np.float32)
    edge = int(e * H)
    out = A.copy()
    land = t0 + dur
    shake = 0
    if t >= land: shake = [9, -6, 4, -2][min(3, int((t - land) * 30))] if t - land < 0.14 else 0
    M = np.float32([[1, 0, 0], [0, 1, edge - H + shake]])
    Bw = cv2.warpAffine(B, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    y = min(H, edge + shake)
    if y > 0: out[:y] = Bw[:y]
    # shadow under leading edge
    if 0 < y < H:
        n = min(70, H - y); g = np.linspace(.55, 1, n, dtype=np.float32)[:, None, None]
        out[y:y + n] *= g
    if p >= 1:
        out = np.roll(B, shake, axis=0) if shake else B
        if shake > 0: out[:shake] = out[shake:shake + 1]
        elif shake < 0: out[shake:] = out[shake - 1:shake][:1]
    return out

def copier_flash(A, B, t, t0, dur=0.45):
    p = sprog(t, t0, dur, 20)
    base = A if p < 0.5 else B
    f = max(0.0, 1 - abs(p - .5) / .42) ** 1.3
    out = base * (1 - f) + f
    bx = int(_lerp(-300, W + 300, p)); bw = 260
    x0, x1 = max(0, bx - bw), min(W, bx + bw)
    if x1 > x0:
        xs = np.arange(x0, x1, dtype=np.float32); prof = np.clip(1 - np.abs(xs - bx) / bw, 0, 1) ** 1.2
        out[:, x0:x1] = out[:, x0:x1] + (1 - out[:, x0:x1]) * (prof * .8)[None, :, None]
        for xx in (bx - bw // 2, bx + bw // 2):
            if 0 <= xx < W - 3: out[:, xx:xx + 3] *= .55
    return out

@functools.lru_cache(None)
def _grids():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); return xx, yy
def _noise1d(n, seed, octs=((140, 26), (47, 12), (15, 5), (5, 2))):
    rg = np.random.default_rng(seed); out = np.zeros(n, np.float32)
    for per, amp in octs:
        k = rg.standard_normal(n // per + 3).astype(np.float32)
        out += np.interp(np.arange(n) / per, np.arange(len(k)), k) * amp
    return out
def torn_reveal(A, B, t, t0, dur=0.55):
    p = sprog(t, t0, dur, 15); e = p ** 1.4
    ang = math.radians(32); n0, n1 = math.cos(ang), math.sin(ang)
    xx, yy = _grids(); s = xx * n0 + yy * n1; tt = (-xx * n1 + yy * n0 + 1200).astype(np.int32)
    nz = _noise1d(4000, 5)[tt]
    fine = _noise1d(4000, 9, octs=((6, 3), (2, 1.5)))[tt]
    lo, hi = -250, W * n0 + H * n1 + 250
    f = _lerp(lo, hi, e)
    sd = s - f + nz + fine           # >0 : old page A still there; <0 : revealed B
    out = B.copy()
    # shadow of old page on B near edge
    shd = np.clip(1 - (np.abs(np.minimum(sd, 0)) / 38), 0, 1) * (sd < 0)
    out *= (1 - .55 * shd)[..., None]
    a_mask = sd > 18
    back = (sd > 0) & (sd <= 18)        # curled paper back
    out = np.where(a_mask[..., None], A, out)
    g = (0.93 - 0.13 * (1 - sd / 18).clip(0, 1))[..., None]
    out = np.where(back[..., None], np.broadcast_to(g, out.shape) * np.array([1, .99, .95], np.float32), out)
    return out

def staple_pop(A, B, t, t0, dur=0.40):
    p = sprog(t, t0, dur, 15)
    if p < 0.12: return A
    q = (p - .12) / .88
    sc = 1.0 + 0.14 * (1 - out_back(q, 1.6))
    out = page_warp(B, 0, sc, 0, 0, border=(0.05, .05, .05))
    if q > .45:
        pg = Page(0, 0); out_ = out.copy()
        st = tex('staple', 0); r = pg._sl(st[..., 3], 40, 34)
        if r is not None:
            sl, mm = r; out_[sl] = out_[sl] * (1 - mm[..., None]) + st[..., :3][:mm.shape[0], :mm.shape[1]] * mm[..., None]
        out = out_
    if .4 < q < .6: out = out * .75 + .25
    return out

def film_damage(img, t, t0, dur, seed):
    """cigarette burn + frame flicker, only inside a transition window"""
    k = int(math.floor(t * 30)); rg = np.random.default_rng(k * 7 + seed)
    amp = CFG['film_damage']['flicker_amp']
    img = np.clip(img * (1 + rg.uniform(-amp, amp * .4)), 0, 1)
    idx = int((t - t0) * 30)
    if 1 <= idx <= 6:
        b = tex('burn', seed % 3); sc = [1.4, 1.0, 1.7][seed % 3]
        b = cv2.resize(b, None, fx=sc, fy=sc)
        px, py = [(1640, 90), (110, 700), (1500, 760), (120, 120), (1720, 300)][seed % 5]
        px += int(rg.integers(-4, 5)); py += int(rg.integers(-4, 5))
        h, w = b.shape[:2]
        x0, y0 = max(0, px), max(0, py); x1, y1 = min(W, px + w), min(H, py + h)
        if x1 > x0 and y1 > y0:
            bb = b[y0 - py:y1 - py, x0 - px:x1 - px]; a = bb[..., 3:4]
            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + bb[..., :3] * a
    return img
