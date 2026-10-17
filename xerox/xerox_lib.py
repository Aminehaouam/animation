"""XEROX CATALOGUE style engine. Everything look-related is read from config/xerox_catalogue.json."""
import json, math, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

CFG = json.load(open('config/xerox_catalogue.json'))
W, H = CFG['canvas']['width'], CFG['canvas']['height']
PINK = np.array(CFG['palette']['pink_rgb'], np.float32) / 255
INK = np.array(CFG['palette']['ink_rgb'], np.float32) / 255
PAPER = np.array(CFG['palette']['paper_rgb'], np.float32) / 255
TILES = [np.array(t, np.float32) / 255 for t in CFG['palette']['ransom_tiles']]
MIS = CFG['misregistration_px']
FONTS = CFG['fonts']

def _load(p, gray=False):
    a = np.asarray(Image.open(p).convert('L' if gray else 'RGB'), np.float32) / 255
    return a
@functools.lru_cache(None)
def tex(kind, i):
    if kind == 'paper': return _load(CFG['textures']['paper'][i % 2])
    if kind == 'toner': return _load(CFG['textures']['toner'][i % 2], True)
    if kind == 'lid': return _load(f"assets/textures/lid_{CFG['textures']['lid'][i % 4]}.png", True)
    if kind == 'burn': return np.asarray(Image.open(CFG['textures']['burn'][i % 3]), np.float32) / 255
    if kind == 'tape': return np.asarray(Image.open(CFG['textures']['tape']), np.float32) / 255
    if kind == 'staple': return np.asarray(Image.open(CFG['textures']['staple']), np.float32) / 255

# ---------- easing / stepped time ----------
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def step_t(t, fps=None): f = fps or CFG['motion']['graphics_fps']; return math.floor(t * f + 1e-6) / f
def out_back(p, s=1.9): p = clamp(p) - 1; return 1 + (s + 1) * p ** 3 + s * p ** 2
def out_expo(p): p = clamp(p); return 1 if p >= 1 else 1 - 2 ** (-10 * p)
def prog(t, t0, dur): return clamp((t - t0) / dur)
def sp(t, t0, dur, ease=out_back):  # stepped progress for graphics
    return ease(prog(step_t(t), t0, dur))
def rng_for(*k): return np.random.default_rng(abs(hash(k)) % (2 ** 32))
def jit(t, seed=0):
    r = np.random.default_rng(int(step_t(t) * 10) * 131 + seed)
    a = CFG['motion']['graphics_jitter_px']
    return (r.uniform(-a, a), r.uniform(-a, a))

# ---------- halftone ----------
def _screen():
    ang = math.radians(CFG['photocopy']['halftone_angle_deg']); s = CFG['photocopy']['halftone_cell_px']
    Hb, Wb = H + 480, W + 480
    yy, xx = np.mgrid[0:Hb, 0:Wb].astype(np.float32)
    u = (xx * math.cos(ang) + yy * math.sin(ang)) / s * 2 * math.pi
    v = (-xx * math.sin(ang) + yy * math.cos(ang)) / s * 2 * math.pi
    return ((np.cos(u) + np.cos(v)) / 4 + 0.5).astype(np.float32)
SCREEN = _screen()
def screen_at(x, y, w, h):
    x = int(clamp(x + 240, 0, SCREEN.shape[1] - w)); y = int(clamp(y + 240, 0, SCREEN.shape[0] - h))
    return SCREEN[y:y + h, x:x + w]
def halftone(d, x=0, y=0, k=7.0):
    """d: ink density 0..1 (1 = black). returns ink mask using the fixed copier screen."""
    h, w = d.shape[:2]
    return np.clip((d - (1 - screen_at(x, y, w, h))) * k + 0.5, 0, 1)
def tone(gray, lo=0.12, hi=0.85, gamma=1.0, toner=None):
    g = np.clip((gray - lo) / (hi - lo), 0, 1) ** gamma
    c0, c1 = CFG['photocopy']['crush']
    return np.clip((1 - g) * c0 + c1, 0, 1)

def cover_view(gray, z=1.0, cx=None, cy=None, w=W, h=H):
    sh, sw = gray.shape[:2]
    scale = max(w / sw, h / sh) * z
    cx = sw / 2 if cx is None else cx; cy = sh / 2 if cy is None else cy
    M = np.float32([[scale, 0, w / 2 - cx * scale], [0, scale, h / 2 - cy * scale]])
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

# ---------- text ----------
@functools.lru_cache(maxsize=4096)
def font(path, size): return ImageFont.truetype(path, size)
@functools.lru_cache(maxsize=2048)
def text_mask(text, fkey, size, rot=0.0, rough=0.0, stroke=0, seed=0):
    path = FONTS.get(fkey, fkey)
    f = font(path, size); pad = 14 + stroke
    l, t, r, b = f.getbbox(text, stroke_width=stroke)
    im = Image.new('L', (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
    m = np.asarray(im, np.float32) / 255
    if rough > 0:
        rg = np.random.default_rng(seed + 5)
        n = cv2.GaussianBlur(rg.standard_normal(m.shape).astype(np.float32), (0, 0), 1.6) * rough * 2.2
        m = np.clip((cv2.GaussianBlur(m, (0, 0), 1.3) + n - 0.5) * 3.2 + 0.5, 0, 1)
    if rot: m = rotate(m, rot)
    return m
def rotate(m, deg):
    h, w = m.shape[:2]; a = math.radians(deg)
    nw = int(abs(w * math.cos(a)) + abs(h * math.sin(a))) + 2; nh = int(abs(w * math.sin(a)) + abs(h * math.cos(a))) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0); M[0, 2] += (nw - w) / 2; M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(m, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
def rect_mask(w, h, rough=3.0, seed=0, torn=False):
    w = max(12, int(w)); h = max(12, int(h))
    m = np.zeros((h, w), np.float32)
    rg = np.random.default_rng(seed)
    n = 40
    top = [(x, rg.uniform(0, rough * (2.5 if torn else 1))) for x in np.linspace(0, w, n)]
    bot = [(x, h - rg.uniform(0, rough * (2.5 if torn else 1))) for x in np.linspace(w, 0, n)]
    pts = np.array([(p[0], p[1]) for p in top] + [(w - rg.uniform(0, rough), h * .5)] + [(p[0], p[1]) for p in bot] + [(rg.uniform(0, rough), h * .5)], np.int32)
    cv2.fillPoly(m, [pts], 1.0)
    return cv2.GaussianBlur(m, (0, 0), 0.8)

# ---------- page / canvas ----------
class Page:
    def __init__(self, paper_i=0, lid_i=0, toner_i=0):
        self.paper = tex('paper', paper_i); self.img = self.paper.copy()
        self.lid_i, self.toner_i = lid_i, toner_i
    def _sl(self, m, x, y):
        h, w = m.shape[:2]; x = int(round(x)); y = int(round(y))
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0: return None
        return (slice(y0, y1), slice(x0, x1)), m[y0 - y:y1 - y, x0 - x:x1 - x]
    def ink(self, m, x, y, s=1.0, col=None):
        r = self._sl(m, x, y)
        if r is None: return
        sl, mm = r; c = INK if col is None else col
        self.img[sl] *= (1 - (mm * s)[..., None] * (1 - c))
    def pink(self, m, x, y, s=1.0, off=True):
        if off: x += MIS['pink_dx']; y += MIS['pink_dy']
        self.ink(m, x, y, s, PINK)
    def fill(self, m, x, y, color, s=1.0, tex_mul=True):
        r = self._sl(m, x, y)
        if r is None: return
        sl, mm = r; mm = (mm * s)[..., None]
        base = np.asarray(color, np.float32)
        if tex_mul: base = base * (self.paper[sl] / PAPER) ** 1.0
        self.img[sl] = self.img[sl] * (1 - mm) + base * mm
    def shadow(self, m, x, y, s=0.42, d=None):
        dx, dy = d or (MIS['shadow_dx'], MIS['shadow_dy'])
        r = self._sl((m > 0.5).astype(np.float32), x + dx, y + dy)
        if r is None: return
        sl, mm = r; self.img[sl] *= (1 - s * mm)[..., None]
    def card(self, m, x, y, tone=None, s=0.42, shadow=True):
        if shadow: self.shadow(m, x, y, s)
        self.fill(m, x, y, TILES[0] if tone is None else tone)
    def tape(self, x, y, rot=0, w=1.0):
        t = tex('tape', 0).copy(); 
        if w != 1.0: t = cv2.resize(t, None, fx=w, fy=w)
        if rot: 
            a = rotate(t[..., 3], rot); c = rotate(t[..., 0], rot); t = np.dstack([c, c, c, a]) 
            t[..., 0] = t[..., 0]; 
        m = t[..., 3]
        self.ink(m, x, y, 0.55, np.array([.78, .72, .5], np.float32))
    def staple(self, x, y):
        t = tex('staple', 0); r = self._sl(t[..., 3], x, y)
        if r is None: return
        sl, mm = r; col = t[..., :3][:mm.shape[0], :mm.shape[1]]
        self.img[sl] = self.img[sl] * (1 - mm[..., None]) + col * mm[..., None]
    def finish(self):
        t = tex('toner', self.toner_i)
        img = self.img * (1 - (t * 0.45)[..., None])
        img *= tex('lid', self.lid_i)[..., None]
        return np.clip(img, 0, 1)

def page_warp(img, deg=0.0, scale=1.0, dx=0.0, dy=0.0, border=(0.12, 0.11, 0.1)):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, scale); M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

# ---------- ransom-note lettering ----------
@functools.lru_cache(maxsize=256)
def ransom_layout(word, size, seed, allow_black=True):
    rg = np.random.default_rng(seed); out = []; x = 0
    pool = FONTS['ransom_pool']
    for i, ch in enumerate(word):
        fp = pool[int(rg.integers(0, len(pool)))]
        fs = int(size * rg.uniform(0.82, 1.08))
        tm = text_mask(ch, fp, fs)
        tw, th = tm.shape[1] - 28, tm.shape[0] - 28
        pw, ph = int(tw + size * 0.30), int(size * rg.uniform(1.0, 1.2))
        tile = rect_mask(pw, ph, rough=2.5, seed=int(seed * 31 + i), torn=True)
        rot = float(rg.uniform(-9, 9))
        ti = int(rg.integers(0, 4 if allow_black else 3))
        if ch.upper() == ch and not allow_black: ti = int(rg.integers(0, 3))
        tile_r = rotate(tile, rot); tm_r = rotate(tm, rot)
        out.append(dict(x=x, y=float(rg.uniform(-0.07, 0.07) * size), tile=tile_r, tm=tm_r, ti=ti))
        x += tile_r.shape[1] * 0.86
    return out, x
def draw_ransom(page, word, x, y, size, seed, t, t0, stagger=0.09, dur=0.12, gj=(0, 0), allow_black=True):
    letters, wtot = ransom_layout(word, size, seed, allow_black)
    for i, L in enumerate(letters):
        p = sp(t, t0 + i * stagger, dur, ease=lambda q: q)
        if p <= 0: continue
        sc = 1.0
        lx = x + L['x'] + gj[0]; ly = y + L['y'] + gj[1] - (1 - p) * 30
        page.card(L['tile'], lx, ly, TILES[L['ti']], s=0.45)
        tcol = INK if L['ti'] != 3 else TILES[0]
        tmx = lx + (L['tile'].shape[1] - L['tm'].shape[1]) / 2; tmy = ly + (L['tile'].shape[0] - L['tm'].shape[0]) / 2
        if L['ti'] == 3: page.fill(L['tm'], tmx, tmy, TILES[0], tex_mul=False)
        else: page.ink(L['tm'], tmx, tmy, 1.0)
    return wtot

# ---------- marker / hand-drawn ----------
def wobble_path(pts, step=9, amp=2.2, seed=0):
    rg = np.random.default_rng(seed); out = []
    for (a, b) in zip(pts[:-1], pts[1:]):
        n = max(2, int(math.dist(a, b) / step))
        for k in range(n):
            q = k / n; out.append((a[0] + (b[0] - a[0]) * q + rg.normal(0, amp), a[1] + (b[1] - a[1]) * q + rg.normal(0, amp)))
    out.append(pts[-1]); return out
def marker(page, pts, p, width=12, seed=0, color=None, pink=False, gj=(0, 0)):
    """progressively draw a wobbly marker stroke through pts, p in 0..1"""
    if p <= 0: return
    path = wobble_path(pts, seed=seed); n = max(2, int(len(path) * p))
    path = np.array(path[:n], np.float32)
    x0, y0 = path.min(0) - width * 2; x1, y1 = path.max(0) + width * 2
    w, h = int(x1 - x0) + 1, int(y1 - y0) + 1
    m = np.zeros((h, w), np.float32)
    P = (path - [x0, y0]).astype(np.int32)
    cv2.polylines(m, [P], False, 1.0, width, cv2.LINE_AA)
    cv2.polylines(m, [P + [1, 2]], False, 0.7, max(2, width - 4), cv2.LINE_AA)
    rg = np.random.default_rng(seed + 9)
    m *= np.clip(cv2.GaussianBlur(rg.random((h, w)).astype(np.float32), (0, 0), 1.2) * 1.6 + 0.35, 0, 1)
    m = np.clip(m * 1.4, 0, 1)
    if pink: page.pink(m, x0 + gj[0], y0 + gj[1])
    else: page.ink(m, x0 + gj[0], y0 + gj[1])

def barcode(page, x, y, w, h, p, seed=3, gj=(0, 0)):
    rg = np.random.default_rng(seed); m = np.zeros((h, w), np.float32); cx = 0
    while cx < w:
        bw = int(rg.choice([2, 3, 3, 4, 6])); 
        if rg.random() > 0.35: m[:, cx:cx + bw] = 1
        cx += bw + int(rg.choice([2, 3, 4]))
    m[:, int(w * p):] = 0
    page.ink(m, x + gj[0], y + gj[1])

def price_tag(page, x, y, size, p, price='$ 4.99', gj=(0, 0), t=0.0, t0=0.0):
    """tag outline + string draw on, label text pops once outline done."""
    w, h = int(size * 1.95), size
    body = np.array([[0.22 * w, 0], [w, 0], [w, h], [0.22 * w, h], [0, h * 0.5]], np.float32)
    ring = (0.2 * w, h * 0.5)
    pts = [tuple(v) for v in body] + [tuple(body[0])]
    m = np.zeros((h + 8, w + 8), np.float32)
    per = sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])); tgt = per * clamp(p * 1.15); acc = 0
    for a, b in zip(pts[:-1], pts[1:]):
        d = math.dist(a, b)
        if acc >= tgt: break
        q = clamp((tgt - acc) / d)
        e = (a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q)
        cv2.line(m, (int(a[0]) + 4, int(a[1]) + 4), (int(e[0]) + 4, int(e[1]) + 4), 1.0, 6, cv2.LINE_AA); acc += d
    if p > 0.5: cv2.circle(m, (int(ring[0]) + 4, int(ring[1]) + 4), int(size * 0.07), 1.0, 5, cv2.LINE_AA)
    page.ink(m, x + gj[0], y + gj[1])
    if p >= 0.95:
        tm = text_mask(price, 'catalogue', int(size * 0.40))
        page.ink(tm, x + w * 0.68 - tm.shape[1] / 2 + gj[0], y + h * 0.5 - tm.shape[0] / 2 + gj[1])
    return w, h

def torn_strip(w, h, seed): return rect_mask(w, h, rough=4, seed=seed, torn=True)

def pink_block(page, w, h, x, y, seed=0, s=1.0, gj=(0, 0)):
    m = rect_mask(w, h, rough=4, seed=seed, torn=True)
    page.card(m, x + gj[0], y + gj[1], TILES[0], s=.4)
    page.pink(m, x + gj[0], y + gj[1], s)
