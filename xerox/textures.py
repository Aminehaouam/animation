"""Generates the reusable XEROX CATALOGUE texture set into assets/textures/."""
import numpy as np, cv2
from PIL import Image, ImageDraw
W, H = 1920, 1080
OUT = 'assets/textures/'

def blur_noise(rng, h, w, sigma, amp):
    n = rng.standard_normal((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / (n.std() + 1e-6) * amp

def paper(seed):
    rng = np.random.default_rng(seed)
    base = np.array([244, 240, 227], np.float32) / 255
    img = np.ones((H, W, 3), np.float32) * base
    low = blur_noise(rng, H, W, 120, 0.018) + blur_noise(rng, H, W, 25, 0.008)
    grain = rng.standard_normal((H, W)).astype(np.float32) * 0.012
    img += (low + grain)[..., None] * np.array([1, 1, 1.15], np.float32)
    fib = np.zeros((H, W), np.float32)
    for _ in range(900):
        x, y = rng.integers(0, W), rng.integers(0, H)
        a = rng.uniform(0, np.pi); l = rng.uniform(6, 22)
        cv2.line(fib, (int(x), int(y)), (int(x + l * np.cos(a)), int(y + l * np.sin(a))), float(rng.uniform(-.05, .035)), 1, cv2.LINE_AA)
    img += fib[..., None]
    # creases: highlight + shadow pair along long lines
    cr = np.zeros((H, W), np.float32)
    for _ in range(5):
        p0 = np.array([rng.uniform(0, W), rng.uniform(0, H)])
        ang = rng.uniform(0, np.pi); d = np.array([np.cos(ang), np.sin(ang)])
        a, b = p0 - d * 2500, p0 + d * 2500
        wob = lambda p: p + rng.normal(0, 1.2, 2)
        cv2.line(cr, tuple(map(int, a)), tuple(map(int, b)), -0.07 * rng.uniform(.5, 1), 2, cv2.LINE_AA)
        n = np.array([-d[1], d[0]]) * 3
        cv2.line(cr, tuple(map(int, a + n)), tuple(map(int, b + n)), 0.05 * rng.uniform(.5, 1), 2, cv2.LINE_AA)
    cr = cv2.GaussianBlur(cr, (0, 0), 1.1)
    img += cr[..., None]
    # staple holes top-left
    for (x, y) in [(70, 60), (70, 120)]:
        cv2.ellipse(img, (x, y), (11, 6), 90, 0, 360, (0.42, 0.40, 0.38), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (13, 8), 90, 0, 360, (0.7, 0.68, 0.64), 2, cv2.LINE_AA)
    return np.clip(img, 0, 1)

def toner(seed):
    rng = np.random.default_rng(seed)
    t = np.zeros((H, W), np.float32)
    for _ in range(26):  # vertical drum streaks
        x = int(rng.integers(0, W)); y0 = int(rng.integers(-200, H)); l = int(rng.integers(200, 1200))
        cv2.line(t, (x, y0), (x + int(rng.integers(-3, 4)), y0 + l), float(rng.uniform(.04, .2)), int(rng.integers(1, 4)), cv2.LINE_AA)
    t = cv2.GaussianBlur(t, (0, 0), 1.2)
    sp = (rng.random((H, W)) > 0.9985).astype(np.float32)
    sp = cv2.GaussianBlur(sp, (0, 0), 1.4) * 9
    t += np.clip(sp, 0, .6) + blur_noise(rng, H, W, 40, 0.02).clip(0, 1)
    return np.clip(t, 0, 1)

def lid(seed, side):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    edge = np.minimum.reduce([xx, W - xx, yy * 1.4, (H - yy) * 1.4])
    sh = 1 - 0.38 * np.exp(-edge / 90)
    # irregular lid shadow band on one side
    n = blur_noise(rng, H, W, 30, 14)
    band_d = {'left': xx, 'right': W - xx, 'top': yy, 'bottom': H - yy}[side]
    band = np.clip(1 - (band_d + n - 8) / 26, 0, 1)
    sh *= 1 - 0.55 * band
    return np.clip(sh, 0, 1)

def burn(seed, size=120):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    r = r + blur_noise(rng, size, size, 5, .02)
    ring = np.exp(-((r - .72) / .18) ** 2)       # dark scorched ring
    core = np.clip(1 - r / .6, 0, 1)             # bright centre
    rgba = np.zeros((size, size, 4), np.float32)
    rgba[..., :3] = np.stack([ring * .12 + core * .98] * 3, -1)
    rgba[..., 3] = np.clip(ring * .95 + core * 1.0, 0, 1) * (r < 1)
    return rgba

def tape():
    w, h = 250, 80
    rng = np.random.default_rng(3)
    a = np.zeros((h, w), np.float32)
    pts = [(0, 6)] + [(i, 6 + rng.integers(-4, 5)) for i in range(6, 12, 3)]
    poly = np.array([[8, 0], [w - 8, 0], [w, 10], [w - 6, 25], [w, 40], [w - 8, h - 1], [8, h - 1], [0, 55], [8, 38], [0, 20]], np.int32)
    cv2.fillPoly(a, [poly], 1.0)
    img = np.zeros((h, w, 4), np.float32)
    img[..., :3] = np.array([232, 224, 190], np.float32) / 255
    img[..., 3] = a * 0.72
    img[..., :3] -= (rng.random((h, w, 1)) * .05)
    return np.clip(img, 0, 1)

def staple():
    im = Image.new('RGBA', (90, 34), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle([4, 14, 86, 20], fill=(150, 152, 158, 255)); d.rectangle([4, 14, 86, 15], fill=(235, 238, 240, 255))
    d.rectangle([4, 20, 86, 21], fill=(70, 72, 78, 255))
    d.rectangle([4, 14, 8, 28], fill=(110, 112, 118, 255)); d.rectangle([82, 14, 86, 28], fill=(110, 112, 118, 255))
    return np.asarray(im, np.float32) / 255

def save(a, name):
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(a).save(OUT + name)

if __name__ == '__main__':
    save(paper(1), 'paper_a.png'); save(paper(7), 'paper_b.png')
    save(toner(2), 'toner_a.png'); save(toner(9), 'toner_b.png')
    for i, s in enumerate(['left', 'top', 'right', 'bottom']): save(lid(i, s), f'lid_{s}.png')
    for i in range(3): save(burn(i + 11), f'burn_{i}.png')
    save(tape(), 'tape.png'); save(staple(), 'staple.png')
    print('textures ok')
