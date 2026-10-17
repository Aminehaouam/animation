"""Audio: original trailer audio ONLY in key scenes; quiet synthesised bed + foley in VO windows (left clear for the user's voice)."""
import json, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt
from v2lib import *
SR = 44100; TOTAL = PLAN['total']; N = int(TOTAL * SR); rg = np.random.default_rng(11); t = np.arange(N) / SR
def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)
def hp(x, f): return sosfilt(butter(2, f, 'high', fs=SR, output='sos'), x)
def lp(x, f): return sosfilt(butter(2, f, 'low', fs=SR, output='sos'), x)
# ---- bed (very quiet, only under VO windows)
chords = [[110, 164.8, 220, 261.6], [87.3, 130.8, 174.6, 261.6], [73.4, 110, 146.8, 220], [82.4, 123.5, 164.8, 207.7]]
pad = np.zeros(N)
for ci in range(int(TOTAL / 4.5) + 2):
    t0 = ci * 4.5; env = np.clip((t - t0) / 1.5, 0, 1) * np.clip(1 - (t - t0 - 4.5) / 1.5, 0, 1) * ((t >= t0) & (t < t0 + 6))
    for f in chords[ci % 4]:
        for d in (-.3, 0, .35): pad += np.sin(2 * np.pi * (f + d) * t + rg.uniform(0, 6.28)) * env * .05
pad = lp(pad, 1100)
vo_gain = np.zeros(N)
for s in PLAN['segs']:
    if s['kind'] == 'vo':
        a, b = s['t0'], s['t1']; g = np.clip((t - a) / .35, 0, 1) * np.clip((b - t) / .35, 0, 1) * ((t >= a) & (t <= b)); vo_gain = np.maximum(vo_gain, g)
bed = pad * vo_gain; bed = bed / np.abs(bed).max() * .09
# ---- original audio in key scenes
orig = np.zeros((N, 2))
for s in PLAN['segs']:
    if s['kind'] != 'key': continue
    a, sr = sf.read(f"v2/audio/{s['id']}_orig.wav"); i = int(s['t0'] * SR); a = a[:N - i]
    n = len(a); fade = np.minimum(1, np.minimum(np.arange(n) / (.06 * SR), (n - np.arange(n)) / (.18 * SR)))
    orig[i:i + n] += a * fade[:, None]
# ---- foley on transitions
sfx = np.zeros(N)
def put(x, tt, g=1.0):
    i = int(max(0, tt) * SR); j = min(N, i + len(x)); 
    if i < N: sfx[i:j] += x[:j - i] * g
def slap(tt):
    x = np.arange(int(.22 * SR)) / SR; put(np.sin(2 * np.pi * np.cumsum(150 * np.exp(-x * 12) + 45) / SR) * np.exp(-x * 16) + bp(rg.standard_normal(len(x)), 900, 5000) * np.exp(-x * 60) * .9, tt)
def flash(tt):
    x = np.arange(int(.5 * SR)) / SR; e = np.sin(np.pi * np.clip(x / .5, 0, 1)) ** 1.5
    put(bp(rg.standard_normal(len(x)), 600, 7000) * e * .6, tt); put(np.sin(2 * np.pi * np.cumsum(70 * np.exp(-x * 6) + 40) / SR) * np.exp(-x * 9) * .9, tt + .2)
def tear(tt):
    x = np.arange(int(.55 * SR)) / SR; c = (rg.random(len(x)) > .985).astype(float); c = np.convolve(c, np.exp(-np.arange(300) / 40), 'same')
    put(hp(rg.standard_normal(len(x)), 2500) * (.25 + .75 * np.clip(c * 4, 0, 1)) * np.sin(np.pi * x / .55) * .8, tt)
def staple(tt):
    x = np.arange(int(.05 * SR)) / SR; put(hp(rg.standard_normal(len(x)), 4000) * np.exp(-x * 140) + np.sin(2 * np.pi * 1700 * x) * np.exp(-x * 110) * .6, tt, .9)
    y = np.arange(int(.2 * SR)) / SR; put(np.sin(2 * np.pi * np.cumsum(120 * np.exp(-y * 14) + 50) / SR) * np.exp(-y * 20), tt, .8)
slap(.4)
for k, s in enumerate(PLAN['segs']):
    if k == 0: continue
    n = s['trans_in'][0]; w0 = s['t0'] - .12
    {'paper_slap': lambda: slap(w0 + .4), 'copier_flash': lambda: flash(w0), 'torn_reveal': lambda: tear(w0), 'staple_pop': lambda: staple(w0 + .08)}.get(n, lambda: None)()
sfx = sfx / max(1e-6, np.abs(sfx).max()) * .30
base = orig * .92 + np.stack([bed, bed], 1) + np.stack([sfx, sfx], 1)
def fin(x): x = np.tanh(x * 1.05) / np.tanh(1.05); return x / max(1e-6, np.abs(x).max()) * .93
sf.write('v2/mix_for_your_vo.wav', fin(base), SR)
# guide VO (reference only)
G = json.load(open('v2/guide.json')); vo = np.zeros(N)
for s in PLAN['segs']:
    if s['kind'] == 'vo':
        a, sr = sf.read(f"v2/audio/{s['id']}.wav")
        from scipy.signal import resample_poly
        a = resample_poly(a, SR, sr); i = int(s['vo_t0'] * SR); vo[i:i + len(a)] += a[:N - i]
vo = vo / np.abs(vo).max() * .8
sf.write('v2/mix_with_guide_vo.wav', fin(base * .9 + np.stack([vo, vo], 1)), SR); print('audio ok')
