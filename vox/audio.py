"""Original trailer audio (ducked, faded) + synthesised paper/copier foley on transitions."""
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt
SR = 44100; rg = np.random.default_rng(7)
a, sr = sf.read('build/clip_audio.wav'); assert sr == SR
N = int(9.91 * SR); a = a[:N]; a = np.pad(a, ((0, N - len(a)), (0, 0)))
t = np.arange(N) / SR
a *= np.clip(t / 0.08, 0, 1)[:, None] * np.clip((9.91 - t) / 0.35, 0, 1)[:, None]
def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)
def hp(x, f): return sosfilt(butter(2, f, 'high', fs=SR, output='sos'), x)
sfx = np.zeros(N)
def put(x, tt, g=1.0):
    i = int(tt * SR); j = min(N, i + len(x));  sfx[i:j] += x[:j - i] * g
def slap(tt):
    x = np.arange(int(.22 * SR)) / SR; th = np.sin(2 * np.pi * np.cumsum(150 * np.exp(-x * 12) + 45) / SR) * np.exp(-x * 16)
    put(th + bp(rg.standard_normal(len(x)), 900, 5000) * np.exp(-x * 60) * .9, tt)
def flash(tt):
    x = np.arange(int(.5 * SR)) / SR; e = np.sin(np.pi * np.clip(x / .5, 0, 1)) ** 1.5
    put(bp(rg.standard_normal(len(x)), 600, 7000) * e * .6, tt)
    put(np.sin(2 * np.pi * np.cumsum(70 * np.exp(-x * 6) + 40) / SR) * np.exp(-x * 9) * .9, tt + .2)
def tear(tt):
    x = np.arange(int(.55 * SR)) / SR; c = (rg.random(len(x)) > .985).astype(float); c = np.convolve(c, np.exp(-np.arange(300) / 40), 'same')
    put(hp(rg.standard_normal(len(x)), 2500) * (.25 + .75 * np.clip(c * 4, 0, 1)) * np.sin(np.pi * x / .55) * .8, tt)
def staple(tt):
    x = np.arange(int(.05 * SR)) / SR
    put(hp(rg.standard_normal(len(x)), 4000) * np.exp(-x * 140) + np.sin(2 * np.pi * 1700 * x) * np.exp(-x * 110) * .6, tt, .9)
    y = np.arange(int(.2 * SR)) / SR; put(np.sin(2 * np.pi * np.cumsum(120 * np.exp(-y * 14) + 50) / SR) * np.exp(-y * 20), tt, .8)
slap(1.70); staple(3.05); slap(4.171); flash(5.45); tear(6.62); slap(9.06)
sfx = sfx / np.abs(sfx).max() * .35
mix = a * .85 + np.stack([sfx, sfx], 1)
mix = np.tanh(mix * 1.05) / np.tanh(1.05); mix /= np.abs(mix).max() / .93
sf.write('build/mix.wav', mix, SR); print('audio ok')
