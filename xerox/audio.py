"""Voice (Kokoro) + synthesised original music + synthesised foley SFX -> audio/mix.wav (44.1 kHz stereo)."""
import json, numpy as np, soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt
SR = 44100; DUR = 30.0; N = int(SR * DUR)
tl = json.load(open('audio/timeline.json')); rg = np.random.default_rng(4)
def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)
def hp(x, f): return sosfilt(butter(2, f, 'high', fs=SR, output='sos'), x)
def lp(x, f): return sosfilt(butter(2, f, 'low', fs=SR, output='sos'), x)
def put(buf, x, t, g=1.0):
    i = int(t * SR); j = min(N, i + len(x))
    if i < N: buf[i:j] += x[:j - i] * g
# ---- voice
voice = np.zeros(N)
for i, l in enumerate(tl['lines']):
    a, sr = sf.read(f'audio/line{i+1}.wav'); a = resample_poly(a, SR, sr)
    put(voice, hp(a, 70), l['start'])
voice = voice / np.abs(voice).max() * 0.85
# ---- music: Am - F - Dm - E, detuned soft pad + low copier pulse + tick
t = np.arange(N) / SR
chords = [[110, 164.8, 220, 261.6, 329.6], [87.3, 130.8, 174.6, 261.6, 349.2], [73.4, 110, 146.8, 220, 293.7], [82.4, 123.5, 164.8, 207.7, 329.6]]
pad = np.zeros(N)
for ci, ch in enumerate(chords * 2):
    t0 = ci * 3.75; seg = (t >= t0) & (t < t0 + 3.75 + 1.2)
    env = np.clip((t - t0) / 1.2, 0, 1) * np.clip(1 - (t - t0 - 3.75) / 1.2, 0, 1) * seg
    for f in ch:
        for d in (-0.35, 0.0, 0.4):
            ph = rg.uniform(0, 6.28); w = np.sin(2 * np.pi * (f + d) * t + ph) + .35 * np.sin(2 * np.pi * 2 * (f + d) * t + ph * 1.3) + .15 * np.sin(2 * np.pi * 3 * (f + d) * t)
            pad += w * env * 0.05
pad = lp(pad, 1400) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.11 * t))
pulse = np.zeros(N); beat = 60 / 72
for k in range(int(DUR / beat)):
    x = np.arange(int(.35 * SR)) / SR; f = 62 * np.exp(-x * 9) + 38
    put(pulse, np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 11), k * beat, 0.5 if k % 4 == 0 else 0.28)
tick = np.zeros(N)
for k in range(int(DUR / (beat / 2))):
    x = hp(rg.standard_normal(int(.02 * SR)), 5000) * np.exp(-np.arange(int(.02 * SR)) / SR * 220)
    put(tick, x, k * beat / 2 + (0.012 if k % 2 else 0), 0.05 if k % 2 == 0 else 0.03)
music = pad * 1.0 + pulse * 0.9 + tick * 0.8
l5 = tl['lines'][4]['start']
g = np.ones(N)
drop0, drop1 = l5 - 1.0, l5
g *= np.clip((drop0 - t) / 0.12 + 1, 0, 1) + (t >= drop1) * np.clip((t - drop1) / 0.9, 0, 1) * 0.0  # fade out into the drop
g = np.where(t < drop0, 1.0, np.where(t < drop1, 0.0, np.clip((t - drop1) / 0.9, 0, 1) * 0.8))
g *= np.clip((DUR - t) / 1.4, 0, 1) * np.clip(t / 0.5, 0, 1)
fade_out = np.clip((drop0 - t) / 0.12, 0, 1); g = np.where(t < drop0, np.minimum(1, fade_out + (t < drop0 - .12)), g)
music = music * g; music = music / np.abs(music).max() * 0.20
# ---- sfx
sfx = np.zeros(N)
def slap(tt, gain=1.0):
    x = np.arange(int(.22 * SR)) / SR; f = 150 * np.exp(-x * 12) + 45
    th = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 16)
    nz = bp(rg.standard_normal(len(x)), 900, 5000) * np.exp(-x * 60) * 0.9
    put(sfx, th + nz, tt, gain)
def flash(tt):
    x = np.arange(int(.5 * SR)) / SR; e = np.sin(np.pi * np.clip(x / .5, 0, 1)) ** 1.5
    nz = bp(rg.standard_normal(len(x)), 600, 7000) * e * 0.6
    put(sfx, nz, tt); 
    th = np.sin(2 * np.pi * np.cumsum(70 * np.exp(-x * 6) + 40) / SR) * np.exp(-x * 9); put(sfx, th * .9, tt + .2)
def tear(tt):
    x = np.arange(int(.55 * SR)) / SR
    crk = (rg.random(len(x)) > 0.985).astype(float); crk = np.convolve(crk, np.exp(-np.arange(300) / 40), 'same')
    nz = hp(rg.standard_normal(len(x)), 2500) * (0.25 + 0.75 * np.clip(crk * 4, 0, 1)) * np.sin(np.pi * x / .55) * 0.8
    put(sfx, nz, tt)
def staple(tt):
    x = np.arange(int(.05 * SR)) / SR
    c = hp(rg.standard_normal(len(x)), 4000) * np.exp(-x * 140) + np.sin(2 * np.pi * 1700 * x) * np.exp(-x * 110) * .6
    put(sfx, c, tt, 0.9); y = np.arange(int(.2 * SR)) / SR
    put(sfx, np.sin(2 * np.pi * np.cumsum(120 * np.exp(-y * 14) + 50) / SR) * np.exp(-y * 20), tt, 0.8)
slap(0.40, .8); slap(5.02, 1.0); flash(9.85); tear(13.9); staple(20.02)
sfx = sfx / max(1e-6, np.abs(sfx).max()) * 0.5
mix = voice + music + sfx
# gentle limiter
mix = np.tanh(mix * 1.1) / np.tanh(1.1); mix = mix / np.abs(mix).max() * 0.92
sf.write('audio/mix.wav', np.stack([mix, mix], 1), SR)
sf.write('audio/music_only.wav', music, SR)
print('rms voice', np.sqrt((voice**2).mean()), 'music', np.sqrt((music**2).mean()), 'drop', drop0, drop1)
