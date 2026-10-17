"""Reference (guide) narration for the user's script: durations + word timings. User records own VO; this is only for timing."""
import json, re, numpy as np, soundfile as sf
from kokoro_onnx import Kokoro
VO = {
 'vo1': ("You are what you own. At least, that's the lie he's been sold.", ['own', 'lie']),
 'vo2': ("Every object is a decision he never made. And he pays for each one with his time.", ['decision', 'never', 'time']),
 'vo3': ("So he builds something violent, anonymous, and real.", ['violent', 'anonymous', 'real']),
 'vo4': ("But a name is a brand. And the rebellion soon has rules, uniforms, and a leader.", ['brand', 'rules', 'uniforms', 'leader']),
 'vo5': ("Fight Club isn't asking whether you're free. It's asking who sold you your freedom.", ['free', 'sold', 'freedom']),
}
k = Kokoro('/tmp/kokoro/kokoro.onnx', '/tmp/kokoro/voices.bin'); out = {}
for name, (text, keys) in VO.items():
    s, sr = k.create(text, voice='bm_george', speed=0.92, lang='en-gb')
    idx = np.where(np.abs(s) > .01)[0]; s = s[idx[0]:idx[-1] + 1]; sf.write(f'v2/audio/{name}.wav', s, sr); dur = len(s) / sr
    hop = int(sr * .01); env = np.array([np.sqrt(np.mean(s[j:j + hop] ** 2)) for j in range(0, len(s) - hop, hop)])
    quiet = env < env.max() * .04; sil = []; j = 0
    while j < len(quiet):
        if quiet[j]:
            e = j
            while e < len(quiet) and quiet[e]: e += 1
            if e - j >= 9 and j > 3 and e < len(quiet) - 3: sil.append((j * .01, e * .01))
            j = e
        else: j += 1
    toks = text.split(); bnd = [n for n, w in enumerate(toks[:-1]) if re.search(r'[.,]$', w)]
    sil = sorted(sorted(sil, key=lambda x: x[0] - x[1])[:len(bnd)])
    segs = []; prev = 0; cur = 0.0
    for n, b in enumerate(bnd):
        a0, a1 = sil[n] if n < len(sil) else (None, None); segs.append((prev, b, cur, a0)); prev = b + 1; cur = a1
    segs.append((prev, len(toks) - 1, cur, dur)); words = []
    if any(a is None or b is None for _, _, a, b in segs):
        W = [len(w) + 1.5 for w in toks]; tt = 0
        for w, wt in zip(toks, W): words.append((w, tt / sum(W) * dur, (tt + wt) / sum(W) * dur)); tt += wt
    else:
        for a, b, t0, t1 in segs:
            ws = toks[a:b + 1]; W = [len(w.strip('.,')) + 1.5 for w in ws]; tt = 0
            for w, wt in zip(ws, W): words.append((w, t0 + tt / sum(W) * (t1 - t0), t0 + (tt + wt) / sum(W) * (t1 - t0))); tt += wt
    out[name] = {'text': text, 'dur': dur, 'words': [{'w': w, 's': round(a, 3), 'e': round(b, 3), 'key': re.sub(r"[^a-z']", '', w.lower()) in keys} for w, a, b in words]}
    print(name, round(dur, 2), len(toks), 'words')
json.dump(out, open('v2/guide.json', 'w'), indent=1)
