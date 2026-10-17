import sys, subprocess, numpy as np
from multiprocessing import Pool
from v2 import *
from transitions import *
TOTAL = PLAN['total']; ORDER = PLAN['segs']
FN = {'paper_slap': paper_slap, 'copier_flash': copier_flash, 'torn_reveal': torn_reveal, 'staple_pop': staple_pop}
DUR = {'paper_slap': .40, 'copier_flash': .45, 'torn_reveal': .55, 'staple_pop': .40}
def windows():
    w = []
    for k, s in enumerate(ORDER):
        name = s['trans_in'][0]
        if k == 0: w.append(('paper_slap', 0.0, .40, None, s)); continue
        if name == 'hard': continue
        w.append((name, s['t0'] - .12, DUR[name], ORDER[k - 1], s))
    return w
WIN = windows()
def frame(i):
    t = i / 30
    for n, (name, t0, dur, a, b) in enumerate(WIN):
        if t0 <= t < t0 + dur + (.16 if name == 'paper_slap' else 0):
            A = seg_frame(a, min(t, a['t1'] - 1e-3)) if a is not None else None; B = seg_frame(b, max(t, b['t0']))
            f = FN[name](A, B, t, t0, dur)
            if name != 'paper_slap' or t < t0 + dur: f = film_damage(f, t, t0, dur, n + 1)
            return (np.clip(f, 0, 1) * 255).astype(np.uint8)
    s = seg_at(t); return (np.clip(seg_frame(s, t), 0, 1) * 255).astype(np.uint8)
if __name__ == '__main__':
    out = sys.argv[1]; n0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0; n1 = int(sys.argv[3]) if len(sys.argv) > 3 else int(TOTAL * 30)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', '30', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for j, f in enumerate(pool.imap(frame, range(n0, n1), chunksize=3)):
            ff.stdin.write(f.tobytes())
            if j % 100 == 0: print(j + n0, flush=True)
    ff.stdin.close(); ff.wait()
