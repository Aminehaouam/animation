import sys, subprocess, numpy as np
from multiprocessing import Pool
from vox_ep import shot_frame, CUTS, NF
from transitions import *
FPS = 30
# (type, window start, dur, from-shot, to-shot)
TR = [('paper_slap', 1.30, .40, 0, 1), ('staple_pop', 2.85, .40, 1, 2), ('hard', 4.171, 0, 2, 3),
      ('copier_flash', 5.45, .45, 3, 4), ('torn_reveal', 6.62, .55, 4, 5), ('paper_slap', 8.66, .40, 5, 6)]
def frame(i):
    t = i / FPS
    for n, (name, t0, dur, a, b) in enumerate(TR):
        if name == 'hard': continue
        if t0 <= t < t0 + dur + (.16 if name == 'paper_slap' else 0):
            A = shot_frame(a, t); B = shot_frame(b, t)
            f = {'paper_slap': paper_slap, 'copier_flash': copier_flash, 'torn_reveal': torn_reveal, 'staple_pop': staple_pop}[name](A, B, t, t0, dur)
            if name != 'paper_slap' or t < t0 + dur: f = film_damage(f, t, t0, dur, n + 1)
            return (np.clip(f, 0, 1) * 255).astype(np.uint8)
    k = 0
    for n, (name, t0, dur, a, b) in enumerate(TR):
        if t >= t0 + dur and not (name == 'hard' and t < t0): k = b
    if t >= 4.171 and t < 4.171 + .1:
        # page shake on the hard cut
        f = shot_frame(3, t); sh = [10, -6, 3][int((t - 4.171) * 30)] if int((t - 4.171) * 30) < 3 else 0
        f = np.roll(f, sh, axis=0); return (np.clip(f, 0, 1) * 255).astype(np.uint8)
    return (np.clip(shot_frame(k, t), 0, 1) * 255).astype(np.uint8)
if __name__ == '__main__':
    out = sys.argv[1]; n0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0; n1 = int(sys.argv[3]) if len(sys.argv) > 3 else NF
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', '30', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for j, f in enumerate(pool.imap(frame, range(n0, n1), chunksize=3)):
            ff.stdin.write(f.tobytes())
            if j % 50 == 0: print(j + n0, flush=True)
    ff.stdin.close(); ff.wait()
