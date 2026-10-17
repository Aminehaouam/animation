import sys, subprocess, math, numpy as np, cv2
from multiprocessing import Pool
from episode import scene_frame, CUTS
from transitions import *
FPS, DUR = 30, 30.0
TR = [('paper_slap', 0.0, 0.40, None, 0), ('paper_slap', 4.62, 0.40, 0, 1), ('copier_flash', 9.85, 0.45, 1, 2),
      ('torn_reveal', 13.9, 0.55, 2, 3), ('staple_pop', 19.8, 0.40, 3, 4)]
def scene_at(t):
    for k in range(5):
        if t < CUTS[k + 1]: return k
    return 4
def frame(i):
    t = i / FPS
    for n, (name, t0, dur, a, b) in enumerate(TR):
        if t0 <= t < t0 + dur + (0.16 if name == 'paper_slap' else 0.0):
            A = scene_frame(a, t) if a is not None else None; B = scene_frame(b, t)
            f = {'paper_slap': paper_slap, 'copier_flash': copier_flash, 'torn_reveal': torn_reveal, 'staple_pop': staple_pop}[name](A, B, t, t0, dur)
            if name != 'paper_slap' or t < t0 + dur: f = film_damage(f, t, t0, dur, n + 1)
            return (np.clip(f, 0, 1) * 255).astype(np.uint8)
    # outside windows: after cut centre the scene is the one whose window ended
    k = 0
    for n, (name, t0, dur, a, b) in enumerate(TR):
        if t >= t0 + dur: k = b
    return (np.clip(scene_frame(k, t), 0, 1) * 255).astype(np.uint8)
if __name__ == '__main__':
    out = sys.argv[1]; n0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0; n1 = int(sys.argv[3]) if len(sys.argv) > 3 else int(FPS * DUR)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for j, f in enumerate(pool.imap(frame, range(n0, n1), chunksize=3)):
            ff.stdin.write(f.tobytes())
            if j % 60 == 0: print(j + n0, flush=True)
    ff.stdin.close(); ff.wait()
