"""Per-frame subject mattes (every 2nd frame -> 15 fps, held = stop-motion cutout feel)."""
import sys, numpy as np
from PIL import Image
from rembg import remove, new_session
s = new_session('u2net_human_seg')
for i in range(1, 300, 2):
    im = Image.open(f'build/fr/f{i:04d}.png').convert('RGB')
    a = np.asarray(remove(im, session=s, only_mask=True))
    Image.fromarray(a).save(f'build/cut/m{i:04d}.png')
    if i % 40 == 1: print(i, flush=True)
