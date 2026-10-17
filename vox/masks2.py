import os, numpy as np
from PIL import Image
from rembg import remove, new_session
from v2lib import *
s_ = new_session('u2net_human_seg'); todo = []
for s in PLAN['segs']:
    if s['kind'] != 'vo': continue
    for k, p in enumerate(s['pieces']):
        L = int(round(p[1] / p[2] * FPS))
        for n in range(1, L + 1, 2): todo.append((path(s, k, n), mpath(s, k, n)))
    todo.append((path(s, 'fz', 1), mpath(s, 'fz', 1)))
print(len(todo), 'masks', flush=True)
for i, (a, b) in enumerate(todo):
    if os.path.exists(b): continue
    m = remove(Image.open(a).convert('RGB'), session=s_, only_mask=True); Image.fromarray(np.asarray(m)).save(b)
    if i % 30 == 0: print(i, flush=True)
