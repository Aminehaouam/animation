import json, math, os
PLAN = json.load(open('v2/plan.json')); SEGS = {s['id']: s for s in PLAN['segs']}; FPS = 30
def seg_at(t):
    for s in PLAN['segs']:
        if t < s['t1']: return s
    return PLAN['segs'][-1]
def locate(s, tl):
    """local time -> (piece_index or 'fz', frame number n (1-based), piece zoom spec, progress 0..1 within zoom span)"""
    acc = 0.0
    for k, p in enumerate(s['pieces']):
        L = p[1] / p[2]
        if tl < acc + L - 1e-9:
            n = min(int(round(L * FPS)), int((tl - acc) * FPS) + 1); return k, max(1, n), p[3], (tl - acc) / L
        acc += L
    fz = s.get('freeze') or s['pieces'][-1][3]; fl = max(.001, s['freeze_len']) if s['freeze_len'] else 1
    return 'fz', 1, fz, min(1.0, (tl - acc) / fl)
def path(s, k, n):
    return f"v2/o/{s['id']}_fz.jpg" if k == 'fz' else f"v2/o/{s['id']}_{k}_{n:04d}.jpg"
def mpath(s, k, n):
    if k != 'fz' and n % 2 == 0: n -= 1
    return f"v2/om/{s['id']}_fz.png" if k == 'fz' else f"v2/om/{s['id']}_{k}_{max(1,n):04d}.png"
