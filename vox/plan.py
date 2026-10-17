"""Builds v2/plan.json: timeline of VO windows (breathers for the user's own voice) and key scenes (original audio)."""
import json
G = json.load(open('v2/guide.json'))
def win(name): d = G[name]['dur']; return round(.5 + d * 1.2 + .8, 2)
# piece: (src_start, src_dur, speed, zoom=(z0,z1,fx,fy))  ; freeze zoom handled in seg
SEGS = [
 dict(id='vo1', kind='vo', dur=win('vo1'), pieces=[(15.782, 1.13, 1, (1.0, 1.25, 640, 175)), (18.385, 1.53, 1, (1.05, 1.28, 590, 150)), (21.088, 1.40, 1, (1.3, 1.0, 590, 160))], freeze=(1.0, 1.4, 585, 175), trans_in=('hard',)),
 dict(id='k1', kind='key', pieces=[(25.759, 1.068, 1, (1.0, 1.12, 640, 278)), (26.827, 3.973, 1, (1.0, 1.0, 640, 278))], src=(25.759, 30.80), trans_in=('copier_flash',),
      zk=[(0, 1.0, 640, 278), (1.06, 1.12, 640, 278), (1.07, 1.0, 640, 278), (2.0, 1.0, 640, 278), (3.1, 1.55, 980, 240), (5.04, 1.65, 980, 240)],
      subs=[("I'd flip through catalogs and wonder.", 25.85, 27.62, []), ("What kind of dining set defines me as a person?", 27.74, 30.79, ['dining', 'set', 'person?'])]),
 dict(id='vo2', kind='vo', dur=win('vo2'), pieces=[(22.556, 1.2, 1, (1.0, 1.15, 560, 230)), (23.757, 1.94, .6, (1.0, 1.35, 700, 250))], freeze=(1.35, 1.7, 760, 260), trans_in=('paper_slap',)),
 dict(id='k2', kind='key', pieces=[(31.05, 3.9, 1, (1.0, 1.0, 640, 278))], src=(31.05, 34.95), trans_in=('staple_pop',),
      zk=[(0, 1.05, 420, 230), (.9, 1.4, 420, 230), (1.0, 1.0, 640, 278), (1.4, 1.0, 560, 200), (2.9, 1.3, 560, 170), (2.92, 1.0, 700, 170), (3.78, 1.35, 700, 170), (3.9, 1.35, 700, 170)],
      subs=[("This is your life and it's ending one minute at a time.", 31.13, 34.92, ['life', 'ending', 'minute'])]),
 dict(id='vo3', kind='vo', dur=win('vo3'), pieces=[(47.85, 2.6, .7, (1.6, 1.0, 520, 330))], freeze=(1.0, 1.0, 640, 300), trans_in=('torn_reveal',)),
 dict(id='k3', kind='key', pieces=[(52.40, 3.8, 1, (1.0, 1.0, 640, 278))], src=(52.40, 56.20), trans_in=('copier_flash',),
      zk=[(0, 1.0, 640, 278), (1.5, 1.0, 640, 278), (2.0, 1.2, 740, 200), (2.5, 1.2, 740, 200), (2.85, 1.6, 740, 170), (3.8, 1.75, 740, 170)],
      subs=[("We just gave it a name.", 52.47, 54.7, ['name']), ("Gentlemen! Welcome to Fight Club.", 54.9, 56.1, ['Fight', 'Club.'])]),
 dict(id='vo4', kind='vo', dur=win('vo4'), pieces=[(33.97, .86, .6, (1.0, 1.5, 700, 170)), (41.31, 1.2, .8, (1.0, 1.3, 660, 200)), (44.2, 1.33, .8, (1.0, 1.25, 400, 250))], freeze=(1.3, 1.6, 660, 190), trans_in=('paper_slap',)),
 dict(id='vo5', kind='vo', dur=win('vo5'), pieces=[(25.759, 1.0, .5, (1.9, 1.0, 793, 125))], freeze=(1.0, 1.0, 640, 278), trans_in=('staple_pop',)),
]
t = 0.0
for s in SEGS:
    if s['kind'] == 'key': s['dur'] = round(s['src'][1] - s['src'][0], 3)
    s['t0'] = round(t, 3); s['t1'] = round(t + s['dur'], 3); t = s['t1']
    # output length of pieces; remainder is freeze
    plen = sum(p[1] / p[2] for p in s['pieces']); s['piece_len'] = round(plen, 3); s['freeze_len'] = round(max(0, s['dur'] - plen), 3)
    if s['kind'] == 'vo': s['vo_t0'] = round(s['t0'] + .5, 3)   # VO starts here in the guide
json.dump({'total': round(t, 3), 'segs': SEGS}, open('v2/plan.json', 'w'), indent=1)
for s in SEGS: print(s['id'], s['kind'], s['t0'], s['t1'], 'pieces', s['piece_len'], 'freeze', s['freeze_len'])
print('TOTAL', round(t, 2))
