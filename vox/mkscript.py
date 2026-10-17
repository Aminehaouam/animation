import json
from v2lib import *
G = json.load(open('v2/guide.json'))
def tc(x): return f"{int(x // 60)}:{x % 60:05.2f}"
WHY = {
 'vo1': "Zoom IN slowly on the direct-eye-contact shot, then pull back from the desk: first he looks at us, then he is just one desk among many. Key word OWN gets the ransom-note cut-out; 'the lie' gets the stamp.",
 'k1': "The film speaks, so it fills the screen. Zoom stays calm on the catalogue, then PUNCHES IN on the on-screen 'closeout' labels exactly on 'dining set': the joke is that he is shopping for an identity.",
 'vo2': "Slow-motion + freeze, then a zoom INTO the room: price tags land on objects as you say 'decision'. 'Paid in time' sets up the next line.",
 'k2': "Push in on the TV, ease into his face on the sofa, hold on the row of identical men: 'your life, ending one minute at a time'.",
 'vo3': "Zoom OUT from one face to the whole crowd: he disappears into it, which is what 'anonymous' means. Three stencil words land on your three adjectives.",
 'k3': "Hold wide, push on the speaker, then a fast SNAP zoom on 'Welcome to Fight Club': naming it is the moment it becomes a brand.",
 'vo4': "Freeze on the uniformed row and zoom into the line (uniforms), then the leader. BRAND, the rules list and LEADER stamp each land on their word.",
 'vo5': "Start tight on the $695 price and zoom OUT: who sold you this? The closing line builds as cut-out words; the freedom price tag stamps SOLD at the end.",
}
L = ["# Script, timing and shot list (zoom reasons)", "", "Record each VO line inside its window. Every window has about 0.5 s of breath before and about 0.8 s after the guide pace, so your own pace has room. Subtitles for VO windows follow the guide timing; edit `v2/guide.json` or retime if your pace is different.", ""]
for s in PLAN['segs']:
    if s['kind'] == 'vo':
        g = G[s['id']]
        L += [f"## {s['id'].upper()}  VO window {tc(s['t0'])} to {tc(s['t1'])}", f"**Say (start about {tc(s['vo_t0'])}, guide length {g['dur']:.1f}s):** \"{g['text']}\"", f"*Why this picture/zoom:* {WHY[s['id']]}", ""]
    else:
        L += [f"## {s['id'].upper()}  ORIGINAL AUDIO {tc(s['t0'])} to {tc(s['t1'])} (trailer source {s['src'][0]:.2f} to {s['src'][1]:.2f})", "Dialogue: " + " / ".join(f'"{x[0]}"' for x in s['subs']), f"*Why this picture/zoom:* {WHY[s['id']]}", ""]
L.append(f"Total length: {tc(PLAN['total'])}")
open('SCRIPT_AND_SHOTLIST.md', 'w').write('\n'.join(L)); print('ok')
