import subprocess, shutil, os, json
from v2lib import *
for s in PLAN['segs']:
    last = None
    for k, (st, du, sp, z) in enumerate(s['pieces']):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(st), '-t', str(du), '-i', 'src/trailer.mp4', '-vf', f'crop=1280:556:0:90,setpts=PTS/{sp},fps=30', '-q:v', '3', f"v2/o/{s['id']}_{k}_%04d.jpg"], check=True)
        fs = sorted(f for f in os.listdir('v2/o') if f.startswith(f"{s['id']}_{k}_")); last = fs[-1]
        print(s['id'], k, len(fs), 'frames', round(len(fs) / 30, 2), 's (expect', round(du / sp, 2), ')')
    shutil.copy('v2/o/' + last, f"v2/o/{s['id']}_fz.jpg")
# audio for key scenes (original)
for s in PLAN['segs']:
    if s['kind'] == 'key':
        a, b = s['src']
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(a), '-t', str(b - a), '-i', 'src/trailer.mp4', '-vn', '-c:a', 'pcm_s16le', '-ar', '44100', f"v2/audio/{s['id']}_orig.wav"], check=True)
