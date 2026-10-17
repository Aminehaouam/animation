import json, numpy as np, soundfile as sf, sys
from kokoro_onnx import Kokoro
LINES=["You are what you own. At least, that's the lie he's been sold.",
"He lives in a catalogue. Every object is a decision he never made.",
"So he builds something violent, anonymous, and real.",
"But the rebellion becomes another brand, with rules, uniforms and a leader.",
"Fight Club isn't asking whether you're free. It's asking who sold you your freedom."]
voice=sys.argv[1]; speed=float(sys.argv[2])
k=Kokoro('/tmp/kokoro/kokoro.onnx','/tmp/kokoro/voices.bin')
out=[]
for i,t in enumerate(LINES):
    s,sr=k.create(t,voice=voice,speed=speed,lang='en-gb' if voice.startswith('b') else 'en-us')
    a=np.abs(s); idx=np.where(a>0.01)[0]; s=s[idx[0]:idx[-1]+1]
    sf.write(f'audio/line{i+1}.wav',s,sr); out.append(len(s)/sr); print(i+1,round(len(s)/sr,2))
print(sum(out))
