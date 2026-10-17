"""Narration (Kokoro, local) -> per-line wavs -> word-level timeline.json"""
import json, re, sys, numpy as np, soundfile as sf
from kokoro_onnx import Kokoro
LINES=["You are what you own. At least, that's the lie he's been sold.",
"He lives in a catalogue. Every object is a decision he never made.",
"So he builds something violent, anonymous, and real.",
"But the rebellion becomes another brand, with rules, uniforms and a leader.",
"Fight Club isn't asking whether you're free. It's asking who sold you your freedom."]
KEY={0:['own','lie'],1:['catalogue','decision','never'],2:['violent','anonymous','real'],
     3:['brand','rules','uniforms','leader'],4:['free','sold','freedom']}
VOICE,SPEED='bm_george',0.93
START0=0.55; GAP=0.62; PRE_LAST_GAP=1.35; TOTAL=30.0
k=Kokoro('/tmp/kokoro/kokoro.onnx','/tmp/kokoro/voices.bin')
t=START0; out=[]
for i,text in enumerate(LINES):
    s,sr=k.create(text,voice=VOICE,speed=SPEED,lang='en-gb')
    idx=np.where(np.abs(s)>0.01)[0]; s=s[idx[0]:idx[-1]+1]
    sf.write(f'audio/line{i+1}.wav',s,sr); dur=len(s)/sr
    # silences inside the line
    hop=int(sr*0.01); env=np.array([np.sqrt(np.mean(s[j:j+hop]**2)) for j in range(0,len(s)-hop,hop)])
    quiet=env<env.max()*0.04; sil=[]; j=0
    while j<len(quiet):
        if quiet[j]:
            e=j
            while e<len(quiet) and quiet[e]: e+=1
            if e-j>=9 and j>3 and e<len(quiet)-3: sil.append((j*0.01,e*0.01))
            j=e
        else: j+=1
    toks=text.split(); bnd=[n for n,w in enumerate(toks[:-1]) if re.search(r'[.,]$',w)]
    sil=sorted(sorted(sil,key=lambda x:x[0]-x[1])[:len(bnd)]); sil.sort()
    segs=[]; prev=0; cur=0.0
    for n,b in enumerate(bnd):
        s0,s1=sil[n] if n<len(sil) else (None,None)
        end_t=s0 if s0 is not None else None
        segs.append((prev,b,cur,end_t)); prev=b+1; cur=s1 if s1 is not None else None
    segs.append((prev,len(toks)-1,cur,dur))
    # fallback if detection failed: proportional over whole line
    words=[]
    if any(a is None or b is None for _,_,a,b in segs):
        W=[len(w)+1.5 for w in toks]; tt=0
        for w,wt in zip(toks,W): words.append((w,tt/sum(W)*dur,(tt+wt)/sum(W)*dur)); tt+=wt
    else:
        for a,b,t0,t1 in segs:
            ws=toks[a:b+1]; W=[len(w.strip('.,'))+1.5 for w in ws]; tt=0
            for w,wt in zip(ws,W): words.append((w,t0+tt/sum(W)*(t1-t0),t0+(tt+wt)/sum(W)*(t1-t0))); tt+=wt
    # snap internal word boundaries to nearest energy valley (+-90 ms) so captions follow the actual speech
    env_s=np.convolve(env,np.ones(3)/3,'same'); loc=[j*0.01 for j in range(2,len(env_s)-2) if env_s[j]<=env_s[j-1] and env_s[j]<=env_s[j+1] and env_s[j]<env_s.max()*0.35]
    def snap(b):
        c=[x for x in loc if abs(x-b)<=0.09]; return min(c,key=lambda x:abs(x-b)) if c else b
    nw=[]; 
    for n,(w,a,b) in enumerate(words):
        nw.append([w,a,b])
    for n in range(len(nw)-1):
        if nw[n][2]==nw[n+1][1]:   # contiguous inside a segment
            sb=snap(nw[n][2]); sb=min(max(sb,nw[n][1]+0.06),nw[n+1][2]-0.06); nw[n][2]=nw[n+1][1]=sb
    words=[tuple(x) for x in nw]
    out.append({'start':t,'dur':dur,'text':text,'sil':sil,'words':[{'w':w,'s':round(t+a,3),'e':round(t+b,3),
        'key':re.sub(r"[^a-z']",'',w.lower()) in KEY[i]} for w,a,b in words]})
    t+=dur+(PRE_LAST_GAP if i==3 else GAP)
json.dump({'total':TOTAL,'voice':VOICE,'speed':SPEED,'lines':out},open('audio/timeline.json','w'),indent=1)
for l in out: print(round(l['start'],2),round(l['start']+l['dur'],2),l['sil'])
