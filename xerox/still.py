import sys, time, numpy as np, cv2
from episode import *
k=int(sys.argv[1]); ts=[float(x) for x in sys.argv[2:]]
for t in ts:
    t0=time.time(); f=scene_frame(k,t); print(k,t,round(time.time()-t0,2),'s')
    cv2.imwrite(f'build/s{k+1}_{t:.2f}.png', cv2.cvtColor((f*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
