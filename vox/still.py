import sys, cv2, numpy as np
from vox_ep import *
k=int(sys.argv[1]); out=[]
for t in map(float,sys.argv[2:]):
    f=shot_frame(k,t); cv2.imwrite(f'build/v{k}_{t:.2f}.png',cv2.cvtColor((f*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
