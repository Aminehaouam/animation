import sys, cv2, numpy as np
from render2 import *
for a in sys.argv[1:]:
    t = float(a); f = frame(int(round(t * 30))); cv2.imwrite(f'build/w_{t:05.2f}.png', cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
