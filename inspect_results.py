from PIL import Image
import numpy as np

paths=['static/results/test123_roi.png','static/results/test123_result.png']
for p in paths:
    im=Image.open(p)
    arr=np.array(im)/255.0
    print(p, 'min',arr.min(), 'max', arr.max(), 'mean', arr.mean())
