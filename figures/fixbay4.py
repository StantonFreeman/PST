"""Replace the odd spacer-bay (bay 4) bar/nut below the front-plate edge with a copy
of the specimen-bay threaded rod + nuts from bay 3 of the same frame."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage
# (bay-3 rod x, bay-4 rod x) per frame, and the vertical span to replace
FRAMES=[(720,905),(1528,1760)]
Y0,Y1,HALF,FEATHER=490,600,40,8
def fix(img):
    a=img.astype(np.float32); out=a.copy()
    h,w=Y1-Y0,2*HALF
    m=np.zeros((h,w),np.float32); m[FEATHER:-FEATHER,FEATHER:-FEATHER]=1
    m=ndimage.gaussian_filter(m,FEATHER/2)[...,None]
    for xs,xd in FRAMES:
        P=a[Y0:Y1,xs-HALF:xs+HALF]; T=a[Y0:Y1,xd-HALF:xd+HALF]
        # match plate brightness using the border ring
        ring=np.ones((h,w),bool); ring[6:-6,6:-6]=False
        g=T[ring].mean(0)/np.maximum(P[ring].mean(0),1)
        out[Y0:Y1,xd-HALF:xd+HALF]=T*(1-m)+P*g*m
    return np.clip(out,0,255).astype(np.uint8)
if __name__=='__main__':
    Image.fromarray(fix(np.array(Image.open(sys.argv[1]).convert('RGB')))).save(sys.argv[2])
