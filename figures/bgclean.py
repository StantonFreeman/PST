"""Smooth the plain studio background in the regions where column shifts left faint
steps (gap between the frames, beside the outer plate ends). Only background pixels
(light, unsaturated) are replaced, by a normalized-convolution blur of the background,
so the edges of the frames and rods are untouched."""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage

REGIONS=[(940,0,1085,735),(0,430,125,735),(1885,430,2012,735)]

def clean(img,sig=14):
    a=img.astype(np.float32); out=a.copy()
    g=a.mean(2); sat=a.max(2)-a.min(2)
    bg=(g>175)&(sat<14)
    bg=ndimage.binary_erosion(bg,iterations=2)          # stay clear of object edges
    for x0,y0,x1,y1 in REGIONS:
        P=24; X0,Y0,X1,Y1=max(x0-P,0),max(y0-P,0),min(x1+P,a.shape[1]),min(y1+P,a.shape[0])
        m=bg[Y0:Y1,X0:X1].astype(np.float32)
        num=cv2.GaussianBlur(a[Y0:Y1,X0:X1]*m[...,None],(0,0),sig)
        den=cv2.GaussianBlur(m,(0,0),sig)[...,None]
        sm=num/np.maximum(den,1e-3)
        sel=bg[y0:y1,x0:x1]
        # keep a little of the original grain so it is not plastic-smooth
        orig=a[y0:y1,x0:x1]; grain=orig-cv2.GaussianBlur(orig,(0,0),1.2)
        rep=sm[y0-Y0:y1-Y0,x0-X0:x1-X0]+0.6*grain
        w=cv2.GaussianBlur(sel.astype(np.float32),(0,0),1.0)[...,None]
        out[y0:y1,x0:x1]=orig*(1-w)+rep*w
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    Image.fromarray(clean(np.array(Image.open(sys.argv[1]).convert('RGB')))).save(sys.argv[2])
