"""Rebuild the front plates so all eight bays carry identical, regularly placed
hardware (threaded rod + nuts, two slotted cap nuts, lower cap nut), as on the real
rig. Each bay is replaced by a clean bay module taken from the right frame (bays 2
and 3, alternated so neighbours are not clones). The plate's top edge band and the
plate end edges are kept from the image so everything above the plate still joins."""
import sys, numpy as np
from PIL import Image

RODS=[[216,440,663,884],[1128,1349,1570,1791]]   # threaded-rod x per frame
TOP=[477,479]                                     # front-plate top edge y per frame
SRC=[(1,1),(1,2)]                                 # module sources: (frame, bay)
HALF=116; FX=10; KEEP_TOP=10; FY=8; BOTTOM=760; FB=14
PBOT=[722,725]                                    # front-plate bottom edge y per frame

def ramp(n,a,b):
    """0..1 ramp over [a,b) along an axis of length n (1 after b)."""
    return np.clip((np.arange(n)-a)/max(b-a,1),0,1)

def plate_edges(img,frame,top):
    g=img.astype(np.float32).mean(2)[top+200]
    e=np.where(np.abs(np.diff(g))>25)[0]
    lo=RODS[frame][0]-125; hi=RODS[frame][-1]+125
    e=e[(e>lo)&(e<hi)]
    return e[0],e[-1]

def build(img):
    out=img.astype(np.float32).copy(); H,W=img.shape[:2]
    mods=[]
    for f,b in SRC:
        x=RODS[f][b]; t=TOP[f]
        mods.append((img[t-20:BOTTOM+20,x-HALF:x+HALF].astype(np.float32),t))
    k=0
    for f in range(2):
        L,R=plate_edges(img,f,TOP[f])
        for b,x in enumerate(RODS[f]):
            m,ts=mods[k%2]; k+=1
            t=TOP[f]; y0=t-20; y1=y0+m.shape[0]
            x0=x-HALF; x1=x+HALF
            # horizontal weight: feather at module sides, but stop 6 px inside plate ends
            wx=np.minimum(ramp(2*HALF,0,FX),1-ramp(2*HALF,2*HALF-FX,2*HALF))
            cols=np.arange(x0,x1)
            wx=np.where(cols<L+6,0,wx)*np.where(cols>R-6,0,1)
            wx=np.minimum(wx,np.clip((cols-(L+6))/6,0,1)); wx=np.minimum(wx,np.clip(((R-6)-cols)/6,0,1))
            if b==0: wx=np.where(cols<x,np.clip((cols-(L+6))/6,0,1),wx)
            if b==3: wx=np.where(cols>x,np.clip(((R-6)-cols)/6,0,1),wx)
            rows=np.arange(y0,y1)
            wy=np.clip((rows-(t+KEEP_TOP))/FY,0,1)*np.clip(((PBOT[f]-10)-rows)/FB,0,1)   # keep this plate's own bottom edge
            w=(wy[:,None]*wx[None,:])[...,None]
            # same vertical alignment: module top edge onto this plate's top edge
            src=m[(ts-20)-(ts-20):,:]   # module already starts 20 rows above its top
            out[y0:y1,x0:x1]=out[y0:y1,x0:x1]*(1-w)+src*w
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    Image.fromarray(build(np.array(Image.open(sys.argv[1]).convert('RGB')))).save(sys.argv[2])
