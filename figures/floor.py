"""Re-render the floor below the front plates as one continuous soft shadow, removing
the vertical steps left where plate columns were shifted. Shadow depth vs. distance
below the plate is measured from clean columns; it fades out over a penumbra at the
plate ends."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage

def plate_span(g,y,lo,hi):
    e=np.where(np.abs(np.diff(ndimage.uniform_filter1d(g[y],3)))>25)[0]
    e=e[(e>lo)&(e<hi)]; return e[0]+1,e[-1]

def bottom(g,x):
    c=ndimage.uniform_filter1d(g[:,x-3:x+4].mean(1),3)
    return 650+int(np.argmax(c[651:780]-c[650:779]))     # dark plate -> light floor

def render(img,frames=((60,1000),(1000,1990)),probe=((330,550,770),(1240,1460,1680)),pen=16):
    a=img.astype(np.float32); g=a.mean(2); H,W=g.shape
    bgx=[x for x in list(range(4,40))+list(range(W-40,W-4))]
    out=a.copy(); X=np.arange(W)
    spans=[plate_span(g,700,*fr) for fr in frames]
    bots=[int(np.median([bottom(g,x) for x in pr])) for pr in probe]
    y0=min(bots)+2
    bg=a[:,bgx].mean(1)                                   # (H,3) floor colour per row
    ratio=np.ones((len(frames),H,3),np.float32)
    for i,pr in enumerate(probe):
        col=np.mean([a[:,x-2:x+3].mean(1) for x in pr],0)
        r=np.clip(col/np.maximum(bg,1),0,1.2)
        # shift so the profile is measured from this frame's plate bottom
        ratio[i,bots[i]+2:]=r[bots[i]+2:]
    val=np.repeat(bg[:,None,:],W,1)
    for i,(L,R) in enumerate(spans):
        w=0.5*(1+np.tanh((X-L)/ (pen/2)))*0.5*(1+np.tanh((R-X)/(pen/2)))
        val=val*(1-w[None,:,None]*(1-ratio[i][:,None,:]))
    # replace from just below each plate's bottom edge (outside plates from y0)
    start=np.full(W,y0)
    for (L,R),b in zip(spans,bots): start[L:R+1]=b+2
    Y=np.arange(H)[:,None]
    m=np.clip((Y-start[None,:])/3,0,1)[...,None]
    out=a*(1-m)+val*m
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    Image.fromarray(render(np.array(Image.open(sys.argv[1]).convert('RGB')))).save(sys.argv[2])
