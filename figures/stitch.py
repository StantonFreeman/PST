"""Replace the bottom of the figure (front plates and everything below the plate-top
edge) with the corresponding part of a second rendering (new7.webp). Each frame is
registered separately (ECC, affine) on the front-plate region, then blended in at the
plate-top bevel."""
import sys, numpy as np, cv2
from PIL import Image

def register(ours,new,x0,x1,y0,y1):
    g1=cv2.GaussianBlur(cv2.cvtColor(ours,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1.5)
    g2=cv2.GaussianBlur(cv2.cvtColor(new,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1.5)
    m=np.zeros(g1.shape,np.uint8); m[y0:y1,x0:x1]=1
    W=np.eye(2,3,dtype=np.float32)
    crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,500,1e-6)
    for s in (0.25,0.5,1.0):                       # coarse to fine
        a=cv2.resize(g1,None,fx=s,fy=s); b=cv2.resize(g2,None,fx=s,fy=s)
        mm=cv2.resize(m,None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
        Ws=W.copy(); Ws[:,2]*=s
        cc,Ws=cv2.findTransformECC(a,b,Ws,cv2.MOTION_AFFINE,crit,mm,5)
        W=Ws.copy(); W[:,2]/=s
    return W,cc

SEAM=12      # seam this many rows above the plate-top bevel (new image supplies the rod tops)
def stitch(ours,new):
    H,Wd=ours.shape[:2]
    new=cv2.copyMakeBorder(new,0,max(H-new.shape[0],0),0,max(Wd-new.shape[1],0),cv2.BORDER_REPLICATE)[:H,:Wd]
    out=ours.astype(np.float32)
    frames=[(100,995,'L'),(1008,1910,'R')]
    go=ours.astype(np.float32).mean(2)
    def top_edges(g,xs):
        """Row of the plate's top edge (dark -> bright step) for each column xs."""
        sub=g[455:505][:,xs]; d=np.diff(cv2.GaussianBlur(sub,(1,3),0),axis=0)
        return 455+1+np.argmax(d,axis=0)
    for x0,x1,_ in frames:
        Wm,cc=register(ours,new,x0,x1,490,725)
        warp=lambda M: cv2.warpAffine(new,M,(Wd,H),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        warped=warp(Wm)
        # snap vertically so the two plates' top edges coincide (ECC fits the plate face;
        # the edge itself sits a few px lower in the second rendering)
        xs=np.arange(x0+30,x1-30,3)
        eo=top_edges(go,xs); en=top_edges(warped.mean(2),xs)
        dy=float(np.median(en-eo))
        M2=Wm.copy(); M2[1,2]+=dy                      # inverse map: sample dy rows lower
        warped=warp(M2)
        print('frame',x0,x1,'ecc %.3f'%cc,'edge shift %.1f px'%dy)
        # per-column seam just above our plate's top edge (smoothed), feathered over 5 rows
        xa=np.arange(Wd); xs_c=np.clip(xa,x0+30,x1-31)
        e=top_edges(go,np.arange(x0+30,x1-30)).astype(np.float32)
        e=np.minimum.accumulate(np.r_[e]) if False else e
        e=cv2.medianBlur(e.astype(np.uint8)[None,:],15)[0].astype(np.float32)
        seam=np.interp(xa,np.arange(x0+30,x1-30),e)-3
        Y=np.arange(H)[:,None]
        wy=np.clip((Y-(seam[None,:]-5))/5,0,1)
        lo=0 if x0<500 else 1001; hi=1001 if x0<500 else Wd
        wx=((np.arange(Wd)>=lo)&(np.arange(Wd)<hi)).astype(np.float32)[None,:]
        w=(wy*wx)[...,None]
        out=out*(1-w)+warped*w
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    ours=np.array(Image.open(sys.argv[1]).convert('RGB'))
    new=np.array(Image.open(sys.argv[2]).convert('RGB'))
    Image.fromarray(stitch(ours,new)).save(sys.argv[3])
