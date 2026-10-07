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

END_TOP=440
END_ZONES={}
CLEAN_BOXES=[(111,141,472,481),(1020,1066,469,481)]
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
        # per-column seam a few rows above the HIGHER of the two plate edges, so no part
        # of either plate's top edge is left above it
        from scipy.ndimage import median_filter, minimum_filter, gaussian_filter1d
        xs_p=np.arange(x0+30,x1-30)
        e=np.minimum(median_filter(top_edges(go,xs_p).astype(np.float32),9),
                     median_filter(top_edges(warped.mean(2),xs_p).astype(np.float32),9))
        e=gaussian_filter1d(minimum_filter(e,9),3)
        xa=np.arange(Wd); seam=np.interp(xa,xs_p,e)-4
        # plate-end zones (outer support rod + background above the plate): take the new
        # image from higher up, so artefacts just above our plate ends are replaced too
        for zl,zr in END_ZONES.get(x0,[]):
            t=np.clip(np.minimum((xa-zl)/15,(zr-xa)/15),0,1)
            seam=seam*(1-t)+np.minimum(seam,END_TOP)*t
        Y=np.arange(H)[:,None]
        # low-frequency colour correction: make the new image match ours along the seam,
        # fading out below it (weights favour pixels where both images agree in structure)
        band=[]
        for c in range(3):
            pass
        rows=(seam[None,:]+np.arange(-9,-3)[:,None]).astype(int)
        oo=ours.astype(np.float32)[rows,xa[None,:]]; nn=warped[rows,xa[None,:]]
        d=oo-nn; wgt=np.exp(-np.abs(d).mean(2)/12)
        num=gaussian_filter1d((d*wgt[...,None]).sum(0),25,axis=0)
        den=gaussian_filter1d(wgt.sum(0),25)[:,None]
        D=num/np.maximum(den,1e-3)
        fade=np.exp(-np.clip(Y-seam[None,:],0,None)/45.0)[...,None]
        warped=warped+D[None,:,:]*fade
        wy=np.clip((Y-(seam[None,:]-5))/5,0,1)
        lo=0 if x0<500 else 1001; hi=1001 if x0<500 else Wd
        # feather the two frames' versions across the gap (background) over 8 px
        xx=np.arange(Wd)
        wx=(np.clip((xx-(lo-4))/8,0,1) if lo>0 else np.ones(Wd))*(np.clip(((hi+4)-xx)/8,0,1) if hi<Wd else np.ones(Wd))
        w=(wy*wx[None,:])[...,None]
        if x0<500: acc_w=w; acc=warped*w
        else:
            tot=acc_w+w; blend=(acc+warped*w)/np.maximum(tot,1e-6)
            out=out*(1-np.clip(tot,0,1))+blend*np.clip(tot,0,1)
    # small leftovers of our plate ends just above the new plate edge: background
    for xl,xr,yt,yb in CLEAN_BOXES:
        for x in range(xl,xr):
            col=out[:,x]
            g=col[yt:yb+6].mean(1)
            # stop above the new plate's top edge (first strong brightening below yt+2)
            d=np.diff(g); k=[i for i in range(2,len(d)) if d[i]>18]
            ye=yt+(k[0] if k else yb-yt)
            ref=out[yt-6:yt-1,x].mean(0)
            out[yt:ye,x]=ref
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    ours=np.array(Image.open(sys.argv[1]).convert('RGB'))
    new=np.array(Image.open(sys.argv[2]).convert('RGB'))
    Image.fromarray(stitch(ours,new)).save(sys.argv[3])
