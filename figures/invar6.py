"""Spacer bay (bay 4, both frames), as in the original photos of the rig: the invar rod
runs out of the roller bore straight across the plate-top bevel and becomes a threaded
stud in the notch, down to the hex nut (instead of the specimen bays' smooth spacer).

Builds on invar4 (hardware centred on the rod, rod at full width under the roller).
The thread texture is the figure's own: taken from the protruding threaded end of the
same bay's lock nut and tiled along the stud."""
import sys, numpy as np, cv2
from PIL import Image
import invar4
from invar3 import FRAMES, axis, cover, rod_profile, sample_prof

THREAD=[dict(x0=879,x1=900,y0=563,y1=576),dict(x0=1769,x1=1790,y0=565,y1=578)]  # 4 pitches
WT=22.0          # stud width (px)

def build(img):
    a=invar4.build(img).astype(np.float32); out=a.copy()
    base=img.astype(np.float32)
    for F,T in zip(FRAMES,THREAD):
        P=F['P']; xa=axis(F,P); w=F['w']; yb=P+12; yh=F['yh']
        prof=rod_profile(base,F)
        patch=a[T['y0']:T['y1'],T['x0']:T['x1']]          # (rows, cols, 3)
        ph,pw=patch.shape[:2]
        y0,y1=P-14,yh+1
        x0,x1=int(xa-w/2-3),int(xa+w/2+4)
        Y,X=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        reg=out[y0:y1,x0:x1]
        # rod: full width over the bevel, short neck down to the stud width
        apex=P-12.0
        def width(y):                            # gradual taper: bore -> notch -> stud
            t=np.clip((y-apex)/(yb-apex),0,1); return w+(WT-w)*t
        def cx(y): return axis(F,np.minimum(y,P))+(xa-axis(F,P))*0
        def in_rod(x,y):
            u=(x-cx(y))/(width(y)/2)
            return (np.abs(u)<=1)&(y<yb)&(y>=apex+9.0*(1-np.sqrt(np.clip(1-u*u,0,1))))
        # clear invar4's wider rod first (roller shadow under it), then draw the tapered rod
        old=cover(lambda x,y:(np.abs(x-cx(y))<=w/2+0.5)&(y>=apex+9.0*(1-np.sqrt(np.clip(1-((x-cx(y))/(w/2+0.5))**2,0,1))))&(y<P),X,Y)
        newc=cover(in_rod,X,Y)
        shadow=np.array([62,58,55],np.float32)          # gap between roller and plate, in shade
        reg[:]=reg*(1-(old*(1-newc))[...,None])+shadow*(old*(1-newc))[...,None]
        rodc=newc
        col=sample_prof(prof,(X-cx(Y))/width(Y)+0.5)
        col=col*(0.80-0.18*np.clip((Y-apex)/(yb-apex),0,1))[...,None]   # in the roller's / notch's shade
        reg[:]=reg*(1-rodc[...,None])+col*rodc[...,None]
        # threaded stud from the bevel bottom to the hex nut
        studc=cover(lambda x,y:(np.abs(x-xa)<=WT/2)&(y>=yb),X,Y)
        u=np.clip((X-(xa-WT/2))/WT,0,1)*(pw-1)
        rr=((Y-yb)%ph).astype(np.float32)
        tex=cv2.remap(patch,u.astype(np.float32),rr,cv2.INTER_LINEAR)
        shade=0.50+0.42*np.clip((Y-yb)/(yh-yb),0,1)                 # darker deep in the notch
        tex=tex*shade[...,None]
        reg[:]=reg*(1-studc[...,None])+tex*studc[...,None]
        # thin dark seam where the neck meets the thread
        seam=np.exp(-((Y-yb)/0.9)**2)*(np.abs(X-xa)<=WT/2)
        reg[:]=reg*(1-0.35*seam[...,None])
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    Image.fromarray(build(np.array(Image.open(sys.argv[1]).convert('RGB')))).save(sys.argv[2])
