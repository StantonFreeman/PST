"""Spacer bay (bay 4, both frames), modelled on the original photos of the rig: the
invar rod comes straight out of the roller, crosses the plate-top bevel at full width,
and continues as a threaded stud through the notch, through the hex nut and on to the
lock nut with its protruding threaded end (threads also show between the two nuts).

Builds on invar4 (hardware centred on the rod axis, rod under the roller). Threads are
rendered as a helix on the rod's own measured shading profile (fine pitch, slanted
crests, serrated silhouette), matching the threaded end of the figure's lock nuts."""
import sys, numpy as np, cv2
from PIL import Image
import invar4, invar3
from invar3 import FRAMES, axis, cover, rod_profile, sample_prof

WR=27.0        # rod width where it leaves the roller / crosses the bevel
WT=24.0        # thread major diameter
PITCH=3.2
SLANT=0.9      # helix: crest rises this many px across the half-width
GAP=[(532,544),(534,546)]          # rows between the hex nut and the lock nut, per frame

def thread_rgb(prof,X,Y,xc,w,light):
    u=(X-xc)/(w/2)
    base=sample_prof(prof,(u+1)/2)
    ph=(Y+SLANT*u)/PITCH
    t=0.5+0.5*np.cos(2*np.pi*ph)                    # 1 on a crest
    crest=np.exp(-((1-t)/0.25)**2)
    col=base*(0.66+0.30*t)[...,None]+25*crest[...,None]*np.clip(base.mean(-1,keepdims=True)/200,0,1)
    hw=w/2-1.3*(1-t)                                # serrated silhouette
    return np.clip(col*light,0,255),hw

def build(img,lift=None):
    if lift:
        # roller lifted up its rod (invar3), rod drawn straight at WR (no cone)
        invar3.S=lift; invar3.WT_CONE=WR; invar3.W_ROD=WR; invar3.CENTER_HOLE=True
        saved=[F['w'] for F in FRAMES]
        prof_img=img.astype(np.float32)
        a=invar3.build(img).astype(np.float32)
    else:
        a=invar4.build(img).astype(np.float32)
    out=a.copy()
    base=img.astype(np.float32)
    for F,(g0,g1) in zip(FRAMES,GAP):
        P=F['P']; xa=axis(F,P); w=F['w']; yb=P+12; yh=F['yh']
        prof=rod_profile(base,F)
        if lift:
            apex=P-1.0; ah=0.0
        else:
            apex=P-12.0; ah=9.0
        y0,y1=int(apex)-2,g1+1
        x0,x1=int(xa-w/2-3),int(xa+w/2+4)
        Y,X=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        reg=out[y0:y1,x0:x1]
        cx=(lambda y: np.zeros_like(y)+xa) if lift else (lambda y: np.where(y<P,axis(F,np.minimum(y,P)),xa))
        arch=lambda x,y,hw: y>=apex+ah*(1-np.sqrt(np.clip(1-((x-cx(y))/hw)**2,0,1)))
        # clear invar4's wider rod under the roller (gap in shade), keep the bore arch
        old=cover(lambda x,y:(np.abs(x-cx(y))<=w/2+0.5)&arch(x,y,w/2+0.5)&(y<P),X,Y)
        shadow=np.array([38,35,33],np.float32)
        if lift: pass
        else:
            reg[:]=reg*(1-old[...,None])+shadow*old[...,None]
        # notch opening in the bevel rows: dark (the copied bay-3 notch shows that bay's rod)
        nd=cover(lambda x,y:(np.abs(x-xa)<=15.0)&(y>=P)&(y<yb+2),X,Y)
        dark=np.array([24,23,23],np.float32)
        reg[:]=reg*(1-nd[...,None])+dark*nd[...,None]
        # straight rod from the bore across the bevel
        rc=cover(lambda x,y:(np.abs(x-cx(y))<=WR/2)&arch(x,y,WR/2)&(y<yb-2),X,Y)
        col=sample_prof(prof,(X-cx(Y))/WR+0.5)*(0.80-0.12*np.clip((Y-apex)/(yb-apex),0,1))[...,None]
        reg[:]=reg*(1-rc[...,None])+col*rc[...,None]
        # short chamfer, then the threaded stud down to the hex nut, and between the nuts
        light=(0.55+0.40*np.clip((Y-yb)/(yh-yb),0,1))[...,None]
        light=np.where((Y>=g0)[...,None],0.80,light)
        trgb,hw=thread_rgb(prof,X,Y,xa,WT,light)
        cham=np.clip((Y-(yb-2))/2,0,1)
        hwc=WR/2*(1-cham)+hw*cham
        tc=cover(lambda x,y:(np.abs(x-xa)<=(WR/2*(1-np.clip((y-(yb-2))/2,0,1))+
                 (WT/2-1.3*(1-(0.5+0.5*np.cos(2*np.pi*(y+SLANT*(x-xa)/(WT/2))/PITCH))))*np.clip((y-(yb-2))/2,0,1)))
                 &(y>=yb-2)&((y<yh)|((y>=g0)&(y<g1))),X,Y)
        reg[:]=reg*(1-tc[...,None])+trgb*tc[...,None]
    if lift:
        # faint vertical seam on the left roller face (left over from the AI image's notch
        # edge, now moved up with the roller): interpolate across it
        for y in range(400,481):
            l,r=out[y,906],out[y,912]
            for i,x in enumerate(range(907,912)):
                t=(i+1)/6
                if abs(out[y,x].mean()-(l.mean()*(1-t)+r.mean()*t))<20: out[y,x]=l*(1-t)+r*t
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    lift=int(sys.argv[3]) if len(sys.argv)>3 else None
    Image.fromarray(build(np.array(Image.open(sys.argv[1]).convert('RGB')),lift)).save(sys.argv[2])
