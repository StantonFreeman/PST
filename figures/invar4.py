"""Spacer bay (bay 4, both frames): where the invar rod meets the front plate.

The roller stays where it is. Two targeted fixes:
  1. the notch / coupling / nut column is centred on the rod's axis (it was 6-11 px
     off) and the notch gets the same spacer as the specimen bays (from bay 3);
  2. the rod showing under the roller was a narrow dark stub (~30 px) unlike the rod
     above it (36-41 px); it is redrawn at the rod's full width with the rod's own
     shading profile, in the roller's shade, down to the plate's top edge.
"""
import sys, numpy as np, cv2
from PIL import Image
from invar3 import FRAMES, axis, cover, rod_profile, sample_prof, NB, YB

def build(img):
    a=img.astype(np.float32); out=a.copy(); H,W=a.shape[:2]
    for F in FRAMES:
        P=F['P']; xa=axis(F,P)
        # 1a. re-centre the hardware column below the plate-top bevel on the rod axis
        d=xa-F['hw0']
        moved=cv2.warpAffine(out,np.float32([[1,0,d],[0,1,0]]),(W,H),flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        Yg,Xg=np.mgrid[0:H,0:W].astype(np.float32)
        wx=np.clip((46-np.abs(Xg-(xa+F['hw0'])/2))/10,0,1)
        wy=np.clip((Yg-(P+12))/2,0,1)*np.clip((YB-Yg)/10,0,1)
        w=(wx*wy)[...,None]; out=out*(1-w)+moved*w
        # 1b. notch (incl. its opening in the bevel) copied from bay 3, centred on xa
        sx=F['b3']-xa
        for y in range(P,F['yh']+2):
            xs=np.arange(int(xa)-NB,int(xa)+NB+1).astype(np.float32)
            srow=np.stack([np.interp(xs+sx,np.arange(W),a[y,:,c]) for c in range(3)],-1)
            fx=np.clip((NB+0.5-np.abs(xs-xa))/4,0,1)[:,None]
            xi=xs.astype(int); out[y,xi]=out[y,xi]*(1-fx)+srow*fx
        # 2. rod under the roller at full width
        prof=rod_profile(a,F); wr=F['w']
        apex=P-12.0; hb=9.0                                 # arch where the rod leaves the roller
        x0,x1=int(xa-wr),int(xa+wr); y0=int(apex-3)
        Y,X=np.mgrid[y0:P,x0:x1].astype(np.float32)
        reg=out[y0:P,x0:x1]
        def in_rod(x,y):
            u=(x-axis(F,y))/(wr/2)
            return (np.abs(u)<=1)&(y>=apex+hb*(1-np.sqrt(np.clip(1-u*u,0,1))))
        rc=cover(in_rod,X,Y)
        col=sample_prof(prof,(X-axis(F,Y))/wr+0.5)
        shade=0.70+0.12*np.clip((Y-apex)/12,0,1)            # in the roller's shade
        col=col*shade[...,None]
        reg[:]=reg*(1-rc[...,None])+col*rc[...,None]
        # dark contact line along the arch (edge of the roller's bore)
        ac=cover(lambda x,y: (np.abs((x-axis(F,y))/(wr/2+1.2))<=1)&
                 (np.abs(y-(apex+hb*(1-np.sqrt(np.clip(1-((x-axis(F,y))/(wr/2+1.2))**2,0,1)))))<=0.9),X,Y)
        reg[:]=reg*(1-0.6*ac[...,None])
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    Image.fromarray(build(np.array(Image.open(sys.argv[1]).convert('RGB')))).save(sys.argv[2])
