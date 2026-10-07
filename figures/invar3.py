"""Spacer bay (bay 4, both frames): rebuild where the invar rod meets the front plate.

In the AI image the lower roller sat flat on the plate and the rod showed only as a
narrow dark stub (~30 px) in an arch under it, much thinner than the rod above
(36-41 px), and the plate hardware was 6-11 px off the rod's axis. As on the real rig,
the rod now leaves the centre of the roller face at full diameter and tapers into the
plate notch. Steps:
  1. centre the notch / coupling / nut column on the rod axis,
  2. give the notch the same spacer as the specimen bays (copied from bay 3),
  3. move the lower roller up its rod by S px and draw the lower half of its face,
     which was hidden behind the plate; fill the area it uncovers with the rods
     behind (copied from higher up, along the rod direction),
  4. render the rod from the face centre at full width with a short cone taper into
     the notch, using the rod's own measured shading profile.
"""
import sys, numpy as np, cv2
from PIL import Image

FRAMES=[
 dict(x350=881.0,k=0.045,w=36,                       # rod axis x at y=350, slope, width
      xs=888,as_=63,yb=458,bs=53,                    # roller silhouette (centre x, half-width, back ellipse)
      xf=889.5,yf=478,af=57,bf=48,                   # roller face ellipse
      P=480,hw0=881.0,b3=662.0,yh=521),              # plate top; hardware centre now; bay-3 notch centre
 dict(x350=1744.5,k=0.243,w=41,
      xs=1773,as_=68,yb=458,bs=53,
      xf=1778,yf=478,af=61.5,bf=48.5,
      P=482,hw0=1787.5,b3=1569.0,yh=523),
]
S=65          # roller shift up the rod (vertical component, px)
SS=3          # supersampling for antialiased coverage
NB=27         # half-width of the copied notch block
YB=640        # bottom of the hardware column that is re-centred

def axis(F,y): return F['x350']+F['k']*(y-350)

def cover(fn,X,Y):
    acc=np.zeros(X.shape,np.float32)
    for i in range(SS):
        for j in range(SS):
            acc+=fn(X+(i+0.5)/SS-0.5,Y+(j+0.5)/SS-0.5)
    return acc/(SS*SS)

def roller_old(F,x,y):
    back=((x-F['xs'])/F['as_'])**2+((y-F['yb'])/F['bs'])**2<=1
    side=(np.abs(x-F['xs'])<=F['as_'])&(y>=F['yb'])&(y<=F['yf'])
    low=((x-F['xs'])/F['as_'])**2+((y-F['yf'])/(F['bf']+3))**2<=1
    return (back|side|low)&(y<F['P'])

def rod_profile(a,F,n=64):
    us=np.linspace(-F['w']/2,F['w']/2,n); acc=np.zeros((n,3))
    for y in range(330,396):
        xs=axis(F,y)+us
        for c in range(3): acc[:,c]+=np.interp(xs,np.arange(a.shape[1]),a[y,:,c])
    return acc/66

def sample_prof(prof,u):
    u=np.clip(u,0,1)*(len(prof)-1)
    return np.stack([np.interp(u,np.arange(len(prof)),prof[:,c]) for c in range(3)],-1)

def build(img):
    a=img.astype(np.float32); out=a.copy(); H,W=a.shape[:2]
    for F in FRAMES:
        k=F['k']; P=F['P']
        xa=axis(F,P)                                   # rod axis where it meets the plate
        # ---- 1. re-centre the hardware column below the plate-top bevel on the rod axis
        d=xa-F['hw0']
        M=np.float32([[1,0,d],[0,1,0]])
        moved=cv2.warpAffine(out,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        Yg,Xg=np.mgrid[0:H,0:W].astype(np.float32)
        wx=np.clip((46-np.abs(Xg-(xa+F['hw0'])/2))/10,0,1)
        wy=np.clip((Yg-(P+12))/2,0,1)*np.clip((YB-Yg)/10,0,1)
        w=(wx*wy)[...,None]; out=out*(1-w)+moved*w
        # ---- 2. notch (incl. its opening in the bevel) copied from bay 3, centred on xa
        sx=F['b3']-xa
        blk_y0,blk_y1=P,F['yh']+2
        for y in range(blk_y0,blk_y1):
            xs=np.arange(int(xa)-NB,int(xa)+NB+1).astype(np.float32)
            srow=np.stack([np.interp(xs+sx,np.arange(W),a[y,:,c]) for c in range(3)],-1)
            fx=np.clip((NB+0.5-np.abs(xs-xa))/4,0,1)[:,None]
            xi=xs.astype(int)
            out[y,xi]=out[y,xi]*(1-fx)+srow*fx
        # ---- 3. roller moved up the rod, lower face drawn, uncovered area filled
        dx=-k*S
        x0,x1=int(F['xs']-F['as_']-14),int(F['xs']+F['as_']+14)
        y0,y1=int(F['yb']-F['bs']-S-6),P
        Y,X=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        reg=out[y0:y1,x0:x1]
        src=lambda xx,yy: cv2.remap(a,xx.astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
        old=cover(lambda x,y: roller_old(F,x,y),X,Y)
        D=90.0
        reg[:]=reg*(1-old[...,None])+src(X-k*D,Y-D)*old[...,None]
        def sil(x,y):
            xo,yo=x-dx,y+S
            back=((xo-F['xs'])/F['as_'])**2+((yo-F['yb'])/F['bs'])**2<=1
            side=(np.abs(xo-F['xs'])<=F['as_'])&(yo>=F['yb'])&(yo<=F['yf'])
            low=((xo-F['xs'])/F['as_'])**2+((yo-F['yf'])/(F['bf']+3))**2<=1
            return back|side|low
        cov=cover(sil,X,Y)
        xo,yo=X-dx,Y+S
        moved=src(xo,yo)
        ym=467                                          # original rows >= ym were hole/plate
        def face_hw(y): return F['af']*np.sqrt(np.clip(1-((y-F['yf'])/F['bf'])**2,0,1))
        def sil_hw(y):
            return np.where(y<=F['yf'],F['as_'],
                   F['as_']*np.sqrt(np.clip(1-((y-F['yf'])/(F['bf']+3))**2,0,1)))
        lower=yo>=ym
        wr=face_hw(ym-2)
        xr=F['xf']+np.clip(xo-F['xf'],-wr+2,wr-2)      # stay inside the face when sampling
        g=1-0.06*np.clip((yo-ym)/F['bf'],0,1)
        face_c=src(xr,np.full_like(yo,ym-2))*g[...,None]
        # ring: map each pixel's position across the ring (face edge -> outline) onto the
        # same relative position at the reference row, so the ring's radial shading carries on
        side=np.where(xo<F['xf'],-1.0,1.0)
        fe=F['xf']+side*face_hw(yo)                          # face edge at this row
        oe=F['xs']+side*sil_hw(yo)                           # outline at this row
        s_=np.clip((xo-fe)/np.where(np.abs(oe-fe)<0.5,0.5*side,oe-fe),0,1)
        fe0=F['xf']+side*wr; oe0=F['xs']+side*F['as_']
        ringx=fe0+s_*(oe0-fe0)
        ring_c=src(ringx,np.full_like(yo,ym-2))*(1-0.30*np.clip((yo-F['yf'])/F['bf'],0,1))[...,None]
        fcov=cover(lambda x,y: (((x-dx)-F['xf'])/F['af'])**2+(((y+S)-F['yf'])/F['bf'])**2<=1,X,Y)
        lowpix=face_c*fcov[...,None]+ring_c*(1-fcov[...,None])
        rollerpix=np.where(lower[...,None],lowpix,moved)
        # darker edge along the lower outline
        e=cover(lambda x,y: (np.abs(((x-dx)-F['xs']))>=sil_hw(y+S)-1.3),X,Y)
        fade=np.clip((yo-(F['yf']-12))/14,0,1)            # outline darkens gradually downward
        rollerpix=rollerpix*(1-0.30*(e*fade)[...,None])
        reg[:]=reg*(1-cov[...,None])+rollerpix*cov[...,None]
        # ---- 4. rod from the face centre, cone into the notch
        prof=rod_profile(a,F); w=F['w']
        # spacer width in the notch (from bay 3): measure lit spacer just above the coupling
        wt=26.0
        yfn=F['yf']-S
        hb=w/2*F['bf']/F['af']
        yc0,yc1=P-15,P
        def cx(y):
            return axis(F,y)
        def width(y):
            t=np.clip((y-yc0)/(yc1-yc0),0,1); t=t*t*(3-2*t)
            return w+(wt-w)*t
        def in_rod(x,y):
            u=(x-cx(y))/(width(y)/2)
            top=yfn-hb*np.sqrt(np.clip(1-u*u,0,1))
            return (np.abs(u)<=1)&(y>=top)&(y<P)
        rc=cover(in_rod,X,Y)
        shc=cv2.GaussianBlur(cover(lambda x,y: in_rod(x-5,y-3),X,Y),(0,0),2.2)
        reg[:]=reg*(1-0.36*(shc*cov)[...,None])
        U=(X-cx(Y))/width(Y)+0.5
        col=sample_prof(prof,U)
        sh=0.88+0.12*np.clip((Y-yfn)/6,0,1)
        t=np.clip((Y-yc0)/(yc1-yc0),0,1)
        sh=sh*(1-0.30*t)                                # cone faces down/away: darker
        col=col*sh[...,None]
        col=col*(1-0.22*np.exp(-((Y-yc0)/1.3)**2))[...,None]   # edge where the taper starts
        reg[:]=reg*(1-rc[...,None])+col*rc[...,None]
        ac=cover(lambda x,y: (np.abs((x-cx(y))/(w/2+1.5))<=1)&
                 (np.abs(y-(yfn-(hb+1.2)*np.sqrt(np.clip(1-((x-cx(y))/(w/2+1.5))**2,0,1))))<=1.0)&(y<yfn),X,Y)
        reg[:]=reg*(1-0.55*ac[...,None])
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    Image.fromarray(build(np.array(Image.open(sys.argv[1]).convert('RGB')))).save(sys.argv[2])
