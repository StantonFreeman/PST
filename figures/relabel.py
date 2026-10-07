"""Replace the (incorrect) specimen labels in the AI-rendered figure with the real
hand-written labels lifted from the original lightbox photos.
Usage: python3 relabel.py ai.webp D.jpg A.jpg out.png
"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage
from flip_label import flip_label

# label boxes (x0,y0,x1,y1) in the original photos, arrows excluded
SRC={'D1':(425,990,515,1120),'D2':(588,959,668,1096),'D3':(743,958,797,1082),
     'A1':(398,968,516,1162),'A2':(576,972,652,1142),'A3':(733,976,813,1142)}
# label boxes on the AI image (where its wrong text is); new text is centred here
# specimen axis angle in the photos (deg from vertical, + = top leans left), measured
# from the cylinder edges; the text is rotated by this so it runs along the specimen
AXIS={'D1':-18.4,'D2':-6.7,'D3':0.0,'A1':-16.0,'A2':-7.1,'A3':3.6}
# one scale per photo so the handwriting size is consistent (A photo is ~7% closer)
SCALE={'D':1.30,'A':1.22}
DST={'D1':(222,172,342,342),'D2':(438,172,562,338),'D3':(652,172,778,332),
     'A1':(1042,186,1152,382),'A2':(1247,183,1352,377),'A3':(1447,183,1562,374)}

def ink_mask(rgb,box,thr,min_size=10,min_peak=0):
    x0,y0,x1,y1=box
    g=rgb[y0:y1,x0:x1].astype(np.float32).mean(2)
    bg=ndimage.median_filter(g,size=25)
    c=bg-g; dark=c>thr
    lab,n=ndimage.label(dark)
    sz=ndimage.sum(dark,lab,range(1,n+1)); pk=ndimage.maximum(c,lab,range(1,n+1))
    # marker ink is much darker than pores/edges: require a strong peak
    return np.isin(lab,1+np.where((sz>=min_size)&(pk>=min_peak))[0])

def keep_text(ink):
    """Drop stray marks (pores, rod edges): keep only components that are part of the
    main cluster of strokes and do not touch the box edge."""
    lab,n=ndimage.label(ink)
    objs=ndimage.find_objects(lab); H,W=ink.shape
    ok=[i+1 for i,o in enumerate(objs) if o[0].start>0 and o[1].start>0 and o[0].stop<H and o[1].stop<W]
    ink=np.isin(lab,ok)
    lab,n=ndimage.label(ink); objs=ndimage.find_objects(lab)
    thin=[i+1 for i,o in enumerate(objs)
          if np.hypot(o[0].stop-o[0].start,o[1].stop-o[1].start)>0.6*H
          or (o[0].stop-o[0].start>0.25*H and o[1].stop-o[1].start<7)]   # rod edges
    ink&=~np.isin(lab,thin)
    grp,m=ndimage.label(ndimage.binary_dilation(ink,iterations=9))
    sz=ndimage.sum(ink,grp,range(1,m+1))
    ink&=np.isin(grp,1+np.where(sz>=0.08*sz.sum())[0])
    lab,n=ndimage.label(ink); sz=ndimage.sum(ink,lab,range(1,n+1))
    return np.isin(lab,1+np.where(sz>=5)[0])

# D1: axis point, axis direction (down the specimen) and radius, measured from the
# cylinder edges at the label height in the D photo (unflipped)
D1_CYL=dict(p0=(498.0,1000.0),d=(-0.3,1.0),R=44.0,t=(-35,165),s=(-0.97,0.97))

def unwrap(rgb,p0,d,R,t,s):
    """Sample the cylinder surface on a (along-axis, arc-length) grid, so text written
    on the curving side is shown at its true proportions; axis ends up vertical."""
    d=np.float32(d)/np.hypot(*d); n=np.float32([d[1],-d[0]])
    if n[0]<0: n=-n
    T=np.arange(t[0],t[1],dtype=np.float32)
    S=np.arange(int(s[0]*R*np.pi/2),int(s[1]*R*np.pi/2),dtype=np.float32)
    TT,SS=np.meshgrid(T,S,indexing='ij'); u=R*np.sin(SS/R)
    mx=p0[0]+TT*d[0]+u*n[0]; my=p0[1]+TT*d[1]+u*n[1]
    return cv2.remap(rgb,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_CUBIC)

def lift_d1(rgb_unflipped,thr=40):
    reg=unwrap(rgb_unflipped,**D1_CYL)[::-1,::-1].copy()   # rotate 180: read like D2/D3
    H,W=reg.shape[:2]; m=12
    ink=np.zeros((H,W),bool); ink[m:-m,m:-m]=keep_text(ink_mask(reg,(m,m,W-m,H-m),thr,min_peak=90))
    mm=ndimage.binary_dilation(ink,iterations=2)
    clean=cv2.inpaint(reg,(mm*255).astype(np.uint8),5,cv2.INPAINT_TELEA).astype(np.float32)
    f=np.where(mm,reg.astype(np.float32).mean(2)/np.maximum(clean.mean(2),1),1.0).clip(0,1)
    f=cv2.erode(f,np.ones((2,2),np.uint8))
    ys,xs=np.nonzero(ink)
    return f[ys.min():ys.max()+2,xs.min():xs.max()+2]

# non-ink marks inside a label box (photo coords): the rod edge next to A3
IGNORE={'A3':[(806,970,816,1096),(790,1137,816,1145)]}

def _comp(f,labels,lab,dil=2):
    """Soft ink factor of the given connected components only (1 elsewhere)."""
    m=ndimage.binary_dilation(np.isin(lab,labels),iterations=dil)
    return np.where(m,f,1.0)

def compose_d1(rgb):
    """D1's label sits on the far side of the specimen in the photo, seen almost edge-on,
    so lifting it gives squashed, heavy strokes. Rebuild 'MT0.5-1 (D)(1)' from the same
    writer's strokes on the other D specimens: 'MT0.5-1 (D)' from D3, and the circled 1
    from D2's circle (its '2' left out) with D2's '1' stroke placed inside."""
    f3,_=lift(rgb,SRC['D3'],AXIS['D3']); f2,_=lift(rgb,SRC['D2'],AXIS['D2'])
    l3,_=ndimage.label(f3<0.7); l2,_=ndimage.label(f2<0.7)
    head=f3[:88]                                         # MT0.5-1 (D)  (D3 comps 1-10)
    ring=_comp(f2,[13],l2)[100:127]                      # D2's circle around the '2'
    one=_comp(f2,[10],l2)[60:70,8:39]                    # D2's '1' stroke
    r=ring.copy(); oy=(r.shape[0]-one.shape[0])//2; ox=(r.shape[1]-one.shape[1])//2
    r[oy:oy+one.shape[0],ox:ox+one.shape[1]]=np.minimum(r[oy:oy+one.shape[0],ox:ox+one.shape[1]],one)
    W=max(head.shape[1],r.shape[1]); gap=4
    out=np.ones((head.shape[0]+gap+r.shape[0],W),np.float32)
    hx=(W-head.shape[1])//2; out[:head.shape[0],hx:hx+head.shape[1]]=head
    rx=(W-r.shape[1])//2; out[head.shape[0]+gap:,rx:rx+r.shape[1]]=r
    return out

def lift(rgb,box,ang,thr=45,ignore=()):
    """Return a grey 'ink factor' patch (1 = no ink), rotated so the text runs vertically."""
    x0,y0,x1,y1=box; pad=30
    X0,Y0,X1,Y1=x0-pad,y0-pad,x1+pad,y1+pad
    reg=rgb[Y0:Y1,X0:X1].astype(np.float32)
    raw=ink_mask(rgb,box,thr,min_size=4,min_peak=80)
    for a0,b0,a1,b1 in ignore: raw[max(b0-y0,0):b1-y0,max(a0-x0,0):a1-x0]=False
    ink=np.zeros(reg.shape[:2],bool); ink[pad:-pad,pad:-pad]=keep_text(raw)
    m=ndimage.binary_dilation(ink,iterations=2)
    clean=cv2.inpaint(reg.astype(np.uint8),(m*255).astype(np.uint8),5,cv2.INPAINT_TELEA).astype(np.float32)
    f=np.where(m,reg.mean(2)/np.maximum(clean.mean(2),1),1.0).clip(0,1)
    ys,xs=np.nonzero(ink); cy,cx=ys.mean(),xs.mean()
    M=cv2.getRotationMatrix2D((cx,cy),-ang,1.0)
    fr=cv2.warpAffine(f,M,(f.shape[1],f.shape[0]),flags=cv2.INTER_CUBIC,borderValue=1.0)
    ir=cv2.warpAffine(ink.astype(np.float32),M,(f.shape[1],f.shape[0]))>0.3
    ys,xs=np.nonzero(ir)
    return fr[ys.min():ys.max()+1,xs.min():xs.max()+1], ang

def erase(img,mask,k=None,pad=40):
    """Remove the AI text with frequency-selective-reconstruction inpainting
    (cv2.xphoto, FSR), which rebuilds the concrete texture rather than smearing it."""
    ys,xs=np.nonzero(mask)
    y0,y1=max(ys.min()-pad,0),ys.max()+pad; x0,x1=max(xs.min()-pad,0),xs.max()+pad
    crop=np.clip(img[y0:y1,x0:x1],0,255).astype(np.uint8)
    valid=(~mask[y0:y1,x0:x1]).astype(np.uint8)*255
    dst=np.zeros_like(crop)
    cv2.xphoto.inpaint(crop,valid,dst,cv2.xphoto.INPAINT_FSR_FAST)
    out=img.copy(); out[y0:y1,x0:x1]=dst.astype(np.float32)
    return out

def main(ai_p,d_p,a_p,out):
    ai=np.array(Image.open(ai_p).convert('RGB'))
    d0=np.array(Image.open(d_p).convert('RGB'))
    a=np.array(Image.open(a_p).convert('RGB'))
    res=ai.astype(np.float32)
    for k,(x0,y0,x1,y1) in DST.items():
        # 1) erase the AI text
        m=ink_mask(ai,(x0,y0,x1,y1),40,min_size=6)
        m=ndimage.binary_dilation(m,iterations=5)
        full=np.zeros(ai.shape[:2],np.uint8); full[y0:y1,x0:x1]=m*255
        res=erase(res,full>0,k)
        # 2) lift the real label and scale it to the AI label's length
        if k=='D1': f,ang=compose_d1(d0),AXIS[k]
        else: f,ang=lift(d0 if k[0]=='D' else a,SRC[k],AXIS[k],ignore=IGNORE.get(k,()))
        ys,xs=np.nonzero(m)
        cx=x0+xs.mean(); cy=y0+(ys.min()+ys.max())/2
        s=SCALE[k[0]]
        f=cv2.resize(f,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC).clip(0,1)
        h,w=f.shape; px,py=int(round(cx-w/2)),int(round(cy-h/2))
        res[py:py+h,px:px+w]*=f[...,None]
        print(k,'rot %.1f deg'%ang,'scale %.2f'%s)
    Image.fromarray(np.clip(res,0,255).astype(np.uint8)).save(out)

if __name__=='__main__':
    main(*sys.argv[1:5])
