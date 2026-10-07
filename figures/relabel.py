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
          if np.hypot(o[0].stop-o[0].start,o[1].stop-o[1].start)>0.6*H]
    ink&=~np.isin(lab,thin)
    grp,m=ndimage.label(ndimage.binary_dilation(ink,iterations=9))
    sz=ndimage.sum(ink,grp,range(1,m+1))
    return ink&np.isin(grp,1+np.where(sz>=0.08*sz.sum())[0])

def lift(rgb,box,ang,thr=45):
    """Return a grey 'ink factor' patch (1 = no ink), rotated so the text runs vertically."""
    x0,y0,x1,y1=box; pad=30
    X0,Y0,X1,Y1=x0-pad,y0-pad,x1+pad,y1+pad
    reg=rgb[Y0:Y1,X0:X1].astype(np.float32)
    ink=np.zeros(reg.shape[:2],bool); ink[pad:-pad,pad:-pad]=keep_text(ink_mask(rgb,box,thr,min_peak=95))
    m=ndimage.binary_dilation(ink,iterations=2)
    clean=cv2.inpaint(reg.astype(np.uint8),(m*255).astype(np.uint8),5,cv2.INPAINT_TELEA).astype(np.float32)
    f=np.where(m,reg.mean(2)/np.maximum(clean.mean(2),1),1.0).clip(0,1)
    ys,xs=np.nonzero(ink); cy,cx=ys.mean(),xs.mean()
    M=cv2.getRotationMatrix2D((cx,cy),-ang,1.0)
    fr=cv2.warpAffine(f,M,(f.shape[1],f.shape[0]),flags=cv2.INTER_CUBIC,borderValue=1.0)
    ir=cv2.warpAffine(ink.astype(np.float32),M,(f.shape[1],f.shape[0]))>0.3
    ys,xs=np.nonzero(ir)
    return fr[ys.min():ys.max()+1,xs.min():xs.max()+1], ang

def main(ai_p,d_p,a_p,out):
    ai=np.array(Image.open(ai_p).convert('RGB'))
    d=np.array(Image.open(d_p).convert('RGB')); flip_label(d,(418,982,522,1132))
    a=np.array(Image.open(a_p).convert('RGB'))
    res=ai.astype(np.float32)
    for k,(x0,y0,x1,y1) in DST.items():
        # 1) erase the AI text
        m=ink_mask(ai,(x0,y0,x1,y1),40,min_size=6)
        m=ndimage.binary_dilation(m,iterations=3)
        full=np.zeros(ai.shape[:2],np.uint8); full[y0:y1,x0:x1]=m*255
        res=cv2.inpaint(np.clip(res,0,255).astype(np.uint8),full,6,cv2.INPAINT_TELEA).astype(np.float32)
        # 2) lift the real label and scale it to the AI label's length
        f,ang=lift(d if k[0]=='D' else a,SRC[k],AXIS[k])
        ys,xs=np.nonzero(m)
        h_ai=ys.max()-ys.min(); cx=x0+xs.mean(); cy=y0+(ys.min()+ys.max())/2
        s=h_ai/f.shape[0]
        f=cv2.resize(f,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC).clip(0,1)
        h,w=f.shape; px,py=int(round(cx-w/2)),int(round(cy-h/2))
        res[py:py+h,px:px+w]*=f[...,None]
        print(k,'rot %.1f deg'%ang,'scale %.2f'%s)
    Image.fromarray(np.clip(res,0,255).astype(np.uint8)).save(out)

if __name__=='__main__':
    main(*sys.argv[1:5])
