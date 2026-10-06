"""Figure 1 (45-degree view): D-series rig left, A-series rig right.

Inputs (original lightbox photos + rembg cutouts of them):
  2.jpg / D_cut.png  -> D rig,   1.jpg / A_cut.png -> A rig
Usage: python3 make_figure1_45deg.py out.png [gap_px] [D.jpg] [A.jpg]
Steps per rig: keep rembg foreground inside a hand-measured rig outline (drops the
long cable runs), mirror the short cable stub to the outer side, deskew so the back
plate sits centred over the front plate, then scale D to A's front-plate width.
The D1 label is rotated 180 deg so it reads like D2/D3.
"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage
from flip_label import flip_label

def main_alpha(path):
    a=np.array(Image.open(path))[:,:,3].astype(np.float32)
    m=a>40; lab,n=ndimage.label(m)
    keep=lab==(np.argmax(ndimage.sum(m,lab,range(1,n+1)))+1)
    return a*ndimage.binary_dilation(keep,iterations=3)/255.

def line(p,q):
    (x0,y0),(x1,y1)=p,q
    return lambda y: x0+(y-y0)*(x1-x0)/(y1-y0)

def build(rgb, alpha, R):
    H,W=alpha.shape
    yy,xx=np.mgrid[0:H,0:W]
    L,Rr=line(*R['left']),line(*R['right'])
    pl,pr=line(*R['plate_l']),line(*R['plate_r'])
    yb,yt,yp,yz=R['top'],R['back_top'],R['plate_top'],R['plate_bot']
    body=((yy>=yb)&(yy<yt)&(xx>=R['top_x'][0])&(xx<=R['top_x'][1])) | \
         ((yy>=yt)&(yy<yp)&(xx>=L(yy))&(xx<=Rr(yy))) | \
         ((yy>=yt)&(yy<R['back_bot'])&(xx>=R['back_x'][0])&(xx<=R['back_x'][1])) | \
         ((yy>=yp)&(yy<=yz)&(xx>=pl(yy))&(xx<=pr(yy)))
    body=ndimage.gaussian_filter(body.astype(np.float32),1.0)
    # short cable stub on the cable side, between back plate and front-plate top
    side=(xx>Rr(yy)) if R['cables']=='right' else (xx<L(yy))
    stub=(side&(yy>=yt)&(yy<yp-8)&~((yy<R['back_bot'])&(xx>=R['back_x'][0])&(xx<=R['back_x'][1]))).astype(np.float32)
    rgba=np.dstack([rgb.astype(np.float32),alpha*body])
    s=np.dstack([rgb.astype(np.float32),alpha*stub])
    C=R['mirror_c']                       # mirror about back-plate centre: x' = C - x
    s=np.roll(s[:,::-1],C-(W-1),axis=1)
    if C-(W-1)<0: s[:,C-(W-1):]=0
    a1,a2=rgba[...,3:],s[...,3:]
    ao=a1+a2*(1-a1)
    co=(rgba[...,:3]*a1+s[...,:3]*a2*(1-a1))/np.maximum(ao,1e-6)
    out=np.dstack([co,ao])
    # deskew above the front plate so the back plate is centred over the front plate
    k=R['skew']/(yp-yt)
    mx=(xx+np.where(yy<yp,k*(yp-yy),0)).astype(np.float32)
    out=cv2.remap(out,mx,yy.astype(np.float32),cv2.INTER_LINEAR,borderValue=0)
    return out

D_R=dict(top=840,back_top=910,back_bot=1000,back_x=(462,945),plate_top=1236,plate_bot=1406,top_x=(455,960),
         left=((463,910),(290,1236)),right=((946,1000),(1014,1232)),
         plate_l=((295,1236),(318,1406)),plate_r=((1070,1236),(1055,1406)),
         cables='right',mirror_c=1407,skew=(463+944)/2-(297+1068)/2)
A_R=dict(top=840,back_top=918,back_bot=980,back_x=(443,949),plate_top=1281,plate_bot=1456,top_x=(438,958),
         left=((446,960),(335,1281)),right=((950,915),(1032,1281)),
         plate_l=((268,1281),(296,1456)),plate_r=((1101,1281),(1077,1456)),
         cables='left',mirror_c=1392,skew=(444+948)/2-(269+1100)/2)

if __name__=='__main__':
    out=sys.argv[1]
    d_rgb=np.array(Image.open(sys.argv[3] if len(sys.argv)>3 else '2.jpg').convert('RGB'))
    flip_label(d_rgb,(418,982,522,1132))
    a_rgb=np.array(Image.open(sys.argv[4] if len(sys.argv)>4 else '1.jpg').convert('RGB'))
    D=build(d_rgb,main_alpha('D_cut.png'),D_R)
    A=build(a_rgb,main_alpha('A_cut.png'),A_R)
    s=(A_R['plate_r'][0][0]-A_R['plate_l'][0][0])/(D_R['plate_r'][0][0]-D_R['plate_l'][0][0])
    D=cv2.resize(D,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
    gap,m=int(sys.argv[2]) if len(sys.argv)>2 else 30,10
    # front-plate reference points (top-left x, bottom y) in each layer
    dL,dR,dB=D_R['plate_l'][0][0]*s,D_R['plate_r'][0][0]*s,D_R['plate_bot']*s
    aL,aR,aB=A_R['plate_l'][0][0],A_R['plate_r'][0][0],A_R['plate_bot']
    top=min(D_R['top']*s-dB,A_R['top']-aB)       # relative to plate bottom
    Hc=int(-top)+2*m+5
    Wc=int((dR-dL)+gap+(aR-aL))+2*m
    canvas=np.ones((Hc,Wc,3),np.float32)*255
    def paste(layer,ox,oy):
        h,w=layer.shape[:2]
        M=np.float32([[1,0,ox],[0,1,oy]])
        L=cv2.warpAffine(layer,M,(Wc,Hc),flags=cv2.INTER_LINEAR,borderValue=0)
        a=L[...,3:]
        canvas[:]=canvas*(1-a)+L[...,:3]*a
    base=Hc-m-5
    paste(D,m-dL,base-dB)
    paste(A,m+(dR-dL)+gap-aL,base-aB)
    Image.fromarray(np.clip(canvas,0,255).astype(np.uint8)).save(out,dpi=(300,300))
    print('scale',s,'size',Wc,Hc)
