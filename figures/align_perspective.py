"""Final step for Figure1_45deg.png: re-project both rigs (built by make_figure1_45deg.py)
onto one shared, symmetric template so the frames have identical perspective lines.

Two homographies per rig, blended across the front-plate top edge:
  upper  (rods/specimens/back plate): back-plate top corners + front-plate top corners
  lower  (front plate):               front-plate top + bottom corners
The threaded rods below the plate edge are erased (inpainted) and re-drawn along the
now-vertical specimen axes so they line up with the specimens.
Template proportions (units of front-plate width) match the target look: back plate
0.81 wide, centred, its top 0.41 above the front plate; front plate 0.235 tall, keeping its natural slight taper so the bolts stay upright.
Usage: python3 align_perspective.py Figure1_45deg.png [gap_px] [D.jpg] [A.jpg]
"""
import sys, numpy as np, cv2
from PIL import Image
from make_figure1_45deg import build, main_alpha, D_R, A_R
from flip_label import flip_label

# measured in the photos: back-top L/R, front-top L/R, front-bottom L/R
P={'D':np.float32([[463,910],[944,910],[297,1234],[1068,1234],[319,1405],[1053,1405]]),
   'A':np.float32([[444,918],[948,918],[269,1282],[1100,1282],[296,1455],[1076,1455]])}
BACK_W,BACK_H,PLATE_H,PLATE_TAPER=0.81,0.41,0.235,0.027
def template(W):
    b=(1-BACK_W)/2
    return np.float32([[b,-BACK_H],[1-b,-BACK_H],[0,0],[1,0],[PLATE_TAPER,PLATE_H],[1-PLATE_TAPER,PLATE_H]])*W

# threaded rods + nuts below the front-plate top edge: source x where each leaves the
# plate top, and how far (source px) it reaches down
RODS={'D':([393,588,777,972],90),'A':([380,576,785,992],96)}

def warp(layer,src,dst,size,seam_y,rods,band=12,half=(40,24),notch=40,feather=5):
    """Upper homography above the front-plate top edge, lower one below it.
    The threaded rods/nuts sticking out of the front plate continue the specimen axes,
    so they are erased from the plate and re-drawn along those (now vertical) axes."""
    Wc,Hc=size
    Hu=cv2.getPerspectiveTransform(src[:4],dst[:4])
    Hl=cv2.getPerspectiveTransform(src[2:],dst[2:])
    Hui=np.linalg.inv(Hu)
    Y,X=np.mgrid[0:Hc,0:Wc].astype(np.float32)
    pts=np.stack([X,Y,np.ones_like(X)],-1)
    def inv(H):
        q=pts@np.linalg.inv(H).T; return q[...,0]/q[...,2], q[...,1]/q[...,2]
    ux,uy=inv(Hu); lx,ly=inv(Hl)
    # "fan": below the seam, follow the source lines that the upper homography makes vertical
    q0=np.stack([X[0],np.full(Wc,seam_y),np.ones(Wc)],-1)@Hui.T
    q1=np.stack([X[0],np.full(Wc,seam_y-60),np.ones(Wc)],-1)@Hui.T
    x0,y0=q0[:,0]/q0[:,2],q0[:,1]/q0[:,2]; x1,y1=q1[:,0]/q1[:,2],q1[:,1]/q1[:,2]
    slope=(x0-x1)/(y0-y1)
    fx=x0[None,:]+slope[None,:]*(ly-y0[None,:])
    w=np.clip((Y-seam_y)/band+0.5,0,1)
    base=((1-w)*ux+w*lx,(1-w)*uy+w*ly)
    fan=((1-w)*ux+w*fx,(1-w)*uy+w*ly)
    # erase the original (tilted) rods from the plate in the source
    xs,reach=rods; ytop=src[2,1]
    rgb=np.clip(layer[...,:3],0,255).astype(np.uint8)
    hole=np.zeros(rgb.shape[:2],np.uint8)
    r=np.zeros((Hc,Wc),np.float32)
    for x in xs:
        p=cv2.perspectiveTransform(np.float32([[[x,ytop]]]),Hu)[0,0]
        e=cv2.perspectiveTransform(np.float32([[[x,ytop+reach]]]),Hl)[0,0]
        k=slope[int(round(p[0]))]
        yy=np.arange(int(ytop)-2,int(ytop+reach)+4)
        for y in yy:                      # wide over the plate notch, narrow over rod/nuts
            h=half[0] if y<ytop+notch else half[1]
            c=x+k*(y-ytop); hole[y,int(c-h-3):int(c+h+4)]=255
        # where the straightened rod goes in the output
        en=cv2.perspectiveTransform(np.float32([[[x,ytop+notch]]]),Hl)[0,0][1]
        hw=np.where(Y<en,half[0],half[1])*(dst[3,0]-dst[2,0])/(src[3,0]-src[2,0])
        t=np.clip((np.abs(X-p[0])-hw)/feather+0.5,0,1); rx=1-t
        t=np.clip((Y-e[1])/8+0.5,0,1); ry=1-t
        r=np.maximum(r,rx*ry*(Y>seam_y))
    clean=layer.copy()
    clean[...,:3]=cv2.inpaint(rgb,hole,7,cv2.INPAINT_TELEA).astype(np.float32)
    f32=lambda m:(m[0].astype(np.float32),m[1].astype(np.float32))
    B=cv2.remap(clean,*f32(base),cv2.INTER_LANCZOS4,borderValue=0)
    R=cv2.remap(layer,*f32(fan),cv2.INTER_LANCZOS4,borderValue=0)
    return B*(1-r[...,None])+R*r[...,None]

if __name__=='__main__':
    out=sys.argv[1]
    d_rgb=np.array(Image.open(sys.argv[3] if len(sys.argv)>3 else '2.jpg').convert('RGB'))
    flip_label(d_rgb,(418,982,522,1132))
    a_rgb=np.array(Image.open(sys.argv[4] if len(sys.argv)>4 else '1.jpg').convert('RGB'))
    D_R['skew']=0; A_R['skew']=0
    layers={'D':build(d_rgb,main_alpha('D_cut.png'),D_R),'A':build(a_rgb,main_alpha('A_cut.png'),A_R)}
    Wf=900; T=template(Wf)
    gap=int(sys.argv[2]) if len(sys.argv)>2 else int(0.054*Wf)
    m=12; top=-T[:,1].min()+140
    Hc=int(top+T[:,1].max()+m); Wc=int(2*Wf+gap+2*m)
    canvas=np.ones((Hc,Wc,3),np.float32)*255
    for i,k in enumerate(['D','A']):
        dst=T+np.float32([m+i*(Wf+gap),top])
        L=np.clip(warp(layers[k],P[k],dst,(Wc,Hc),top,RODS[k]),0,None); L[...,3]=np.clip(L[...,3],0,1)
        a=L[...,3:]; canvas=canvas*(1-a)+L[...,:3]*a
    rows=np.where((canvas<250).any((1,2)))[0]
    canvas=canvas[max(rows[0]-m,0):]
    Image.fromarray(np.clip(canvas,0,255).astype(np.uint8)).save(out,dpi=(300,300))
    print('size',canvas.shape[1],canvas.shape[0])
