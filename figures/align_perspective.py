"""Final step for Figure1_45deg.png: warp both rigs (built by make_figure1_45deg.py)
onto one shared, left/right-symmetric perspective template so the front plate, back
plate and converging rod lines are identical for the two frames.
Usage: python3 align_perspective.py Figure1_45deg.png [gap_px] [D.jpg] [A.jpg]
"""
import sys, numpy as np, cv2
from PIL import Image
from make_figure1_45deg import build, main_alpha, D_R, A_R
from flip_label import flip_label

# key points: back-plate top L/R, front-plate top L/R, front-plate bottom L/R
P={'D':np.float32([[463,910],[944,910],[297,1234],[1068,1234],[319,1405],[1053,1405]]),
   'A':np.float32([[444,918],[948,918],[269,1282],[1100,1282],[296,1455],[1076,1455]])}

def norm(p):                       # origin = front-top-left, unit = front-top width
    o=p[2]; w=p[3,0]-p[2,0]; return (p-o)/w
def template(width):
    t=(norm(P['D'])+norm(P['A']))/2
    # make it left/right symmetric about the front-plate centre
    m=t.copy(); m[:,0]=1-t[:,0]; m=m[[1,0,3,2,5,4]]
    t=(t+m)/2
    return t*width

if __name__=='__main__':
    out=sys.argv[1]; gap=int(sys.argv[2]) if len(sys.argv)>2 else 30
    d_rgb=np.array(Image.open(sys.argv[3] if len(sys.argv)>3 else '2.jpg').convert('RGB'))
    flip_label(d_rgb,(418,982,522,1132))
    a_rgb=np.array(Image.open(sys.argv[4] if len(sys.argv)>4 else '1.jpg').convert('RGB'))
    D_R['skew']=0; A_R['skew']=0
    layers={'D':build(d_rgb,main_alpha('D_cut.png'),D_R),'A':build(a_rgb,main_alpha('A_cut.png'),A_R)}
    Wf=P['A'][3,0]-P['A'][2,0]                     # front-plate width in output px
    T=template(Wf)
    m=10
    top=-T[:,1].min()+90                            # room for gauges/cables above back plate
    Hc=int(top+T[:,1].max()+m)
    Wc=int(2*Wf+gap+2*m)
    canvas=np.ones((Hc,Wc,3),np.float32)*255
    for i,k in enumerate(['D','A']):
        dst=T+np.float32([m+i*(Wf+gap),top])
        H,_=cv2.findHomography(P[k],dst,0)
        L=cv2.warpPerspective(layers[k],H,(Wc,Hc),flags=cv2.INTER_LINEAR,borderValue=0)
        a=L[...,3:]; canvas=canvas*(1-a)+L[...,:3]*a
        err=cv2.perspectiveTransform(P[k][None],H)[0]-dst
        print(k,'max corner error px',np.abs(err).max().round(1))
    # trim empty rows at top
    rows=np.where((canvas<250).any((1,2)))[0]
    canvas=canvas[max(rows[0]-m,0):]
    Image.fromarray(np.clip(canvas,0,255).astype(np.uint8)).save(out,dpi=(300,300))
    print('size',canvas.shape[1],canvas.shape[0])
