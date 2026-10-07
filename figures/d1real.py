"""D1's label from its own hand-written strokes in the D photo, straightened.
Each glyph is mapped with one linear transform that takes the text line direction to
'down' and the letters' up direction to 'right' (the layout of D2/D3), then placed on a
straight line keeping its spacing along the text."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
import relabel

BOX=(418,985,535,1130)
GLYPHS=[[10],[9],[8],[7],[6],[5],[4],[2,3],[1]]      # M T 0 . 5 - 1 (D) (1)  (component ids)

def _pca(mask):
    ys,xs=np.nonzero(mask); P=np.stack([xs-xs.mean(),ys-ys.mean()]); w,v=np.linalg.eigh(np.cov(P))
    return v[:,1]

def build(d):
    ink=relabel.keep_text(relabel.ink_mask(d,BOX,40,min_size=4,min_peak=80))
    lab,_=ndimage.label(ink)
    x0,y0,x1,y1=BOX
    reg=d[y0:y1,x0:x1].astype(np.float32)
    m=ndimage.binary_dilation(ink,iterations=2)
    clean=cv2.inpaint(reg.astype(np.uint8),(m*255).astype(np.uint8),5,cv2.INPAINT_TELEA).astype(np.float32)
    f=np.where(m,reg.mean(2)/np.maximum(clean.mean(2),1),1.0).clip(0,1)
    C=np.array([np.array(ndimage.center_of_mass(np.isin(lab,g)))[::-1] for g in GLYPHS])
    t=np.arange(len(C)); b=np.array([np.polyfit(t,C[:,0],1)[0],np.polyfit(t,C[:,1],1)[0]]); b/=np.linalg.norm(b)
    u=_pca(lab==4); u=u*np.sign(-u[0])                 # the '1' stroke gives the letters' up direction
    L=np.array([[0.,1.],[1.,0.]])@np.linalg.inv(np.stack([b,u],1))
    s=(C-C[0])@b; pad=40
    out=np.ones((int(s[-1]+3*pad+60),120),np.float32)
    for g,c,si in zip(GLYPHS,C,s):
        gf=np.where(ndimage.binary_dilation(np.isin(lab,g),iterations=3),f,1.0)
        A=np.hstack([L,(np.array([60.0,pad+si])-L@c)[:,None]]).astype(np.float32)
        out=np.minimum(out,np.clip(cv2.warpAffine(gf,A,(out.shape[1],out.shape[0]),flags=cv2.INTER_CUBIC,borderValue=1.0),0,1))
    ys,xs=np.nonzero(out<0.8)
    return out[max(ys.min()-1,0):ys.max()+2,max(xs.min()-1,0):xs.max()+2]
