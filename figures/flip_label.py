import numpy as np, cv2
from PIL import Image
from scipy import ndimage
def flip_label(rgb, box, min_size=12, thr=35):
    """Rotate the hand-written label inside box (x0,y0,x1,y1) by 180 deg in place.
    Only the ink moves: original strokes are inpainted, then re-applied rotated."""
    x0,y0,x1,y1=box
    reg=rgb[y0:y1,x0:x1].astype(np.float32)
    g=reg.mean(2)
    bg=ndimage.median_filter(g,size=21)
    dark=(bg-g)>thr
    lab,n=ndimage.label(dark)
    sz=ndimage.sum(dark,lab,range(1,n+1))
    objs=ndimage.find_objects(lab)
    H,W=g.shape
    keepl=[i+1 for i,o in enumerate(objs) if sz[i]>=min_size and o[0].start>0 and o[1].start>0 and o[0].stop<H and o[1].stop<W]
    ink=np.isin(lab,keepl)
    m=ndimage.binary_dilation(ink,iterations=2)
    # ink strength (0..1) relative to local background, as multiplicative factor
    clean=cv2.inpaint(reg.astype(np.uint8),(m*255).astype(np.uint8),5,cv2.INPAINT_TELEA).astype(np.float32)
    ratio=np.where(m[...,None], reg/np.maximum(clean,1), 1.0)
    ratio=np.clip(ratio,0,1)
    # 180 deg rotation about the ink centroid, so the label stays where it was
    cy,cx=ndimage.center_of_mass(ink)
    yy,xx=np.mgrid[0:H,0:W]
    sy=np.clip(np.round(2*cy-yy).astype(int),0,H-1); sx=np.clip(np.round(2*cx-xx).astype(int),0,W-1)
    ratio_r=ratio[sy,sx]
    out=clean*ratio_r
    rgb[y0:y1,x0:x1]=np.clip(out,0,255).astype(np.uint8)
    return ink
