"""Even out the bay spacing of the left frame with content-aware seam carving.
Seams are confined to the gaps between bays, so specimens/rollers keep their shape.
Ops (band in current image x, count): negative = remove, positive = insert."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage
R=990                                    # only columns [0,R) (the left frame) change
def energy(img):
    g=img.astype(np.float32).mean(2)
    return np.abs(ndimage.sobel(g,1))+np.abs(ndimage.sobel(g,0))
def find_seam(e,lo,hi):
    H,W=e.shape; big=1e9
    c=np.full((H,W),big,np.float32); c[:,lo:hi]=e[:,lo:hi]
    back=np.zeros((H,W),np.int32)
    for y in range(1,H):
        p=c[y-1]; l=np.r_[big,p[:-1]]; r=np.r_[p[1:],big]
        st=np.stack([l,p,r]); k=st.argmin(0); c[y]+=st[k,np.arange(W)]; back[y]=np.arange(W)+k-1
    s=np.zeros(H,np.int32); s[-1]=int(c[-1].argmin())
    for y in range(H-1,0,-1): s[y-1]=back[y,s[y]]
    return s
def remove(img,s):
    out=img.copy()
    for y,x in enumerate(s): out[y,x:R-1]=img[y,x+1:R]
    return out
def insert(img,s):
    out=img.copy()
    for y,x in enumerate(s):
        out[y,x+1]=((img[y,x].astype(np.int32)+img[y,x+1])//2).astype(img.dtype)
        out[y,x+2:R]=img[y,x+1:R-1]
    return out
def run(img,ops):
    pen=np.zeros(img.shape[:2],np.float32)
    for lo,hi,n in ops:
        for i in range(abs(n)):
            e=energy(img)+pen
            s=find_seam(e,lo,hi if n>0 else hi-i)
            if n<0:
                img=remove(img,s); pen=remove(pen[...,None],s)[...,0]
            else:
                img=insert(img,s); pen=insert(pen[...,None],s)[...,0]
                for y,x in enumerate(s): pen[y,x:x+3]+=60   # spread insertions out
    return img
def stretch_path(img,path,n,w=6):
    """Widen the image by n px along a hand-picked path through plain areas: in each row
    the w px around path[y] are resampled to w+n px, everything right of it shifts."""
    out=img.copy(); H=img.shape[0]
    src=np.linspace(0,w-1,w+n)
    for y in range(H):
        p=int(path[y]); seg=img[y,p:p+w].astype(np.float32)
        new=np.stack([np.interp(src,np.arange(w),seg[:,c]) for c in range(3)],-1)
        out[y,p:p+w+n]=new.astype(img.dtype)
        out[y,p+w+n:R]=img[y,p+w:R-n]
    return out

if __name__=='__main__':
    img=np.array(Image.open(sys.argv[1]).convert('RGB'))
    # bays (rod x at the front plate): 235, 487, 718, 905 -> equal 223 spacing
    img=run(img,[(362,438,-29),(560,650-29,-7)])
    # widen bay 3->4 by 36 px along a path in plain areas (original x 805: gap between the
    # two rods, back plate, plate face; then over to x 768 to pass through the empty part
    # of the slot instead of its cap nut). Coordinates here are after the 36 px removal.
    y=np.arange(img.shape[0])
    path=np.interp(y,[0,535,570,img.shape[0]],[805,805,768,768])-36-3
    Image.fromarray(stretch_path(img,path,36)).save(sys.argv[2])
