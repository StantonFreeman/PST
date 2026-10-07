"""Fix frame proportions in the edited AI figure so both frames match the real rig:
bays centred in the front plate (end margins = half a bay, as measured on the real
rig: 96 px margins at 193 px spacing) and the back plate centred over the specimens.
Plain plate is added/removed between the end nut and the plate edge, inside the
plate's own rows only, so no hardware is distorted."""
import sys, numpy as np
from PIL import Image

FRONT=(474,794)      # rows of the front plate (+ its shadow)
BACK=(78,296)        # rows of the back plate
W6=6                 # width of the plain strip that gets stretched

def expand(img,rows,x,n,side,lim):
    """Stretch the plain strip [x,x+W6) to W6+n px. Content between the strip and
    `lim` (a background column in the gap) moves outward; background past it is lost."""
    out=img.copy(); r0,r1=rows
    src=np.linspace(0,W6-1,W6+n)
    for y in range(r0,r1):
        seg=img[y,x:x+W6].astype(np.float32)
        new=np.stack([np.interp(src,np.arange(W6),seg[:,c]) for c in range(3)],-1).astype(img.dtype)
        if side=='right':
            out[y,x+W6+n:lim]=img[y,x+W6:lim-n]; out[y,x:x+W6+n]=new
        else:
            out[y,lim:x-n]=img[y,lim+n:x]; out[y,x-n:x+W6]=new
    return out

def shrink(img,rows,x0,x1,side,lim):
    """Remove columns [x0,x1); content between them and `lim` moves inward and the
    freed columns next to `lim` are filled with that background column."""
    out=img.copy(); r0,r1=rows; n=x1-x0
    for y in range(r0,r1):
        if side=='left':
            out[y,lim+n:x1]=img[y,lim:x0]; out[y,lim:lim+n]=img[y,lim]
        else:
            out[y,x0:lim-n]=img[y,x1:lim]; out[y,lim-n:lim]=img[y,lim-1]
    return out

def fix(img,cut=988,D=68):
    # make room: move the right frame D px to the right
    bg=np.repeat(img[:,cut:cut+1],D,1)
    img=np.concatenate([img[:,:cut],bg,img[:,cut:]],1)
    # left frame: front margins 135/76 -> 111/111, back 105/77 -> 105/105
    mid=cut+D//2; W=img.shape[1]
    img=shrink(img,FRONT,175,199,'left',0)
    img=expand(img,FRONT,971,35,'right',mid)
    img=expand(img,BACK,962,28,'right',mid)
    # right frame (+D): front margins 81/123 -> 114/114, back 78/98 -> 98/98
    img=expand(img,FRONT,1003+D,29,'left',mid)
    img=shrink(img,FRONT,1869+D,1874+D,'right',W)
    img=expand(img,BACK,1008+D,20,'left',mid)
    return img[:,19:]     # equal outer margins



def even_right_frame(img):
    """Right frame bays were 221/233/232 px: remove 12 and 11 px (seam carving, confined
    to the gaps between specimens) so all three are ~221, like the left frame."""
    import respace
    respace.R=img.shape[1]
    return respace.run(img,[(1400,1470,-12),(1613,1678,-11)])

def finish(img):
    img=even_right_frame(fix(img))
    # trim so the outer margins (plate edge to image edge) are equal
    g=img.astype(np.float32).mean(2)[650]
    e=np.where(np.abs(np.diff(g))>25)[0]; l,r=e[0],e[-1]
    m=min(l,img.shape[1]-r)
    return img[:,l-m:r+m]

if __name__=='__main__':
    Image.fromarray(finish(np.array(Image.open(sys.argv[1]).convert('RGB')))).save(sys.argv[2])
