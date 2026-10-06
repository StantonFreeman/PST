# Builds figures/Figure1_45deg.png: D-series rig (left) and A-series rig (right), 45-degree view.
# Inputs: D_cut.png / A_cut.png = the two lightbox photos with background removed by
# rembg (isnet-general-use). Usage: python3 make_figure1_45deg.py Figure1_45deg.png
from PIL import Image
import numpy as np
from scipy import ndimage
import sys
def cut(f):
    im=np.array(Image.open(f)).astype(float)
    a=im[:,:,3]; m=a>40
    lab,n=ndimage.label(m)
    keep=lab==(np.argmax(ndimage.sum(m,lab,range(1,n+1)))+1)
    keep=ndimage.binary_dilation(keep,iterations=3)
    im[:,:,3]=a*keep
    out=Image.fromarray(im.astype('uint8'),'RGBA')
    return out.crop(out.split()[3].point(lambda v:255 if v>10 else 0).getbbox())
def plate(img):
    # front plate = widest solid run of rows; return (width, bottom row, left x)
    a=np.array(img.split()[3])>128
    w=a.sum(1); rows=np.where(w>0.8*w.max())[0]
    cols=np.where(a[rows].any(0))[0]
    return cols[-1]-cols[0], rows[-1], cols[0]
D,A=cut('D_cut.png'),cut('A_cut.png')
(wd,bd,_),(wa,ba,_)=plate(D),plate(A)
s=wa/wd
D=D.resize((round(D.width*s),round(D.height*s)),Image.LANCZOS); bd=round(bd*s)
print('scale D by',s)
top=max(bd,ba); below=max(D.height-bd,A.height-ba)
pad,gap=60,120
W=D.width+A.width+gap+2*pad; H=top+below+2*pad
c=Image.new('RGB',(W,H),'white')
c.paste(D,(pad,pad+top-bd),D)
c.paste(A,(pad+D.width+gap,pad+top-ba),A)
c.save(sys.argv[1],dpi=(300,300)); print(c.size)
# whiten faint residual floor shadow below the front plates (cables are dark, untouched)
arr=np.array(c); y0=pad+top+2
reg=arr[y0:]; light=reg.min(2)>150; reg[light]=255
Image.fromarray(arr).save(sys.argv[1],dpi=(300,300))
