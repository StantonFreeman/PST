"""Bay 4 (spacer bay): line up the front-plate hardware (notch, spacer rod, hex
coupling, nut, threaded end) with the invar rod coming out of the lower roller.
In the AI image the rod and the hardware below it were 5-8 px apart, so the rod
seemed to step sideways where it enters the plate. The hardware column is moved
sideways onto the rod axis; the window is feathered through plain plate."""
import sys, numpy as np, cv2
from PIL import Image

# per frame: notch axis x, rod (roller exit) axis x, first row below the plate-top bevel
SHIFTS=[(882,887,492),(1788,1780,494)]   # y = bottom of the plate-top bevel
HALF=46; FX=10; Y1=640; FY=10

def fix(img):
    a=img.astype(np.float32); out=a.copy(); H,W=a.shape[:2]
    for xn,xr,yt in SHIFTS:
        d=xr-xn
        M=np.float32([[1,0,d],[0,1,0]])
        moved=cv2.warpAffine(a,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        X=np.arange(W)[None,:]; Y=np.arange(H)[:,None]
        c=(xn+xr)/2
        wx=np.clip((HALF-np.abs(X-c))/FX,0,1)
        wy=np.clip((Y-(yt-2))/3,0,1)*np.clip((Y1-Y)/FY,0,1)
        # in the plate-top bevel rows move only the notch walls / bevel, not the rod
        # (the rod there already sits on the roller axis)
        w=(wx*wy)[...,None]
        out=out*(1-w)+moved*w
        # plate-top bevel rows: clean bevel either side of the notch (sampled from plain
        # bevel further along the same row) and the rod continued straight down from
        # the row just above the bevel, so notch walls and rod are both straight
        for y in range(yt-13,yt):
            for x in range(xr-HALF,xr+HALF+1):
                dx=abs(x-xr)
                bevel=a[y,xr-HALF-2] if x<xr else a[y,xr+HALF+2]   # plain bevel, same side
                if dx<=15:
                    out[y,x]=a[yt-14,x]*(1-0.35*(y-(yt-13))/12)
                elif dx<=15+1:
                    out[y,x]=0.5*out[y,x]+0.5*bevel*0.6         # dark notch wall edge
                elif dx<HALF-FX:
                    out[y,x]=bevel
        # roller hole: trim it to the rod's width so it is symmetric about the axis
        for y in range(yt-40,yt-13):
            for side in (-1,1):
                face=a[y,xr+side*30]
                for dx in range(16,30):
                    x=xr+side*dx
                    if a[y,x].mean()<face.mean()-20: out[y,x]=face
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    Image.fromarray(fix(np.array(Image.open(sys.argv[1]).convert('RGB')))).save(sys.argv[2])
