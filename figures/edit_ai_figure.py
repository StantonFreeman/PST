"""Builds figures/Figure1_45deg_edited.png from the AI-rendered two-frame image:
  1. relabel.py   - replace the wrong specimen labels with the real hand-written ones
                    lifted from the lightbox photos (D1 rotated 180 deg like D2/D3)
  2. fixbay4.py   - give the spacer bay the same threaded rod + nuts as the specimen bays
  3. respace.py   - even out the left frame's bay spacing (seam carving / plain-area stretch)
  4. proportions.py - centre the bays in each front plate (end margins = half a bay, as on
                    the real rig), centre the back plates, even out the right frame's bays
  5. plates.py    - give all eight bays the same regularly placed plate hardware
  6. floor.py     - one continuous floor shadow under the plates
  7. bgclean.py   - smooth the studio background where columns were shifted
  8. invar6.py    - spacer bay, as on the real rig: the invar rod tapers out of the roller
                    into a threaded stud in the notch, then the hex nut and lock nut
Needs opencv-contrib-python (cv2.xphoto) for the label erase.
Usage: python3 edit_ai_figure.py ai.webp D.jpg A.jpg out.png
"""
import sys, os, tempfile, numpy as np
from PIL import Image
import relabel, fixbay4, respace, proportions, plates, floor, bgclean, invar6
ai,d,a,out=sys.argv[1:5]
tmp=os.path.join(tempfile.mkdtemp(),'relabeled.png')
relabel.main(ai,d,a,tmp)
img=fixbay4.fix(np.array(Image.open(tmp).convert('RGB')))
img=respace.run(img,[(362,438,-29),(560,650-29,-7)])
y=np.arange(img.shape[0])
path=np.interp(y,[0,535,570,img.shape[0]],[805,805,768,768])-36-3
img=respace.stretch_path(img,path,36)
img=proportions.finish(img)
img=invar6.build(bgclean.clean(floor.render(plates.build(img))))
Image.fromarray(img).save(out)
