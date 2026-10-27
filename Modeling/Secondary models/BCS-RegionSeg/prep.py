# Build data.npz: images + (6,H,W) multi-label masks in canonical class order
import json,glob,sys,os,cv2,numpy as np
sys.path.insert(0,'/w'); from rle import rle_to_mask
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']
lab=json.load(open('/w/labels.json')); imgs,masks,names=[],[],[]
for f in sorted(glob.glob('/d/*.json')):
    d=json.load(open(f)); L=lab[os.path.basename(f)]; M=np.zeros((6,1080,1920),np.uint8)
    for i,a in enumerate(d['annotations']): M[CLASSES.index(L[str(i)])]=rle_to_mask(a['segmentation'])
    imgs.append(cv2.imread(f.replace('.json','.jpg'))[:,:,::-1]); masks.append(M); names.append(os.path.basename(f)[:30])
    ov=(M.sum(0)>1).sum()/max((M.sum(0)>0).sum(),1); print(names[-1],'overlap frac of labeled px: %.2f'%ov)
np.savez_compressed('/w/data.npz',imgs=np.stack(imgs),masks=np.stack(masks),names=names)
