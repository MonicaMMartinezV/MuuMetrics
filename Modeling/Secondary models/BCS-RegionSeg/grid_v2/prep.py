# Build data.npz from all splits (half resolution): images, (6,H,W) multi-label masks, split, cow ID
import json,sys,os,cv2,numpy as np
sys.path.insert(0,'/w'); from rle import rle_to_mask
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']
EXCLUDE={'valid/2025-06-22-19-51-38'}  # ambiguous annotation (cow cut off, unclear hips/loin)
H,W=540,960
lab=json.load(open('/w/labels.json')); imgs,masks,names,splits,cows=[],[],[],[],[]
for k in sorted(lab):
    if any(k.startswith(e) for e in EXCLUDE): continue
    d=json.load(open('/d/'+k)); M=np.zeros((6,H,W),np.uint8)
    for i,a in enumerate(d['annotations']):
        M[CLASSES.index(lab[k][str(i)])]=cv2.resize(rle_to_mask(a['segmentation']),(W,H),interpolation=cv2.INTER_NEAREST)
    imgs.append(cv2.resize(cv2.imread('/d/'+k.replace('.json','.jpg')),(W,H),interpolation=cv2.INTER_AREA)[:,:,::-1])
    masks.append(M); names.append(os.path.basename(k)[:30]); splits.append(k.split('/')[0])
    cows.append(d['image']['extra'].get('user_metadata',{}).get('ID',-1))
np.savez_compressed('/w/data.npz',imgs=np.stack(imgs),masks=np.stack(masks),names=names,splits=splits,cows=np.array(cows))
print(len(names),{s:splits.count(s) for s in set(splits)})
