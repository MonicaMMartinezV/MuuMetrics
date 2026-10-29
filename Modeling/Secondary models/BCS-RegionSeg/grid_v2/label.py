# Assign anatomical class to each unlabeled mask via Hungarian matching on normalized geometry
import json,glob,sys,os,numpy as np
import itertools
sys.path.insert(0,'/w'); from rle import rle_to_mask
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']
# prototype features: (cx, cy, w, h) normalized to union bbox of all regions
PROTO={'loin':(.5,.12,.8,.25),'hips':(.5,.3,1,.3),'rump':(.5,.65,.8,.4),
       'tailhead':(.5,.55,.45,.3),'thigh_L':(.15,.6,.3,.8),'thigh_R':(.85,.6,.3,.8)}
out={}
for f in sorted(glob.glob(sys.argv[1]+'/*/*.json')):
    d=json.load(open(f)); ms=[rle_to_mask(a['segmentation']) for a in d['annotations']]
    U=np.any(ms,0); ys,xs=np.nonzero(U); x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
    W,H=x1-x0,y1-y0; feats=[]
    for m in ms:
        yy,xx=np.nonzero(m)
        feats.append(((xx.mean()-x0)/W,(yy.mean()-y0)/H,(xx.max()-xx.min())/W,(yy.max()-yy.min())/H))
    C=np.array([[np.sum((np.array(fe)-np.array(PROTO[c]))**2*[3,3,1,1]) for c in CLASSES] for fe in feats])
    c=min(itertools.permutations(range(6)),key=lambda p:sum(C[i,p[i]] for i in range(6))); r=range(6)
    out[os.path.relpath(f,sys.argv[1])]={int(i):CLASSES[j] for i,j in zip(r,c)}
    print(os.path.relpath(f,sys.argv[1])[:40], [CLASSES[j] for j in c], round(float(sum(C[i,c[i]] for i in range(6))),3))
json.dump(out,open('/w/labels.json','w'),indent=1)
