# Post-hoc on out-of-fold probabilities: per-class threshold tuning and ensembles (no retraining)
import glob,itertools,json,os,sys,cv2,numpy as np
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']; TH=np.round(np.arange(.25,.71,.05),2)
D=np.load('/w/data.npz'); NM=list(D['names'])
GT={n:np.stack([cv2.resize(m,(480,270),interpolation=cv2.INTER_NEAREST) for m in D['masks'][i]])>0 for i,n in enumerate(NM)}
OOF={os.path.basename(f)[:-4]:dict(np.load(f)) for f in sorted(glob.glob('/w/oof/*.npz'))}

def scores(P,th,cls=range(6)):  # P: {name:(len(cls),h,w)} -> per-image per-class IoU at per-class thresholds
    return np.array([[((P[n][j]>th[j])&GT[n][c]).sum()/max(((P[n][j]>th[j])|GT[n][c]).sum(),1) for j,c in enumerate(cls)] for n in P])

def tune(P):  # pick threshold per class maximizing mean IoU over OOF images (slightly optimistic; report both)
    best=[max(TH,key=lambda t:scores({n:P[n][[c]] for n in P},[t],[c]).mean()) for c in range(6)]
    return best,scores(P,best)

rows=[]
for tag,P in OOF.items():
    P={n:v.astype(np.float32) for n,v in P.items()}; base=scores(P,[.5]*6); th,tun=tune(P)
    rows.append((tag,base.mean(),tun.mean(),th))
for k in (2,3):
    for combo in itertools.combinations(sorted(t for t in OOF if '_frozen' not in t or k==2),k):
        common=set.intersection(*[set(OOF[t]) for t in combo])
        P={n:np.mean([OOF[t][n].astype(np.float32) for t in combo],0) for n in common}
        base=scores(P,[.5]*6); th,tun=tune(P); rows.append(('+'.join(combo),base.mean(),tun.mean(),th))
rows.sort(key=lambda r:-r[1])
print(f'{"config":75s} mIoU@0.5  mIoU@tuned  thresholds')
for t,b,u,th in rows[:25]: print(f'{t:75s} {b:.4f}    {u:.4f}     {th}')
json.dump([dict(config=t,miou_05=float(b),miou_tuned=float(u),th=[float(x) for x in th]) for t,b,u,th in rows],open('/w/results/posthoc.json','w'),indent=1)
