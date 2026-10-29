# Render GT | prediction for every image in an OOF/test prob file, sorted worst->best, with per-image mIoU
import json,sys,cv2,numpy as np
tag=sys.argv[1]; th=json.loads(sys.argv[2]) if len(sys.argv)>2 else [.5]*6
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']
COL=[(230,25,75),(60,180,75),(255,225,25),(0,130,200),(245,130,48),(145,30,180)]  # RGB
D=np.load('/w/data.npz'); NM=list(D['names']); P=dict(np.load(f'/w/oof/{tag}.npz')); W,H=480,270
def draw(img,M,title):
    f=img.astype(np.float32)
    for c in range(6): m=M[c]>0; f[m]=f[m]*.75+np.array(COL[c])*.25
    o=f.astype(np.uint8)
    for c in range(6):
        cs,_=cv2.findContours((M[c]>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(o,cs,-1,COL[c],2)
    cv2.rectangle(o,(0,0),(W,24),(0,0,0),-1); cv2.putText(o,title,(6,17),0,.5,(255,255,255),1); return o
rows=[]
for n,p in P.items():
    i=NM.index(n); img=cv2.resize(D['imgs'][i],(W,H),interpolation=cv2.INTER_AREA)
    G=np.stack([cv2.resize(m,(W,H),interpolation=cv2.INTER_NEAREST) for m in D['masks'][i]])>0
    pr=np.stack([p[c].astype(np.float32)>th[c] for c in range(6)])
    iou=[(pr[c]&G[c]).sum()/max((pr[c]|G[c]).sum(),1) for c in range(6)]
    rows.append((np.mean(iou),np.hstack([draw(img,G,f'GT {n[:19]}'),draw(img,pr,f'Pred mIoU={np.mean(iou):.2f}  tail={iou[3]:.2f}')])))
rows.sort(key=lambda r:r[0])
leg=np.zeros((30,2*W,3),np.uint8)
for c,nm in enumerate(CLASSES): cv2.rectangle(leg,(8+c*160,8),(26+c*160,24),COL[c],-1); cv2.putText(leg,nm,(32+c*160,22),0,.5,(255,255,255),1)
tiles=[r for _,r in rows]
if len(tiles)%2: tiles.append(np.zeros_like(tiles[0]))
grid=np.vstack([np.hstack(tiles[k:k+2]) for k in range(0,len(tiles),2)])
cv2.imwrite(f'/w/figs/{tag}_all.jpg',np.vstack([np.hstack([leg,leg]),grid])[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,85])
print(f'{tag}: {len(rows)} imgs, mIoU={np.mean([m for m,_ in rows]):.4f}, worst={rows[0][0]:.3f}, best={rows[-1][0]:.3f}')
