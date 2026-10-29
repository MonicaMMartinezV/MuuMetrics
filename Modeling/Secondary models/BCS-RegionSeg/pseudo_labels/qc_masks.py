# QC sheet: random crops + crops with an empty region, overlaid with their pseudo-label masks (thr 0.5)
import csv,random,cv2,numpy as np
COL=[(230,25,75),(60,180,75),(255,225,25),(0,130,200),(245,130,48),(145,30,180)]; CL=['loin','hips','rump','tail','thL','thR']
L=[r for r in csv.DictReader(open('/out/masks_log.csv')) if r['status']=='ok']
empty=[r for r in L if any(float(r[k])==0 for k in r if k.startswith('cov_'))]; random.seed(0)
pick=random.sample([r for r in L if r not in empty],18)+empty
tiles=[]
for r in pick:
    img=cv2.imread('/data/'+r['crop'])[:,:,::-1]; Z=np.load('/out/'+r['crop'].split('/',1)[1][:-4]+'.npz')['masks']
    h=360; w=int(img.shape[1]*h/img.shape[0]); img=cv2.resize(img,(w,h)); M=np.stack([cv2.resize(m,(w,h)) for m in Z])>127
    f=img.astype(np.float32)
    for c in range(6): f[M[c]]=f[M[c]]*.75+np.array(COL[c])*.25
    o=f.astype(np.uint8)
    for c in range(6):
        cs,_=cv2.findContours(M[c].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(o,cs,-1,COL[c],2)
    t=np.zeros((h+22,320,3),np.uint8); o=o[:, :320] if w>=320 else np.pad(o,((0,0),(0,320-w),(0,0)))
    t[22:]=o; cv2.putText(t,('EMPTY ' if r in empty else '')+r['crop'].split('/')[1][:14],(3,16),0,.45,(255,255,255),1); tiles.append(t)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
leg=np.zeros((26,6*320,3),np.uint8)
for c,n in enumerate(CL): cv2.rectangle(leg,(8+c*320,6),(24+c*320,20),COL[c],-1); cv2.putText(leg,n,(30+c*320,19),0,.5,(255,255,255),1)
cv2.imwrite('/w/qc_masks.jpg',np.vstack([leg]+[np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)])[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,85]); print(len(pick))
