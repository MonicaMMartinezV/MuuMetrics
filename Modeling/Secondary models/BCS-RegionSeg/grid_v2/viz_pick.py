# Large GT | Pred views for worst/median/best images of an OOF prob file
import sys,cv2,numpy as np
tag=sys.argv[1]; CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']
COL=[(230,25,75),(60,180,75),(255,225,25),(0,130,200),(245,130,48),(145,30,180)]
D=np.load('/w/data.npz'); NM=list(D['names']); P=dict(np.load(f'/w/oof/{tag}.npz')); W,H=960,540
def draw(img,M,title):
    f=img.astype(np.float32)
    for c in range(6): m=M[c]>0; f[m]=f[m]*.72+np.array(COL[c])*.28
    o=f.astype(np.uint8)
    for c in range(6):
        cs,_=cv2.findContours((M[c]>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(o,cs,-1,COL[c],3)
    cv2.rectangle(o,(0,0),(W,40),(0,0,0),-1); cv2.putText(o,title,(10,29),0,.9,(255,255,255),2); return o
items=[]
for n,p in P.items():
    i=NM.index(n); G=D['masks'][i]>0; pr=np.stack([cv2.resize(c.astype(np.float32),(W,H)) for c in p])>.5
    iou=[(pr[c]&G[c]).sum()/max((pr[c]|G[c]).sum(),1) for c in range(6)]; items.append((np.mean(iou),n,i,pr,iou))
items.sort(key=lambda r:r[0]); k=len(items)//2
pick=[('PEOR',items[0]),('PEOR',items[1]),('TIPICA',items[k-1]),('TIPICA',items[k]),('MEJOR',items[-2]),('MEJOR',items[-1])]
leg=np.zeros((50,2*W,3),np.uint8)
for c,nm in enumerate(CLASSES): cv2.rectangle(leg,(15+c*320,12),(45+c*320,38),COL[c],-1); cv2.putText(leg,nm,(55+c*320,35),0,.9,(255,255,255),2)
for j in range(0,6,2):
    rows=[leg]
    for lab,(m,n,i,pr,iou) in pick[j:j+2]:
        img=D['imgs'][i]; s=' '.join(f'{c[:4]}={v:.2f}' for c,v in zip(CLASSES,iou))
        row=np.hstack([draw(img,D['masks'][i],f'ANOTACION  {n[:19]}'),draw(img,pr,f'{lab}  mIoU={m:.2f}')])
        bar=np.zeros((36,2*W,3),np.uint8); cv2.putText(bar,s,(10,26),0,.8,(220,220,220),2); rows+= [row,bar]
    cv2.imwrite(f'/w/figs/{tag}_{pick[j][0].lower()}.jpg',np.vstack(rows)[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,88])
print('ok')
