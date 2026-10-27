# YOLO11-seg instance segmentation (one instance per region => overlap allowed), leave-one-image-out CV
import os,sys,json,cv2,numpy as np
from ultralytics import YOLO
MODEL=sys.argv[1] if len(sys.argv)>1 else 'yolo11s-seg.pt'; EP=int(sys.argv[2]) if len(sys.argv)>2 else 150
D=np.load('/w/data.npz'); IM,MK,NM=D['imgs'],D['masks'],list(D['names'])
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']; R='/w/yolo'
os.makedirs(f'{R}/images',exist_ok=True); os.makedirs(f'{R}/labels',exist_ok=True)
for i,n in enumerate(NM):
    cv2.imwrite(f'{R}/images/{n}.jpg',IM[i][:,:,::-1]); lines=[]
    for c in range(6):
        cs,_=cv2.findContours(MK[i,c],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cn=max(cs,key=cv2.contourArea)[:,0]
        lines.append(' '.join([str(c)]+[f'{x/1920:.5f} {y/1080:.5f}' for x,y in cn]))
    open(f'{R}/labels/{n}.txt','w').write('\n'.join(lines))
res={}
for k,n in enumerate(NM):
    tr=[f'{R}/images/{m}.jpg' for j,m in enumerate(NM) if j!=k]; open(f'{R}/tr{k}.txt','w').write('\n'.join(tr))
    open(f'{R}/va{k}.txt','w').write(f'{R}/images/{n}.jpg')
    open(f'{R}/f{k}.yaml','w').write(f"train: {R}/tr{k}.txt\nval: {R}/va{k}.txt\nnames: {CLASSES}\n")
    m=YOLO(MODEL); m.train(data=f'{R}/f{k}.yaml',epochs=EP,imgsz=640,batch=4,fliplr=0.0,degrees=15,scale=.4,
        mosaic=0.0,overlap_mask=False,project=f'{R}/runs',name=f'f{k}',exist_ok=True,verbose=False,plots=False,val=False,workers=2)
    r=m.predict(f'{R}/images/{n}.jpg',conf=.05,retina_masks=True,verbose=False)[0]
    G=MK[k]>0; iou=[]
    for c in range(6):
        idx=[i for i,cl in enumerate(r.boxes.cls.tolist()) if int(cl)==c]
        if not idx: iou.append(0.); continue
        b=max(idx,key=lambda i:r.boxes.conf[i]); p=r.masks.data[b].cpu().numpy()>.5
        iou.append(float((p&G[c]).sum()/(p|G[c]).sum()))
    res[n]=iou; print('RES',n,' '.join(f'{c}={v:.2f}' for c,v in zip(CLASSES,iou)),f'mIoU={np.mean(iou):.3f}',flush=True)
A=np.array(list(res.values())); print('RES MEAN per class:',' '.join(f'{c}={v:.3f}' for c,v in zip(CLASSES,A.mean(0))),f'| mIoU={A.mean():.3f}')
json.dump(res,open(f'/w/res_yolo_{MODEL.split(".")[0]}.json','w'),indent=1)
