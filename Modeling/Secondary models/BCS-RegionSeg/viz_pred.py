# Side-by-side: GT | multi-label | YOLO-seg, held-out predictions, each region as fill+contour (overlaps visible)
import os,json,cv2,numpy as np
from ultralytics import YOLO
D=np.load('/w/data.npz'); IM,MK,NM=D['imgs'],D['masks'],list(D['names'])
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']
COL=[(230,25,75),(60,180,75),(255,225,25),(0,130,200),(245,130,48),(145,30,180)]  # RGB
ML=[('ConvNeXt-T','convnext_tiny','/w/pred_ml'),('DINOv2-B','vit_base_patch14_dinov2_frozen','/w/pred_vit_base_patch14_dinov2_frozen'),('SAM2-Hiera-B+','sam2_hiera_base_plus_frozen','/w/pred_sam2_hiera_base_plus_frozen')]
RS={t:json.load(open(f'/w/res_ml_{t}.json')) for _,t,_ in ML}; ry=json.load(open('/w/res_yolo_yolo11s-seg.json'))
W,H=480,270; os.makedirs('/w/cmp',exist_ok=True)
def draw(img,M,title):
    o=img.copy(); f=img.astype(np.float32)
    for c in range(6):
        m=M[c]>0; f[m]=f[m]*.75+np.array(COL[c])*.25
    o=f.astype(np.uint8)
    for c in range(6):
        cs,_=cv2.findContours((M[c]>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o,cs,-1,COL[c],3)
    cv2.rectangle(o,(0,0),(W,34),(0,0,0),-1); cv2.putText(o,title,(8,25),0,.6,(255,255,255),2)
    return o
legend=np.zeros((40,5*W,3),np.uint8)
for c,n in enumerate(CLASSES):
    cv2.rectangle(legend,(10+c*400,10),(40+c*400,30),COL[c],-1); cv2.putText(legend,n,(50+c*400,28),0,.7,(255,255,255),2)
rows=[]
for k,n in enumerate(NM):
    img=cv2.resize(IM[k],(W,H),interpolation=cv2.INTER_AREA)
    G=np.stack([cv2.resize(m,(W,H),interpolation=cv2.INTER_NEAREST) for m in MK[k]])
    w=f'/w/yolo/runs/f{k}/weights/best.pt'; w=w if os.path.exists(w) else w.replace('best','last')
    r=YOLO(w).predict(f'/w/yolo/images/{n}.jpg',conf=.05,retina_masks=True,verbose=False)[0]; Y=np.zeros((6,H,W),np.uint8)
    for c in range(6):
        idx=[i for i,cl in enumerate(r.boxes.cls.tolist()) if int(cl)==c]
        if idx: Y[c]=cv2.resize((r.masks.data[max(idx,key=lambda i:r.boxes.conf[i])].cpu().numpy()>.5).astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
    Ps=[np.stack([cv2.resize(p,(W,H)) for p in np.load(f'{d}/{k}.npy').astype(np.float32)])>.5 for _,_,d in ML]
    row=np.hstack([draw(img,G,f'GT {n[:19]}')]+[draw(img,P,f'{nm} mIoU={np.mean(RS[t][n]):.2f}') for P,(nm,t,_) in zip(Ps,ML)]+[draw(img,Y,f'YOLO11s-seg mIoU={np.mean(ry[n]):.2f}')])
    cv2.imwrite(f'/w/cmp/fold{k}.jpg',row[:,:,::-1]); rows.append(row)
cv2.imwrite('/w/cmp/comparison_all.jpg',np.vstack([legend]+rows)[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,85])
print('done')
