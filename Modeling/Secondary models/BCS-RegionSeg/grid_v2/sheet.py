# Contact sheet: every image with its 6 masks colored by assigned class, sorted by labeler cost (worst first)
import json,sys,cv2,numpy as np
sys.path.insert(0,'/w'); from rle import rle_to_mask
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']
COL=[(75,25,230),(75,180,60),(25,225,255),(200,130,0),(48,130,245),(180,30,145)]  # BGR
lab=json.load(open('/w/labels.json')); keys=sorted(lab); tiles=[]
for k in keys:
    d=json.load(open('/d/'+k)); img=cv2.resize(cv2.imread('/d/'+k.replace('.json','.jpg')),(480,270)); o=img.astype(np.float32)
    for i,a in enumerate(d['annotations']):
        c=CLASSES.index(lab[k][str(i)]); m=cv2.resize(rle_to_mask(a['segmentation']),(480,270),interpolation=cv2.INTER_NEAREST)
        o[m>0]=o[m>0]*.8+np.array(COL[c])*.2
    o=o.astype(np.uint8)
    for i,a in enumerate(d['annotations']):
        c=CLASSES.index(lab[k][str(i)]); m=cv2.resize(rle_to_mask(a['segmentation']),(480,270),interpolation=cv2.INTER_NEAREST)
        cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(o,cs,-1,COL[c],3)
        if cs:
            M=cv2.moments(max(cs,key=cv2.contourArea)); cx,cy=int(M['m10']/max(M['m00'],1)),int(M['m01']/max(M['m00'],1))
            cv2.putText(o,CLASSES[c][:5],(cx-20,cy),0,.5,(255,255,255),2)
    cv2.rectangle(o,(0,0),(480,22),(0,0,0),-1); cv2.putText(o,f'{len(tiles)} {k[:30]}',(4,16),0,.5,(255,255,255),1)
    tiles.append(o)
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]
for s in range(0,len(rows),4): cv2.imwrite(f'/w/sheet{s//4}.jpg',np.vstack(rows[s:s+4]))
print(len(keys))
