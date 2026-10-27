import json,glob,sys,os,cv2,numpy as np
sys.path.insert(0,os.path.dirname(__file__)); from rle import rle_to_mask
src=sys.argv[1]; out=sys.argv[2]; os.makedirs(out,exist_ok=True)
cols=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
for f in sorted(glob.glob(src+'/*.json')):
    d=json.load(open(f)); img=cv2.imread(f.replace('.json','.jpg'))
    tiles=[]
    for i,a in enumerate(d['annotations']):
        m=rle_to_mask(a['segmentation']).astype(bool)
        t=img.copy(); t[m]=(0.5*t[m]+0.5*np.array(cols[i])).astype(np.uint8)
        cv2.putText(t,f"#{i} id{a['id']} cov={m.mean():.2f}",(20,60),0,2,(255,255,255),4)
        tiles.append(cv2.resize(t,(640,360)))
    grid=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:6])])
    cv2.imwrite(f"{out}/{os.path.basename(f)[:30]}.jpg",grid)
    print(f[-60:], [int(rle_to_mask(a['segmentation']).sum()) for a in d['annotations']])
