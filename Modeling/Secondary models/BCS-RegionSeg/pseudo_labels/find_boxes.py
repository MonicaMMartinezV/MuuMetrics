# Recover the crop box of each Dataset_Fr_BCS image inside its original full frame (template matching)
import glob,os,sys,csv,cv2,numpy as np
D='/data'; LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 0
orig={os.path.basename(p):p for p in glob.glob(f'{D}/Images/*/*.jpg')}
crops=sorted(glob.glob(f'{D}/Dataset_Fr_BCS/*/*.jpg')); crops=crops[:LIMIT] if LIMIT else crops
rows=[]
for p in crops:
    n=os.path.basename(p).split('__',1)[1]; r=dict(crop=os.path.relpath(p,D),orig='',x1='',y1='',x2='',y2='',score='')
    if n in orig:
        F=cv2.imread(orig[n],0); C=cv2.imread(p,0); h,w=C.shape; s=4
        res=cv2.matchTemplate(cv2.resize(F,None,fx=1/s,fy=1/s,interpolation=cv2.INTER_AREA),cv2.resize(C,(w//s,h//s),interpolation=cv2.INTER_AREA),cv2.TM_CCOEFF_NORMED)
        _,_,_,(x,y)=cv2.minMaxLoc(res); x,y=x*s,y*s                      # coarse
        x0,y0=max(0,x-2*s),max(0,y-2*s); win=F[y0:min(F.shape[0],y+h+2*s),x0:min(F.shape[1],x+w+2*s)]
        res=cv2.matchTemplate(win,C,cv2.TM_CCOEFF_NORMED); _,sc,_,(dx,dy)=cv2.minMaxLoc(res)   # full-res refine
        r.update(orig=os.path.relpath(orig[n],D),x1=x0+dx,y1=y0+dy,x2=x0+dx+w,y2=y0+dy+h,score=round(float(sc),4))
    rows.append(r)
out='/w/fr_bcs_boxes.csv' if not LIMIT else '/w/fr_bcs_boxes_test.csv'
csv.DictWriter(open(out,'w',newline=''),fieldnames=list(rows[0])).writeheader() or None
with open(out,'a',newline='') as f: csv.DictWriter(f,fieldnames=list(rows[0])).writerows(rows)
sc=[r['score'] for r in rows if r['score']!='']
print(f'{len(rows)} crops | con original {len(sc)} | score min={min(sc):.3f} median={np.median(sc):.3f} | <0.95: {sum(s<0.95 for s in sc)}')
