# Pseudo-label Dataset_Fr_BCS: segment each ORIGINAL full frame with the production region model,
# crop the soft masks with the recovered crop box, save uint8 (6,h,w) at half crop resolution.
import csv,glob,os,cv2,numpy as np,torch,torch.nn as nn,torch.nn.functional as F,timm
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']; SWAP=[0,1,2,3,5,4]
ENC='vit_base_patch14_dinov2.lvd142m'; H,W=378,672; DEC=128; dev='cuda'
MEAN=np.array([.485,.456,.406]); STD=np.array([.229,.224,.225]); D='/data'; OUT='/out'

class Net(nn.Module):  # same architecture as grid.py (finetune mode)
    def __init__(s):
        super().__init__(); s.e=timm.create_model(ENC,pretrained=False,features_only=True,dynamic_img_size=True,out_indices=(2,5,8,11))
        ch=s.e.feature_info.channels(); s.lat=nn.ModuleList([nn.Conv2d(c,DEC,1) for c in ch])
        s.head=nn.Sequential(nn.Conv2d(DEC*len(ch),DEC,3,padding=1),nn.BatchNorm2d(DEC),nn.ReLU(),nn.Conv2d(DEC,6,1))
    def forward(s,x):
        fs=s.e(x); t=fs[0].shape[-2:]; p=[]; y=None
        for f,l in zip(fs[::-1],list(s.lat)[::-1]):
            z=l(f); y=z if y is None else z+F.interpolate(y,size=z.shape[-2:],mode='bilinear'); p.append(F.interpolate(y,size=t,mode='bilinear'))
        return F.interpolate(s.head(torch.cat(p,1)),size=x.shape[-2:],mode='bilinear')

net=Net(); net.load_state_dict(torch.load('/weights/final_all40.pt',map_location='cpu')); net=net.to(dev).eval()
fallback={os.path.basename(p).split('_jpg.rf.')[0]+'.jpg':p for p in glob.glob(f'{D}/BCS_masking.sam2/*/*.jpg')}
rows=list(csv.DictReader(open('/w/fr_bcs_boxes.csv'))); log=[]
for r in rows:
    crop=cv2.imread(f"{D}/{r['crop']}")[:,:,::-1]; ch,cw=crop.shape[:2]; name=os.path.basename(r['crop']).split('__',1)[1]
    if r['orig']: frame=cv2.imread(f"{D}/{r['orig']}")[:,:,::-1]; box=[int(r[k]) for k in ('x1','y1','x2','y2')]
    elif name in fallback:  # original only exists in the segmentation export: locate crop there
        frame=cv2.imread(fallback[name])[:,:,::-1]; g=cv2.matchTemplate(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.TM_CCOEFF_NORMED)
        _,sc,_,(x,y)=cv2.minMaxLoc(g); box=[x,y,x+cw,y+ch]; r['score']=round(float(sc),4)
    else: log.append(dict(crop=r['crop'],status='sin original')); continue
    fh,fw=frame.shape[:2]; x=torch.from_numpy(((cv2.resize(frame,(W,H),interpolation=cv2.INTER_AREA)/255.-MEAN)/STD).transpose(2,0,1)).float()[None].to(dev)
    with torch.no_grad(),torch.autocast('cuda'):
        p=(torch.sigmoid(net(x))+torch.sigmoid(net(x.flip(-1))).flip(-1)[:,SWAP])/2
    P=np.stack([cv2.resize(c,(fw,fh)) for c in p[0].float().cpu().numpy()])[:,box[1]:box[3],box[0]:box[2]]
    M=np.stack([cv2.resize(c,(cw//2,ch//2),interpolation=cv2.INTER_AREA) for c in P]); M=(np.clip(M,0,1)*255).round().astype(np.uint8)
    out=f"{OUT}/{os.path.splitext(r['crop'].split('/',1)[1])[0]}.npz"; os.makedirs(os.path.dirname(out),exist_ok=True)
    np.savez_compressed(out,masks=M,classes=np.array(CLASSES),box=np.array(box),crop_hw=np.array([ch,cw]))
    cov=(M>127).mean((1,2)); log.append(dict(crop=r['crop'],status='ok',match=r['score'],**{f'cov_{c}':round(float(v),3) for c,v in zip(CLASSES,cov)}))
keys=sorted({k for l in log for k in l},key=lambda k:(k!='crop',k!='status',k))
with open(f'{OUT}/masks_log.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(log)
ok=[l for l in log if l['status']=='ok']; empty={c:sum(l[f'cov_{c}']==0 for l in ok) for c in CLASSES}
print(f'{len(ok)}/{len(rows)} masks saved | images with EMPTY region:',empty)
