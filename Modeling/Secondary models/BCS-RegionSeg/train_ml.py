# Multi-label (overlapping) region segmentation, leave-one-image-out CV
import sys,os,json,math,cv2,numpy as np,torch,torch.nn as nn,torch.nn.functional as F,timm
torch.manual_seed(0); np.random.seed(0)
ENC=sys.argv[1] if len(sys.argv)>1 else 'convnext_tiny.fb_in22k'; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 1500
FREEZE=len(sys.argv)>3 and sys.argv[3]=='freeze'; TAG=ENC.split('.')[0]+('_frozen' if FREEZE else '')
KW=dict(dynamic_img_size=True,out_indices=(2,5,8,11)) if 'dinov2' in ENC else {}
H,W=(378 if 'patch14' in ENC else 384),672; dev='cuda'
D=np.load('/w/data.npz'); IM,MK,NM=D['imgs'],D['masks'],list(D['names'])
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']; MEAN=np.array([.485,.456,.406]); STD=np.array([.229,.224,.225])

def augment(img,m):
    h,w=img.shape[:2]
    if np.random.rand()<.5:  # horizontal flip swaps left/right thighs
        img,m=img[:,::-1],m[:,:,::-1][[0,1,2,3,5,4]]
    s=np.exp(np.random.uniform(-.3,.3)); a=np.random.uniform(-20,20)
    A=cv2.getRotationMatrix2D((w/2+np.random.uniform(-.15,.15)*w,h/2+np.random.uniform(-.15,.15)*h),a,s)
    A[:,2]+=np.random.uniform(-.1,.1,2)*[w,h]
    img=cv2.warpAffine(np.ascontiguousarray(img),A,(w,h),borderMode=cv2.BORDER_REFLECT)
    m=np.stack([cv2.warpAffine(np.ascontiguousarray(c),A,(w,h),flags=cv2.INTER_NEAREST) for c in m])
    img=img.astype(np.float32)*np.random.uniform(.6,1.4)+np.random.uniform(-30,30)
    hsv=cv2.cvtColor(np.clip(img,0,255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[...,1]=np.clip(hsv[...,1]*np.random.uniform(.6,1.4),0,255); img=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB)
    if np.random.rand()<.3: img=cv2.GaussianBlur(img,(0,0),np.random.uniform(.5,2))
    return img,m

def to_t(img): return torch.from_numpy(((img/255.-MEAN)/STD).transpose(2,0,1)).float()
def rs(img,m): return cv2.resize(img,(W,H),interpolation=cv2.INTER_AREA),np.stack([cv2.resize(c,(W,H),interpolation=cv2.INTER_NEAREST) for c in m])

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.e=timm.create_model(ENC,pretrained=True,features_only=True,**KW)
        if FREEZE: s.e.requires_grad_(False)
        ch=s.e.feature_info.channels(); s.lat=nn.ModuleList([nn.Conv2d(c,128,1) for c in ch])
        s.head=nn.Sequential(nn.Conv2d(128*len(ch),128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.Conv2d(128,6,1))
    def forward(s,x):
        if FREEZE:
            s.e.eval()
            with torch.no_grad(): fs=s.e(x)
        else: fs=s.e(x)
        t=fs[0].shape[-2:]; p=[]; y=None
        for f,l in zip(fs[::-1],list(s.lat)[::-1]):   # top-down FPN
            z=l(f); y=z if y is None else z+F.interpolate(y,size=z.shape[-2:],mode='bilinear'); p.append(F.interpolate(y,size=t,mode='bilinear'))
        return F.interpolate(s.head(torch.cat(p,1)),size=x.shape[-2:],mode='bilinear')

def loss_fn(lo,y):
    p=torch.sigmoid(lo); inter=(p*y).sum((2,3)); dice=1-(2*inter+1)/(p.sum((2,3))+y.sum((2,3))+1)
    return F.binary_cross_entropy_with_logits(lo,y)+dice.mean()

def predict(net,img):
    x=to_t(cv2.resize(img,(W,H),interpolation=cv2.INTER_AREA))[None].to(dev)
    with torch.no_grad(),torch.autocast('cuda'):
        p=torch.sigmoid(net(x)); pf=torch.sigmoid(net(x.flip(-1))).flip(-1)[:,[0,1,2,3,5,4]]  # flip TTA
    return ((p+pf)/2)[0].float().cpu().numpy()

LR=1e-3 if FREEZE else 3e-4
PD='/w/pred_ml' if TAG=='convnext_tiny' else f'/w/pred_{TAG}'
res={}; os.makedirs(PD,exist_ok=True)
small=[rs(IM[i],MK[i]) for i in range(len(IM))]
for k in range(len(IM)):
    tr=[i for i in range(len(IM)) if i!=k]; net=Net().to(dev)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=LR,weight_decay=1e-4); sc=torch.amp.GradScaler()
    for it in range(ITERS):
        lr=LR*min(1,(it+1)/100)*0.5*(1+math.cos(math.pi*it/ITERS))
        for g in opt.param_groups: g['lr']=lr
        b=[augment(*small[np.random.choice(tr)]) for _ in range(4)]
        x=torch.stack([to_t(a) for a,_ in b]).to(dev); y=torch.from_numpy(np.stack([m for _,m in b])).float().to(dev)
        with torch.autocast('cuda'): l=loss_fn(net(x).float(),y)
        opt.zero_grad(); sc.scale(l).backward(); sc.step(opt); sc.update()
    net.eval(); P=predict(net,IM[k]); G=small[k][1]; pr=P>.5
    iou=[float((pr[c]&(G[c]>0)).sum()/max((pr[c]|(G[c]>0)).sum(),1)) for c in range(6)]
    res[NM[k]]=iou; np.save(f'{PD}/{k}.npy',P.astype(np.float16))
    print(NM[k],' '.join(f'{c}={v:.2f}' for c,v in zip(CLASSES,iou)),f'mIoU={np.mean(iou):.3f}',flush=True)
A=np.array(list(res.values())); print('MEAN per class:',' '.join(f'{c}={v:.3f}' for c,v in zip(CLASSES,A.mean(0))),f'| mIoU={A.mean():.3f}')
json.dump(res,open(f'/w/res_ml_{TAG}.json','w'),indent=1)
