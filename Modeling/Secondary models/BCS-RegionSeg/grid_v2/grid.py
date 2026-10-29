# Multi-label (overlapping) BCS region segmentation: one config, K-fold CV on train+valid (or --final: fit all, eval test)
import argparse,json,math,os,time,cv2,numpy as np,torch,torch.nn as nn,torch.nn.functional as F,timm
ap=argparse.ArgumentParser()
ap.add_argument('--enc',default='convnext_tiny.fb_in22k'); ap.add_argument('--mode',default='finetune',choices=['finetune','frozen'])
ap.add_argument('--enc_lr_mult',type=float,default=1.0); ap.add_argument('--lr',type=float,default=3e-4)
ap.add_argument('--h',type=int,default=384); ap.add_argument('--w',type=int,default=672)
ap.add_argument('--iters',type=int,default=2000); ap.add_argument('--bs',type=int,default=4)
ap.add_argument('--loss',default='bcedice',choices=['bcedice','focaldice','dice']); ap.add_argument('--dec',type=int,default=128)
ap.add_argument('--folds',type=int,default=5); ap.add_argument('--final',action='store_true'); ap.add_argument('--all',action='store_true'); ap.add_argument('--ntrain',type=int,default=0); ap.add_argument('--tag',required=True)
a=ap.parse_args(); dev='cuda'
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']; SWAP=[0,1,2,3,5,4]
MEAN=np.array([.485,.456,.406]); STD=np.array([.229,.224,.225])
D=np.load('/w/data.npz'); IM,MK,NM,SP=D['imgs'],D['masks'],list(D['names']),list(D['splits'])
KW=dict(dynamic_img_size=True) if 'vit_' in a.enc else {}
if 'dinov2' in a.enc: KW['out_indices']=(2,5,8,11)

def augment(img,m):
    h,w=img.shape[:2]
    if np.random.rand()<.5: img,m=img[:,::-1],m[:,:,::-1][SWAP]   # flip swaps left/right thighs
    s=np.exp(np.random.uniform(-.3,.3)); ang=np.random.uniform(-20,20)
    A=cv2.getRotationMatrix2D((w/2+np.random.uniform(-.15,.15)*w,h/2+np.random.uniform(-.15,.15)*h),ang,s)
    A[:,2]+=np.random.uniform(-.1,.1,2)*[w,h]
    img=cv2.warpAffine(np.ascontiguousarray(img),A,(w,h),borderMode=cv2.BORDER_REFLECT)
    m=np.stack([cv2.warpAffine(np.ascontiguousarray(c),A,(w,h),flags=cv2.INTER_NEAREST) for c in m])
    img=img.astype(np.float32)*np.random.uniform(.6,1.4)+np.random.uniform(-30,30)
    hsv=cv2.cvtColor(np.clip(img,0,255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1]=np.clip(hsv[...,1]*np.random.uniform(.6,1.4),0,255); img=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB)
    if np.random.rand()<.3: img=cv2.GaussianBlur(img,(0,0),np.random.uniform(.5,2))
    return img,m

def to_t(img): return torch.from_numpy(((img/255.-MEAN)/STD).transpose(2,0,1)).float()
def rs(img,m): return cv2.resize(img,(a.w,a.h),interpolation=cv2.INTER_AREA),np.stack([cv2.resize(c,(a.w,a.h),interpolation=cv2.INTER_NEAREST) for c in m])

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.e=timm.create_model(a.enc,pretrained=True,features_only=True,**KW)
        if a.mode=='frozen': s.e.requires_grad_(False)
        elif 'vit_' in a.enc:
            try: s.e.set_grad_checkpointing(True)
            except Exception: pass
        ch=s.e.feature_info.channels(); s.lat=nn.ModuleList([nn.Conv2d(c,a.dec,1) for c in ch])
        s.head=nn.Sequential(nn.Conv2d(a.dec*len(ch),a.dec,3,padding=1),nn.BatchNorm2d(a.dec),nn.ReLU(),nn.Conv2d(a.dec,6,1))
    def forward(s,x):
        if a.mode=='frozen':
            s.e.eval()
            with torch.no_grad(): fs=s.e(x)
        else: fs=s.e(x)
        t=fs[0].shape[-2:]; p=[]; y=None
        for f,l in zip(fs[::-1],list(s.lat)[::-1]):   # top-down FPN
            z=l(f); y=z if y is None else z+F.interpolate(y,size=z.shape[-2:],mode='bilinear'); p.append(F.interpolate(y,size=t,mode='bilinear'))
        return F.interpolate(s.head(torch.cat(p,1)),size=x.shape[-2:],mode='bilinear')

def loss_fn(lo,y):
    p=torch.sigmoid(lo); inter=(p*y).sum((2,3)); dice=(1-(2*inter+1)/(p.sum((2,3))+y.sum((2,3))+1)).mean()
    if a.loss=='dice': return dice
    if a.loss=='focaldice':
        bce=F.binary_cross_entropy_with_logits(lo,y,reduction='none'); pt=torch.exp(-bce)
        return (0.25*(1-pt)**2*bce).mean()*10+dice
    return F.binary_cross_entropy_with_logits(lo,y)+dice

def predict(net,img):
    x=to_t(cv2.resize(img,(a.w,a.h),interpolation=cv2.INTER_AREA))[None].to(dev)
    with torch.no_grad(),torch.autocast('cuda'):
        p=torch.sigmoid(net(x)); pf=torch.sigmoid(net(x.flip(-1))).flip(-1)[:,SWAP]  # flip TTA
    P=((p+pf)/2)[0].float().cpu().numpy(); H,W=img.shape[:2]
    return np.stack([cv2.resize(c,(W,H)) for c in P])

def fit(tr):
    torch.manual_seed(0); np.random.seed(0); net=Net().to(dev)
    enc=[p for p in net.e.parameters() if p.requires_grad]; dec=[p for n,p in net.named_parameters() if not n.startswith('e.')]
    groups=[{'params':dec,'lr0':a.lr}]+([{'params':enc,'lr0':a.lr*a.enc_lr_mult}] if enc else [])
    opt=torch.optim.AdamW(groups,lr=a.lr,weight_decay=1e-4); sc=torch.amp.GradScaler(); small=[rs(IM[i],MK[i]) for i in tr]
    for it in range(a.iters):
        f=min(1,(it+1)/100)*0.5*(1+math.cos(math.pi*it/a.iters))
        for g in opt.param_groups: g['lr']=g['lr0']*f
        b=[augment(*small[np.random.randint(len(small))]) for _ in range(a.bs)]
        x=torch.stack([to_t(i) for i,_ in b]).to(dev); y=torch.from_numpy(np.stack([m for _,m in b])).float().to(dev)
        with torch.autocast('cuda'): out=net(x)
        l=loss_fn(out.float(),y); opt.zero_grad(); sc.scale(l).backward(); sc.step(opt); sc.update()
    return net.eval()

def iou(P,G,th=.5): pr=P>th; g=G>0; return [float((pr[c]&g[c]).sum()/max((pr[c]|g[c]).sum(),1)) for c in range(6)]

t0=time.time(); os.makedirs('/w/results',exist_ok=True); os.makedirs('/w/oof',exist_ok=True)
pool=[i for i,s in enumerate(SP) if s!='test']; test=[i for i,s in enumerate(SP) if s=='test']
if a.all: splits=[(pool+test,[])]
elif a.final: splits=[(pool,test)]
else:
    rng=np.random.RandomState(42); perm=rng.permutation(pool); splits=[([i for i in pool if i not in set(perm[k::a.folds])],sorted(perm[k::a.folds])) for k in range(a.folds)]
res={}; oof={}
for k,(tr,va) in enumerate(splits):
    if a.ntrain: tr=sorted(np.random.RandomState(100+k).choice(tr,a.ntrain,replace=False))  # learning-curve subsample
    net=fit(tr)
    for i in va:
        P=predict(net,IM[i]); res[NM[i]]=iou(P,MK[i]); oof[NM[i]]=np.stack([cv2.resize(c,(480,270)) for c in P]).astype(np.float16)
    print(f'[{a.tag}] fold {k} done, running mIoU={np.mean(list(res.values())):.3f} ({time.time()-t0:.0f}s)',flush=True)
    if a.final or a.all: torch.save(net.state_dict(),f'/w/results/{a.tag}.pt')
    del net; torch.cuda.empty_cache()
if a.all: print(f'RESULT {a.tag} trained on {len(pool+test)} imgs (no eval) | {(time.time()-t0)/60:.1f} min'); raise SystemExit
A=np.array(list(res.values()))
out=dict(config=vars(a),per_image=res,per_class=dict(zip(CLASSES,A.mean(0).round(4).tolist())),miou=float(A.mean()),miou_std_img=float(A.mean(1).std()),secs=time.time()-t0)
json.dump(out,open(f'/w/results/{a.tag}.json','w'),indent=1); np.savez_compressed(f'/w/oof/{a.tag}.npz',**oof)
print(f'RESULT {a.tag} mIoU={out["miou"]:.4f} ±{out["miou_std_img"]:.3f} | '+' '.join(f'{c}={v:.3f}' for c,v in out['per_class'].items())+f' | {out["secs"]/60:.1f} min',flush=True)
