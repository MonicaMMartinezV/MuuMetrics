# Print summary table of all CV configs and emit CLI args of the best one (by mIoU) to best_args.txt
import glob,json
R=[json.load(open(f)) for f in glob.glob('results/s[12]_*.json')]; R.sort(key=lambda d:-d['miou'])
with open('summary.txt','w') as f:
    f.write(f"{'config':24s} {'mIoU':>7s} {'±img':>6s}  "+' '.join(f'{c:>8s}' for c in R[0]['per_class'])+'   min\n')
    for d in R: f.write(f"{d['config']['tag']:24s} {d['miou']:7.4f} {d['miou_std_img']:6.3f}  "+' '.join(f'{v:8.3f}' for v in d['per_class'].values())+f"  {d['secs']/60:5.1f}\n")
c=R[0]['config']; keys=['enc','mode','enc_lr_mult','lr','h','w','iters','bs','loss','dec']
open('best_args.txt','w').write(' '.join(f'--{k} {c[k]}' for k in keys)); open('best_tag.txt','w').write(c['tag'])
print(open('summary.txt').read()); print('BEST',c['tag'],open('best_args.txt').read())
