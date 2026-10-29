# Learning curve: mean CV mIoU (and per class) vs number of training images
import json,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
CLASSES=['loin','hips','rump','tailhead','thigh_L','thigh_R']
pts=[(n,json.load(open(f'/w/results/lc_n{n}.json'))) for n in (8,15,22)]+[(29.6,json.load(open('/w/results/s1_dinov2_ft.json')))]
x=[n for n,_ in pts]; y=[d['miou'] for _,d in pts]
se=[np.std([np.mean(v) for v in d['per_image'].values()])/np.sqrt(len(d['per_image'])) for _,d in pts]
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].errorbar(x,y,yerr=se,marker='o',lw=2,capsize=4,color='#1f4e79'); ax[0].set_xlabel('Imágenes de entrenamiento'); ax[0].set_ylabel('mIoU (5-fold CV)')
ax[0].set_title('Curva de aprendizaje — DINOv2 fine-tuned'); ax[0].grid(alpha=.3)
for xi,yi in zip(x,y): ax[0].annotate(f'{yi:.3f}',(xi,yi),textcoords='offset points',xytext=(0,8),ha='center')
for c in CLASSES: ax[1].plot(x,[d['per_class'][c] for _,d in pts],marker='o',label=c)
ax[1].set_xlabel('Imágenes de entrenamiento'); ax[1].set_ylabel('IoU'); ax[1].set_title('Por región'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('/w/figs/learning_curve.png',dpi=130)
print('LC',' '.join(f'n={n}:{v:.4f}' for n,v in zip(x,y)))
