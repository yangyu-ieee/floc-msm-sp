"""UCR all-loss comparison: MSE vs L1 vs Huber vs Charbonnier vs FLOC.

Parameterized runner for real-data robustness checks. The original experiment
used alpha=1.5; journal claims need explicit alpha/scale provenance.
"""
import argparse
import hashlib
import sys
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, time, os, json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from aeon.datasets import load_classification
from stable_noise import draw_cms_variates, from_cms_variates
device='cuda' if torch.cuda.is_available() else 'cpu'; t0=time.time()

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

parser=argparse.ArgumentParser()
parser.add_argument('--alpha',type=float,default=1.5)
parser.add_argument('--scale',type=float,default=0.30)
parser.add_argument('--seeds',type=int,default=5)
parser.add_argument('--shots',type=str,default='5,10')
parser.add_argument('--datasets',type=str,default='')
parser.add_argument('--pt-epochs',type=int,default=20)
parser.add_argument('--ft-epochs',type=int,default=50)
parser.add_argument('--scratch-epochs',type=int,default=80)
parser.add_argument('--log-dir',type=str,default='')
parser.add_argument('--normalization',choices=['maxabs','clean_global_rms'],default='maxabs')
parser.add_argument('--seed',type=int,default=20261007)
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
if args.smoke:
    args.datasets='ECG200'; args.seeds=1; args.shots='5'; args.pt_epochs=1; args.ft_epochs=1; args.scratch_epochs=1
LOG=Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parents[1]/'logs'/f"ucr_cms_{args.normalization}"
LOG.mkdir(parents=True,exist_ok=True)

# ═══════ Model ═══════
class PosEnc(nn.Module):
    def __init__(self,d,max_len=400):
        super().__init__()
        pe=torch.zeros(max_len,d); pos=torch.arange(0,max_len).unsqueeze(1).float()
        dv=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*dv); pe[:,1::2]=torch.cos(pos*dv)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1),:]

class SignalEncoder(nn.Module):
    def __init__(self,d=64,heads=4,n_layers=4):
        super().__init__()
        self.tok=nn.Conv1d(1,d,8,stride=4,padding=2)
        self.pe=PosEnc(d,400); self.d=d
        enc_layer=nn.TransformerEncoderLayer(d,heads,256,dropout=0.1,batch_first=True)
        self.trf=nn.TransformerEncoder(enc_layer,n_layers)
    def forward(self,x,cls=None):
        if x.dim()==2: x=self.pe(self.tok(x.unsqueeze(1)).transpose(1,2))
        if cls is not None: x=torch.cat([cls,x],dim=1)
        return self.trf(x)

# ═══════ Data ═══════
def add_alpha_noise(X,alpha=1.5,scale=0.30,normalization='maxabs',clean_train_rms=None):
    X=np.asarray(X,dtype=np.float64)
    u,w=draw_cms_variates(X.shape,np.random)
    noise=from_cms_variates(alpha,u,w,scale=scale)
    if normalization=='maxabs':
        Xn=X+noise
        Xn/=(np.max(np.abs(Xn),axis=1,keepdims=True)+1e-8)
    elif normalization=='clean_global_rms':
        if clean_train_rms is None or not np.isfinite(clean_train_rms) or clean_train_rms<=0:
            raise ValueError('clean_global_rms requires a positive training-only RMS')
        Xn=X/clean_train_rms+noise
    else:
        raise ValueError(normalization)
    if not np.isfinite(Xn).all(): raise FloatingPointError('non-finite noisy UCR data')
    return Xn.astype(np.float32)

# ═══════ Pre-training ═══════
def pretrain_msm(X_unlabeled, loss_type, epochs=20):
    Xt=torch.from_numpy(X_unlabeled)
    dl=DataLoader(TensorDataset(Xt),batch_size=min(128,len(Xt)),shuffle=True)
    enc=SignalEncoder().to(device)
    dec=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,X_unlabeled.shape[1])).to(device)
    mask_tok=nn.Parameter(torch.randn(1,1,64,device=device)*0.02)
    opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters())+[mask_tok],lr=1e-3)

    for _ in range(epochs):
        enc.train(); dec.train()
        for xb, in dl:
            xb=xb.to(device)
            tok=enc.pe(enc.tok(xb.unsqueeze(1)).transpose(1,2))
            mask=torch.rand(xb.shape[0],tok.shape[1],1,device=device)<0.5
            masked=torch.where(mask,mask_tok.expand(xb.shape[0],tok.shape[1],-1),tok)
            recon=dec(enc.trf(masked).mean(dim=1))
            r=recon-xb
            if loss_type=='mse': loss=r.pow(2).mean()
            elif loss_type=='l1': loss=r.abs().mean()
            elif loss_type=='huber': loss=F.huber_loss(recon,xb,delta=1.0)
            elif loss_type=='charbonnier': loss=torch.sqrt(r.pow(2)+0.01**2).mean()
            elif loss_type=='floc': loss=r.abs().pow(1.2).mean()
            else: raise ValueError(loss_type)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
    return {k:v.clone() for k,v in enc.state_dict().items()}

# ═══════ Evaluation ═══════
def eval_cls(encoder_weights,X_train,y_train,X_test,y_test,shots,n_classes,epochs=50):
    indices=[]
    for c in range(n_classes):
        c_idx=np.where(y_train==c)[0]
        indices.append(np.random.choice(c_idx,min(shots,len(c_idx)),replace=False))
    idx=np.concatenate(indices); np.random.shuffle(idx)

    Xl=torch.from_numpy(X_train[idx]); yl=torch.from_numpy(y_train[idx])
    Xte=torch.from_numpy(X_test); yte=torch.from_numpy(y_test)
    tl=DataLoader(TensorDataset(Xl,yl),batch_size=min(32,len(Xl)),shuffle=True)

    enc=SignalEncoder().to(device)
    if encoder_weights: enc.load_state_dict({k:v for k,v in encoder_weights.items() if not k.startswith('_')})
    cls_tok=nn.Parameter(torch.randn(1,1,64,device=device)*0.02)
    head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,n_classes)).to(device)
    opt=torch.optim.Adam(list(enc.parameters())+[cls_tok]+list(head.parameters()),lr=1e-3)

    for _ in range(epochs):
        enc.train(); head.train()
        for xb,yb in tl:
            xb,yb=xb.to(device),yb.to(device)
            c=cls_tok.expand(xb.shape[0],1,-1)
            loss=F.cross_entropy(head(enc(xb,cls=c)[:,0,:]),yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        c=cls_tok.expand(len(Xte),1,-1)
        pred=head(enc(Xte.to(device),cls=c)[:,0,:]).cpu().argmax(1)
    return (pred==yte).float().mean().item()

# ═══════ Main ═══════
DEFAULT_DATASETS=['ECG200','ECG5000','Wafer','FordA','GunPoint','CBF','FaceAll','SwedishLeaf','ItalyPowerDemand','ChlorineConcentration']
DATASETS=[x.strip() for x in args.datasets.split(',') if x.strip()] if args.datasets else DEFAULT_DATASETS
LOSSES=['scratch','mse','l1','huber','charbonnier','floc']
SHOTS=[int(x.strip()) for x in args.shots.split(',') if x.strip()]
N_SEEDS=args.seeds
results={}
dataset_diagnostics={}
config={
    'experiment':'UCR corrected-CMS robustness check', 'alpha':args.alpha, 'scale':args.scale,
    'generator':'symmetric CMS S1; exp(-|scale*t|^alpha)', 'normalization':args.normalization,
    'normalization_reference':'clean training split only' if args.normalization=='clean_global_rms' else 'each corrupted sequence',
    'datasets':DATASETS, 'losses':LOSSES, 'shots':SHOTS, 'n_seeds':N_SEEDS,
    'pt_epochs':args.pt_epochs, 'ft_epochs':args.ft_epochs, 'scratch_epochs':args.scratch_epochs,
    'seed':args.seed, 'device':device, 'torch':torch.__version__, 'numpy':np.__version__,
    'python':sys.version, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(LOG/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')

print(f'Device={device} | alpha={args.alpha} | scale={args.scale} | seeds={N_SEEDS} | shots={SHOTS}')
print(f'Datasets={DATASETS}')

for ds_i,ds_name in enumerate(DATASETS):
    print(f'\n{"="*50}')
    print(f'{ds_name}')
    try:
        X_train_np,y_train_np=load_classification(ds_name,split='train')
        X_test_np,y_test_np=load_classification(ds_name,split='test')
    except Exception as e:
        print(f'  Skip: {e}'); continue

    X_train=X_train_np.squeeze().astype(np.float32); y_train=y_train_np.astype(np.int64)
    X_test=X_test_np.squeeze().astype(np.float32); y_test=y_test_np.astype(np.int64)

    unique_lbls=np.unique(np.concatenate([y_train,y_test]))
    if unique_lbls[0]!=0 or unique_lbls[-1]!=len(unique_lbls)-1:
        lbl_map={l:i for i,l in enumerate(unique_lbls)}
        y_train=np.array([lbl_map[l] for l in y_train]); y_test=np.array([lbl_map[l] for l in y_test])
    n_classes=len(unique_lbls)
    if len(X_train)<20: continue

    clean_train_rms=float(np.sqrt(np.mean(X_train.astype(np.float64)**2)))
    np.random.seed(args.seed+20000+ds_i*2)
    X_train_n=add_alpha_noise(X_train,args.alpha,args.scale,args.normalization,clean_train_rms)
    np.random.seed(args.seed+20001+ds_i*2)
    X_test_n=add_alpha_noise(X_test,args.alpha,args.scale,args.normalization,clean_train_rms)
    dataset_diagnostics[ds_name]={
        'train_shape':list(X_train.shape),'test_shape':list(X_test.shape),'n_classes':n_classes,
        'clean_train_global_rms':clean_train_rms,'noisy_train_abs_q99':float(np.quantile(np.abs(X_train_n),0.99)),
        'noisy_test_abs_q99':float(np.quantile(np.abs(X_test_n),0.99)),
    }

    n_aug=max(1,3000//len(X_train_n))
    np.random.seed(args.seed+30000+ds_i)
    X_pt=np.tile(X_train_n,(n_aug,1))+np.random.randn(n_aug*len(X_train_n),X_train_n.shape[1]).astype(np.float32)*0.02
    if args.normalization=='maxabs':
        X_pt/=(np.abs(X_pt).max(axis=1,keepdims=True)+1e-8)

    weights={}
    for lt in ['mse','l1','huber','charbonnier','floc']:
        print(f'  PT {lt}...')
        torch.manual_seed(args.seed+40000+ds_i)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed+40000+ds_i)
        weights[lt]=pretrain_msm(X_pt,lt,args.pt_epochs)

    for shots in SHOTS:
        line=f'  {shots}shot:'
        for lt in LOSSES:
            accs=[]
            for seed in range(N_SEEDS):
                np.random.seed(seed); torch.manual_seed(seed)
                if lt=='scratch': a=eval_cls(None,X_train_n,y_train,X_test_n,y_test,shots,n_classes,args.scratch_epochs)
                else: a=eval_cls(weights[lt],X_train_n,y_train,X_test_n,y_test,shots,n_classes,args.ft_epochs)
                accs.append(a)
            m,s=np.mean(accs),np.std(accs)
            results[f'{ds_name}/shots={shots}/loss={lt}']={'mean':float(m),'std':float(s),'seeds':[float(x) for x in accs]}
            line+=f' {lt}={m:.3f}'
            json.dump(results,open(LOG/'results_partial.json','w'),indent=2)
        print(line)
    (LOG/'dataset_diagnostics.json').write_text(json.dumps(dataset_diagnostics,indent=2),encoding='utf-8')

# Summary
win_count={lt:0 for lt in LOSSES}
total=0
for ds_name in DATASETS:
    for shots in SHOTS:
        best_loss,best_acc='',-1
        for lt in LOSSES:
            k=f'{ds_name}/shots={shots}/loss={lt}'
            if k in results and results[k]['mean']>best_acc:
                best_acc=results[k]['mean']; best_loss=lt
        if best_loss: win_count[best_loss]+=1
        total+=1

print(f'\nWin count ({total} conditions):')
for lt in LOSSES: print(f'  {lt:<15}: {win_count[lt]}')
for lt in LOSSES: print(f'  {lt:<15}: {win_count[lt]/total:.1%}')

json.dump(results,open(LOG/'results.json','w'),indent=2)
json.dump({
    'win_count':win_count,
    'total':total,
    'losses':LOSSES,
    'datasets':DATASETS,
    'alpha':args.alpha,
    'scale':args.scale,
    'shots':SHOTS,
    'n_seeds':N_SEEDS,
    'pt_epochs':args.pt_epochs,
    'ft_epochs':args.ft_epochs,
    'scratch_epochs':args.scratch_epochs,
    'device':device,'normalization':args.normalization,'generator':'symmetric CMS S1',
    'seed':args.seed,'config':config
},open(LOG/'summary.json','w'),indent=2)
json.dump({'elapsed_seconds':time.time()-t0},open(LOG/'run_metadata.json','w'),indent=2)
print(f'\nDone | {LOG} | {time.time()-t0:.0f}s')
