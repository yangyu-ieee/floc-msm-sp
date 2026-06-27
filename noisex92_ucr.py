"""NOISEX-92 real impulsive noise → UCR benchmark.
Replaces synthetic alpha-stable with real impulsive recordings: machinegun, factory1, factory2.
Tests FLOC p=1.2 vs MSE vs L1 under REAL impulsive noise.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, time, os, json
from torch.utils.data import DataLoader, TensorDataset
from aeon.datasets import load_classification
import wave, random
device='cuda'; t0=time.time()

# ═══════ Load NOISEX-92 ═══════
NOISEX='D:/datasets/NOISEX-92'
IMPULSIVE_SOURCES=['machinegun.wav','factory1.wav','factory2.wav']
noise_bank=[]
for fname in IMPULSIVE_SOURCES:
    fp=os.path.join(NOISEX,fname)
    with wave.open(fp,'rb') as wf:
        nch=wf.getnchannels(); sw=wf.getsampwidth(); nf=wf.getnframes()
        raw=wf.readframes(nf)
        fmt={1:'int8',2:'int16',4:'int32'}[sw]
        data=np.frombuffer(raw,dtype=fmt).astype(np.float32)
        if nch>1: data=data.reshape(-1,nch).mean(axis=1)  # stereo->mono
        data/=(np.abs(data).max()+1e-8)
        noise_bank.append(data)
        print(f'  {fname}: {nch}ch, {sw}B, {nf} frames -> {len(data)} samples')
noise_len=min(len(n) for n in noise_bank)
noise_bank=np.stack([n[:noise_len] for n in noise_bank],axis=0)
print(f'NOISEX-92 impulsive: {len(IMPULSIVE_SOURCES)} sources, {noise_len} samples each')

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

# ═══════ Noise injection ═══════
def add_real_impulsive_noise(X, noise_bank, snr_db=5):
    """Add real impulsive noise from NOISEX-92 at given SNR."""
    n,L=X.shape; Xn=X.copy()
    for i in range(n):
        # Random noise source and start position
        src=random.randint(0,noise_bank.shape[0]-1)
        start=random.randint(0,noise_bank.shape[1]-L-1)
        noise=noise_bank[src,start:start+L]
        # Scale to target SNR
        sig_rms=np.sqrt(np.mean(X[i]**2)+1e-12)
        noise_rms=np.sqrt(np.mean(noise**2)+1e-12)
        scale=sig_rms*10**(-snr_db/20)/(noise_rms+1e-12)
        Xn[i]+=noise*scale
    Xn/=(np.abs(Xn).max(axis=1,keepdims=True)+1e-8)
    return Xn.astype(np.float32)

# ═══════ Pretraining ═══════
def pretrain(X_unlabeled, loss_type, epochs=20):
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
            recon=dec(enc.trf(masked).mean(dim=1)); r=recon-xb
            if loss_type=='mse': loss=r.pow(2).mean()
            elif loss_type=='l1': loss=r.abs().mean()
            elif loss_type=='floc': loss=r.abs().pow(1.2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
    return {k:v.clone() for k,v in enc.state_dict().items()}

# ═══════ Evaluation ═══════
def eval_cls(encoder_weights, X_train, y_train, X_test, y_test, shots, n_classes, epochs=50):
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
DATASETS=['ECG200','ECG5000','Wafer','GunPoint','CBF','FaceAll','SwedishLeaf','ChlorineConcentration']
SNR=5  # strong real impulsive noise
results={}

for ds_name in DATASETS:
    print(f'\n{ds_name}')
    try:
        X_train_np,y_train_np=load_classification(ds_name,split='train')
        X_test_np,y_test_np=load_classification(ds_name,split='test')
    except: continue
    X_train=X_train_np.squeeze().astype(np.float32); y_train=y_train_np.astype(np.int64)
    X_test=X_test_np.squeeze().astype(np.float32); y_test=y_test_np.astype(np.int64)
    # 0-index
    ul=np.unique(np.concatenate([y_train,y_test]))
    if ul[0]!=0 or ul[-1]!=len(ul)-1:
        m={l:i for i,l in enumerate(ul)}; y_train=np.array([m[l] for l in y_train]); y_test=np.array([m[l] for l in y_test])
    n_classes=len(ul)
    if len(X_train)<20: continue
    # Inject real impulsive noise
    X_train_n=add_real_impulsive_noise(X_train, noise_bank, SNR)
    X_test_n=add_real_impulsive_noise(X_test, noise_bank, SNR)
    # Augment to ~3K for pretraining
    n_aug=max(1,3000//len(X_train_n))
    X_pt=np.tile(X_train_n,(n_aug,1))+np.random.randn(n_aug*len(X_train_n),X_train_n.shape[1]).astype(np.float32)*0.02
    X_pt/=(np.abs(X_pt).max(axis=1,keepdims=True)+1e-8)
    # Pretrain
    w_mse=pretrain(X_pt,'mse',20); w_l1=pretrain(X_pt,'l1',20); w_floc=pretrain(X_pt,'floc',20)
    # Evaluate few-shot
    for shots in [5,10]:
        accs={lt:[] for lt in ['scratch','mse','l1','floc']}
        for seed in range(5):
            np.random.seed(seed); torch.manual_seed(seed)
            accs['scratch'].append(eval_cls(None,X_train_n,y_train,X_test_n,y_test,shots,n_classes,80))
            accs['mse'].append(eval_cls(w_mse,X_train_n,y_train,X_test_n,y_test,shots,n_classes,50))
            accs['l1'].append(eval_cls(w_l1,X_train_n,y_train,X_test_n,y_test,shots,n_classes,50))
            accs['floc'].append(eval_cls(w_floc,X_train_n,y_train,X_test_n,y_test,shots,n_classes,50))
        line=f'  {shots}shot:'
        for lt in ['scratch','mse','l1','floc']:
            m,s=np.mean(accs[lt]),np.std(accs[lt])
            results[f'{ds_name}/shots={shots}/loss={lt}']={'mean':float(m),'std':float(s),'seeds':[float(x) for x in accs[lt]]}
            line+=f' {lt}={m:.3f}'
        print(line)

# ── Summary ──
win_count={lt:0 for lt in ['scratch','mse','l1','floc']}; total=0
for ds_name in DATASETS:
    for shots in [5,10]:
        best_loss,best_acc='',0
        for lt in ['scratch','mse','l1','floc']:
            k=f'{ds_name}/shots={shots}/loss={lt}'
            if k in results and results[k]['mean']>best_acc:
                best_acc=results[k]['mean']; best_loss=lt
        if best_loss: win_count[best_loss]+=1
        total+=1
print(f'\nNOISEX-92 UCR win count ({total} conditions):')
for lt in ['scratch','mse','l1','floc']: print(f'  {lt}: {win_count[lt]}')

LOG='D:/deepl/paper2_floc_msm/logs/noisex92_ucr'
os.makedirs(LOG,exist_ok=True)
json.dump(results,open(f'{LOG}/results.json','w'),indent=2)
json.dump({'win_count':win_count,'SNR':SNR,'noise_sources':IMPULSIVE_SOURCES},open(f'{LOG}/summary.json','w'),indent=2)
print(f'\nSaved to {LOG} | {time.time()-t0:.0f}s')
