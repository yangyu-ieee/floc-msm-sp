"""10-seed main experiment — key losses only (MSE/L1/Huber/FLOC p=1.2)."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, time, os, json
from torch.utils.data import DataLoader, TensorDataset
device='cuda'; t0=time.time()

class PosEnc(nn.Module):
    def __init__(self,d,max_len=200):
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
        self.pe=PosEnc(d,200); self.d=d
        enc_layer=nn.TransformerEncoderLayer(d,heads,256,dropout=0.1,batch_first=True)
        self.trf=nn.TransformerEncoder(enc_layer,n_layers)
    def forward(self,x,cls=None):
        if x.dim()==2: x=self.pe(self.tok(x.unsqueeze(1)).transpose(1,2))
        if cls is not None: x=torch.cat([cls,x],dim=1)
        return self.trf(x)

def gen_unlabeled(n,seq_len=128,alpha=1.5):
    t=np.arange(seq_len); X=np.zeros((n,seq_len),dtype=np.float32)
    for i in range(n):
        st=np.random.randint(0,5)
        if st==0: sig=np.sin(2*np.pi*np.random.uniform(0.01,0.2)*t)+0.5*np.sin(2*np.pi*np.random.uniform(0.01,0.15)*t)
        elif st==1: f0,f1=np.random.uniform(0.01,0.1,2); sig=np.sin(2*np.pi*(f0+(f1-f0)*t/seq_len)*t)
        elif st==2:
            period=np.random.randint(10,40); sig=np.zeros(seq_len)
            for j in range(0,seq_len,period): sig[j:j+2]=np.random.uniform(0.5,1.5)
        elif st==3: tau=np.random.uniform(20,80); sig=np.sin(2*np.pi*np.random.uniform(0.03,0.15)*t)*np.exp(-t/tau)
        else: sig=np.cumsum(np.random.randn(seq_len)*0.1)
        sig+=0.02*np.random.randn(seq_len)
        U=np.random.uniform(-np.pi/2,np.pi/2,seq_len); W=np.random.exponential(1,seq_len)
        noise=np.sin(alpha*U)*(np.cos(U)/W)**((1-alpha)/alpha)/(np.cos(U)**(1/alpha))
        sig+=noise.astype(np.float32)*0.15
        X[i]=sig.astype(np.float32)
    X/=(np.abs(X).max(axis=1,keepdims=True)+1e-8)
    return torch.from_numpy(X)

def gen_cls(n,seq_len=128,alpha=1.5):
    t=np.arange(seq_len); X=np.zeros((n,seq_len),dtype=np.float32); y=np.zeros(n,dtype=np.int64)
    for i in range(n):
        cls=i%5
        if cls==0: sig=np.sin(2*np.pi*0.02*t)+0.3*np.sin(2*np.pi*0.05*t)
        elif cls==1: sig=np.sin(2*np.pi*0.08*t)+0.5*np.sin(2*np.pi*0.15*t)
        elif cls==2: sig=np.sign(np.sin(2*np.pi*0.03*t))
        elif cls==3: sig=2.0*(t%20)/20.0-1.0
        else:
            sig=np.zeros(seq_len)
            for j in range(0,seq_len,25): sig[j:j+2]=1.0
        sig+=0.05*np.random.randn(seq_len)
        U=np.random.uniform(-np.pi/2,np.pi/2,seq_len); W=np.random.exponential(1,seq_len)
        noise=np.sin(alpha*U)*(np.cos(U)/W)**((1-alpha)/alpha)/(np.cos(U)**(1/alpha))
        sig+=noise.astype(np.float32)*0.3
        X[i]=sig.astype(np.float32); y[i]=cls
    X/=(np.abs(X).max(axis=1,keepdims=True)+1e-8)
    return torch.from_numpy(X),torch.from_numpy(y)

def pretrain(data_loader, loss_type, epochs=20):
    enc=SignalEncoder().to(device)
    dec=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,128)).to(device)
    mask_tok=nn.Parameter(torch.randn(1,1,64,device=device)*0.02)
    opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters())+[mask_tok],lr=1e-3)
    for _ in range(epochs):
        enc.train(); dec.train()
        for xb, in data_loader:
            xb=xb.to(device)
            tok=enc.pe(enc.tok(xb.unsqueeze(1)).transpose(1,2))
            mask=torch.rand(xb.shape[0],tok.shape[1],1,device=device)<0.5
            masked=torch.where(mask,mask_tok.expand(xb.shape[0],tok.shape[1],-1),tok)
            recon=dec(enc.trf(masked).mean(dim=1)); r=recon-xb
            if loss_type=='mse': loss=r.pow(2).mean()
            elif loss_type=='l1': loss=r.abs().mean()
            elif loss_type=='huber': loss=F.huber_loss(recon,xb,delta=1.0)
            elif loss_type=='floc': loss=r.abs().pow(1.2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
    return {k:v.clone() for k,v in enc.state_dict().items()}

def evaluate(encoder_weights, n_train, n_test=500, epochs=50):
    encoder=SignalEncoder().to(device)
    if encoder_weights: encoder.load_state_dict({k:v for k,v in encoder_weights.items() if not k.startswith('_')})
    cls_tok=nn.Parameter(torch.randn(1,1,64,device=device)*0.02)
    head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,5)).to(device)
    opt=torch.optim.Adam(list(encoder.parameters())+[cls_tok]+list(head.parameters()),lr=1e-3)
    Xtr,ytr=gen_cls(n_train,128,1.5); Xte,yte=gen_cls(n_test,128,1.5)
    tl=DataLoader(TensorDataset(Xtr,ytr),batch_size=min(32,n_train),shuffle=True)
    for _ in range(epochs):
        encoder.train(); head.train()
        for xb,yb in tl:
            xb,yb=xb.to(device),yb.to(device)
            c=cls_tok.expand(xb.shape[0],1,-1)
            loss=F.cross_entropy(head(encoder(xb,cls=c)[:,0,:]),yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.0); opt.step()
    encoder.eval(); head.eval()
    with torch.no_grad():
        c=cls_tok.expand(n_test,1,-1)
        pred=head(encoder(Xte.to(device),cls=c)[:,0,:]).cpu().argmax(1)
    return (pred==yte).float().mean().item()

# ── Main ──
N_SEEDS=10; LOSSES=['mse','l1','huber','floc']; LABELS=[50,100,200,500]
X_pt=gen_unlabeled(10000,128,1.5)
pt_loader=DataLoader(TensorDataset(X_pt),batch_size=128,shuffle=True)

all_results={}
for loss_type in LOSSES:
    print(f'\nPre-training ({loss_type})...')
    weights=pretrain(pt_loader, loss_type, 20)
    all_results[loss_type]={}
    for n_labeled in LABELS:
        accs=[]
        epochs_ft=30 if n_labeled<=200 else 50
        for seed in range(N_SEEDS):
            torch.manual_seed(seed); np.random.seed(seed)
            a=evaluate(weights, n_labeled, 500, epochs_ft)
            accs.append(a)
        m,s=np.mean(accs),np.std(accs)
        all_results[loss_type][n_labeled]={'mean':float(m),'std':float(s),'seeds':[float(x) for x in accs]}
        print(f'  n={n_labeled:>3}: {m:.3f}±{s:.3f}')

# Scratch
print('\nScratch...')
all_results['scratch']={}
for n_labeled in LABELS:
    accs=[]
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        a=evaluate(None, n_labeled, 500, 80)
        accs.append(a)
    m,s=np.mean(accs),np.std(accs)
    all_results['scratch'][n_labeled]={'mean':float(m),'std':float(s),'seeds':[float(x) for x in accs]}
    print(f'  n={n_labeled:>3}: {m:.3f}±{s:.3f}')

# ── Statistical phase map ──
print(f'\n=== Statistical Phase Map ===')
from scipy import stats
for n_labeled in LABELS:
    print(f'\nn={n_labeled}:')
    # Find best
    best_loss,best_mean='',-1
    for lt in LOSSES:
        m=all_results[lt][n_labeled]['mean']
        if m>best_mean: best_mean=m; best_loss=lt
    # Pairwise vs best
    best_seeds=all_results[best_loss][n_labeled]['seeds']
    for lt in LOSSES:
        if lt==best_loss: continue
        lt_seeds=all_results[lt][n_labeled]['seeds']
        t,p=stats.ttest_rel(best_seeds, lt_seeds)
        d=best_mean-all_results[lt][n_labeled]['mean']
        sig='***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
        print(f'  {best_loss} vs {lt}: Δ={d:+.4f} p={p:.4f} {sig}')

LOG='D:/deepl/paper2_floc_msm/logs/floc_10seed'
os.makedirs(LOG,exist_ok=True)
json.dump(all_results,open(f'{LOG}/results.json','w'),indent=2)
print(f'\nSaved to {LOG} | {time.time()-t0:.0f}s')
