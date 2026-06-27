"""TSP-level experiment: FLOC-MSM vs L1/Huber/Charbonnier/MSE under α-stable noise.

Tests:
1. All robust losses across α=2.0, 1.5, 1.0
2. p vs α ablation
3. Statistical significance (5 seeds)
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, time, os, json
from torch.utils.data import DataLoader, TensorDataset
device='cuda'; t0=time.time()

# ═══════════════════════ Model ═══════════════════════
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
        self.pe=PosEnc(d,300); self.d=d
        enc_layer=nn.TransformerEncoderLayer(d,heads,256,dropout=0.1,batch_first=True)
        self.trf=nn.TransformerEncoder(enc_layer,n_layers)
    def forward(self,x,cls=None):
        if x.dim()==2: x=self.pe(self.tok(x.unsqueeze(1)).transpose(1,2))
        if cls is not None: x=torch.cat([cls,x],dim=1)
        return self.trf(x)

# ═══════════════════════ Data ═══════════════════════
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
        if alpha<2.0:
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
        if alpha<2.0:
            U=np.random.uniform(-np.pi/2,np.pi/2,seq_len); W=np.random.exponential(1,seq_len)
            noise=np.sin(alpha*U)*(np.cos(U)/W)**((1-alpha)/alpha)/(np.cos(U)**(1/alpha))
            sig+=noise.astype(np.float32)*0.3
        X[i]=sig.astype(np.float32); y[i]=cls
    X/=(np.abs(X).max(axis=1,keepdims=True)+1e-8)
    return torch.from_numpy(X),torch.from_numpy(y)

# ═══════════════════════ Loss Functions ═══════════════════════

def loss_mse(residual): return residual.pow(2).mean()                  # MSE
def loss_l1(residual): return residual.abs().mean()                    # L1
def loss_huber(residual, delta=1.0):                                   # Huber
    abs_r=residual.abs()
    return torch.where(abs_r<=delta, 0.5*residual.pow(2), delta*(abs_r-0.5*delta)).mean()
def loss_charbonnier(residual, eps=0.01):                              # Charbonnier (sqrt(x²+eps²))
    return torch.sqrt(residual.pow(2)+eps**2).mean()
def loss_floc(residual, p): return residual.abs().pow(p).mean()        # FLOC

def pretrain_with_loss(data_loader, loss_fn_name, alpha=1.5, epochs=20, **kwargs):
    """Pre-train MSM with specified reconstruction loss."""
    enc=SignalEncoder().to(device)
    dec=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,128)).to(device)
    mask_tok=nn.Parameter(torch.randn(1,1,64,device=device)*0.02)

    params=list(enc.parameters())+list(dec.parameters())+[mask_tok]

    # Handle learnable params
    if loss_fn_name=='floc_learnable':
        p_raw=nn.Parameter(torch.tensor(math.log((1.5-0.5)/(2.0-1.5))), requires_grad=True)
        params.append(p_raw)

    if loss_fn_name=='huber_learnable':
        delta_raw=nn.Parameter(torch.tensor(0.0), requires_grad=True)  # log(delta)
        params.append(delta_raw)

    opt=torch.optim.Adam(params,lr=1e-3)
    p_vals=[]

    for _ in range(epochs):
        enc.train(); dec.train()
        for xb, in data_loader:
            xb=xb.to(device)
            tok=enc.pe(enc.tok(xb.unsqueeze(1)).transpose(1,2))
            mask=torch.rand(xb.shape[0],tok.shape[1],1,device=device)<0.5
            masked=torch.where(mask,mask_tok.expand(xb.shape[0],tok.shape[1],-1),tok)
            recon=dec(enc.trf(masked).mean(dim=1))
            r=recon-xb

            if loss_fn_name=='mse': loss=loss_mse(r)
            elif loss_fn_name=='l1': loss=loss_l1(r)
            elif loss_fn_name=='huber': loss=loss_huber(r)
            elif loss_fn_name=='charbonnier': loss=loss_charbonnier(r)
            elif loss_fn_name=='floc_1.8': loss=loss_floc(r,1.8)
            elif loss_fn_name=='floc_1.5': loss=loss_floc(r,1.5)
            elif loss_fn_name=='floc_1.2': loss=loss_floc(r,1.2)
            elif loss_fn_name=='floc_1.0': loss=loss_floc(r,1.0)
            elif loss_fn_name=='floc_learnable':
                p=0.5+1.5*torch.sigmoid(p_raw); loss=loss_floc(r,p)
            elif loss_fn_name=='huber_learnable':
                delta=torch.exp(delta_raw)+0.01; loss=loss_huber(r,delta)
            else: raise ValueError(loss_fn_name)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()

    result={kk:vv.clone() for kk,vv in enc.state_dict().items()}
    if loss_fn_name=='floc_learnable': result['_final_p']=p.item()
    return result

# ═══════════════════════ Evaluation ═══════════════════════
def eval_downstream(encoder_weights, n_train, n_test=500, n_classes=5, epochs=50, alpha=1.5):
    encoder=SignalEncoder().to(device)
    if encoder_weights:
        encoder.load_state_dict({k:v for k,v in encoder_weights.items() if not k.startswith('_')})
    cls_tok=nn.Parameter(torch.randn(1,1,64,device=device)*0.02)
    head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,n_classes)).to(device)
    opt=torch.optim.Adam(list(encoder.parameters())+[cls_tok]+list(head.parameters()),lr=1e-3)

    Xtr,ytr=gen_cls(n_train,128,alpha); Xte,yte=gen_cls(n_test,128,alpha)
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

# ═══════════════════════ Main ═══════════════════════
LOSSES=[
    'mse', 'l1', 'huber', 'charbonnier',
    'floc_1.8', 'floc_1.5', 'floc_1.2', 'floc_1.0',
    'floc_learnable',
]

results={}
N_LABELED=[50,100,200,500]
N_SEEDS=5

for alpha in [2.0, 1.5, 1.0]:
    print(f'\n{"="*70}')
    print(f'α={alpha}')
    print(f'{"="*70}')

    X_pt=gen_unlabeled(10000,128,alpha)
    pt_loader=DataLoader(TensorDataset(X_pt),batch_size=128,shuffle=True)

    weights={}
    for loss_name in LOSSES:
        print(f'  Pre-training with {loss_name}...')
        weights[loss_name]=pretrain_with_loss(pt_loader,loss_name,alpha,20)

    # Scratch baseline
    print(f'  Scratch baseline...')

    for n_labeled in N_LABELED:
        print(f'\n  n={n_labeled}:')
        for loss_name in ['scratch']+LOSSES:
            accs=[]
            for seed in range(N_SEEDS):
                torch.manual_seed(seed); np.random.seed(seed)
                if loss_name=='scratch':
                    a=eval_downstream(None,n_labeled,500,5,80,alpha=alpha)
                else:
                    a=eval_downstream(weights[loss_name],n_labeled,500,5,30 if n_labeled<=200 else 50,alpha=alpha)
                accs.append(a)
            m,s=np.mean(accs),np.std(accs)

            k=f'a={alpha}/loss={loss_name}/n={n_labeled}'
            results[k]={'mean':float(m),'std':float(s),'seeds':[float(x) for x in accs]}
            print(f'    {loss_name:<20}: {m:.3f}±{s:.3f}')

# ═══════════════════════ Summary ═══════════════════════
print(f'\n{"="*70}')
print('TSP SUMMARY: Robust Loss Comparison')
print(f'{"="*70}')

for alpha in [2.0,1.5,1.0]:
    print(f'\nα={alpha}:')
    header=f'  {"Loss":<20}'
    for nl in N_LABELED: header+=f' {"n="+str(nl):>12}'
    print(header)
    print('  '+'-'*68)
    for loss_name in ['scratch','mse','l1','huber','charbonnier','floc_1.5','floc_1.2','floc_1.0','floc_learnable']:
        row=f'  {loss_name:<20}'
        for nl in N_LABELED:
            v=results[f'a={alpha}/loss={loss_name}/n={nl}']['mean']
            row+=f' {v:12.3f}'
        print(row)

# Find best loss per condition
print(f'\n{"="*70}')
print('BEST LOSS PER CONDITION')
print(f'{"="*70}')
for alpha in [2.0,1.5,1.0]:
    for nl in N_LABELED:
        best_loss,best_acc='',0
        for loss_name in LOSSES:
            v=results[f'a={alpha}/loss={loss_name}/n={nl}']['mean']
            if v>best_acc: best_acc=v; best_loss=loss_name
        print(f'  α={alpha} n={nl:>3}: {best_loss:<20} @ {best_acc:.3f}')

SAVE=os.environ.get('FLOC_MSM_SAVE','D:/codex/paper2_floc_msm_work/tsp_loss_compare_fixed')
os.makedirs(SAVE,exist_ok=True)
json.dump(results,open(f'{SAVE}/results.json','w'),indent=2)
print(f'\nSaved to {SAVE} | {time.time()-t0:.0f}s')
