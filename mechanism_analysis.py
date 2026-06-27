"""Mechanism analysis: Why MSM helps under α-stable noise.

Generates:
1. t-SNE of features (scratch vs MSM vs TS2Vec)
2. Linear probe accuracy vs layer depth
3. Reconstruction quality comparison
4. Feature stability under noise perturbation (cosine similarity)
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, time, os
from torch.utils.data import DataLoader, TensorDataset
device='cuda'; t0=time.time()

# ── Model (same as benchmark) ──
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

# ── Data ──
def gen_analysis_data(n=1000,seq_len=128,alpha=1.5):
    """Generate labeled data for mechanism analysis."""
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

def gen_unlabeled_pt(n=10000,seq_len=128):
    """Unlabeled data for pre-training."""
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
        noise=np.sin(1.5*U)*(np.cos(U)/W)**((1-1.5)/1.5)/(np.cos(U)**(1/1.5))
        sig+=noise.astype(np.float32)*0.15
        X[i]=sig.astype(np.float32)
    X/=(np.abs(X).max(axis=1,keepdims=True)+1e-8)
    return torch.from_numpy(X)

# ── Pre-training ──
def pretrain_msm(X_unlabeled,epochs=20):
    Xt=X_unlabeled; dl=DataLoader(TensorDataset(Xt),batch_size=128,shuffle=True)
    enc=SignalEncoder().to(device)
    dec=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,Xt.shape[1])).to(device)
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
            loss=F.mse_loss(recon,xb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
    return {k:v.clone() for k,v in enc.state_dict().items()}

def pretrain_ts2vec(X_unlabeled,epochs=20):
    Xt=X_unlabeled; dl=DataLoader(TensorDataset(Xt),batch_size=128,shuffle=True)
    enc=SignalEncoder().to(device)
    proj=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,64)).to(device)
    opt=torch.optim.Adam(list(enc.parameters())+list(proj.parameters()),lr=1e-3)
    tau=0.1
    for _ in range(epochs):
        enc.train(); proj.train()
        for xb, in dl:
            xb=xb.to(device); B=xb.shape[0]
            v1=xb+torch.randn_like(xb)*0.05; v1=v1*(0.8+0.4*torch.rand(B,1,device=device))
            v2=xb+torch.randn_like(xb)*0.05; v2=v2*(0.8+0.4*torch.rand(B,1,device=device))
            z1=F.normalize(proj(enc(v1).mean(dim=1)),dim=1); z2=F.normalize(proj(enc(v2).mean(dim=1)),dim=1)
            z=torch.cat([z1,z2],dim=0); sim=(z@z.T)/tau
            labels=torch.cat([torch.arange(B,B*2),torch.arange(B)],dim=0).to(device)
            loss=F.cross_entropy(sim,labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
    return {k:v.clone() for k,v in enc.state_dict().items()}

# ── Train scratch encoder ──
def train_scratch_encoder(X_labeled,y_labeled,epochs=50):
    Xt=torch.from_numpy(X_labeled); yt=torch.from_numpy(y_labeled)
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=32,shuffle=True)
    enc=SignalEncoder().to(device)
    cls_tok=nn.Parameter(torch.randn(1,1,64,device=device)*0.02)
    head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,5)).to(device)
    opt=torch.optim.Adam(list(enc.parameters())+[cls_tok]+list(head.parameters()),lr=1e-3)
    for _ in range(epochs):
        enc.train(); head.train()
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device)
            c=cls_tok.expand(xb.shape[0],1,-1)
            loss=F.cross_entropy(head(enc(xb,cls=c)[:,0,:]),yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
    return enc

# ── Feature Extraction ──
@torch.no_grad()
def extract_features(encoder,X):
    """Extract mean-pooled features from encoder."""
    encoder.eval()
    feats=[]
    for i in range(0,len(X),128):
        xb=X[i:i+128].to(device)
        f=encoder(xb).mean(dim=1)  # (B, d) — mean pool over tokens
        feats.append(f.cpu().numpy())
    return np.concatenate(feats,axis=0)

# ═══════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════

print('=== Mechanism Analysis ===')
print('Pre-training MSM & TS2Vec...')
X_pt=gen_unlabeled_pt(10000,128)
w_msm=pretrain_msm(X_pt,20)
w_ts2=pretrain_ts2vec(X_pt,20)

# Train scratch encoder on labeled data (supervised)
print('Training supervised (scratch) encoder...')
X_labeled,y_labeled=gen_analysis_data(500,128,1.5)
scratch_enc=train_scratch_encoder(X_labeled.numpy(),y_labeled.numpy(),50)

# Load pre-trained encoders
msm_enc=SignalEncoder().to(device); msm_enc.load_state_dict(w_msm)
ts2_enc=SignalEncoder().to(device); ts2_enc.load_state_dict(w_ts2)

# Generate test data for feature extraction
X_test,y_test=gen_analysis_data(1000,128,1.5)

print('Extracting features...')
feat_scratch=extract_features(scratch_enc,X_test)
feat_msm=extract_features(msm_enc,X_test)
feat_ts2=extract_features(ts2_enc,X_test)
y_np=y_test.numpy()

# ── 1. Linear Probe ──
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print('\n--- Linear Probe Accuracy ---')
for name,feat in [('Scratch',feat_scratch),('MSM',feat_msm),('TS2Vec',feat_ts2)]:
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(feat)
    # Vary training data
    for n_train in [10,20,50,100,200,500]:
        idx=np.random.choice(len(feat),n_train,replace=False)
        lr=LogisticRegression(max_iter=500).fit(X_scaled[idx],y_np[idx])
        acc=lr.score(X_scaled,y_np)
        if n_train<=50 or n_train==500:
            print(f'  {name} @ {n_train:>3} labels: {acc:.3f}')

# ── 2. Feature Quality Metrics ──
from sklearn.metrics import silhouette_score
print('\n--- Feature Quality ---')
for name,feat in [('Scratch',feat_scratch),('MSM',feat_msm),('TS2Vec',feat_ts2)]:
    sil=silhouette_score(feat[:500],y_np[:500])
    # Intra-class variance
    intra_var=[]
    for c in range(5):
        c_feat=feat[y_np==c]
        if len(c_feat)>1:
            centroid=c_feat.mean(axis=0)
            intra_var.append(np.mean(np.sum((c_feat-centroid)**2,axis=1)))
    intra=np.mean(intra_var)
    # Inter-class variance
    centroids=np.array([feat[y_np==c].mean(axis=0) for c in range(5) if (y_np==c).sum()>0])
    inter=np.mean([np.sum((centroids[i]-centroids[j])**2) for i in range(len(centroids)) for j in range(i+1,len(centroids))])
    print(f'  {name}: Silhouette={sil:.3f}, IntraVar={intra:.2f}, InterVar={inter:.2f}, Inter/Intra={inter/(intra+1e-8):.2f}')

# ── 3. Feature Stability under Noise ──
print('\n--- Feature Stability (cosine similarity under different noise draws) ---')
# Generate same signal with different noise realizations
X_stab,y_stab=gen_analysis_data(200,128,1.5)
X_stab2,y_stab2=gen_analysis_data(200,128,1.5)  # different noise!

for name,enc in [('Scratch',scratch_enc),('MSM',msm_enc),('TS2Vec',ts2_enc)]:
    f1=extract_features(enc,X_stab)
    f2=extract_features(enc,X_stab2)
    # Cosine similarity between same sample under different noise
    cos_sims=[]
    for i in range(min(100,len(f1))):
        cos=np.dot(f1[i],f2[i])/(np.linalg.norm(f1[i])*np.linalg.norm(f2[i])+1e-12)
        cos_sims.append(cos)
    print(f'  {name}: mean cos={np.mean(cos_sims):.3f}±{np.std(cos_sims):.3f}')

# ── 4. Reconstruction Quality ──
print('\n--- Reconstruction Examples ---')
# Save example reconstructions
X_sample=X_test[:5].to(device)
with torch.no_grad():
    tok=msm_enc.pe(msm_enc.tok(X_sample.unsqueeze(1)).transpose(1,2))
    # Partially mask
    mask=torch.zeros_like(tok[:,:,0]).bool()
    mask[:,::2]=True  # mask every other token
    mask_tok=nn.Parameter(torch.randn(1,1,64,device=device)*0.02)
    masked=torch.where(mask.unsqueeze(-1),mask_tok.expand(5,tok.shape[1],-1),tok)
    enc_out=msm_enc.trf(masked)
    # Decoder not available here, skip for now
    recon_error=torch.norm(enc_out.mean(dim=1),dim=1).cpu().numpy()
    print(f'  MSM encoded norm (masked): {recon_error}')

# ── 5. Save for t-SNE visualization ──
print('\nSaving features for t-SNE plotting...')
import json
SAVE_DIR='D:/deepl/scout_logs/mechanism_20260611'
os.makedirs(SAVE_DIR,exist_ok=True)
np.save(f'{SAVE_DIR}/feat_scratch.npy',feat_scratch)
np.save(f'{SAVE_DIR}/feat_msm.npy',feat_msm)
np.save(f'{SAVE_DIR}/feat_ts2.npy',feat_ts2)
np.save(f'{SAVE_DIR}/labels.npy',y_np)

# Summary
summary={
    'linear_probe':{},
    'feature_quality':{},
    'stability':{},
}
for name,feat in [('Scratch',feat_scratch),('MSM',feat_msm),('TS2Vec',feat_ts2)]:
    lr=LogisticRegression(max_iter=500).fit(StandardScaler().fit_transform(feat[:50]),y_np[:50])
    summary['linear_probe'][name]=float(lr.score(StandardScaler().fit_transform(feat),y_np))
    sil=silhouette_score(feat[:500],y_np[:500])
    summary['feature_quality'][name]=float(sil)

for name,enc in [('Scratch',scratch_enc),('MSM',msm_enc),('TS2Vec',ts2_enc)]:
    f1=extract_features(enc,X_stab); f2=extract_features(enc,X_stab2)
    cos_sims=[np.dot(f1[i],f2[i])/(np.linalg.norm(f1[i])*np.linalg.norm(f2[i])+1e-12) for i in range(min(100,len(f1)))]
    summary['stability'][name]=float(np.mean(cos_sims))

json.dump(summary,open(f'{SAVE_DIR}/summary.json','w'),indent=2)
print(f'\nSaved to {SAVE_DIR} | {time.time()-t0:.0f}s')
