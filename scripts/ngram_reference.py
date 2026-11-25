"""Plain n-gram tables, scored on exactly the arena's val/test windows: the bar for every rule.

  fixed   Dirichlet backoff with k=4, not tuned (what grandmother_cells-style rules should be read against)
  tuned   Witten-Bell backoff, smoothing strength picked on val per order

Order n uses n-1 characters of context, never more than the 128-char window holds.
Tables are unlimited memory: they do NOT fit in the arena's 0.83M-parameter model.

    python scripts/ngram_reference.py
"""
import numpy as np, torch, itertools
from bbp.data import CharData
d=CharData(128,"cpu"); V=65; tr=d.splits["train"].numpy().astype(np.int64)
def windows(split,nb=50):
    g=torch.Generator().manual_seed(1234); X=[];Y=[]
    for _ in range(nb):
        x,y=d.batch(split,32,g); X.append(x.numpy()); Y.append(y.numpy())
    return np.concatenate(X).astype(np.int64),np.concatenate(Y).astype(np.int64)
W={s:windows(s) for s in("val","test")}
NM=10  # context length up to 9 chars (int64 keys)
def ctxkey(seq,n): # seq (...,T): key of last n chars at each t; invalid t<n-1 -> -1
    out=np.full(seq.shape,-1,dtype=np.int64); T=seq.shape[-1]
    if n==0: return np.zeros(seq.shape,np.int64)
    k=sum(seq[...,n-1-i:T-i]*V**i for i in range(n)); out[...,n-1:]=k; return out
tab={}
for n in range(0,NM):
    c=ctxkey(tr[:-1],n); m=c>=0
    k=c[m]*V+tr[1:][m]
    if n==0: pass
    pk,pc=np.unique(k,return_counts=True)
    cx=pk//V; uc,st=np.unique(cx,return_index=True); tot=np.add.reduceat(pc,st); dist=np.diff(np.append(st,len(pk)))
    tab[n]=(pk,pc,uc,tot,dist)
def look(keys,vals,q):
    i=np.searchsorted(keys,q); i=np.minimum(i,len(keys)-1); return np.where(keys[i]==q,vals[i],0)
def ctxstats(split):
    X,Y=W[split]; S=[]
    for n in range(NM):
        c=ctxkey(X,n); pk,pc,uc,tot,dist=tab[n]
        cnt=look(pk,pc,c*V+Y)*(c>=0); t=look(uc,tot,c)*(c>=0); ds=look(uc,dist,c)*(c>=0)
        S.append((cnt,t,ds))
    return S
ST={s:ctxstats(s) for s in W}
def ev(split,mode,par,N):
    p=np.full(W[split][1].shape,1/V)
    for n in range(N):
        cnt,t,ds=ST[split][n]
        if mode=="dir": num=cnt+par*p; den=t+par
        elif mode=="wb": num=cnt+par*ds*p; den=t+par*ds
        elif mode=="kn": num=np.maximum(cnt-par,0)+par*ds*p; den=t   # abs discounting, plain counts
        q=np.where(den>0,num/np.maximum(den,1e-9),p); p=np.where(t>0,q,p)
    return -np.log(p).mean()


if __name__ == "__main__":
    print("order  contexts  (context, next) pairs   fixed k=4 val/test     tuned val/test (par)")
    for N in range(2, NM + 1):
        v, a = min((ev("val", "wb", a, N), a) for a in (1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64))
        print(f"{N:5d} {len(tab[N-1][2]):9,d} {len(tab[N-1][0]):12,d}          {ev('val','dir',4,N):.3f} / {ev('test','dir',4,N):.3f}        {v:.3f} / {ev('test','wb',a,N):.3f} ({a})")
