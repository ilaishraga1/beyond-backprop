"""Plain n-gram tables with backoff (k=4, not tuned): the bar that any counting rule should be read against.

    python scripts/ngram_reference.py
"""
import math
from collections import Counter
from bbp.data import CharData
d=CharData(128,"cpu"); s=d.splits["train"].tolist(); V=65
M={}
for n in range(1,6):
    c=Counter(tuple(s[i-n+1:i+1]) for i in range(n-1,len(s))); x=Counter()
    for k,v in c.items(): x[k[:-1]]+=v
    M[n]=(c,x)
def p(h,y,n,k):
    c,x=M[n]
    if n==1: return (c[(y,)]+.1)/(x[()]+6.5)
    t=tuple(h[-(n-1):]); return (c[t+(y,)]+k*p(h,y,n-1,k))/(x[t]+k)
for sp in ("val","test"):
    v=d.splits[sp].tolist()
    print(sp,{n:round(sum(-math.log(p(v[i-5:i],v[i],n,4.0)) for i in range(5,len(v)))/(len(v)-5),3) for n in M}, {n:len(M[n][1]) for n in M} if sp=="test" else "")
