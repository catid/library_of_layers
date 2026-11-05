import os,sys,time,statistics,torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from pathlib import Path
_repo_root = Path(os.environ.get("GDN2_ROOT", "/home/catid"))
_fla_root = Path(os.environ.get("FLA_ROOT", "/home/catid/gdn2-gb300/flash-linear-attention"))
_cudnn_root = Path(os.environ.get("CUDNN_ROOT", "/home/catid/cudnn132"))
sys.path[:0] = [str(_repo_root), str(_fla_root), str(_cudnn_root), str(_cudnn_root / "nvidia_cutlass_dsl/dsl_packages")]
from gdn2bench.gdn2 import GatedDeltaNet2
D=3072; H=48; KV=4; HD=64; W=128; FF=12000; S=4096; B=1
class FFN(nn.Module):
 def __init__(self):
  super().__init__(); self.n=nn.RMSNorm(D,eps=1e-5); self.g=nn.Linear(D,2*FF,bias=False); self.d=nn.Linear(FF,D,bias=False)
 def forward(self,x):
  z=self.g(self.n(x)); return x+self.d(F.silu(z[...,:FF])*z[...,FF:])
class G(nn.Module):
 def __init__(self):
  super().__init__(); self.n=nn.RMSNorm(D,eps=1e-5); self.m=GatedDeltaNet2(hidden_size=D,expand_v=2,head_dim=64,num_heads=H,mode='chunk',use_short_conv=False,backend='cudnn').to(dtype=torch.bfloat16); self.f=FFN()
 def forward(self,x): return self.f(x+self.m(self.n(x))[0])
class A(nn.Module):
 def __init__(self,local):
  super().__init__(); self.local=local; self.n=nn.RMSNorm(D,eps=1e-5); self.qkv=nn.Linear(D,D+2*KV*HD,bias=False); self.o=nn.Linear(D,D,bias=False); self.f=FFN()
 def forward(self,x):
  y=self.n(x); q,k,v=self.qkv(y).split((D,KV*HD,KV*HD),-1); q=q.view(B,S,H,HD); k=k.view(B,S,KV,HD); v=v.view(B,S,KV,HD)
  if self.local:
   nw=S//W; q=q.view(B*nw,W,H,HD).transpose(1,2); k=k.view(B*nw,W,KV,HD).transpose(1,2); v=v.view(B*nw,W,KV,HD).transpose(1,2)
   with sdpa_kernel(SDPBackend.FLASH_ATTENTION): z=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
   z=z.transpose(1,2).reshape(B,S,D)
  else:
   q=q.transpose(1,2); k=k.transpose(1,2); v=v.transpose(1,2)
   k=k.repeat_interleave(H//KV,dim=1); v=v.repeat_interleave(H//KV,dim=1)
   with sdpa_kernel(SDPBackend.CUDNN_ATTENTION): z=F.scaled_dot_product_attention(q,k,v,is_causal=True)
   z=z.transpose(1,2).reshape(B,S,D)
  return self.f(x+self.o(z))
class M(nn.Module):
 def __init__(self): super().__init__(); self.layers=nn.ModuleList([G(),A(True),A(True),A(False)]); self.out=nn.Linear(D,D,bias=False)
 def forward(self,x):
  for l in self.layers:x=l(x)
  return self.out(x)
torch.manual_seed(1); m=M().cuda().to(torch.bfloat16).train(); print('params',sum(p.numel() for p in m.parameters())/1e6,flush=True); x=torch.randn(B,S,D,device='cuda',dtype=torch.bfloat16); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,fused=True)
def step():
 opt.zero_grad(set_to_none=True); y=m(x); loss=y.float().square().mean(); loss.backward(); opt.step()
for _ in range(2):step()
torch.cuda.synchronize(); ts=[]
for _ in range(5):
 st,en=torch.cuda.Event(True),torch.cuda.Event(True);st.record();step();en.record();en.synchronize();ts.append(st.elapsed_time(en))
print('times',ts,'median',statistics.median(ts),'mean',statistics.mean(ts),flush=True)
