"""Per-layer forward/backward timing for the 600M GDN2→LoL→LoL→FA repeat."""
import os,sys,time,statistics,torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from pathlib import Path
_repo_root = Path(os.environ.get("GDN2_ROOT", "/home/catid"))
_fla_root = Path(os.environ.get("FLA_ROOT", "/home/catid/gdn2-gb300/flash-linear-attention"))
_cudnn_root = Path(os.environ.get("CUDNN_ROOT", "/home/catid/cudnn132"))
sys.path[:0] = [str(_repo_root), str(_fla_root), str(_cudnn_root), str(_cudnn_root / "nvidia_cutlass_dsl/dsl_packages")]
# Prefer the user's optimized GDN2 checkout; fall back to the machine reference.
_gdn2_root = Path(os.environ.get("GDN2_ROOT", "/home/catid/gdn2"))
if (_gdn2_root / "lit_gpt/gdn2.py").exists():
 import types
 _lit = types.ModuleType("lit_gpt"); _lit.__path__ = [str(_gdn2_root / "lit_gpt")]
 sys.modules.setdefault("lit_gpt", _lit)
 from lit_gpt.gdn2 import GatedDeltaNet2
else:
 from gdn2bench.gdn2 import GatedDeltaNet2
D=3072; H=48; KV=4; HD=64; W=128; FF=12000; S=4096; B=1
class FFN(nn.Module):
 def __init__(self):
  super().__init__(); self.n=nn.RMSNorm(D,eps=1e-5); self.g=nn.Linear(D,2*FF,bias=False); self.d=nn.Linear(FF,D,bias=False)
 def forward(self,x):
  z=self.g(self.n(x)); return x+self.d(F.silu(z[...,:FF])*z[...,FF:])
class G(nn.Module):
 def __init__(self):
  super().__init__(); self.n=nn.RMSNorm(D,eps=1e-5); self.m=GatedDeltaNet2(hidden_size=D,expand_v=2,head_dim=64,num_heads=H,mode='chunk',use_short_conv=False,backend='cudnn',fused_projections=True,disable_recompute=True).to(dtype=torch.bfloat16); self.f=FFN()
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
 def __init__(self):
  super().__init__(); self.layers=nn.ModuleList([G(),A(True),A(True),A(False)]); self.out=nn.Linear(D,D,bias=False)
  self.active = None
  self.layer_times = [[] for _ in self.layers]
  for i, layer in enumerate(self.layers):
   layer.register_full_backward_pre_hook(self._bwd_pre(i))
   layer.register_full_backward_hook(self._bwd_post(i))
 def _bwd_pre(self, i):
  def hook(module, grad):
   self.active[i]['b0'].record()
  return hook
 def _bwd_post(self, i):
  def hook(module, grad_in, grad_out):
   self.active[i]['b1'].record()
  return hook
 def forward(self,x):
  self.active = {i: {'f0': torch.cuda.Event(True), 'f1': torch.cuda.Event(True), 'b0': torch.cuda.Event(True), 'b1': torch.cuda.Event(True)} for i in range(len(self.layers))}
  for i,l in enumerate(self.layers):
   self.active[i]['f0'].record(); x=l(x); self.active[i]['f1'].record()
  return self.out(x)
 def collect(self):
  vals=[]
  for a in self.active.values():
   vals.append((a['f0'].elapsed_time(a['f1']), a['b0'].elapsed_time(a['b1'])))
  return vals
torch.manual_seed(1); m=M().cuda().to(torch.bfloat16).train(); print('params',sum(p.numel() for p in m.parameters())/1e6,flush=True); x=torch.randn(B,S,D,device='cuda',dtype=torch.bfloat16,requires_grad=True); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,fused=True)
def step():
 opt.zero_grad(set_to_none=True); y=m(x); loss=y.float().square().mean(); loss.backward(); opt.step()
for _ in range(2):step()
torch.cuda.synchronize(); ts=[]; parts=[]
for _ in range(5):
 st,en=torch.cuda.Event(True),torch.cuda.Event(True);st.record();step();en.record();en.synchronize();ts.append(st.elapsed_time(en)); parts.append(m.collect())
med=[statistics.median([p[i][j] for p in parts]) for i in range(4) for j in range(2)]
print('times',ts,'median',statistics.median(ts),'mean',statistics.mean(ts),flush=True)
for i,name in enumerate(('GDN2','LoL-1','LoL-2','FA')):
 f=med[2*i]; b=med[2*i+1]; print(name,'forward_ms',f,'backward_ms',b,'total_ms',f+b,'total_pct',100*(f+b)/statistics.median(ts),'forward_pct',100*f/statistics.median(ts),'backward_pct',100*b/statistics.median(ts),flush=True)
