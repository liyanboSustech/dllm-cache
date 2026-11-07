import math,torch
from k4_correction_core import js_from_logp,selections,make_cache,Backend,TAG
from priority_kv_core import PriorityCache,importance_scores
from discovery_common import ROOT,write_json
torch.set_num_threads(2);torch.manual_seed(2026110703)
p=torch.tensor([[.1,.9],[.6,.4]],dtype=torch.float32);q=p.flip(-1)
m=(p+q)/2;manual=.5*((p*(p/m).log()).sum(-1)+(q*(q/m).log()).sum(-1))
assert torch.allclose(js_from_logp(p.log(),q.log()),manual,atol=1e-7)
assert torch.allclose(js_from_logp(p.log(),q.log()),js_from_logp(q.log(),p.log()),atol=1e-7)
assert float(js_from_logp(p.log(),p.log()).abs().max())<1e-7
assert float(js_from_logp(p.log(),q.log()).max())<=math.log(2.)
ref=torch.tensor([[.8,.15,.05]]);base=torch.tensor([[.4,.5,.1]]);good=torch.tensor([[.7,.2,.1]]);bad=torch.tensor([[.1,.8,.1]])
d=lambda x:js_from_logp(ref.log(),x.log()).mean()
assert d(base)-d(good)>0 and d(base)-d(bad)<0
cases=0
for heads in [4,8]:
 for plen in [1,16,64]:
  for block in [0,3,6,7]:
   for scale in [1.,1000.]:
    start=plen+32*block;n=plen+256
    k=(torch.randn(1,heads,n,8)*scale).half();v=torch.randn_like(k);q=(torch.randn(1,8,32,8)*scale).half()
    ek=torch.cat((k[:,:,:start],k[:,:,start+32:]),dim=2);ev=torch.cat((v[:,:,:start],v[:,:,start+32:]),dim=2)
    sel=importance_scores(ek,q,3)[0].topk(int(.25*(n-32))).indices
    anchor={0:{'k':ek,'v':ev,'q':q,'selection':sel}}
    plans=selections(anchor,start)
    b=object.__new__(Backend);b.n=1;b.device=torch.device('cpu')
    for layers in [[],[0]]:
     official=PriorityCache(1,'cpu',kernel_size=3,keep_ratio=.25,priority_layers=layers);official.cache[0]={'k':k,'v':v};official.filter_cache(0,q,start,32)
     actual=make_cache(b,anchor,plans,layers)
     for key in ['k','v']:assert torch.equal(official.cache[0][key],actual.cache[0][key])
     assert actual.cache[0]['k'].shape[-2]==int(.25*(n-32))
    cases+=1
write_json(ROOT/'reports'/TAG/'cpu_validation.json',{'passed':True,'selection_cases':cases,'JS_formula_symmetry_null_signed_gain':True,'native_priority_exact_including_inf_GQA_lastblock':True})
print('CPU validation passed',cases)
