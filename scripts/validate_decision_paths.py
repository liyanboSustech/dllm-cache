import torch
from discovery_common import ROOT,write_json
from decision_paths_core import Backend,actions,CONDITIONS,mix_decisions
from types import SimpleNamespace
torch.set_num_threads(2);cases=0
for model in ['llada','dream']:
 b=object.__new__(Backend);b.name=model;b.mask=126336 if model=='llada' else 151666;b.sampler_options={'temperature':0.,'top_k':None,'top_p':None}
 for seed in range(8):
  torch.manual_seed(seed)
  for step,remaining in [(2,30),(2,1),(4,28),(20,12),(31,3)]:
   x=torch.full((1,40),b.mask,dtype=torch.long);x[:,:8]=1;x[:,8+remaining:]=2
   zs=torch.randn(1,32,9,dtype=torch.float16);zf=torch.randn_like(zs)
   if seed==0:zs.zero_();zf.zero_()
   for c in CONDITIONS:
    p,t,r=actions(b,zs,zf,x,8,step,c)
    expected={'BASE':(r['sparse_positions'],r['sparse_tokens']),'ORDER':(r['full_positions'],r['sparse_tokens_at_full_positions']),'CONTENT':(r['sparse_positions'],r['full_tokens_at_sparse_positions']),'BOTH':(r['full_positions'],r['full_tokens'])}[c]
    assert (p,t)==expected
    np,nt,_=actions(b,zs,zs,x,8,step,c);bp,bt,_=b.choice(zs,x,8,step,False);assert (np,nt)==(bp,bt)
    cases+=1
# Explicit positional swap and word swap, including empty and multi-token decisions.
assert mix_decisions([1],[10],[2],[20],[11],[21],'ORDER')==([2],[11])
assert mix_decisions([1],[10],[2],[20],[11],[21],'CONTENT')==([1],[21])
for c in CONDITIONS:assert mix_decisions([],[],[],[],[],[],c)==([],[])
write_json(ROOT/'reports/decision_paths_20261106/cpu_validation.json',{'passed':True,'cases':cases,'both_native_samplers':True,'null_and_tie_and_multi_and_empty_transfer':True,'content_at_chosen_position_not_other_branch_position':True});print('Passed',cases,'decision-source tests')
