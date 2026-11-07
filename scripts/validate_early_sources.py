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
# Synthetic region partitions and restoration order/count/value checks.
from early_sources_core import restoration_plan,region_cache
for block in [0,1]:
 for model in ['llada','dream']:
  b=object.__new__(Backend);b.name=model;b.mask=126336 if model=='llada' else 151666;b.n=2;b.device=torch.device('cpu')
  plen=24;start=plen+32*block;total=plen+96;external=total-32
  x=torch.full((1,total),b.mask,dtype=torch.long);x[:,:start]=7;x[0,start]=8
  for seed in range(20):
   torch.manual_seed(seed);anchor={li:{'k':torch.randn(1,4,external,8),'v':torch.randn(1,4,external,8),'selection':torch.randperm(external)[:external//4]} for li in range(2)}
   cp={'x':x,'anchor_input':x,'start':start,'anchor':anchor}
   plans={src:restoration_plan(b,{'prompt_length':plen},cp,src) for src in ['NONE','ALL','MASK','TEXT','PROMPT','HISTORY']}
   for src,plan in plans.items():
    cache=region_cache(b,cp,plan)
    for rr in plan:
     li=rr['layer'];assert rr['selected'][:len(rr['kept'])]==rr['kept']
     assert torch.equal(cache.cache[li]['k'],anchor[li]['k'].index_select(-2,torch.tensor(rr['selected'])))
   for li in range(2):
    ma=set(plans['MASK'][li]['added']);te=set(plans['TEXT'][li]['added']);al=set(plans['ALL'][li]['added']);assert not ma&te and ma|te==al
    assert set(plans['PROMPT'][li]['added'])|set(plans['HISTORY'][li]['added'])==te
   if block==0:assert plans['TEXT']==plans['PROMPT'] and all(not p['added'] for p in plans['HISTORY'])
write_json(ROOT/'reports/early_sources_20261106/cpu_validation.json',{'passed':True,'region_cases':80,'cases':cases,'both_native_samplers':True,'null_and_tie_and_multi_and_empty_transfer':True,'content_at_chosen_position_not_other_branch_position':True});print('Passed',cases,'decision-source tests')
