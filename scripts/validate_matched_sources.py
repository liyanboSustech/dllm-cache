import torch
from matched_sources_core import Backend,plan_for,region_cache,TAG
from discovery_common import ROOT,write_json
cases=0
for model in ['llada','dream']:
 b=object.__new__(Backend);b.name=model;b.mask=126336 if model=='llada' else 151666;b.n=2;b.device=torch.device('cpu')
 for plen in [1,8,32,128]:
  total=plen+256;external=total-32;x=torch.full((1,total),b.mask,dtype=torch.long);x[:,:plen]=7;s={'prompt_length':plen,'sample_id':'test'}
  for seed in range(12):
   torch.manual_seed(seed);anchor={li:{'k':torch.randn(1,4,external,8),'v':torch.randn(1,4,external,8),'selection':torch.randperm(external)[:external//4]} for li in range(2)};cp={'x':x,'anchor_input':x,'start':plen,'anchor':anchor}
   for draw in range(3):
    m=plan_for(b,s,cp,f'MASK_{draw}');t=plan_for(b,s,cp,f'TEXT_{draw}');assert m==plan_for(b,s,cp,f'MASK_{draw}')
    for mr,tr in zip(m,t):
     assert len(mr['added'])==len(tr['added'])==min(32,mr['available_mask'],mr['available_text']);assert mr['kept']==tr['kept'];assert not set(mr['added'])&set(tr['added']);assert all(p>=plen+32 for p in mr['added_absolute']);assert all(p<plen for p in tr['added_absolute'])
    for plan in [m,t]:
     cache=region_cache(b,cp,plan)
     for r in plan:
      assert len(r['selected'])==len(set(r['selected'])) and r['selected'][:len(r['kept'])]==r['kept']
      for n in ['k','v']:assert torch.equal(cache.cache[r['layer']][n],anchor[r['layer']][n].index_select(-2,torch.tensor(r['selected'])))
    cases+=1
write_json(ROOT/'reports'/TAG/'cpu_validation.json',{'passed':True,'cases':cases,'same_count_layer_values_order_age':True,'small_zero_budget_deterministic_subset':True});print('passed',cases)
