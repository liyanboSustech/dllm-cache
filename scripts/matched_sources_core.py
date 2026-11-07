"""Preregistered equal-per-layer MASK/text restoration diagnostic."""
import hashlib, random, torch
from early_sources_core import Backend, trajectory, checkpoint, restoration_plan, region_cache, cache_new
from decision_paths_core import cache_stamp
TAG='matched_sources_20261106'
SEEDS=[2026110631,2026110632,2026110633]
def conditions(cohort):
 return ['BASE','ALL']+[f'{src}_{i}' for i in (range(3) if cohort=='development' else range(1)) for src in ['MASK','TEXT']]
def plan_for(b,s,cp,condition):
 if condition in ['BASE','ALL']:return restoration_plan(b,s,cp,'NONE' if condition=='BASE' else 'ALL')
 source,draw=condition.split('_');draw=int(draw)
 plans={r:restoration_plan(b,s,cp,r) for r in ['MASK','TEXT']};result=[]
 for m,t in zip(plans['MASK'],plans['TEXT']):
  assert m['layer']==t['layer'] and m['kept']==t['kept']
  n=min(32,len(m['added']),len(t['added']));r=dict(m if source=='MASK' else t)
  seed=int.from_bytes(hashlib.sha256(f"{SEEDS[draw]}:{s['sample_id']}:{b.name}:{r['layer']}:{source}".encode()).digest()[:8],'big')
  chosen=sorted(random.Random(seed).sample(r['added'],n));positions=dict(zip(r['added'],r['added_absolute']))
  r.update(added=chosen,selected=r['kept']+chosen,added_absolute=[positions[i] for i in chosen],matched_count=n,available_mask=len(m['added']),available_text=len(t['added']),draw_seed=SEEDS[draw]);result.append(r)
 return result
@torch.inference_mode()
def run(b,s,cp,plan,check):
 x=cp['x'].clone();plen=s['prompt_length'];ref=region_cache(b,cp,plan);stamp=cache_stamp(ref);finite=torch.ones((),device=b.device,dtype=torch.bool);commits=[]
 for block in range(8):
  check();start=plen+32*block;official=cache_new(b)
  for step in range(2 if block==0 else 0,32):
   full=block>0 and step<2
   z=b.forward(x if full else x[:,start:start+32],start,step if full else 2,official if block else ref)
   finite &= torch.isfinite(z).all();pos,tok,_=b.choice(z,x,start,step,full);del z
   assert b.mask not in tok and all(int(x[0,p])==b.mask for p in pos)
   commits.append({'block':block,'step':step,'positions':pos,'tokens':tok})
   if pos:x[0,pos]=torch.tensor(tok,device=b.device)
  del official
 assert finite.item() and stamp==cache_stamp(ref)
 for r in plan:
  for name in ['k','v']:assert torch.equal(ref.cache[r['layer']][name],cp['anchor'][r['layer']][name].index_select(-2,torch.tensor(r['selected'],device=b.device)))
 ids=x[0,plen:].tolist();assert len(ids)==256 and b.mask not in ids
 return {'generated_token_ids':ids,'commits':commits,'cache_audit':{'source_values_exact':True,'same_age':True,'kept_order_preserved':True,'added_counts':[len(r['added']) for r in plan],'reference_counts':[len(r['selected']) for r in plan],'external_counts':[r['external'] for r in plan],'reference_bytes':sum(v.numel()*v.element_size() for e in ref.cache.values() for v in e.values())}}
