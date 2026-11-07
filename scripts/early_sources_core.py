"""First sparse-read interventions with age-matched regional KV restoration."""
import torch
from kv_generalization_core import Backend,cache_new,cache_from_sources
from kv_history_phase_core import trajectory
from decision_paths_core import actions,cache_stamp
TAG='early_sources_20261106'
ARMS=['BASE','ORDER','CONTENT','BOTH']
PHASES=['b0_early','b0_late','b1_clean_early']
def conditions(phase):
 refs=['ALL'] if phase=='b0_late' else ['ALL','MASK','TEXT']
 return [('NONE','BASE')]+[(s,a) for s in refs for a in ARMS[1:]]+([('PROMPT','BOTH'),('HISTORY','BOTH')] if phase=='b1_clean_early' else [])
def key(source,arm):return 'BASE' if arm=='BASE' else source+'_'+arm

def checkpoint(b,sample,run,block,early):
 old=run['checkpoints'][f'b{block}_r04'];cp={**old}
 if early:
  x=torch.full_like(old['x'],b.mask);plen=sample['prompt_length'];x[:,:plen]=torch.tensor([sample['prompt_ids']],device=b.device)
  for c in run['commits']:
   if (c['block'],c['step'])==(block,2):break
   if c['positions']:x[0,c['positions']]=torch.tensor(c['tokens'],device=b.device)
  cp.update(x=x,step=2,actual_revealed=int((x[:,old['start']:old['start']+32]!=b.mask).sum()),target_revealed=None,native=None)
 cp.update(any_stop_already_present=any(t in b.stops for t in cp['x'][0,sample['prompt_length']:].tolist()),stop_in_previous_blocks=any(t in b.stops for t in cp['x'][0,sample['prompt_length']:cp['start']].tolist()))
 cp['upstream_policy']='C100' if block else 'S25';cp['first_sparse_read_global']=early;cp['prior_stop_commit']=None
 return cp

def restoration_plan(b,sample,cp,source):
 """Keep native S25 slots unchanged and append only missing region slots.
 ALL uses the same append convention; late controls rerun to isolate timing.
 Regions refer to tokens at cache construction, not future information.
 """
 plen=sample['prompt_length'];start=cp['start'];end=start+32;total=cp['x'].shape[1]
 absolute=list(range(start))+list(range(end,total));tokens=cp['anchor_input'][0].tolist()
 regions={name:[] for name in ['MASK','PROMPT','HISTORY','TEXT','ALL','NONE']}
 for i,p in enumerate(absolute):
  name='PROMPT' if p<plen else 'HISTORY' if p<start and tokens[p]!=b.mask else 'MASK'
  assert name!='MASK' or tokens[p]==b.mask,(p,start)
  regions[name].append(i);regions['ALL'].append(i)
 regions['TEXT']=regions['PROMPT']+regions['HISTORY'];assert set(regions['MASK']).isdisjoint(regions['TEXT']) and set(regions['MASK'])|set(regions['TEXT'])==set(regions['ALL'])
 plan=[]
 for li,e in sorted(cp['anchor'].items()):
  kept=e['selection'].tolist();ks=set(kept);add=[i for i in regions[source] if i not in ks];selected=kept+add
  assert len(selected)==len(set(selected)) and selected[:len(kept)]==kept
  plan.append({'layer':li,'kept':kept,'added':add,'selected':selected,'added_absolute':[absolute[i] for i in add],'region_size':len(regions[source]),'external':len(absolute)})
 return plan

def region_cache(b,cp,plan):
 c=cache_new(b);c.cache={r['layer']:{n:cp['anchor'][r['layer']][n].index_select(-2,torch.tensor(r['selected'],device=b.device,dtype=torch.long)) for n in ['k','v']} for r in plan};return c

@torch.inference_mode()
def continue_branch(b,sample,cp,source,arm,check,reference_only=False,null=False):
 x=cp['x'].clone();plen=sample['prompt_length'];first=cp['block'];plan=restoration_plan(b,sample,cp,'NONE' if null else source)
 sparse=cache_from_sources(b,cp['anchor'],{},'S25');ref=region_cache(b,cp,plan);stamps=cache_stamp(sparse),cache_stamp(ref)
 events=[];commits=[];finite=torch.ones((),device=b.device,dtype=torch.bool)
 for block in range(first,8):
  check();start=plen+32*block;official=cache_new(b)
  for step in range(cp['step'] if block==first else 0,32):
   if block==first:
    before=x[:,start:start+32].clone()
    if reference_only:
     z=b.forward(x[:,start:start+32],start,2,ref);finite &= torch.isfinite(z).all();pos,tok,_=b.choice(z,x,start,step,False);del z
    else:
     zs=b.forward(x[:,start:start+32],start,2,sparse);zr=b.forward(x[:,start:start+32],start,2,ref)
     finite &= torch.isfinite(zs).all() & torch.isfinite(zr).all();assert torch.equal(before,x[:,start:start+32])
     pos,tok,event=actions(b,zs,zr,x,start,step,arm)
     events.append({'block':block,'step':step,'start':start,'current_block_before':before[0].tolist(),**event});del zs,zr
   else:
    z=b.forward(x if step<2 else x[:,start:start+32],start,step if step<2 else 2,official)
    finite &= torch.isfinite(z).all();pos,tok,_=b.choice(z,x,start,step,step<2);del z
   assert b.mask not in tok and all(int(x[0,p])==b.mask for p in pos)
   commits.append({'block':block,'step':step,'positions':pos,'tokens':tok})
   if pos:x[0,pos]=torch.tensor(tok,device=b.device)
  del official
 assert finite.item() and stamps==(cache_stamp(sparse),cache_stamp(ref))
 for r in plan:
  li=r['layer'];e=cp['anchor'][li]
  for n in ['k','v']:
   assert torch.equal(sparse.cache[li][n],e[n].index_select(-2,e['selection']))
   assert torch.equal(ref.cache[li][n],e[n].index_select(-2,torch.tensor(r['selected'],device=b.device)))
 ids=x[0,plen:].tolist();assert len(ids)==256 and b.mask not in ids
 audit={'same_anchor_same_age':True,'source_values_exact_before_after':True,'native_kept_order_preserved':True,'reference_on_own_current_state':True,'sparse_counts':[len(r['kept']) for r in plan],'reference_counts':[len(r['selected']) for r in plan],'added_counts':[len(r['added']) for r in plan],'sparse_bytes':sum(v.numel()*v.element_size() for e in sparse.cache.values() for v in e.values()),'reference_bytes':sum(v.numel()*v.element_size() for e in ref.cache.values() for v in e.values()),'diagnostic_extra_memory_compute':True}
 return {'generated_token_ids':ids,'events':events,'commits':commits,'cache_audit':audit}
