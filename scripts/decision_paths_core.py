"""Online 2x2 decision-source intervention; frozen age-matched external KV."""
import hashlib
import torch
from kv_generalization_core import Backend,cache_new,cache_from_sources
from kv_history_phase_core import trajectory,native_forced_tokens
CONDITIONS=['BASE','ORDER','CONTENT','BOTH']
PHASES=['b0_r04','b1_r04']

def cache_stamp(cache):
 return [(li,n,v.data_ptr(),tuple(v.shape)) for li,e in sorted(cache.cache.items()) for n,v in sorted(e.items())]

def mix_decisions(spos,stok,fpos,ftok,s_at_f,f_at_s,condition):
 assert len(spos)==len(fpos)==len(stok)==len(ftok)==len(s_at_f)==len(f_at_s)
 return {'BASE':(spos,stok),'ORDER':(fpos,s_at_f),'CONTENT':(spos,f_at_s),'BOTH':(fpos,ftok)}[condition]

@torch.inference_mode()
def actions(b,zs,zf,x,start,step,condition):
 sp,st,sc=b.choice(zs,x,start,step,False);fp,ft,fc=b.choice(zf,x,start,step,False)
 assert native_forced_tokens(b,zs,start,False,sp)==st
 assert native_forced_tokens(b,zf,start,False,fp)==ft
 sf=native_forced_tokens(b,zs,start,False,fp);fs=native_forced_tokens(b,zf,start,False,sp)
 pos,tok=mix_decisions(sp,st,fp,ft,sf,fs,condition)
 assert len(pos)==len(set(pos)) and all(int(x[0,p])==b.mask for p in pos) and b.mask not in tok
 row={'sparse_positions':sp,'sparse_tokens':st,'sparse_confidence':sc,'full_positions':fp,'full_tokens':ft,'full_confidence':fc,
      'sparse_tokens_at_full_positions':sf,'full_tokens_at_sparse_positions':fs,
      'chosen_positions':pos,'chosen_tokens':tok,'order_set_changed':set(sp)!=set(fp),
      'content_changed_at_sparse_positions':st!=fs,'content_changed_at_full_positions':sf!=ft}
 return pos,tok,row

@torch.inference_mode()
def continue_branch(b,sample,cp,condition,check,reference_same_as_sparse=False,full_only=False):
 """Reference recomputed on this branch's x_t, not another branch's trajectory.
 Subsequent blocks always native S25. full_only is an independent warmup oracle replay.
 """
 x=cp['x'].clone();plen=sample['prompt_length'];first=cp['block'];start=cp['start']
 sparse=cache_from_sources(b,cp['anchor'],{},'S25');full=cache_from_sources(b,cp['anchor'],{},'S25' if reference_same_as_sparse else 'C100')
 stamps=(cache_stamp(sparse),cache_stamp(full));external=x.shape[1]-32
 assert all(e['k'].shape[-2]==int(external*.25) for e in sparse.cache.values())
 assert all(e['k'].shape[-2]==(int(external*.25) if reference_same_as_sparse else external) for e in full.cache.values())
 for li,e in cp['anchor'].items():
  for n in ['k','v']:
   assert torch.equal(sparse.cache[li][n],e[n].index_select(-2,e['selection']))
   if not reference_same_as_sparse:assert torch.equal(full.cache[li][n],e[n])
 events=[];commits=[];finite=torch.ones((),device=x.device,dtype=torch.bool)
 for block in range(first,8):
  check();start=plen+32*block;official=cache_new(b)
  for step in range(cp['step'] if block==first else 0,32):
   if block==first:
    before=x[:,start:start+32].clone()
    if full_only:
     z=b.forward(x[:,start:start+32],start,2,full);finite &= torch.isfinite(z).all();pos,tok,_=b.choice(z,x,start,step,False);del z
    else:
     zs=b.forward(x[:,start:start+32],start,2,sparse)
     zf=b.forward(x[:,start:start+32],start,2,full)
     finite &= torch.isfinite(zs).all() & torch.isfinite(zf).all()
     assert torch.equal(before,x[:,start:start+32])
     pos,tok,event=actions(b,zs,zf,x,start,step,condition)
     events.append({'block':block,'step':step,'start':start,'current_block_before':before[0].tolist(),**event})
     del zs,zf
   else:
    if step<2:z=b.forward(x,start,step,official)
    else:z=b.forward(x[:,start:start+32],start,2,official)
    finite &= torch.isfinite(z).all();pos,tok,_=b.choice(z,x,start,step,step<2);del z
   assert b.mask not in tok
   commits.append({'block':block,'step':step,'positions':pos,'tokens':tok})
   if pos:x[0,pos]=torch.tensor(tok,device=b.device)
  del official
 assert finite.item(),'Nonfinite logits'
 assert stamps==(cache_stamp(sparse),cache_stamp(full)),'Read-only reference replaced cache'
 for li,e in cp['anchor'].items():
  for n in ['k','v']:
   assert torch.equal(sparse.cache[li][n],e[n].index_select(-2,e['selection']))
   expected=e[n].index_select(-2,e['selection']) if reference_same_as_sparse else e[n]
   assert torch.equal(full.cache[li][n],expected),'Read-only reference changed values'
 ids=x[0,plen:].tolist();assert len(ids)==256 and b.mask not in ids
 audit={'same_anchor_same_age':True,'source_vectors_bitwise_checked':True,'cache_storage_and_values_unchanged':True,
        'sparse_counts':[e['k'].shape[-2] for e in sparse.cache.values()], 'reference_counts':[e['k'].shape[-2] for e in full.cache.values()],
        'sparse_bytes':sum(v.numel()*v.element_size() for e in sparse.cache.values() for v in e.values()),
        'reference_bytes':sum(v.numel()*v.element_size() for e in full.cache.values() for v in e.values()),
        'diagnostic_not_equal_total_memory_or_speed':True,'reference_evaluated_on_own_branch_current_state':True}
 return {'generated_token_ids':ids,'events':events,'commits':commits,'cache_audit':audit}
