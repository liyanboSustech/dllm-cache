"""Fixed-budget next-block priority. Vendored model/generator remain unchanged."""
from contextlib import contextmanager
from functools import partial
import torch
import torch.nn.functional as F
from discovery_common import CustomCache
from vendor.sparse_llada import llada_generate

METHODS={'S25':(),'K4_NEXT':(4,5,6,21),'W4_NEXT':(0,1,30,31)}


def importance_scores(k,q,pool):
    ka=k.repeat_interleave(q.size(1)//k.size(1),dim=1) if q.size(1)!=k.size(1) else k
    scores=(q.mean(dim=-2).unsqueeze(-2)@ka.transpose(-2,-1)).squeeze(-2).mean(dim=1)
    if pool is not None:scores=F.max_pool1d(scores.unsqueeze(1),pool,stride=1,padding=pool//2).squeeze(1)
    return scores


class PriorityCache(CustomCache):
    def __init__(self,*args,priority_layers=(),audit_records=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.priority_layers=priority_layers
        self.audit_records=audit_records

    def filter_cache(self,layer_id,q_block,cur_filtered_len,block_len):
        cached=self.get_cache(layer_id)
        expected=int((cached['k'].shape[-2]-block_len)*self.keep_ratios[layer_id])
        priority=layer_id in self.priority_layers
        if priority or self.audit_records is not None:
            start=cur_filtered_len;end=start+block_len
            k=torch.cat((cached['k'][:,:,:start],cached['k'][:,:,end:]),dim=2)
            v=torch.cat((cached['v'][:,:,:start],cached['v'][:,:,end:]),dim=2)
            scores=importance_scores(k,q_block,self.pool_kernel_size)
            original=scores.clone() if self.audit_records is not None else None
            next_end=min(start+block_len,k.shape[-2])
            reserve=max(0,next_end-start) if priority else 0
            if priority:
                if reserve>expected:
                    selected=scores[:,start:next_end].topk(expected,dim=-1).indices.squeeze(0)+start
                else:
                    # Positions after the removed block start at the same offset.
                    # Under the unchanged block schedule all these positions are MASK.
                    scores[:,start:next_end]=torch.inf
                    selected=scores.topk(expected,dim=-1).indices.squeeze(0)
                heads=torch.arange(k.shape[1],device=k.device)[:,None]
                self.cache[layer_id]={'k':k[:,heads,selected],'v':v[:,heads,selected]}
            else:super().filter_cache(layer_id,q_block,cur_filtered_len,block_len)
        else:super().filter_cache(layer_id,q_block,cur_filtered_len,block_len)
        # Shape-only invariant: no device synchronization; identical check for all methods.
        assert self.cache[layer_id]['k'].shape[-2]==expected
        assert self.cache[layer_id]['v'].shape[-2]==expected
        if self.audit_records is not None:
            if not priority:selected=original.topk(expected,dim=-1).indices.squeeze(0)
            chosen=selected.tolist();assert len(set(chosen))==expected
            assert torch.equal(self.cache[layer_id]['k'],k.index_select(-2,selected))
            assert torch.equal(self.cache[layer_id]['v'],v.index_select(-2,selected))
            mandatory=set(range(start,next_end)) if priority else set()
            assert len(mandatory&set(chosen))==min(reserve,expected)
            if priority and reserve<=expected:
                residual=original.clone();residual[:,start:next_end]=-torch.inf
                allowed=set(residual.topk(expected-reserve,dim=-1).indices.squeeze(0).tolist()) if expected>reserve else set()
                # Numerical ties allow multiple equally ranked boundary positions.
                rest=[j for j in chosen if j not in mandatory]
                if set(rest)!=allowed and rest:
                    cutoff=float(residual.topk(expected-reserve,dim=-1).values.min())
                    assert all(float(original[0,j])>=cutoff for j in rest)
            self.audit_records.append({'layer':layer_id,'block_start':start,'external_before':k.shape[-2],
                'retained':expected,'priority':priority,'reserved_next':min(reserve,expected),
                'cache_bytes':sum(t.numel()*t.element_size() for t in self.cache[layer_id].values()),'selected_external_indices':chosen})


@contextmanager
def use_cache(method,audit_records=None):
    original=llada_generate.CustomCache
    llada_generate.CustomCache=partial(PriorityCache,priority_layers=METHODS[method],audit_records=audit_records)
    try:yield
    finally:llada_generate.CustomCache=original


@torch.inference_mode()
def generate(model,prompt,method,audit_records=None,budget=256):
    with use_cache(method,audit_records):
        return llada_generate.generate(model,prompt,steps=budget,gen_length=budget,block_length=32,temperature=0,cfg_scale=0)
