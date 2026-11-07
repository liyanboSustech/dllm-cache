"""One frozen diagnostic: signed JS reduction under the native equal-budget NEXT swap."""
import math
import torch
from kv_generalization_core import Backend, cache_new
from priority_kv_core import PriorityCache, importance_scores

TAG='k4_correction_20261107'
LAYERS={'llada':[4,5,6,21],'dream':[3,4,5,18]}

def js_from_logp(a,b):
    # FP32 log-space mixture; no clipping, temperature, top-k, or EOS weighting.
    assert a.dtype==b.dtype==torch.float32 and a.shape==b.shape
    m=torch.logaddexp(a,b)-math.log(2.)
    return .5*((a.exp()*(a-m)).sum(-1)+(b.exp()*(b-m)).sum(-1))

def selections(anchor,start):
    result={}
    for li,e in anchor.items():
        scores=importance_scores(e['k'],e['q'],3)
        n=e['k'].shape[-2];count=int(.25*n);end=min(start+32,n)
        assert torch.equal(scores[0].topk(count).indices,e['selection'])
        if end-start>count:
            nxt=scores[:,start:end].topk(count,dim=-1).indices.squeeze(0)+start
        else:
            bumped=scores.clone();bumped[:,start:end]=torch.inf
            nxt=bumped.topk(count,dim=-1).indices.squeeze(0)
        # Exact native inf/tie semantics retained, including native Dream overflow.
        result[li]={'base':e['selection'],'next':nxt,'qk_nonfinite':int((~torch.isfinite(scores)).sum()),
                    'next_available':max(0,end-start),'next_retained':int(((nxt>=start)&(nxt<end)).sum())}
    return result

def make_cache(b,anchor,plans,layers=(),full=False):
    c=cache_new(b);chosen=set(layers)
    for li,e in anchor.items():
        if full:c.cache[li]={k:e[k] for k in ['k','v']}
        else:
            ix=plans[li]['next' if li in chosen else 'base']
            c.cache[li]={k:e[k].index_select(-2,ix) for k in ['k','v']}
            assert c.cache[li]['k'].shape[-2]==int(.25*e['k'].shape[-2])
    return c

def tensor_plan(plans):
    return {str(li):{k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in p.items()} for li,p in plans.items()}

def candidate_sets(b,randoms):
    out={'S25':[],'K4':LAYERS[b.name]}
    out.update({f'R{i+1}':r for i,r in enumerate(randoms)})
    out.update({f'L{li:02d}':[li] for li in range(b.n)})
    out.update({f'K4_MINUS_{li:02d}':[i for i in LAYERS[b.name] if i!=li] for li in LAYERS[b.name]})
    return out

@torch.inference_mode()
def probe(b,sample,x,start,block,anchor,official,randoms,check=None):
    assert block<7
    plans=selections(anchor,start)
    for li,e in anchor.items():
        for k in ['k','v']:
            assert torch.equal(official.cache[li][k],e[k].index_select(-2,plans[li]['base']))
    masks=x[0,start:start+32]==b.mask
    assert masks.any()
    ext=torch.cat((torch.arange(start,device=x.device),torch.arange(start+32,x.shape[1],device=x.device)))
    assert (x[0,start+32:start+64]==b.mask).all()
    stamps={li:{k:e[k].clone() for k in ['k','v']} for li,e in anchor.items()}
    inp=x[:,start:start+32].clone()
    fullcache=make_cache(b,anchor,plans,full=True)
    logits=b.forward(inp,start,2,fullcache)
    assert torch.isfinite(logits).all()
    ref=b.aligned_logits(logits,start,False)[0,masks].float().log_softmax(-1)
    del logits,fullcache
    rows=[];base_logits=None
    for name,layers in candidate_sets(b,randoms).items():
        if check:check()
        c=make_cache(b,anchor,plans,layers)
        logits=b.forward(inp,start,2,c)
        assert torch.isfinite(logits).all()
        lp=b.aligned_logits(logits,start,False)[0,masks].float().log_softmax(-1)
        js=js_from_logp(ref,lp)
        assert torch.isfinite(js).all() and float(js.min())>=-1e-6
        row={'condition':name,'layers':layers,'js_per_position':js.tolist(),'js':float(js.mean()),
             'kv_bytes':sum(v.numel()*v.element_size() for e in c.cache.values() for v in e.values()),
             'external_counts':[c.cache[li]['k'].shape[-2] for li in range(b.n)]}
        rows.append(row)
        if name=='S25':
            base_logits=logits.clone()
            native=b.forward(inp,start,2,official)
            assert torch.equal(native,logits),'Reconstructed S25 differs from native cache'
            del native
        del logits,lp,js,c
    for li,e in anchor.items():
        for k in ['k','v']:assert torch.equal(e[k],stamps[li][k]),'Anchor mutated'
    base=rows[0]['js']
    for row in rows:row['gain']=base-row['js']
    out={'block':block,'step':2,'start':start,'x_ids':x[0].tolist(),'active_positions':torch.where(masks)[0].tolist(),
         'prior_stop_present':any(t in b.stops for t in x[0,sample['prompt_length']:].tolist()),
         'anchor_age':'original block step1; no refresh','native_cache_exact':True,'anchor_values_unchanged':True,
         'plans':tensor_plan(plans),'external_absolute_positions':ext.tolist(),'rows':rows}
    return out,base_logits

@torch.inference_mode()
def generate(b,sample,layers=(),randoms=None,probe_callback=None,check=None):
    plen=sample['prompt_length'];x=torch.full((1,plen+256),b.mask,dtype=torch.long,device=b.device)
    x[:,:plen]=torch.tensor([sample['prompt_ids']],device=b.device)
    commits=[];finite=torch.ones((),dtype=torch.bool,device=b.device)
    for block in range(8):
        if check:check()
        start=plen+32*block
        c=PriorityCache(b.n,b.device,kernel_size=3,keep_ratio=.25,priority_layers=tuple(layers))
        anchor=None
        for step in range(32):
            if step==1 and probe_callback is not None and block<7:
                assert not layers
                anchor={};anchor_input=x.clone()
                logits=b.forward(x,start,1,c,anchor)
            elif step==2 and probe_callback is not None and block<7:
                data,logits=probe(b,sample,x,start,block,anchor,c,randoms,check)
                data['anchor_input_ids']=anchor_input[0].tolist()
                probe_callback(data)
                del data,anchor,anchor_input
                anchor=None
            else:logits=b.forward(x if step<2 else x[:,start:start+32],start,min(step,2),c)
            finite &= torch.isfinite(logits).all()
            pos,tok,conf=b.choice(logits,x,start,step,step<2)
            assert b.mask not in tok
            commits.append({'block':block,'step':step,'positions':pos,'tokens':tok,'confidence':conf})
            if pos:x[0,pos]=torch.tensor(tok,device=b.device)
            del logits
        del c
    assert finite.item()
    ids=x[0,plen:].tolist();assert b.mask not in ids
    return {'commits':commits,**b.score(sample,ids)}
