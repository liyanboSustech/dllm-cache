"""Equal-budget mechanistic KV interventions; diagnostic, not speed methods.

All conditions keep S25 tensor geometry. Missing next-block positions replace
the lowest-ranked non-next S25 slots at fixed layers. Hybrid K/V use this fixed
slot correspondence. Normalization probes explicitly alter attention algebra;
they are not a deployment proposal or logit/EOS intervention.
"""
from contextlib import contextmanager, nullcontext
import math
import torch
from discovery_common import CustomCache
from interruption_core import native_choice, slim
from run_discovery_diagnostics import capture
from priority_kv_core import importance_scores

LAYERS=(4,5,6,21)
CONDITIONS=['S25','KV_NEXT','K_NEXT','V_NEXT','NEXTK_OLDPOS','OLDK_NEXTPOS',
            'NOCTX_K','NOCTX_V','NOCTX_KV','REPEAT_NEXT','REPEAT_REPHASE',
            'NORM_ONLY','NUMERATOR_ONLY']
NOCTX={'NOCTX_K','NOCTX_V','NOCTX_KV'}


def rephase(key,source,target,theta):
    """R(target) R(source)^-1 on already-RoPE K, using float32 delta rotation."""
    if torch.equal(source,target):return key.clone()
    dim=key.shape[-1]
    inv=1.0/(theta**(torch.arange(0,dim,2,device=key.device,dtype=torch.float32)/dim))
    angle=(target-source).float()[:,None]*inv[None,:]
    angle=torch.cat((angle,angle),-1)[None,None]
    a,b=key.float().chunk(2,-1)
    rotated=torch.cat((-b,a),-1)
    return (key.float()*angle.cos()+rotated*angle.sin()).to(key.dtype)


def plan(entry,start,end,length,theta):
    selection=entry['selection']
    absolute=torch.cat((torch.arange(start,device=selection.device),torch.arange(end,length,device=selection.device)))
    near=torch.arange(start,min(start+end-start,len(absolute)),device=selection.device)
    target=selection.clone()
    missing=near[~torch.isin(near,selection)]
    candidates=torch.where(~torch.isin(selection,near))[0]
    scores=importance_scores(entry['k'],entry['q'],3)[0]
    # Every primary sample has budget >= block length. Explicitly fail otherwise.
    assert len(selection)>=len(near)
    evict=candidates[torch.argsort(scores[selection[candidates]],stable=True)[:len(missing)]]
    # A fixed positional correspondence makes hybrid K/V comparisons auditable.
    evict=evict[torch.argsort(absolute[selection[evict]])]
    target[evict]=missing
    assert len(torch.unique(target))==len(target)
    assert torch.isin(near,target).all()
    # Membership check against the same priority top-k rule, allowing boundary ties.
    rest=target[~torch.isin(target,near)]
    allowed=scores.clone();allowed[near]=-torch.inf
    if len(rest):
        cutoff=allowed.topk(len(rest)).values.min()
        assert (scores[rest]>=cutoff).all()
    base={n:entry[n].index_select(-2,selection) for n in ['k','v']}
    nxt={n:entry[n].index_select(-2,target) for n in ['k','v']}
    return {'base':base,'next':nxt,'selection':selection,'target':target,'slots':evict,
            'near':near,'absolute':absolute,'theta':theta}


def plans(model,anchor,start,end,length,layers=LAYERS):
    return {li:plan(e,start,end,length,model.config.rope_theta) for li,e in anchor.items() if li in layers}


def make_cache(model,anchor,mapping,condition,contextless=None,reverse=False):
    c=CustomCache(model.config.n_layers,next(model.parameters()).device,kernel_size=3,keep_ratio=.25)
    for li,e in anchor.items():
        original={n:e[n].index_select(-2,e['selection']) for n in ['k','v']}
        if li not in mapping or condition in ['S25','NORM_ONLY']:
            c.cache[li]=original;continue
        p=mapping[li];slots=p['slots'];target=p['target'];absolute=p['absolute']
        base=p['base'];nxt=p['next']
        k=base['k'].clone();v=base['v'].clone()
        src=slots.flip(0) if reverse else slots
        if condition in ['KV_NEXT','K_NEXT','NEXTK_OLDPOS','NOCTX_V','REPEAT_NEXT','REPEAT_REPHASE','NUMERATOR_ONLY']:
            k[:,:,slots]=nxt['k'][:,:,src]
        if condition in ['KV_NEXT','V_NEXT','NEXTK_OLDPOS','NOCTX_K','REPEAT_NEXT','REPEAT_REPHASE','NUMERATOR_ONLY']:
            v[:,:,slots]=nxt['v'][:,:,src]
        if condition=='NEXTK_OLDPOS':
            k[:,:,slots]=rephase(nxt['k'][:,:,slots],absolute[target[slots]],absolute[p['selection'][slots]],p['theta'])
        elif condition=='OLDK_NEXTPOS':
            k[:,:,slots]=rephase(base['k'][:,:,slots],absolute[p['selection'][slots]],absolute[target[slots]],p['theta'])
        if condition in NOCTX:
            assert contextless is not None
            if condition in ['NOCTX_K','NOCTX_KV']:k[:,:,slots]=contextless[li]['k'].index_select(-2,target[slots])
            if condition in ['NOCTX_V','NOCTX_KV']:v[:,:,slots]=contextless[li]['v'].index_select(-2,target[slots])
        if condition in ['REPEAT_NEXT','REPEAT_REPHASE'] and len(slots):
            # First next position always exists, even if S25 retained no future KV.
            donor=p['near'][0:1]
            kr=e['k'].index_select(-2,donor).expand(-1,-1,len(slots),-1)
            vr=e['v'].index_select(-2,donor).expand(-1,-1,len(slots),-1)
            if condition=='REPEAT_REPHASE':
                kr=rephase(kr,absolute[donor].expand(len(slots)),absolute[target[slots]],p['theta'])
            k[:,:,slots]=kr;v[:,:,slots]=vr
        c.cache[li]={'k':k,'v':v}
        assert k.shape==original['k'].shape and v.shape==original['v'].shape
    return c


@contextmanager
def normalization_probe(model,mapping,condition,audit=None):
    """2x2 numerator/denominator intervention, all queries in a target layer.

NORM_ONLY uses S numerator / T denominator; NUMERATOR_ONLY T numerator / S
denominator. S and T are evaluated with the same on-path query/current-block K.
The ratio does not clamp, gate EOS, or inject a residual direction.
"""
    originals=[]
    if condition not in ['NORM_ONLY','NUMERATOR_ONLY']:
        yield;return
    for li,p in mapping.items():
        block=model.model.transformer.blocks[li]
        original=block._scaled_dot_product_attention;originals.append((block,original))
        def wrapped(q,k,v,*args,p=p,li=li,original=original,**kwargs):
            out=original(q,k,v,*args,**kwargs)
            n=p['base']['k'].shape[-2]
            current=k[:,:,n:]
            ks=torch.cat((p['base']['k'],current),-2).float()
            kt=torch.cat((p['next']['k'],current),-2).float()
            scale=math.sqrt(q.shape[-1])
            zs=torch.logsumexp(q.float()@ks.transpose(-1,-2)/scale,-1,keepdim=True)
            zt=torch.logsumexp(q.float()@kt.transpose(-1,-2)/scale,-1,keepdim=True)
            logratio=zs-zt if condition=='NORM_ONLY' else zt-zs
            value=(out.float()*logratio.exp()).to(out.dtype)
            assert torch.isfinite(value).all()
            if audit is not None:
                audit.append({'layer':li,'log_ratio_min':float(logratio.min()),'log_ratio_max':float(logratio.max())})
            return value
        block._scaled_dot_product_attention=wrapped
    try:yield
    finally:
        for block,original in originals:block._scaled_dot_product_attention=original


@torch.inference_mode()
def empty_context(model,length,start,end,device):
    x=torch.full((1,length),126336,dtype=torch.long,device=device)
    cache=CustomCache(model.config.n_layers,device,kernel_size=3,keep_ratio=.25)
    with capture(model,start,end,detailed=False) as cap:
        unused=model(x,start,0,cache).logits
    del unused
    return cap


@torch.inference_mode()
def continuation(model,sample,fork,condition,check=None,budget=256,audit=None):
    x=torch.tensor([fork['input_ids']],device=next(model.parameters()).device)
    plen=sample['prompt_length'];finite=torch.ones((),device=x.device,dtype=torch.bool)
    for block in range(fork['block'],budget//32):
        if check:check()
        start=plen+block*32;end=start+32
        first=fork['step'] if block==fork['block'] else 0
        anchor=fork['anchor'] if block==fork['block'] else None
        official=CustomCache(model.config.n_layers,x.device,kernel_size=3,keep_ratio=.25)
        mapping=plans(model,anchor,start,end,x.shape[1]) if anchor is not None else None
        noctx=empty_context(model,x.shape[1],start,end,x.device) if condition in NOCTX else None
        cache=make_cache(model,anchor,mapping,condition,noctx) if anchor is not None else None
        for step in range(first,32):
            if step==0:logits=model(x,start,0,official).logits
            elif step==1:
                with capture(model,start,end) as captured:logits=model(x,start,1,official).logits
                anchor=captured
                mapping=plans(model,anchor,start,end,x.shape[1])
                cache=make_cache(model,anchor,mapping,condition,noctx)
            else:
                with normalization_probe(model,mapping,condition):
                    logits=model(x[:,start:end],start,2,cache).logits
            finite &= torch.isfinite(logits).all()
            pos,token,_=native_choice(logits,x,start,end,step<2)
            x[0,pos]=token
            del logits
        counts=[int(cache.cache[li]['k'].shape[-2]) for li in range(model.config.n_layers)]
        assert all(n==int((x.shape[1]-32)*.25) for n in counts)
        if audit is not None:audit.append({'block':block,'external_counts':counts,'replaced_slots':{li:len(p['slots']) for li,p in mapping.items()},'kv_bytes':sum(t.numel()*t.element_size() for e in cache.cache.values() for t in e.values())})
        del cache,mapping,anchor,noctx,official
    assert finite.item()
    ids=x[0,plen:].tolist();assert 126336 not in ids
    return ids
