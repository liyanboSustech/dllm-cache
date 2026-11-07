"""Frozen, model-specific decoding plus shared equal-budget interventions."""
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
import hashlib,json,math
import torch
import torch.nn.functional as F
from discovery_common import ROOT,CustomCache,load_model
from priority_kv_core import importance_scores
from rescore_discovery import parse_final_field
import kv_factor_core as factor

MODEL_PATHS={'llada':'/dataset/money-shredder-group/tv/LLaDA-8B-Instruct',
             'dream':'/dataset/money-shredder-group/tv/Dream-v0-Instruct-7B'}
LAYERS={'llada':(4,5,6,21),'dream':(3,4,5,18)}


class Backend:
    def __init__(self,name,model,tokenizer):
        self.name,self.model,self.tokenizer=name,model,tokenizer
        self.n=model.config.n_layers if name=='llada' else model.config.num_hidden_layers
        self.mask=126336 if name=='llada' else 151666
        self.stops={126081,126348} if name=='llada' else {151643,151645}
        self.layers=LAYERS[name]
        self.device=next(model.parameters()).device
        self.model.config.n_layers=self.n
        self.model.config.keep_ratio=.25
        self.model.config.kernel_size=3
        self.sampler_options={'temperature':0.}
        if name=='dream':
            g=model._prepare_generation_config(None,temperature=0,alg='maskgit_plus',eps=.001)
            self.sampler_options.update(top_k=g.top_k,top_p=g.top_p)

    @classmethod
    def load(cls,name):
        if name=='llada':m,t,info=load_model()
        else:
            from transformers import AutoTokenizer
            from vendor.sparse_dream.configuration_dream import DreamConfig
            from vendor.sparse_dream.modeling_dream import DreamModel
            torch.set_num_threads(4);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
            torch.backends.cuda.matmul.allow_tf32=False
            t=AutoTokenizer.from_pretrained(MODEL_PATHS[name],trust_remote_code=True,local_files_only=True)
            cfg=DreamConfig.from_pretrained(MODEL_PATHS[name],local_files_only=True)
            cfg.block_len=32;cfg.kernel_size=3;cfg.keep_ratio=.25;cfg.use_cache=False
            # The upstream override assumes a model, not HF's (model, info) tuple.
            m=DreamModel.from_pretrained(MODEL_PATHS[name],config=cfg,torch_dtype=torch.float16,
                device_map={'':'cuda:0'},local_files_only=True,attn_implementation='sdpa')
            expected=set(m.state_dict())
            stored=set(json.loads((Path(MODEL_PATHS[name])/'model.safetensors.index.json').read_text())['weight_map'])
            assert expected==stored,(expected-stored,stored-expected)
            info={'key_names_exactly_match_safetensors_index':True,'tensor_key_count':len(expected),
                'default_loader_shape_check':True,'missing_keys':[],'unexpected_keys':[],'mismatched_keys':[],'error_msgs':[]}
            m.eval()
        assert all(p.device.type=='cuda' and p.dtype==torch.float16 for p in m.parameters())
        return cls(name,m,t),info

    @contextmanager
    def operator(self,captured=None,start=0,end=32,mode='anchor',layers=None):
        """Preserve 4-head Dream storage; repeat heads only for V100 SDPA."""
        original=F.scaled_dot_product_attention;counter=0
        def wrapped(q,k,v,*args,**kwargs):
            nonlocal counter
            li=counter;counter+=1
            if captured is not None and (layers is None or li in layers):
                if mode=='anchor':
                    ext=torch.cat((torch.arange(start,device=k.device),torch.arange(end,k.shape[-2],device=k.device)))
                    ek=k.index_select(-2,ext);ev=v.index_select(-2,ext);qb=q[:,:,start:end].clone()
                    sel=importance_scores(ek,qb,3)[0].topk(int(ek.shape[-2]*.25)).indices
                    captured[li]={'k':ek,'v':ev,'q':qb,'selection':sel}
                else:captured[li]={'k':k.clone(),'v':v.clone()}
            if kwargs.pop('enable_gqa',False):
                assert q.shape[1]%k.shape[1]==0
                k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
                v=v.repeat_interleave(q.shape[1]//v.shape[1],dim=1)
            return original(q,k,v,*args,**kwargs)
        F.scaled_dot_product_attention=wrapped
        try:yield
        finally:
            F.scaled_dot_product_attention=original
            if captured is not None:assert counter==self.n,(counter,self.n)

    def forward(self,x,start,state,cache,captured=None,mode='anchor',layers=None):
        with self.operator(captured,start,start+32,mode,layers):
            if self.name=='llada':return self.model(x,start,state,cache).logits
            return self.model(input_ids=x,position_offset=start,cache_state=state,customcache=cache).logits

    def aligned_logits(self,logits,start,full):
        if self.name=='dream':logits=torch.cat((logits[:,:1],logits[:,:-1]),dim=1)
        return logits[:,start:start+32] if full else logits

    def choice(self,logits,x,start,step,full):
        """Match each official sampler, including shape/tie semantics."""
        if self.name=='llada':
            from interruption_core import native_choice
            p,t,c=native_choice(logits,x,start,start+32,full)
            return [int(p)],[int(t)],[float(c)]
        from vendor.sparse_dream.generation_utils import sample_tokens
        z=self.aligned_logits(logits,start,full)
        if step==0:
            c,t=sample_tokens(z,**self.sampler_options)
            return [start],[int(t[0,0])],[float(c[0,0])]
        masked=x[:,start:start+32]==self.mask
        c,t=sample_tokens(z[masked],**self.sampler_options)
        times=torch.linspace(1,.001,33,device=x.device)
        n=int((masked.sum()/masked.shape[0])*(1-times[step+1]/times[step])) if step<31 else int(masked.sum())
        scores=torch.full_like(x[:,start:start+32],-torch.inf,dtype=z.dtype);scores[masked]=c
        fill=torch.full_like(x[:,start:start+32],self.mask);fill[masked]=t
        pos=scores.topk(n,dim=-1).indices[0]
        return (pos+start).tolist(),fill[0,pos].tolist(),scores[0,pos].tolist()

    def score(self,sample,ids):
        stops=[i for i,t in enumerate(ids) if t in self.stops];end=min(stops) if stops else len(ids)
        text=self.tokenizer.decode(ids[:end],skip_special_tokens=True).strip();parsed=parse_final_field(text)
        correct=parsed is not None and Decimal(parsed)==Decimal(sample['answer'].replace(',',''))
        return {'generated_token_ids':ids,'raw_output':self.tokenizer.decode(ids,skip_special_tokens=False),
            'answer_before_stop':text,'first_stop_position':min(stops) if stops else None,
            'remaining_mask_tokens':ids.count(self.mask),'parsed_answer':parsed,'correct':bool(correct),
            'gold_answer':sample['answer'],'no_stop_within_budget':not stops}


def cache_new(backend):return CustomCache(backend.n,backend.device,kernel_size=3,keep_ratio=.25)


@torch.inference_mode()
def context_forward(backend,anchor_input,start,prompt_ids=None,empty=False,layers=None):
    x=anchor_input.clone()
    if empty:x.fill_(backend.mask)
    elif prompt_ids is not None:x[:,:len(prompt_ids)]=torch.tensor([prompt_ids],device=x.device)
    captured={}
    logits=backend.forward(x,start,0,cache_new(backend),captured,mode='absolute',layers=layers or backend.layers)
    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite donor logits')
    del logits
    return captured


def norm_match(donor,reference):
    dn=donor.float().norm(dim=-1,keepdim=True);rn=reference.float().norm(dim=-1,keepdim=True)
    assert (dn>0).all()
    return (donor.float()*(rn/dn)).to(reference.dtype)


def cache_from_sources(backend,anchor,mapping,condition,donor=None,noctx=None):
    model=backend.model
    if condition=='C100':
        cache=cache_new(backend)
        cache.cache={li:{n:e[n] for n in ['k','v']} for li,e in anchor.items()}
        return cache
    base_name={'KV_ALL':'KV_NEXT','K_ALL':'K_NEXT','V_ALL':'V_NEXT'}.get(condition,condition)
    if condition.startswith('CROSS_') or condition in ['NOCTX_V','ZERO_NEXT_V']:
        cache=factor.make_cache(model,anchor,mapping,'KV_NEXT')
        for li,p in mapping.items():
            slots=p['slots'];absolute=p['absolute'][p['target'][slots]]
            if condition=='ZERO_NEXT_V':cache.cache[li]['v'][:,:,slots]=0;continue
            if len(slots)==0:continue
            source=noctx if condition=='NOCTX_V' else donor
            which='V' if condition=='NOCTX_V' else condition.split('_')[1]
            for name in ['k','v']:
                if name.upper() not in which:continue
                tensor=source[li][name].index_select(-2,absolute)
                if condition.endswith('_NORM'):tensor=norm_match(tensor,p['next'][name][:,:,slots])
                cache.cache[li][name][:,:,slots]=tensor
        return cache
    return factor.make_cache(model,anchor,mapping,base_name)


@torch.inference_mode()
def build_cache(backend,sample,anchor,anchor_input,start,condition,noctx=None):
    layers=range(backend.n) if condition.endswith('_ALL') else backend.layers
    mapping=factor.plans(backend.model,anchor,start,start+32,anchor_input.shape[1],layers=layers)
    donor=None
    if condition.startswith('CROSS_') and any(len(p['slots']) for p in mapping.values()):
        key='B' if condition.endswith('_B') else 'A'
        d=sample['donors'][key];assert d['prompt_length']==sample['prompt_length']
        donor=context_forward(backend,anchor_input,start,d['prompt_ids'])
    cache=cache_from_sources(backend,anchor,mapping,condition,donor,noctx)
    return cache,mapping,donor


def audit_cache(backend,cache,condition,length,block,mapping):
    expected=length-32 if condition=='C100' else int((length-32)*.25)
    counts=[]
    for e in cache.cache.values():
        assert e['k'].shape==e['v'].shape
        assert e['k'].shape[-2]==expected
        if not (torch.isfinite(e['k']).all() and torch.isfinite(e['v']).all()):
            raise FloatingPointError('Nonfinite cache vectors')
        counts.append(e['k'].shape[-2])
    return {'block':block,'external_counts':counts,
        'kv_bytes':sum(t.numel()*t.element_size() for e in cache.cache.values() for t in e.values()),
        'replaced_slots':{li:len(p['slots']) for li,p in mapping.items()}}


@torch.inference_mode()
def rollout(backend,sample,condition,budget=256,keep_checkpoints=False,check=None):
    plen=sample['prompt_length'];length=plen+budget
    x=torch.full((1,length),backend.mask,dtype=torch.long,device=backend.device)
    x[:,:plen]=torch.tensor([sample['prompt_ids']],device=x.device)
    checkpoints={};commits=[];audit=[];first_stop=None;finite=torch.ones((),device=x.device,dtype=torch.bool)
    noctx=context_forward(backend,x,plen,empty=True) if condition=='NOCTX_V' else None
    for block in range(budget//32):
        if check:check()
        start=plen+block*32;official=cache_new(backend);anchor=None;anchor_input=None;mapping=None
        for step in range(32):
            if step==0:logits=backend.forward(x,start,0,official)
            elif step==1:
                anchor={};anchor_input=x.clone()
                logits=backend.forward(x,start,1,official,anchor)
                # Baseline selection exactly reproduces the official cache.
                for li,e in anchor.items():
                    for n in ['k','v']:assert torch.equal(official.cache[li][n],e[n].index_select(-2,e['selection']))
                cache,mapping,donor=build_cache(backend,sample,anchor,anchor_input,start,condition,noctx)
                audit.append(audit_cache(backend,cache,condition,length,block,mapping));del donor
            else:logits=backend.forward(x[:,start:start+32],start,2,cache)
            finite &= torch.isfinite(logits).all()
            pos,tok,conf=backend.choice(logits,x,start,step,step<2)
            stop_positions=[p for p,t in zip(pos,tok) if t in backend.stops]
            row={'block':block,'step':step,'positions':pos,'tokens':tok,'confidence':conf}
            if keep_checkpoints and block==0 and step==2:
                checkpoints['early']={'x':x.clone(),'anchor':anchor,'anchor_input':anchor_input,'start':start,
                    'step':step,'event_position':pos[0] if pos else start+1,'native':row}
            if first_stop is None and stop_positions:
                first_stop={'block':block,'step':step,'position':min(stop_positions)}
                if keep_checkpoints and step>=2:
                    checkpoints['stop']={'x':x.clone(),'anchor':anchor,'anchor_input':anchor_input,'start':start,
                        'step':step,'event_position':min(stop_positions),'native':row}
            commits.append(row)
            if pos:x[0,pos]=torch.tensor(tok,device=x.device)
            del logits
        del cache,mapping,anchor,anchor_input,official
    if not finite.item():raise FloatingPointError('Nonfinite generation logits')
    ids=x[0,plen:].tolist();assert backend.mask not in ids,'Unfilled MASK after full budget'
    return {'ids':ids,'cache_audit':audit,'commits':commits,'first_stop_commit':first_stop,'checkpoints':checkpoints}


@torch.inference_mode()
def local_probe(backend,sample,cp,condition,noctx):
    x=cp['x'];start=cp['start'];anchor=cp['anchor']
    cache,mapping,donor=build_cache(backend,sample,anchor,cp['anchor_input'],start,condition,noctx)
    if condition=='ROPE_ROUNDTRIP':raise AssertionError('Use roundtrip_probe')
    logits=backend.forward(x[:,start:start+32],start,2,cache)
    row=local_metrics(backend,cp,logits)
    row.update(condition=condition,cache_audit=audit_cache(backend,cache,condition,x.shape[1],(start-sample['prompt_length'])//32,mapping))
    if donor is not None:
        stats={}
        for li,p in mapping.items():
            slots=p['slots'];indices=p['absolute'][p['target'][slots]]
            if not len(slots):continue
            stats[li]={}
            for n in ['k','v']:
                a=p['next'][n][:,:,slots].float();b=donor[li][n].index_select(-2,indices).float()
                stats[li][n]={'cosine_mean':float(F.cosine_similarity(a,b,dim=-1).mean()),
                    'norm_ratio_mean':float((b.norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-12)).mean()),
                    'relative_l2':float((a-b).norm()/a.norm().clamp_min(1e-12))}
        row['donor_statistics']=stats
    return row


def local_metrics(backend,cp,logits):
    assert torch.isfinite(logits).all()
    pos,tok,conf=backend.choice(logits,cp['x'],cp['start'],cp['step'],False)
    z=backend.aligned_logits(logits,cp['start'],False)[0,cp['event_position']-cp['start']].float()
    top=torch.topk(z,len(backend.stops)+1)
    other=next(float(v) for v,i in zip(top.values,top.indices) if int(i) not in backend.stops)
    return {'stop_margin':max(float(z[t]) for t in backend.stops)-other,
        'stop_probability':float(z.softmax(-1)[list(backend.stops)].sum()),
        'selected_positions':pos,'selected_tokens':tok,'selected_confidences':conf,
        'any_selected_stop':any(t in backend.stops for t in tok),'number_commits':len(pos)}


@torch.inference_mode()
def roundtrip_probe(backend,sample,cp):
    cache,mapping,_=build_cache(backend,sample,cp['anchor'],cp['anchor_input'],cp['start'],'S25')
    for li,p in mapping.items():
        slots=p['slots'];a=p['absolute'][p['selection'][slots]];b=p['absolute'][p['target'][slots]]
        k=cache.cache[li]['k'][:,:,slots]
        cache.cache[li]['k'][:,:,slots]=factor.rephase(factor.rephase(k,a,b,p['theta']),b,a,p['theta'])
    logits=backend.forward(cp['x'][:,cp['start']:cp['start']+32],cp['start'],2,cache)
    return {'condition':'ROPE_ROUNDTRIP',**local_metrics(backend,cp,logits)}
