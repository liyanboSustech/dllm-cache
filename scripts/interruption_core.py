"""Actual block-only S25 trajectories and paired KV policy continuations."""
from decimal import Decimal
import json
import hashlib
import torch
import torch.nn.functional as F
from discovery_common import CustomCache, score
from run_discovery_diagnostics import capture
from rescore_discovery import parse_final_field

POLICIES=['A25','A100','F25','F100']
STOP_IDS={126081,126348}


def slim(captured):
    return {li:{name:entry[name] for name in ['k','v','selection']} for li,entry in captured.items()}


def cache_for(model,anchor,current,policy):
    cache=CustomCache(model.config.n_layers,torch.device('cuda' if torch.cuda.is_available() else 'cpu'),kernel_size=3,keep_ratio=0.25)
    source=current if policy.startswith('F') else anchor
    for li,entry in source.items():
        indices=None
        if policy in ['A25','F25']:
            indices=anchor[li]['selection']
        elif policy=='F25_current':
            indices=current[li]['selection']
        cache.cache[li]={name:entry[name] if indices is None else entry[name].index_select(-2,indices) for name in ['k','v']}
    return cache


def native_choice(logits,x,start,end,full):
    inp=x if full else x[:,start:end]
    masked=inp==126336
    predicted=logits.argmax(-1)
    probabilities=F.softmax(logits,dim=-1)
    confidence=probabilities.gather(-1,predicted.unsqueeze(-1)).squeeze(-1)
    if full:confidence[:,end:]=-torch.inf
    confidence=torch.where(masked,confidence,-torch.inf)
    local=torch.topk(confidence[0],k=1).indices[0]
    absolute=local if full else local+start
    token=predicted[0,local]
    selected_confidence=confidence[0,local]
    return absolute,token,selected_confidence


@torch.inference_mode()
def step_logits(model,x,start,end,anchor,policy,cached=None):
    current=None
    if policy.startswith('F'):
        dummy=CustomCache(model.config.n_layers,x.device,kernel_size=3,keep_ratio=0.25)
        with capture(model,start,end,detailed=policy=='F25_current') as captured:
            unused=model(x,start,0,dummy).logits
        del unused
        current=captured
        cached=cache_for(model,anchor,current,policy)
    elif cached is None:
        cached=cache_for(model,anchor,None,policy)
    logits=model(x[:,start:end],start,2,cached).logits
    return logits


@torch.inference_mode()
def rollout(model,prompt_ids,policy,budget=256,fork=None,keep_trace=False,check=None):
    device=next(model.parameters()).device
    plen=len(prompt_ids)
    if fork is None:
        x=torch.full((1,plen+budget),126336,dtype=torch.long,device=device)
        x[:,:plen]=torch.tensor([prompt_ids],device=device)
        first_block,first_step=0,0
    else:
        x=torch.tensor([fork['input_ids']],device=device)
        first_block,first_step=fork['block'],fork['step']
    anchors={}
    traces=[]
    all_finite=torch.ones((),dtype=torch.bool,device=device)
    for block in range(first_block,budget//32):
        if check:check()
        start,end=plen+block*32,plen+(block+1)*32
        initial_step=first_step if block==first_block else 0
        anchor=fork['anchor'] if fork is not None and block==first_block and initial_step>=2 else None
        cached=cache_for(model,anchor,None,policy) if anchor is not None and policy.startswith('A') else None
        official_cache=CustomCache(model.config.n_layers,device,kernel_size=3,keep_ratio=0.25)
        if anchor is not None and keep_trace:anchors[block]=anchor
        for step in range(initial_step,32):
            if step<2:
                if step==1:
                    with capture(model,start,end) as captured:
                        logits=model(x,start,1,official_cache).logits
                    anchor=slim(captured)
                    del captured
                    # Recorded anchor selections must match the actual upstream cache.
                    for li,a in anchor.items():
                        for name in ['k','v']:
                            assert torch.equal(official_cache.cache[li][name],a[name].index_select(-2,a['selection']))
                    if keep_trace:anchors[block]=anchor
                    if policy.startswith('A'):cached=cache_for(model,anchor,None,policy)
                else:
                    logits=model(x,start,0,official_cache).logits
            else:
                logits=step_logits(model,x,start,end,anchor,policy,cached)
            all_finite &= torch.isfinite(logits).all()
            pos,token,confidence=native_choice(logits,x,start,end,full=step<2)
            pos_i,tok_i=int(pos.item()),int(token.item())
            if keep_trace:
                traces.append({'block':block,'step':step,'global_step':block*32+step,
                    'input_ids':x[0].tolist(),'position':pos_i,'relative_position':pos_i-plen,
                    'token':tok_i,'confidence':float(confidence.item()),'is_stop':tok_i in STOP_IDS})
            x[0,pos]=token
            del logits
        official_cache.clear()
    assert all_finite.item(), 'Nonfinite logits'
    ids=x[0,plen:].tolist()
    assert 126336 not in ids,'Unfilled MASK positions'
    return {'ids':ids,'trace':traces,'anchors':anchors,'finite':True}


def make_fork(trace,anchors,block,step):
    row=next(r for r in trace if r['block']==block and r['step']==step)
    return {'input_ids':row['input_ids'],'block':block,'step':step,'anchor':anchors[block]}


def evaluate(tokenizer,sample,ids):
    result=score(tokenizer,sample,ids)
    parsed=parse_final_field(result['answer_before_stop'])
    correct=parsed is not None and Decimal(parsed)==Decimal(sample['answer'].replace(',',''))
    result.update({'parsed_answer':parsed,'correct':bool(correct),'em':float(bool(correct)),
                   'missing_final_answer':parsed is None,'has_final_answer_field':parsed is not None})
    return result


@torch.inference_mode()
def stop_counterfactual(model,tokenizer,sample,fork,event):
    x=torch.tensor([fork['input_ids']],device=next(model.parameters()).device)
    start=sample['prompt_length']+fork['block']*32
    end=start+32
    rows=[]
    for policy in POLICIES:
        logits=step_logits(model,x,start,end,fork['anchor'],policy)
        pos,token,conf=native_choice(logits,x,start,end,full=False)
        original_position=event['position']-start
        vector=logits[0,original_position].float()
        probability=F.softmax(vector,-1)
        best_stop=max(float(vector[t].item()) for t in STOP_IDS)
        top=torch.topk(vector,3)
        best_other=next(float(value) for value,index in zip(top.values.tolist(),top.indices.tolist()) if index not in STOP_IDS)
        predicted=int(vector.argmax().item())
        row={'sample_id':sample['sample_id'],'policy':policy,'block':fork['block'],'step':fork['step'],
            'original_position':event['relative_position'],'original_token':event['token'],
            'selected_position':int(pos.item())-sample['prompt_length'],'selected_token':int(token.item()),
            'selected_token_text':tokenizer.decode([int(token.item())]),'selected_is_stop':int(token.item()) in STOP_IDS,
            'selected_confidence':float(conf.item()),'original_position_top1':predicted,
            'original_position_top1_text':tokenizer.decode([predicted]),
            'original_position_top1_is_stop':predicted in STOP_IDS,
            'original_position_stop_probability':float(sum(probability[t] for t in STOP_IDS).item()),
            'original_position_stop_minus_best_nonstop_logit':best_stop-best_other,
            'committed_pair_changed':int(pos.item())!=event['position'] or int(token.item())!=event['token']}
        if policy=='A25':
            assert not row['committed_pair_changed']
            assert row['selected_confidence']==event['confidence']
        rows.append(row)
    return rows
