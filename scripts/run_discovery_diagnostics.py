"""Teacher-forced KV interventions on the frozen dense trajectory.

Primary model interventions preserve full sequence GEMM geometry to remove
FP16 shape/reordering confounds. At every layer, external K/V are overwritten;
only the current block's evolving state can influence its next-layer state.
This is algebraically block-only propagation, but deliberately computes unused
external queries/pointwise operations. It is an expensive diagnostic, NOT a
deployable acceleration. A genuine block-only fresh-KV null is also recorded.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from discovery_common import *
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from rescore_discovery import rescore

CONDITIONS=['F100','A100','F25','A25']


def external_indices(length,start,end,device):
    return torch.cat((torch.arange(start,device=device),torch.arange(end,length,device=device)))


def select(q,k):
    # Exactly the vendored CustomCache rule, including FP16 operations/order.
    scores=torch.matmul(q.mean(dim=-2).unsqueeze(-2),k.transpose(-2,-1)).squeeze(-2)
    importance=F.max_pool1d(scores.mean(dim=1).unsqueeze(1),3,stride=1,padding=1).squeeze(1)
    return torch.topk(importance,k=int(k.shape[-2]*0.25),dim=-1).indices.squeeze(0)


def index_hash(entries):
    h=hashlib.sha256()
    for layer in sorted(entries):
        h.update(entries[layer]['selection'].cpu().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def capture(model,start,end,detailed=True):
    saved={}
    originals=[]
    for li,block in enumerate(model.model.transformer.blocks):
        original=block._scaled_dot_product_attention
        originals.append((block,original))
        def wrapper(q,k,v,*args,li=li,original=original,**kwargs):
            result=original(q,k,v,*args,**kwargs)
            ext=external_indices(k.shape[-2],start,end,k.device)
            record={'k':k.index_select(-2,ext),'v':v.index_select(-2,ext)}
            if detailed:
                record.update({'q':q[:,:,start:end].clone(),'bk':k[:,:,start:end].clone(),
                               'bv':v[:,:,start:end].clone(),'o':result[:,:,start:end].clone()})
                record['selection']=select(record['q'],record['k'])
            saved[li]=record
            return result
        block._scaled_dot_product_attention=wrapper
    try:
        yield saved
    finally:
        for block,original in originals:
            block._scaled_dot_product_attention=original


def pack(k,v,bk,bv,start,indices=None):
    if indices is None:
        # Reinsert block into its original sequence positions (full-size null).
        return (torch.cat((k[:,:,:start],bk,k[:,:,start:]),dim=-2),
                torch.cat((v[:,:,:start],bv,v[:,:,start:]),dim=-2))
    return torch.cat((k.index_select(-2,indices),bk),dim=-2),torch.cat((v.index_select(-2,indices),bv),dim=-2)


@contextmanager
def inject(model,start,end,condition,anchor,current):
    originals=[]
    for li,block in enumerate(model.model.transformer.blocks):
        original=block._scaled_dot_product_attention
        originals.append((block,original))
        def wrapper(q,k,v,*args,li=li,original=original,**kwargs):
            source=anchor[li] if condition.startswith('A') else current[li]
            indices=None
            if condition in ['F25','A25']:
                indices=anchor[li]['selection']
            elif condition=='F25_current':
                indices=current[li]['selection']
            keys,values=pack(source['k'],source['v'],k[:,:,start:end],v[:,:,start:end],start,indices)
            return original(q,keys,values,*args,**kwargs)
        block._scaled_dot_product_attention=wrapper
    try:
        yield
    finally:
        for block,original in originals:
            block._scaled_dot_product_attention=original


def decide(logits,x,start,end):
    # logits are current-block logits; retain full-vector top-k tie semantics.
    predicted=logits.argmax(-1)
    confidence=F.softmax(logits,dim=-1).gather(-1,predicted.unsqueeze(-1)).squeeze(-1)
    masked=x[:,start:end]==126336
    confidence=torch.where(masked,confidence,-torch.inf)
    full=torch.full_like(x,-torch.inf,dtype=confidence.dtype)
    full[:,start:end]=confidence
    pos=torch.topk(full[0],k=1).indices[0]
    token=predicted[0,pos-start]
    return pos,token,confidence,predicted


def summarize(values):
    flat=values.float().flatten()
    if flat.numel()==0:
        return {'mean':None,'median':None,'p10':None,'p95':None}
    q=torch.quantile(flat,torch.tensor([0.1,0.5,0.95],device=flat.device)).tolist()
    return {'mean':flat.mean().item(),'median':q[1],'p10':q[0],'p95':q[2]}


def drift(current,other):
    current,other=current.float(),other.float()
    diff=current-other
    per_token=diff.norm(dim=-1)/(current.norm(dim=-1)+1e-8)
    per_head=diff.norm(dim=(-2,-1))/(current.norm(dim=(-2,-1))+1e-8)
    return {'relative_l2':(diff.norm()/(current.norm()+1e-8)).item(),
            'head_token_distribution':summarize(per_token),'per_head':per_head[0].tolist()}


def compare_logits(logits,ref,x,start,end,stop):
    remaining=x[0,start:end]==126336
    refpos,reftok,refconfidence,refpred=decide(ref,x,start,end)
    pos,token,confidence,predicted=decide(logits,x,start,end)
    logp=F.log_softmax(ref.float(),-1)
    logq=F.log_softmax(logits.float(),-1)
    logm=torch.logaddexp(logp,logq)-math.log(2)
    js=(0.5*((logp.exp()*(logp-logm)).sum(-1)+(logq.exp()*(logq-logm)).sum(-1)))[0]
    top2=torch.topk(ref.float(),2,dim=-1).values
    margins=(top2[...,0]-top2[...,1])[0]
    topconf=torch.topk(refconfidence[0],2).values.float()
    result={'committed_position':int(pos.item()),'committed_token':int(token.item()),
        'reference_position':int(refpos.item()),'reference_token':int(reftok.item()),
        'position_changed':bool((pos!=refpos).item()),
        'committed_token_changed':bool((token!=reftok).item()),
        'committed_pair_changed':bool(((pos!=refpos)|(token!=reftok)).item()),
        'reference_selected_token_changed_at_same_position':bool((predicted[0,refpos-start]!=refpred[0,refpos-start]).item()),
        'top1_change_fraction_remaining':(predicted[0,remaining]!=refpred[0,remaining]).float().mean().item(),
        'js_remaining':summarize(js[remaining]),
        'relative_logit_l2':((logits.float()-ref.float()).norm()/(ref.float().norm()+1e-8)).item(),
        'max_abs_logit_error':(logits.float()-ref.float()).abs().max().item(),
        'reference_token_margin_remaining':summarize(margins[remaining]),
        'reference_position_confidence_gap':float((topconf[0]-topconf[1]).item()),
        'reference_selected_confidence':float(refconfidence[0,refpos-start].item()),
        'logits_finite':bool(torch.isfinite(logits).all().item()),
        'reference_selected_before_final_stop':int(refpos.item())<stop,
        'selected_before_reference_final_stop':int(pos.item())<stop}
    relevant=remaining & (torch.arange(start,end,device=x.device)<stop)
    result['js_remaining_before_reference_stop']=summarize(js[relevant])
    result['remaining_positions_before_reference_stop']=int(relevant.sum().item())
    result['top1_change_fraction_remaining_before_reference_stop']=(predicted[0,relevant]!=refpred[0,relevant]).float().mean().item() if relevant.any().item() else None
    return result


def local_metrics(anchor,current,previous,start,end,remaining,base,path):
    for li,c in current.items():
        a=anchor[li]
        keys,values=pack(c['k'],c['v'],c['bk'],c['bv'],start)
        reference=F.scaled_dot_product_attention(c['q'],keys,values,is_causal=False)
        ext=external_indices(keys.shape[-2],start,end,keys.device)
        probabilities=F.softmax(torch.matmul(c['q'].float(),keys.float().transpose(-2,-1))/math.sqrt(keys.shape[-1]),dim=-1)
        extprob=probabilities.index_select(-1,ext)
        extmass=extprob.sum(-1)
        retained=extprob.index_select(-1,a['selection']).sum(-1)
        coverage=retained/(extmass+1e-8)
        overlap=torch.isin(a['selection'],c['selection']).float().mean().item()
        row={**base,'layer':li,'selection_overlap':overlap,
             'external_attention_mass_all':summarize(extmass),
             'coverage_all':summarize(coverage),'coverage_remaining':summarize(coverage[:,:,remaining]),
             'k_anchor':drift(c['k'],a['k']),'v_anchor':drift(c['v'],a['v']),
             'k_adjacent':drift(c['k'],previous[li]['k']),'v_adjacent':drift(c['v'],previous[li]['v']),
             'local_fresh_vs_dense_output_relative_l2':((reference.float()-c['o'].float()).norm()/(c['o'].float().norm()+1e-8)).item(),
             'operator_errors':{}}
        for cond in CONDITIONS:
            source=a if cond.startswith('A') else c
            selected=a['selection'] if '25' in cond else None
            k,v=pack(source['k'],source['v'],c['bk'],c['bv'],start,selected)
            output=F.scaled_dot_product_attention(c['q'],k,v,is_causal=False)
            delta=output.float()-reference.float()
            row['operator_errors'][cond]={'relative_l2':(delta.norm()/(reference.float().norm()+1e-8)).item(),
                'head_query_relative_error':summarize(delta.norm(dim=-1)/(reference.float().norm(dim=-1)+1e-8))}
        row['operator_error_interaction']=row['operator_errors']['A25']['relative_l2']-row['operator_errors']['A100']['relative_l2']-row['operator_errors']['F25']['relative_l2']
        append(path,row)


def run_conditions(model,x,start,end,anchor,current,reference,base,stop,path,conditions):
    rows=[]
    dummy=CustomCache(model.config.n_layers,x.device,kernel_size=3,keep_ratio=1)
    for cond in conditions:
        with inject(model,start,end,cond,anchor,current):
            logits=model(x,start,0,dummy).logits[:,start:end].contiguous()
        if cond in ['F100','A100']:
            all_indices=torch.arange(x.shape[1]-(end-start),dtype=torch.int64).numpy().tobytes()
            kept_hash=hashlib.sha256(all_indices*len(anchor)).hexdigest()
        else:
            kept_hash=index_hash(current) if cond=='F25_current' else index_hash(anchor)
        row={**base,'condition':cond,'anchor_selection_sha256':index_hash(anchor),
            'selection_kind':'all_external' if cond in ['F100','A100'] else 'current_top25' if cond=='F25_current' else 'anchor_top25',
            'selected_set_sha256':kept_hash,
            **compare_logits(logits,reference,x,start,end,stop)}
        if cond=='F100':
            row['exact_logits_equal']=torch.equal(logits,reference)
            assert row['exact_logits_equal'] and not row['committed_pair_changed'], 'Fresh/full null does not align'
        assert row['logits_finite']
        append(path,row)
        rows.append(row)
        del logits
    return rows


def baseline_rows(path):
    return {(r['sample_id'],r['mode']):r for r in rescore(path)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baselines',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args()
    assert (args.baselines/'completed.json').exists()
    frozen=json.loads((args.baselines/'frozen_protocol.json').read_text())
    samples=read_jsonl(DATA/'samples.jsonl')
    samples=[s for s in samples if s['sample_id'] in frozen['diagnostic_ids']]
    assert len(samples)==4 and frozen['samples_sha256']==sha(DATA/'samples.jsonl')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'command.json',{'argv':sys.argv,'cwd':os.getcwd()})
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu)
    uuid=wait_idle(args.gpu,60,args.output)
    assert uuid==json.loads((args.baselines/'environment.json').read_text())['gpu_uuid']
    model,tokenizer,loading=load_model()
    monitor=Monitor(uuid,args.output/'gpu_monitor.jsonl')
    write_json(args.output/'environment.json',{'gpu_uuid':uuid,'dtype':'float16','backend':'EFFICIENT_ATTENTION',
        'diagnostic_implementation':'Full sequence shapes, external KV overwritten at every layer; current-block propagation is algebraically block-only. Unused external queries computed to guarantee exact F100 null. Not acceleration timing.',
        'files_sha256':{str(t.relative_to(ROOT)):sha(t) for t in [Path(__file__),ROOT/'scripts/discovery_common.py',ROOT/'scripts/run_baseline_smoke.py']},
        'baseline_environment':str((args.baselines/'environment.json').resolve()),'loading_info':loading,
        'aggregation':'KV drift per-head/token vector relative L2; head-level L2 across external tokens; local attention errors before W_O. JS in FP32, decisions in native FP16.',
        'selection':'Anchor and current raw-score maxpool/top-k match upstream. Fixed anchor selection for all primary sparse interventions.'})
    baselines=baseline_rows(args.baselines)
    budget=frozen['gen_length']
    all_rows=[]
    saved=[]
    validations=[]
    try:
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for sample_index,s in enumerate(samples):
                monitor.check()
                plen=s['prompt_length']
                x=torch.full((1,plen+budget),126336,dtype=torch.long,device='cuda')
                x[:,:plen]=torch.tensor([s['prompt_ids']],device='cuda')
                dense=baselines[(s['sample_id'],'D')]
                stop=plen+(dense['first_stop_position'] if dense['first_stop_position'] is not None else budget)
                dummy=CustomCache(32,x.device,kernel_size=3,keep_ratio=1)
                for block in [0,1]:
                    start=plen+block*32
                    end=start+32
                    anchor=None
                    previous=None
                    for step in range(32):
                        monitor.check()
                        current=None
                        if step in [1,7,8,23,24]:
                            with capture(model,start,end,detailed=step not in [7,23]) as captured:
                                full=model(x,start,0,dummy).logits
                            if step==1:
                                anchor=captured
                                anchor_input=x[0].tolist()
                            elif step in [7,23]:
                                previous=captured
                            else:
                                current=captured
                        else:
                            full=model(x,start,0,dummy).logits
                        reference=full[:,start:end].contiguous()
                        del full
                        pos,tok,_,_=decide(reference,x,start,end)
                        if step==1 and block==0 and sample_index==0:
                            # Anchor identity and selector equivalence, before scientific comparisons.
                            sanity={}
                            for cond in CONDITIONS:
                                with inject(model,start,end,cond,anchor,anchor):
                                    sanity[cond]=model(x,start,0,dummy).logits[:,start:end].contiguous()
                            assert torch.equal(sanity['F100'],sanity['A100']) and torch.equal(sanity['F25'],sanity['A25'])
                            assert torch.equal(sanity['F100'],reference)
                            a=anchor[0]
                            k,v=pack(a['k'],a['v'],a['bk'],a['bv'],start)
                            original_cache=CustomCache(32,x.device,kernel_size=3,keep_ratio=0.25)
                            original_cache.update_cache(0,k,v)
                            original_cache.filter_cache(0,a['q'],start,32)
                            assert torch.equal(original_cache.cache[0]['k'],a['k'].index_select(-2,a['selection']))
                            assert torch.equal(original_cache.cache[0]['v'],a['v'].index_select(-2,a['selection']))
                            validations.append({'anchor_null_exact':True,'selection_matches_upstream':True})
                            del sanity,original_cache,k,v
                        if current is not None:
                            base={'sample_id':s['sample_id'],'task':s['task'],'block':block,'step':step,'anchor_step':1,
                                'dense_final_correct':dense['correct'],'dense_final_stop_absolute':stop,
                                'block_overlaps_reference_answer':start<stop,'remaining_queries':int((x[0,start:end]==126336).sum().item()),
                                'input_sha256':hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()}
                            emit('diagnostic_checkpoint',**base)
                            # Genuine block-only null records FP16 geometry differences separately.
                            oracle=CustomCache(32,x.device,kernel_size=3,keep_ratio=1)
                            oracle.cache={li:{'k':c['k'],'v':c['v']} for li,c in current.items()}
                            native=model(x[:,start:end],start,2,oracle).logits
                            append(args.output/'block_only_null.jsonl',{**base,**compare_logits(native,reference,x,start,end,stop)})
                            del native,oracle
                            local_metrics(anchor,current,previous,start,end,x[0,start:end]==126336,base,args.output/'local_metrics.jsonl')
                            rows=run_conditions(model,x,start,end,anchor,current,reference,base,stop,args.output/'conditions.jsonl',CONDITIONS)
                            all_rows.extend(rows)
                            snap={**base,'input_ids':x[0].tolist(),'anchor_input_ids':anchor_input,'prompt_length':plen,
                                  'anchor_selection_sha256':index_hash(anchor),'current_selection_sha256':index_hash(current)}
                            append(args.output/'checkpoints.jsonl',snap)
                            saved.append(snap)
                            previous=None
                            current=None
                        x[0,pos]=tok
                        del reference
                    anchor=None
                agreement=x[0,plen:plen+64].tolist()==dense['generated_token_ids'][:64]
                assert agreement, 'Captured dense trajectory differs from baseline prefix'
                validations.append({'sample_id':s['sample_id'],'dense_first_64_tokens_match_baseline':agreement})
                emit('diagnostic_sample_complete',sample_id=s['sample_id'])
                clear()
            write_json(args.output/'validation.json',validations)
            assert len(all_rows)==64 and len(saved)==16
            changed=[r for r in all_rows if r['condition']=='F25' and r['committed_pair_changed']]
            trigger=len(changed)>=4 and len({r['sample_id'] for r in changed})>=2
            write_json(args.output/'supplement_trigger.json',{'triggered':trigger,'changed_checkpoints':len(changed),'samples':sorted({r['sample_id'] for r in changed}),'predeclared_rule':frozen['diagnosis']['additional_current_selection_trigger']})
            if trigger:
                emit('current_selection_supplement_triggered',checkpoints=len(changed))
                for snap in saved:
                    monitor.check()
                    start=snap['prompt_length']+snap['block']*32
                    end=start+32
                    xa=torch.tensor([snap['anchor_input_ids']],device='cuda')
                    x=torch.tensor([snap['input_ids']],device='cuda')
                    with capture(model,start,end) as anchor:
                        unused=model(xa,start,0,dummy).logits
                    del unused
                    with capture(model,start,end) as current:
                        reference=model(x,start,0,dummy).logits[:,start:end].contiguous()
                    assert index_hash(anchor)==snap['anchor_selection_sha256'] and index_hash(current)==snap['current_selection_sha256']
                    base={k:v for k,v in snap.items() if k not in ['input_ids','anchor_input_ids']}
                    run_conditions(model,x,start,end,anchor,current,reference,base,snap['dense_final_stop_absolute'],args.output/'supplement_conditions.jsonl',['F25_current'])
                    del anchor,current,reference
            monitor.check()
            write_json(args.output/'completed.json',{'status':'complete','primary_conditions':64,'local_layer_checkpoints':512,'extra_current_selection_conditions':16 if trigger else 0,'gpu_uuid':uuid})
            emit('diagnostics_complete',output=str(args.output),supplement=trigger)
    finally:
        monitor.close()


if __name__=='__main__':
    main()
