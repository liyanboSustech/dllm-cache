"""Prospective recipient shard, fixed conditions and audited cross-prompt donors."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from discovery_common import ROOT,wait_idle,Monitor,write_json,read_jsonl,append,sha,emit,clear,gpu_snapshot
from kv_generalization_core import Backend,rollout,context_forward,local_probe,roundtrip_probe


def source_files():
    base=['scripts/kv_generalization_core.py','scripts/run_kv_generalization_worker.py',
        'scripts/kv_factor_core.py','scripts/discovery_common.py','scripts/interruption_core.py',
        'scripts/priority_kv_core.py','scripts/run_baseline_smoke.py','scripts/rescore_discovery.py']
    return base+[str(p.relative_to(ROOT)) for folder in ['vendor/sparse_llada','vendor/sparse_dream'] for p in sorted((ROOT/folder).glob('*.py'))]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['llada','dream'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(a.gpu)
    report=ROOT/'reports/kv_generalization_20261104';data=ROOT/'data/kv_generalization_20261104'
    protocol=json.loads((data/'protocol.json').read_text());spec=protocol['models'][a.model]
    assert sha(ROOT/spec['path'])==spec['sha256']
    out=ROOT/'results/kv_generalization_20261104'/a.model/f'gpu{a.gpu}';out.mkdir(parents=True,exist_ok=False)
    code_hashes={f:sha(ROOT/f) for f in source_files()}
    write_json(out/'sources_at_start.json',code_hashes)
    preflight=report/f'preflight_{a.model}_fixed'/'passed.json'
    emit('waiting_for_preflight',model=a.model,path=str(preflight))
    while not preflight.exists():time.sleep(5)
    pre=json.loads(preflight.read_text())
    assert pre['passed'] and pre['sources_sha256']['scripts/kv_generalization_core.py']==code_hashes['scripts/kv_generalization_core.py']
    assert json.loads((report/'cpu_validation.json').read_text())['passed']
    # Preflight writes its verdict just before releasing the loaded model.
    while True:
        snap=gpu_snapshot();gpurow=next(r.split(', ') for r in snap['gpus'].splitlines() if int(r.split(',')[0])==a.gpu)
        if gpurow[1] not in snap['processes'] and int(gpurow[3])<=1000:break
        time.sleep(5)
    uuid=wait_idle(a.gpu,30,out);b,loading=Backend.load(a.model)
    monitor=Monitor(uuid,out/'gpu_monitor.jsonl')
    samples=read_jsonl(ROOT/spec['path'])[a.shard::2]
    write_json(out/'environment.json',{'model':a.model,'gpu':a.gpu,'gpu_uuid':uuid,'pid_namespace':os.getpid(),
        'argv':sys.argv,'torch':torch.__version__,'dtype':'float16','loading':loading,'sampler_options':b.sampler_options,
        'layer_scope':list(b.layers),'sources_sha256':code_hashes,'protocol_sha256':sha(data/'protocol.json'),
        'samples_sha256':spec['sha256'],'sample_ids':[s['sample_id'] for s in samples]})
    try:
      with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        # Each physical GPU validates official decoding on the excluded old warmup.
        old=next(r for r in read_jsonl(ROOT/'data/discovery_20261103/preflight.jsonl') if r['task']=='gsm8k')
        text=b.tokenizer.apply_chat_template([{'role':'user','content':old['prompt']}],tokenize=False,add_generation_prompt=True)
        ids=b.tokenizer(text)['input_ids'];warm={**old,'prompt_ids':ids,'prompt_length':len(ids),'donors':{}}
        r=rollout(b,warm,'S25',check=monitor.check);inp=torch.tensor([ids],device=b.device)
        if a.model=='llada':
            from vendor.sparse_llada.llada_generate import generate
            native=generate(b.model,inp,steps=256,gen_length=256,block_length=32,temperature=0,cfg_scale=0)
        else:
            with b.operator():native=b.model.diffusion_generate(inp,max_new_tokens=256,steps=256,temperature=0,alg='maskgit_plus',eps=.001)
        assert r['ids']==native[0,len(ids):].tolist()
        write_json(out/'warmup_validation.json',{'official_exact':True,'sample_id':warm['sample_id'],'generated_token_ids':r['ids']})
        del r,native,inp
        for sample in samples:
            monitor.check();d=out/sample['sample_id'];d.mkdir();write_json(d/'sample.json',sample)
            checkpoints=None;baseline_signature=None
            for condition in protocol['conditions']:
                begin=time.monotonic()
                try:r=rollout(b,sample,condition,keep_checkpoints=condition=='S25',check=monitor.check)
                except FloatingPointError as exc:
                    row={'sample_id':sample['sample_id'],'model':a.model,'condition':condition,'gpu_uuid':uuid,
                        'numerical_failure':True,'failure':str(exc),'diagnostic_wall_seconds':time.monotonic()-begin}
                    append(d/'continuations.jsonl',row);emit('condition_numerical_failure',**row);continue
                sig=[(x['external_counts'],x['kv_bytes']) for x in r['cache_audit']]
                if condition=='S25':baseline_signature=sig;checkpoints=r['checkpoints']
                elif condition!='C100':assert sig==baseline_signature
                row={'sample_id':sample['sample_id'],'model':a.model,'condition':condition,'gpu_uuid':uuid,
                    'diagnostic_wall_seconds':time.monotonic()-begin,'cache_audit':r['cache_audit'],
                    'commits':r['commits'],'first_stop_commit':r['first_stop_commit'],**b.score(sample,r['ids'])}
                append(d/'continuations.jsonl',row)
                emit('condition_complete',model=a.model,sample_id=sample['sample_id'],condition=condition,
                    strict_correct=row['correct'],stop_position=row['first_stop_position'],seconds=round(row['diagnostic_wall_seconds'],2))
                del row,r
                if condition=='S25':
                    write_json(d/'checkpoints.json',{name:{'input_ids':cp['x'][0].tolist(),
                        'anchor_input_ids':cp['anchor_input'][0].tolist(),'start':cp['start'],'step':cp['step'],
                        'event_position':cp['event_position'],'native':cp['native']} for name,cp in checkpoints.items()})
                    noctx=context_forward(b,checkpoints['early']['x'],sample['prompt_length'],empty=True)
                    for name,cp in checkpoints.items():
                        for c in protocol['conditions']:
                            monitor.check();lr=local_probe(b,sample,cp,c,noctx)
                            if c=='S25':
                                assert lr['selected_positions']==cp['native']['positions']
                                assert lr['selected_tokens']==cp['native']['tokens']
                                assert lr['selected_confidences']==cp['native']['confidence']
                            append(d/'local.jsonl',{'checkpoint':name,'model':a.model,'sample_id':sample['sample_id'],**lr})
                        append(d/'local.jsonl',{'checkpoint':name,'model':a.model,'sample_id':sample['sample_id'],**roundtrip_probe(b,sample,cp)})
                    emit('local_complete',model=a.model,sample_id=sample['sample_id'],checkpoints=list(checkpoints))
                    del noctx,checkpoints,cp
            saved=read_jsonl(d/'continuations.jsonl');assert len(saved)==len(protocol['conditions'])
            assert len({r['condition'] for r in saved})==len(saved)
            monitor.check();write_json(d/'completed.json',{'conditions':len(saved),'numerical_failures':sum(r.get('numerical_failure',False) for r in saved),'equal_sparse_count_and_bytes':True})
            clear()
        assert all(sha(ROOT/f)==h for f,h in code_hashes.items())
        write_json(out/'completed.json',{'status':'complete','samples':len(samples),'conditions_per_sample':len(protocol['conditions'])})
    finally:monitor.close()

if __name__=='__main__':main()
