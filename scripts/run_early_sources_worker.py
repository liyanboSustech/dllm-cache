import argparse,json,os,time,sys
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from discovery_common import ROOT,sha,write_json,read_jsonl,append,wait_idle,Monitor,emit,clear
from run_decision_paths_worker import sources as old_sources,compact
from early_sources_core import Backend,trajectory,checkpoint,continue_branch,restoration_plan,TAG,PHASES,conditions,key,ARMS

def sources():return sorted(set(old_sources()+['scripts/run_decision_paths_worker.py']+['scripts/'+f for f in ['early_sources_core.py','prepare_early_sources.py','run_early_sources_worker.py','launch_early_sources.py','validate_early_sources.py','analyze_early_sources.py']]))
def states(b,s,base,dense):return {'b0_early':checkpoint(b,s,base,0,True),'b0_late':checkpoint(b,s,base,0,False),'b1_clean_early':checkpoint(b,s,dense,1,True)}
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--model',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']==str(a.gpu)
 D=ROOT/'data'/TAG;P=ROOT/'reports'/TAG;protocol=json.loads((D/'protocol.json').read_text());spec=protocol['models'][a.model]
 assert sha(D/'protocol.json')==json.loads((P/'protocol_frozen.json').read_text())['sha256'];assert sha(ROOT/spec['samples'])==spec['sha256'];assert json.loads((P/'cpu_validation.json').read_text())['passed']
 hashes={f:sha(ROOT/f) for f in sources()};assert hashes==json.loads((P/'sources_frozen.json').read_text())
 out=ROOT/'results'/TAG/a.model/f'gpu{a.gpu}';out.mkdir(parents=True,exist_ok=False);uuid=wait_idle(a.gpu,30,out);b,loading=Backend.load(a.model);monitor=Monitor(uuid,out/'gpu_monitor.jsonl');samples=read_jsonl(ROOT/spec['samples'])[a.shard::2]
 write_json(out/'environment.json',{'gpu':a.gpu,'gpu_uuid':uuid,'model':a.model,'pid':os.getpid(),'torch':torch.__version__,'dtype':'float16','attention':'EFFICIENT_ATTENTION','loading':loading,'sampler_options':b.sampler_options,'sources_sha256':hashes,'protocol_sha256':sha(D/'protocol.json'),'sample_ids':[s['sample_id'] for s in samples],'argv':sys.argv})
 try:
  with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
   w=next(s for s in read_jsonl(ROOT/'data/discovery_20261103/preflight.jsonl') if s['task']=='gsm8k');text=b.tokenizer.apply_chat_template([{'role':'user','content':w['prompt']}],tokenize=False,add_generation_prompt=True);ids=b.tokenizer(text)['input_ids'];w={**w,'prompt_ids':ids,'prompt_length':len(ids)}
   base=trajectory(b,w,'S25',check=monitor.check);dense=trajectory(b,w,'C100',check=monitor.check);cps=states(b,w,base,dense);warm={}
   # Frozen previous native warmup makes the new baseline an exact official replay.
   oldwarm=json.loads((ROOT/'results/decision_paths_20261106'/a.model/f'gpu{a.gpu}'/'warmup_tokens.json').read_text());assert base['ids']==oldwarm['b0_r04:BASE']
   for ph in PHASES:
    cp=cps[ph]
    for src,arm in conditions(ph):
     z=continue_branch(b,w,cp,src,arm,monitor.check);warm[ph+':'+key(src,arm)]=z['generated_token_ids']
     if arm=='BASE' and ph!='b1_clean_early':assert z['generated_token_ids']==base['ids']
     if src=='ALL' and arm=='BOTH':
      independent=continue_branch(b,w,cp,src,arm,monitor.check,reference_only=True);assert z['generated_token_ids']==independent['generated_token_ids'];del independent
     del z
    emit('warm_phase_complete',gpu=a.gpu,phase=ph)
   for arm in ARMS:
    z=continue_branch(b,w,cps['b0_early'],'ALL',arm,monitor.check,null=True);assert z['generated_token_ids']==base['ids'];del z
   # Block0 has no external generated history: history restoration is null; TEXT==PROMPT exactly.
   assert all(not r['added'] for r in restoration_plan(b,w,cps['b0_early'],'HISTORY'))
   assert restoration_plan(b,w,cps['b0_early'],'PROMPT')==restoration_plan(b,w,cps['b0_early'],'TEXT')
   write_json(out/'warmup_tokens.json',warm);write_json(P/f'preflight_gpu{a.gpu}.json',{'passed':True,'native_S25_previous_official_exact':True,'all_conditions_checked':26,'null_four_arms_exact':True,'BOTH_independent_reference_exact_three_phases':True,'b0_HISTORY_empty_TEXT_equals_PROMPT':True});emit('preflight_passed',gpu=a.gpu,model=a.model)
   del base,dense,cps,cp,w,warm;clear()
   while not all((P/f'preflight_gpu{g}.json').exists() for g in range(4)):monitor.check();time.sleep(3)
   for index,s in enumerate(samples):
    monitor.check();case=out/s['sample_id'];case.mkdir();write_json(case/'sample.json',s)
    base=trajectory(b,s,'S25',check=monitor.check);dense=trajectory(b,s,'C100',check=monitor.check);cps=states(b,s,base,dense)
    old=read_jsonl(ROOT/'results/kv_generalization_20261104'/a.model/f'gpu{a.gpu}'/s['sample_id']/'continuations.jsonl')
    for policy,run in [('S25',base),('C100',dense)]:
     previous=next(r for r in old if r['condition']==policy);assert previous['gpu_uuid']==uuid and previous['generated_token_ids']==run['ids'],('Native replay mismatch',policy,s['sample_id'])
     write_json(case/f'native_{policy}.json',{'gpu_uuid':uuid,'model':a.model,'sample_id':s['sample_id'],'commits':run['commits'],'historical_exact':True,**b.score(s,run['ids'])})
    for ph in PHASES:
     cp=cps[ph];write_json(case/f'checkpoint_{ph}.json',compact(cp))
     for src in sorted({src for src,arm in conditions(ph)}):write_json(case/f'plan_{ph}_{src}.json',restoration_plan(b,s,cp,src))
     order=conditions(ph);rot=index%len(order);order=order[rot:]+order[:rot]
     for src,arm in order:
      monitor.check();begin=time.monotonic();z=continue_branch(b,s,cp,src,arm,monitor.check)
      if arm=='BASE' and ph!='b1_clean_early':assert z['generated_token_ids']==base['ids']
      events=z.pop('events');ids=z.pop('generated_token_ids');cond=key(src,arm);write_json(case/f'events_{ph}_{cond}.json',events)
      append(case/'outputs.jsonl',{'model':a.model,'sample_id':s['sample_id'],'phase':ph,'condition':cond,'source':src,'arm':arm,'gpu_uuid':uuid,'step':cp['step'],'block':cp['block'],'actual_revealed':cp['actual_revealed'],'upstream_policy':cp['upstream_policy'],'any_stop_already_present':cp['any_stop_already_present'],'stop_in_previous_blocks':cp['stop_in_previous_blocks'],'diagnostic_wall_seconds':time.monotonic()-begin,**z,**b.score(s,ids)})
      emit('continuation_complete',gpu=a.gpu,model=a.model,sample_id=s['sample_id'],phase=ph,condition=cond)
    assert len(read_jsonl(case/'outputs.jsonl'))==26;write_json(case/'completed.json',{'continuations':26,'native_replays':2});del base,dense,cps,cp;clear()
   monitor.check();assert all(sha(ROOT/f)==h for f,h in hashes.items());write_json(out/'completed.json',{'samples':len(samples),'continuations':26*len(samples),'status':'complete'});emit('worker_complete',gpu=a.gpu,model=a.model)
 finally:monitor.close()
if __name__=='__main__':main()
