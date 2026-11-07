import argparse,json,os,time,sys
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from discovery_common import ROOT,sha,write_json,read_jsonl,append,wait_idle,Monitor,emit,clear
from run_early_sources_worker import sources as old_sources
from run_decision_paths_worker import compact
from early_sources_core import continue_branch
from matched_sources_core import TAG,Backend,trajectory,checkpoint,conditions,plan_for,run

def sources():return sorted(set(old_sources()+['scripts/run_early_sources_worker.py']+['scripts/'+f for f in ['matched_sources_core.py','prepare_matched_sources.py','run_matched_sources_worker.py','launch_matched_sources.py','validate_matched_sources.py','analyze_matched_sources.py']]))
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
   base=trajectory(b,w,'S25',check=monitor.check);cp=checkpoint(b,w,base,0,True);warm={}
   oldwarm=json.loads((ROOT/'results/early_sources_20261106'/a.model/f'gpu{a.gpu}'/'warmup_tokens.json').read_text());assert base['ids']==oldwarm['b0_early:BASE']
   for cond in conditions('development'):
    plan=plan_for(b,w,cp,cond);z=run(b,w,cp,plan,monitor.check);warm[cond]=z['generated_token_ids']
    if cond=='BASE':assert z['generated_token_ids']==base['ids']
    if cond=='ALL':
     independent=continue_branch(b,w,cp,'ALL','BOTH',monitor.check);assert z['generated_token_ids']==independent['generated_token_ids'];del independent
    del z
   for i in range(3):assert [len(r['selected']) for r in plan_for(b,w,cp,f'MASK_{i}')]==[len(r['selected']) for r in plan_for(b,w,cp,f'TEXT_{i}')]
   write_json(out/'warmup_tokens.json',warm);write_json(P/f'preflight_gpu{a.gpu}.json',{'passed':True,'native_previous_official_exact':True,'all_8_conditions_checked':True,'BASE_null_exact':True,'ALL_independent_four_arm_BOTH_exact':True,'matched_per_layer':True});emit('preflight_passed',gpu=a.gpu,model=a.model)
   del base,cp,w,warm;clear()
   while not all((P/f'preflight_gpu{g}.json').exists() for g in range(4)):monitor.check();time.sleep(3)
   for cohort in ['development','confirmation']:
    current=[s for s in samples if s['cohort']==cohort]
    for index,s in enumerate(current):
     monitor.check();case=out/s['sample_id'];case.mkdir();write_json(case/'sample.json',s)
     begin=time.monotonic();base=trajectory(b,s,'S25',check=monitor.check);native_seconds=time.monotonic()-begin;cp=checkpoint(b,s,base,0,True)
     historical=False
     if cohort=='development':
      previous=json.loads((ROOT/'results/early_sources_20261106'/a.model/f'gpu{a.gpu}'/s['sample_id']/'native_S25.json').read_text());assert previous['gpu_uuid']==uuid and previous['generated_token_ids']==base['ids'];historical=True
     write_json(case/'native_S25.json',{'gpu_uuid':uuid,'model':a.model,'sample_id':s['sample_id'],'commits':base['commits'],'historical_exact':historical,**b.score(s,base['ids'])});write_json(case/'checkpoint.json',compact(cp))
     for cond in conditions(cohort):write_json(case/f'plan_{cond}.json',plan_for(b,s,cp,cond))
     append(case/'outputs.jsonl',{'model':a.model,'sample_id':s['sample_id'],'cohort':cohort,'condition':'BASE','gpu_uuid':uuid,'diagnostic_wall_seconds':native_seconds,'timing_scope':'native full capture with probes; not deploy latency','commits':base['commits'],**b.score(s,base['ids'])})
     order=conditions(cohort)[1:];rot=index%len(order);order=order[rot:]+order[:rot]
     if index%2:order=order[::-1]
     for cond in order:
      monitor.check();begin=time.monotonic();plan=plan_for(b,s,cp,cond);z=run(b,s,cp,plan,monitor.check);ids=z.pop('generated_token_ids')
      append(case/'outputs.jsonl',{'model':a.model,'sample_id':s['sample_id'],'cohort':cohort,'condition':cond,'gpu_uuid':uuid,'diagnostic_wall_seconds':time.monotonic()-begin,'timing_scope':'continuation only; anchor capture excluded, no deploy claim',**z,**b.score(s,ids)});emit('continuation_complete',gpu=a.gpu,cohort=cohort,model=a.model,sample_id=s['sample_id'],condition=cond)
     assert len(read_jsonl(case/'outputs.jsonl'))==len(conditions(cohort));write_json(case/'completed.json',{'outputs':len(conditions(cohort)),'native_captures':1});del base,cp;clear()
    write_json(P/f'{cohort}_gpu{a.gpu}_complete.json',{'questions':len(current)})
    if cohort=='development':
     while not all((P/f'development_gpu{g}_complete.json').exists() for g in range(4)):monitor.check();time.sleep(3)
   monitor.check();assert all(sha(ROOT/f)==h for f,h in hashes.items());write_json(out/'completed.json',{'samples':len(samples),'outputs':384,'status':'complete'});emit('worker_complete',gpu=a.gpu,model=a.model)
 finally:monitor.close()
if __name__=='__main__':main()
