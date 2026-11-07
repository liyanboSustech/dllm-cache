import argparse,json,os,time,sys
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from discovery_common import ROOT,sha,write_json,read_jsonl,append,wait_idle,Monitor,emit,clear
from run_kv_generalization_worker import source_files
from decision_paths_core import Backend,trajectory,continue_branch,CONDITIONS,PHASES
TAG='decision_paths_20261106'
def sources():return sorted(set(source_files()+['scripts/kv_history_phase_core.py']+['scripts/'+f for f in ['decision_paths_core.py','prepare_decision_paths.py','run_decision_paths_worker.py','launch_decision_paths.py','validate_decision_paths.py','analyze_decision_paths.py']]))
def compact(cp):return {**{k:v for k,v in cp.items() if k not in ['x','anchor','anchor_input']},'x_t':cp['x'][0].tolist(),'anchor_input':cp['anchor_input'][0].tolist()}
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--model',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']==str(a.gpu)
 d=ROOT/'data'/TAG;r=ROOT/'reports'/TAG;protocol=json.loads((d/'protocol.json').read_text());spec=protocol['models'][a.model]
 assert sha(d/'protocol.json')==json.loads((r/'protocol_frozen.json').read_text())['sha256']
 assert sha(ROOT/spec['samples'])==spec['sha256'];assert json.loads((r/'cpu_validation.json').read_text())['passed']
 hashes={f:sha(ROOT/f) for f in sources()};assert hashes==json.loads((r/'sources_frozen.json').read_text())
 out=ROOT/'results'/TAG/a.model/f'gpu{a.gpu}';out.mkdir(parents=True,exist_ok=False);uuid=wait_idle(a.gpu,30,out)
 b,loading=Backend.load(a.model);monitor=Monitor(uuid,out/'gpu_monitor.jsonl');samples=read_jsonl(ROOT/spec['samples'])[a.shard::2]
 write_json(out/'environment.json',{'gpu':a.gpu,'gpu_uuid':uuid,'model':a.model,'pid':os.getpid(),'torch':torch.__version__,'dtype':'float16','attention':'EFFICIENT_ATTENTION','loading':loading,'sampler_options':b.sampler_options,'sources_sha256':hashes,'protocol_sha256':sha(d/'protocol.json'),'sample_ids':[s['sample_id'] for s in samples],'argv':sys.argv})
 try:
  with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
   w=next(s for s in read_jsonl(ROOT/'data/discovery_20261103/preflight.jsonl') if s['task']=='gsm8k');text=b.tokenizer.apply_chat_template([{'role':'user','content':w['prompt']}],tokenize=False,add_generation_prompt=True);ids=b.tokenizer(text)['input_ids'];w={**w,'prompt_ids':ids,'prompt_length':len(ids)}
   base=trajectory(b,w,'S25',check=monitor.check);inp=torch.tensor([ids],device=b.device)
   if a.model=='llada':
    from vendor.sparse_llada.llada_generate import generate
    z=generate(b.model,inp,steps=256,gen_length=256,block_length=32,temperature=0,cfg_scale=0)
   else:
    with b.operator():z=b.model.diffusion_generate(inp,max_new_tokens=256,steps=256,temperature=0,alg='maskgit_plus',eps=.001)
   assert z[0,len(ids):].tolist()==base['ids'];del z,inp
   warm={}
   for phase in PHASES:
    cp=base['checkpoints'][phase];assert cp['actual_revealed']==4
    for c in CONDITIONS:
     monitor.check();z=continue_branch(b,w,cp,c,monitor.check)
     warm[phase+':'+c]=z['generated_token_ids']
     if c=='BASE':assert z['generated_token_ids']==base['ids']
     if c=='BOTH':
      independent=continue_branch(b,w,cp,c,monitor.check,full_only=True);assert z['generated_token_ids']==independent['generated_token_ids'];del independent
     del z
   for c in CONDITIONS:
    z=continue_branch(b,w,base['checkpoints'][PHASES[0]],c,monitor.check,reference_same_as_sparse=True);assert z['generated_token_ids']==base['ids'];del z
   write_json(out/'warmup_tokens.json',warm);write_json(r/f'preflight_gpu{a.gpu}.json',{'passed':True,'model':a.model,'official_S25_exact':True,'four_arm_null_exact':True,'BASE_exact_both_phases':True,'BOTH_independent_reference_exact_both_phases':True,'same_state_age_readonly_checks':True,'sources':hashes});emit('preflight_passed',model=a.model,gpu=a.gpu)
   del base,w,warm;clear()
   while not all((r/f'preflight_gpu{g}.json').exists() for g in range(4)):monitor.check();time.sleep(3)
   for index,s in enumerate(samples):
    monitor.check();case=out/s['sample_id'];case.mkdir();write_json(case/'sample.json',s)
    base=trajectory(b,s,'S25',check=monitor.check)
    old=next(x for x in read_jsonl(ROOT/'results/kv_generalization_20261104'/a.model/f'gpu{a.gpu}'/s['sample_id']/'continuations.jsonl') if x['condition']=='S25')
    assert old['gpu_uuid']==uuid and old['generated_token_ids']==base['ids'],('Historical S25 replay failed',s['sample_id'])
    write_json(case/'baseline.json',{'gpu_uuid':uuid,'model':a.model,'sample_id':s['sample_id'],'commits':base['commits'],'historical_tokens_exact':True,**b.score(s,base['ids'])})
    for phase in PHASES:
     cp=base['checkpoints'][phase];assert cp['actual_revealed']==4;meta=compact(cp);write_json(case/f'checkpoint_{phase}.json',meta)
     order=CONDITIONS[index%4:]+CONDITIONS[:index%4]
     for c in order:
      monitor.check();begin=time.monotonic();z=continue_branch(b,s,cp,c,monitor.check)
      if c=='BASE':assert z['generated_token_ids']==base['ids']
      ids=z.pop('generated_token_ids');events=z.pop('events');write_json(case/f'events_{phase}_{c}.json',events)
      append(case/'outputs.jsonl',{'model':a.model,'sample_id':s['sample_id'],'phase':phase,'condition':c,'gpu_uuid':uuid,'any_stop_already_present':cp['any_stop_already_present'],'stop_in_previous_blocks':cp['stop_in_previous_blocks'],'block':cp['block'],'step':cp['step'],'diagnostic_wall_seconds':time.monotonic()-begin,**z,**b.score(s,ids)})
      emit('continuation_complete',model=a.model,gpu=a.gpu,sample_id=s['sample_id'],phase=phase,condition=c)
     del cp
    assert len(read_jsonl(case/'outputs.jsonl'))==8;write_json(case/'completed.json',{'continuations':8,'baseline_replay_exact':True});del base;clear()
   monitor.check();assert all(sha(ROOT/f)==h for f,h in hashes.items());write_json(out/'completed.json',{'samples':len(samples),'continuations':8*len(samples),'status':'complete'});emit('worker_complete',model=a.model,gpu=a.gpu)
 finally:monitor.close()
if __name__=='__main__':main()
