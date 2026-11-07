import argparse,json,os,sys,time
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from discovery_common import ROOT,sha,write_json,read_jsonl,append,wait_idle,Monitor,emit,clear
from run_matched_sources_worker import sources as previous_sources
from k4_correction_core import TAG,Backend,generate,LAYERS

def sources():
 return sorted(set(previous_sources()+['scripts/'+s for s in ['run_matched_sources_worker.py','k4_correction_core.py','prepare_k4_correction.py','validate_k4_correction.py','run_k4_correction_worker.py','launch_k4_correction.py','analyze_k4_correction.py','audit_k4_correction.py']]))

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--model',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(a.gpu)
 P=ROOT/'reports'/TAG;D=ROOT/'data'/TAG;protocol=json.loads((D/'protocol.json').read_text());spec=protocol['models'][a.model]
 assert sha(D/'protocol.json')==json.loads((P/'protocol_frozen.json').read_text())['sha256'];assert sha(ROOT/spec['samples'])==spec['sha256']
 hashes={f:sha(ROOT/f) for f in sources()};assert hashes==json.loads((P/'sources_frozen.json').read_text());assert json.loads((P/'cpu_validation.json').read_text())['passed']
 out=ROOT/'results'/TAG/a.model/f'gpu{a.gpu}';out.mkdir(parents=True,exist_ok=False)
 uuid=wait_idle(a.gpu,30,out);b,loading=Backend.load(a.model);monitor=Monitor(uuid,out/'gpu_monitor.jsonl')
 ss=read_jsonl(ROOT/spec['samples'])[a.shard::2]
 write_json(out/'environment.json',{'gpu':a.gpu,'gpu_uuid':uuid,'pid':os.getpid(),'model':a.model,'dtype':'float16','attention':'EFFICIENT_ATTENTION','torch':torch.__version__,'loading':loading,'sources':hashes,'protocol_sha256':sha(D/'protocol.json'),'sample_ids':[s['sample_id'] for s in ss],'argv':sys.argv})
 try:
  with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
   w=next(s for s in read_jsonl(ROOT/'data/discovery_20261103/preflight.jsonl') if s['task']=='gsm8k')
   rendered=b.tokenizer.apply_chat_template([{'role':'user','content':w['prompt']}],tokenize=False,add_generation_prompt=True);ids=b.tokenizer(rendered)['input_ids'];w={**w,'prompt_ids':ids,'prompt_length':len(ids)}
   warm=[];base=generate(b,w,randoms=spec['random_four'],probe_callback=lambda row:warm.append(row),check=monitor.check)
   old=json.loads((ROOT/'results/early_sources_20261106'/a.model/f'gpu{a.gpu}'/'warmup_tokens.json').read_text())['b0_early:BASE'];assert base['generated_token_ids']==old
   z=generate(b,w,LAYERS[a.model],check=monitor.check)
   if a.model=='llada':
    from priority_kv_core import generate as official_generate
    native=official_generate(b.model,torch.tensor([w['prompt_ids']],device=b.device),'K4_NEXT');assert native[0,w['prompt_length']:].tolist()==z['generated_token_ids']
   write_json(out/'warmup.json',{'S25':base['generated_token_ids'],'K4':z['generated_token_ids'],'probes':warm})
   write_json(P/f'preflight_gpu{a.gpu}.json',{'passed':True,'S25_old_exact':True,'K4_llada_official_exact':a.model=='llada','all_probe_conditions_and_seven_blocks':True})
   del w,base,z,warm;clear();emit('preflight_passed',gpu=a.gpu,model=a.model)
   while not all((P/f'preflight_gpu{g}.json').exists() for g in range(4)):monitor.check();time.sleep(3)
   for s in ss:
    monitor.check();case=out/s['sample_id'];case.mkdir();write_json(case/'sample.json',s)
    z=generate(b,s,randoms=spec['random_four'],probe_callback=lambda row:append(case/'probes.jsonl',row),check=monitor.check)
    old=json.loads((ROOT/'results/early_sources_20261106'/a.model/f'gpu{a.gpu}'/s['sample_id']/'native_S25.json').read_text())
    assert old['gpu_uuid']==uuid and old['generated_token_ids']==z['generated_token_ids']
    append(case/'outputs.jsonl',{'model':a.model,'sample_id':s['sample_id'],'condition':'S25','gpu_uuid':uuid,**z})
    assert len(read_jsonl(case/'probes.jsonl'))==7
    write_json(case/'scores_frozen.json',{'probes_sha256':sha(case/'probes.jsonl'),'native_S25_exact':True});emit('scores_frozen',gpu=a.gpu,model=a.model,sample_id=s['sample_id']);del z;clear()
   write_json(P/f'scores_gpu{a.gpu}_complete.json',{'samples':len(ss)})
   while not all((P/f'scores_gpu{g}_complete.json').exists() for g in range(4)):monitor.check();time.sleep(3)
   for index,s in enumerate(ss):
    case=out/s['sample_id'];conds=[('K4',LAYERS[a.model])]+[(f'R{i+1}',r) for i,r in enumerate(spec['random_four'])]
    offset=index%4;conds=conds[offset:]+conds[:offset]
    if index%2:conds.reverse()
    for name,layers in conds:
     monitor.check();z=generate(b,s,layers,check=monitor.check)
     append(case/'outputs.jsonl',{'model':a.model,'sample_id':s['sample_id'],'condition':name,'gpu_uuid':uuid,**z});emit('generation_complete',gpu=a.gpu,model=a.model,sample_id=s['sample_id'],condition=name);del z
    assert sha(case/'probes.jsonl')==json.loads((case/'scores_frozen.json').read_text())['probes_sha256'];write_json(case/'completed.json',{'outputs':5,'checkpoints':7});clear()
   monitor.check();assert all(sha(ROOT/f)==h for f,h in hashes.items());write_json(out/'completed.json',{'samples':len(ss),'outputs':len(ss)*5});emit('worker_complete',gpu=a.gpu)
 finally:monitor.close()
if __name__=='__main__':main()
