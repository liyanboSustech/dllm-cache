"""Machine/path adaptation only. The archived scientific core is never edited."""
import argparse,importlib.metadata,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from .common import *

def check_environment(paths,gpus=None):
 verify_export()
 import torch,transformers
 versions={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','numpy','scipy','safetensors','tokenizers']}
 for name,expected in [('torch','2.6.0'),('transformers','4.57.1'),('numpy','1.26.4'),('scipy','1.12.0')]:
  if versions[name].split('+')[0]!=expected:raise RuntimeError(f'{name}: expected {expected}, found {versions[name]}; use the pinned environment')
 reference=read(ROOT/'references/model_reference.json');models={}
 from transformers import AutoTokenizer
 for model,path in paths.items():
  directory=Path(path).expanduser().resolve()
  if not directory.is_dir():raise FileNotFoundError(f'{model} model directory missing: {directory}')
  info=reference['models'][model]
  for f,h in info['metadata_sha256'].items():
   if not (directory/f).is_file() or sha(directory/f)!=h:raise RuntimeError(f'{model} metadata differs from frozen model: {f}')
  for f,size in info['weight_files'].items():
   if not (directory/f).is_file() or (directory/f).stat().st_size!=size:raise RuntimeError(f'{model} missing/truncated weight shard: {f}')
  tokenizer=AutoTokenizer.from_pretrained(directory,trust_remote_code=True,local_files_only=True)
  samples=rows(ROOT/'data'/TAG/f'{model}_samples.jsonl')
  for s in samples:
   assert tokenizer(s['rendered_prompt'])['input_ids']==s['prompt_ids'],s['sample_id']
   assert len(s['prompt_ids'])==s['prompt_length']
  models[model]={'path':str(directory),'metadata_exact':True,'weight_sizes_exact':True,'all32_prompt_tokens_exact':True}
 out={'python':sys.version,'versions':versions,'torch_cuda':torch.version.cuda,'models':models,'weights_note':'Shard names/sizes verified, not full weight SHA256. Use the original model revisions.'}
 if gpus is not None:
  snap=gpu_snapshot();out.update(gpus=snap,requested_gpus=gpus,gpus_idle=gpu_ready(snap,gpus),cuda_available=torch.cuda.is_available())
 return out

def self_test():
 verify_export();code_path()
 import runpy,torch
 # The frozen CPU validator writes into a temporary run, not tracked reports.
 import discovery_common,k4_correction_core,tempfile
 with tempfile.TemporaryDirectory() as tmp:
  old=discovery_common.ROOT;discovery_common.ROOT=Path(tmp)
  (Path(tmp)/'reports'/TAG).mkdir(parents=True)
  try:runpy.run_path(str(ROOT/'scripts/validate_k4_correction.py'),run_name='__main__')
  finally:discovery_common.ROOT=old
  result=read(Path(tmp)/'reports'/TAG/'cpu_validation.json')
 assert result['passed'];verify_export();return result

def freeze_run(run,paths,gpus,environment):
 run=Path(run).resolve();run.mkdir(parents=True,exist_ok=False)
 for f in ['data','reports','results']:(run/f).mkdir()
 for f in ['protocol.json','llada_samples.jsonl','dream_samples.jsonl']:shutil.copy2(ROOT/'data'/TAG/f,run/'data'/f)
 protocol=read(run/'data/protocol.json')
 write(run/'execution.json',{'models':{k:str(Path(v).expanduser().resolve()) for k,v in paths.items()},'gpus':gpus,'protocol_sha256':sha(run/'data/protocol.json'),
  'scientific_change':False,'hardware_amendment':'User requested g8 migration. All compared conditions recomputed on the new per-question GPU. Original V100 UUIDs and token equality are NOT hard gates across devices. Historical token match is reported only. Same-device official sampler/null checks and same-model two-GPU warmups ARE gates.',
  'scope':'Only first-step diagnostic. No selector, metric tuning, new samples, or later-stage launch.','counts':protocol['counts'],'environment_check':environment})
 manifest=source_manifest();write(run/'source_manifest.json',manifest)
 write(run/'reports/sources_frozen.json',manifest)
 write(run/'reports/analysis_frozen.json',{'sha256':sha(ROOT/'scripts/analyze_k4_correction.py')})
 write(run/'reports/protocol_frozen.json',{'sha256':sha(run/'data/protocol.json')})
 write(run/'reports/review_semantic.json',{})
 write(run/'status.json',{'phase':'prepared','run_dir':str(run),'pid':os.getpid()})
 return run

def wait_gpus(run,gpus,wait):
 stable=None
 while True:
  snap=gpu_snapshot();append(run/'reports/idle_checks.jsonl',snap)
  if gpu_ready(snap,gpus):
   stable=stable or time.monotonic()
   if time.monotonic()-stable>=30:return snap
  else:
   stable=None
   if not wait:raise RuntimeError('Selected GPUs are occupied. Choose four idle GPUs, or use --wait. Other jobs were not touched.')
  write(run/'status.json',{'phase':'waiting_for_idle_GPUs','gpus':gpus,'pid':os.getpid()})
  print('Waiting for four GPUs idle for 30 seconds...',flush=True);time.sleep(5)

def launch(args):
 paths={'llada':args.llada_model,'dream':args.dream_model};gpus=parse_gpus(args.gpus)
 environment=check_environment(paths,gpus);validation=self_test()
 run=freeze_run(args.run_dir,paths,gpus,environment);write(run/'reports/cpu_validation.json',validation)
 jobs={};handles={}
 try:
  snap=wait_gpus(run,gpus,args.wait);by={d['index']:d for d in snap['devices']}
  specs=[(gpus[0],'llada',0),(gpus[1],'llada',1),(gpus[2],'dream',0),(gpus[3],'dream',1)]
  for gpu,model,shard in specs:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=by[gpu]['uuid'],PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
   log=(run/'reports'/f'{model}_gpu{gpu}.log').open('x');handles[gpu]=log
   cmd=[sys.executable,str(ROOT/'run_k4.py'),'_worker','--run-dir',str(run),'--gpu',str(gpu),'--model',model,'--shard',str(shard)]
   jobs[gpu]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  write(run/'reports/worker_pids.json',{g:p.pid for g,p in jobs.items()})
  write(run/'status.json',{'phase':'running_first_step','gpus':gpus,'pid':os.getpid()})
  print(f'Running four workers. Logs: {run / "reports"}',flush=True)
  while any(p.poll() is None for p in jobs.values()):
   if any(p.poll() not in [None,0] for p in jobs.values()):raise RuntimeError('A worker failed; inspect its log. No score/selection changes were made.')
   time.sleep(5)
  codes={g:p.wait() for g,p in jobs.items()};write(run/'reports/exit_codes.json',codes)
  if any(c!=0 for c in codes.values()):raise RuntimeError('Worker failure')
  verify_run(run)
  from .audit import audit
  audit(run)
  progress=analyze(run,prepare=True)
  write(run/'status.json',{'phase':'awaiting_blinded_review','review':progress,'run_dir':str(run)})
  print(json.dumps({'generations_complete':True,'audited':True,'review':progress,'next':f'python run_k4.py review --run-dir {run}'},ensure_ascii=False),flush=True)
 except BaseException as exc:
  for child in jobs.values():
   if child.poll() is None:child.terminate()
  for child in jobs.values():
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  if jobs:write(run/'reports/exit_codes.json',{g:p.returncode for g,p in jobs.items()})
  write(run/'status.json',{'phase':'failed_or_interrupted','error':repr(exc),'only_owned_workers_stopped':True});raise
 finally:
  for h in handles.values():h.close()

def worker(args):
 run=Path(args.run_dir).resolve();manifest=verify_run(run);execution=read(run/'execution.json');protocol=read(run/'data/protocol.json')
 gpu=args.gpu;model=args.model;spec=protocol['models'][model];gpus=execution['gpus']
 assert (gpu,model,args.shard) in [(gpus[0],'llada',0),(gpus[1],'llada',1),(gpus[2],'dream',0),(gpus[3],'dream',1)]
 configure_models(execution['models'])
 import torch
 from torch.nn.attention import sdpa_kernel,SDPBackend
 from discovery_common import wait_idle,Monitor,clear,emit
 from k4_correction_core import Backend,generate,LAYERS
 out=run/'results'/model/f'gpu{gpu}';out.mkdir(parents=True,exist_ok=False)
 uuid=wait_idle(gpu,30,out);assert os.environ['CUDA_VISIBLE_DEVICES']==uuid
 b,loading=Backend.load(model);monitor=Monitor(uuid,out/'gpu_monitor.jsonl')
 all_samples=rows(run/'data'/f'{model}_samples.jsonl');samples=all_samples[args.shard::2]
 env={'gpu':gpu,'gpu_uuid':uuid,'model':model,'pid':os.getpid(),'torch':torch.__version__,'dtype':'float16','attention':'EFFICIENT_ATTENTION','loading':loading,'sources':manifest,'protocol_sha256':sha(run/'data/protocol.json'),'sample_ids':[s['sample_id'] for s in samples],'model_paths':execution['models']}
 write(out/'environment.json',env)
 historical=read(ROOT/'references/g4_native_tokens.json')
 try:
  with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
   w=next(s for s in rows(ROOT/'data/discovery_20261103/preflight.jsonl') if s['task']=='gsm8k')
   text=b.tokenizer.apply_chat_template([{'role':'user','content':w['prompt']}],tokenize=False,add_generation_prompt=True);ids=b.tokenizer(text)['input_ids'];w={**w,'prompt_ids':ids,'prompt_length':len(ids)}
   probes=[];base=generate(b,w,randoms=spec['random_four'],probe_callback=probes.append,check=monitor.check)
   inp=torch.tensor([ids],device=b.device)
   if model=='llada':
    from vendor.sparse_llada.llada_generate import generate as official
    native=official(b.model,inp,steps=256,gen_length=256,block_length=32,temperature=0,cfg_scale=0)
   else:
    with b.operator():native=b.model.diffusion_generate(inp,max_new_tokens=256,steps=256,temperature=0,alg='maskgit_plus',eps=.001)
   assert base['generated_token_ids']==native[0,len(ids):].tolist(),'S25 differs from same-device official sampler'
   k4=generate(b,w,LAYERS[model],check=monitor.check)
   if model=='llada':
    from priority_kv_core import generate as priority_official
    native_k4=priority_official(b.model,inp,'K4_NEXT')
    assert k4['generated_token_ids']==native_k4[0,len(ids):].tolist()
    del native_k4
   else:
    repeat=generate(b,w,LAYERS[model],check=monitor.check)
    assert k4['generated_token_ids']==repeat['generated_token_ids'];del repeat
   write(out/'warmup.json',{'S25':base['generated_token_ids'],'K4':k4['generated_token_ids'],'probes':probes})
   legacy=next(iter(historical['warmup'][model].values()))
   write(run/'reports'/f'preflight_gpu{gpu}.json',{'passed':True,'S25_same_device_official_exact':True,'K4_same_device_checked':True,'historical_V100_S25_exact':base['generated_token_ids']==legacy,'historical_match_is_not_cross_device_gate':True})
   del base,k4,native,inp,probes,w;clear();emit('preflight_passed',gpu=gpu,model=model)
   while not all((run/'reports'/f'preflight_gpu{g}.json').exists() for g in gpus):monitor.check();time.sleep(3)
   pair=gpus[:2] if model=='llada' else gpus[2:]
   a,c=[read(run/'results'/model/f'gpu{g}'/'warmup.json') for g in pair]
   assert all(a[k]==c[k] for k in ['S25','K4']),'Same-model GPUs have different warmup tokens; do not pool silently'
   del a,c
   for s in samples:
    monitor.check();case=out/s['sample_id'];case.mkdir();write(case/'sample.json',s)
    result=generate(b,s,randoms=spec['random_four'],probe_callback=lambda cp:append(case/'probes.jsonl',cp),check=monitor.check)
    assert len(rows(case/'probes.jsonl'))==7
    append(case/'outputs.jsonl',{'model':model,'sample_id':s['sample_id'],'condition':'S25','gpu_uuid':uuid,**result})
    write(case/'scores_frozen.json',{'probes_sha256':sha(case/'probes.jsonl'),'historical_V100_S25_exact':result['generated_token_ids']==historical['samples'][model][s['sample_id']],'historical_match_is_not_cross_device_gate':True})
    emit('scores_frozen',model=model,gpu=gpu,sample_id=s['sample_id']);del result;clear()
   write(run/'reports'/f'scores_gpu{gpu}_complete.json',{'samples':len(samples)})
   while not all((run/'reports'/f'scores_gpu{g}_complete.json').exists() for g in gpus):monitor.check();time.sleep(3)
   for index,s in enumerate(samples):
    case=out/s['sample_id'];conditions=[('K4',LAYERS[model])]+[(f'R{i+1}',r) for i,r in enumerate(spec['random_four'])]
    offset=index%4;conditions=conditions[offset:]+conditions[:offset]
    if index%2:conditions.reverse()
    for name,layers in conditions:
     monitor.check();result=generate(b,s,layers,check=monitor.check)
     append(case/'outputs.jsonl',{'model':model,'sample_id':s['sample_id'],'condition':name,'gpu_uuid':uuid,**result});emit('generation_complete',model=model,gpu=gpu,sample_id=s['sample_id'],condition=name);del result
    assert sha(case/'probes.jsonl')==read(case/'scores_frozen.json')['probes_sha256']
    write(case/'completed.json',{'outputs':5,'checkpoints':7});clear()
   monitor.check();verify_run(run);write(out/'completed.json',{'samples':16,'outputs':80});emit('worker_complete',gpu=gpu,model=model)
 finally:monitor.close()

def analyze(run,prepare=False):
 run=Path(run).resolve();verify_run(run);code_path()
 import analyze_k4_correction as frozen
 frozen.REPORT=run/'reports';frozen.RESULT=run/'results';frozen.DATA=run/'data'
 if prepare:return frozen.prepare(*frozen.load())
 if not read(run/'reports/audit.json')['passed']:raise RuntimeError('Audit must pass first')
 frozen.analyze()
 summary=read(run/'reports/final/summary.json')
 write(run/'status.json',{'phase':'complete_first_step_only','decision':summary['decision'],'cross_model_supported':summary['cross_model_supported'],'no_next_stage_started':True})
 return summary['decision']

def label(run,path):
 run=Path(run).resolve();progress=analyze(run,prepare=True)
 pending={x['review_id'] for x in read(run/'reports/review_pending.json')}
 current=read(run/'reports/review_semantic.json');supplied=read(path)
 if not isinstance(supplied,dict):raise ValueError('Label input must be an object keyed by review_id')
 for key,value in supplied.items():
  if key not in pending:raise ValueError('Unknown review_id: '+key)
  for field in ['complete','correct']:
   if not isinstance(value.get(field),bool):raise ValueError('complete/correct must be booleans')
  if value['correct'] and not value['complete']:raise ValueError('Correct target answers must be complete')
  if not value.get('reason'):raise ValueError('Every manual label needs a reason')
  if key in current and current[key]!=value:raise ValueError('Refusing to replace an existing frozen label: '+key)
 current.update(supplied);write(run/'reports/review_semantic.json',current)
 return analyze(run,prepare=True)

def main():
 parser=argparse.ArgumentParser(description='Frozen K4 same-budget correction diagnostic, portable g8 runner')
 sub=parser.add_subparsers(dest='command',required=True)
 sub.add_parser('self-test',help='CPU formula/selection tests and immutable export integrity')
 for command in ['check','run']:
  p=sub.add_parser(command);p.add_argument('--llada-model',default=os.environ.get('LLADA_MODEL','/dataset/money-shredder-group/tv/LLaDA-8B-Instruct'))
  p.add_argument('--dream-model',default=os.environ.get('DREAM_MODEL','/dataset/money-shredder-group/tv/Dream-v0-Instruct-7B'))
  p.add_argument('--gpus',default='0,1,2,3')
  if command=='run':p.add_argument('--run-dir',required=True);p.add_argument('--wait',action='store_true')
 for command in ['audit','review','analyze','label']:
  p=sub.add_parser(command);p.add_argument('--run-dir',required=True)
  if command=='label':p.add_argument('--labels',required=True)
 p=sub.add_parser('_worker',help=argparse.SUPPRESS);p.add_argument('--run-dir',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--model',choices=['llada','dream'],required=True);p.add_argument('--shard',type=int,choices=[0,1],required=True)
 args=parser.parse_args()
 if args.command=='self-test':print(json.dumps(self_test()))
 elif args.command=='check':print(json.dumps(check_environment({'llada':args.llada_model,'dream':args.dream_model},parse_gpus(args.gpus)),ensure_ascii=False,indent=2))
 elif args.command=='run':launch(args)
 elif args.command=='_worker':worker(args)
 elif args.command=='audit':
  from .audit import audit
  print(json.dumps(audit(Path(args.run_dir).resolve())))
 elif args.command=='review':print(json.dumps(analyze(args.run_dir,prepare=True)))
 elif args.command=='analyze':print(json.dumps(analyze(args.run_dir)))
 elif args.command=='label':print(json.dumps(label(args.run_dir,args.labels)))

if __name__=='__main__':main()
