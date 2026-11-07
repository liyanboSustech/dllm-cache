"""Wait for four idle GPUs, launch only owned workers, fail closed on interference."""
import json,os,subprocess,time,sys
from discovery_common import ROOT,append,write_json,gpu_snapshot
from k4_correction_core import TAG
P=ROOT/'reports'/TAG
def status(phase,running):
 path=ROOT/'status.json';s=json.loads(path.read_text());s.update(experiments_running=running,current_task=TAG,next_action=phase)
 s[TAG]={'phase':phase,'launcher_pid':os.getpid(),'report':str(P.relative_to(ROOT)),'step2_authorized':False,'step3_authorized':False}
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)
status('waiting_for_four_idle_GPUs',False)
write_json(P/'launcher_pid.json',{'pid':os.getpid(),'state':'waiting_for_four_idle_GPUs'})
while True:
 snap=gpu_snapshot();append(P/'queue_gpu_checks.jsonl',snap)
 rows=[r.split(', ') for r in snap['gpus'].splitlines()]
 idle=all(int(r[3])<=1000 and int(r[4])<=5 and r[1] not in snap['processes'] for r in rows if int(r[0])<4)
 if len(rows)>=4 and idle:break
 print(json.dumps({'event':'waiting_for_foreign_jobs','snapshot':snap}),flush=True);time.sleep(30)
status('four_GPU_first_step_running',True)
jobs={};logs={}
for gpu,model,shard in [(0,'llada',0),(1,'llada',1),(2,'dream',0),(3,'dream',1)]:
 log=(P/f'{model}_gpu{gpu}.log').open('x');logs[gpu]=log
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
 cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_k4_correction_worker.py'),'--model',model,'--gpu',str(gpu),'--shard',str(shard)]
 jobs[gpu]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
write_json(P/'worker_pids.json',{g:p.pid for g,p in jobs.items()});print('workers', {g:p.pid for g,p in jobs.items()},flush=True)
while any(p.poll() is None for p in jobs.values()):
 if any(p.poll() not in [None,0] for p in jobs.values()):
  for p in jobs.values():
   if p.poll() is None:p.terminate()
  break
 time.sleep(5)
codes={g:p.wait() for g,p in jobs.items()};write_json(P/'exit_codes.json',codes)
for log in logs.values():log.close()
print('exit_codes',codes,flush=True)
if not all(c==0 for c in codes.values()):
 status('worker_failure_inspect_logs_no_metric_change',False);sys.exit(1)
for script,args in [('audit_k4_correction.py',[]),('analyze_k4_correction.py',['--prepare'])]:
 with (P/(script+'.log')).open('w') as log:
  result=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  status('audit_or_review_preparation_failure_inspect_logs',False);sys.exit(result.returncode)
status('generations_audited_awaiting_blinded_review_and_final_analysis',False)

