"""Launch four owned workers; abort peers if any validation/run fails."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
report=ROOT/'reports/decision_paths_20261106'
jobs={};logs={}
for gpu,model,shard in [(0,'llada',0),(1,'llada',1),(2,'dream',0),(3,'dream',1)]:
    log=(report/f'{model}_gpu{gpu}.log').open('x');logs[gpu]=log
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
    command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_decision_paths_worker.py'),
             '--model',model,'--gpu',str(gpu),'--shard',str(shard)]
    jobs[gpu]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
(report/'worker_pids.json').write_text(json.dumps({g:p.pid for g,p in jobs.items()},indent=2)+'\n')
print(json.dumps({'workers':{g:p.pid for g,p in jobs.items()}}),flush=True)
while any(p.poll() is None for p in jobs.values()):
    if any(p.poll() not in [None,0] for p in jobs.values()):
        # These exact child PIDs belong to this launcher; never inspect/stop other jobs.
        for process in jobs.values():
            if process.poll() is None:process.terminate()
        break
    time.sleep(5)
codes={gpu:process.wait() for gpu,process in jobs.items()}
(report/'exit_codes.json').write_text(json.dumps(codes,indent=2)+'\n')
for log in logs.values():log.close()
print(json.dumps({'exit_codes':codes}),flush=True)
raise SystemExit(0 if all(code==0 for code in codes.values()) else 1)
