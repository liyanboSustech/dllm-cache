from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
TAG='k4_correction_20261107'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def append(p,value):
 with Path(p).open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def code_path():
 sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
def verify_export():
 manifest=read(ROOT/'references/export_manifest.json')
 for f,h in manifest['files'].items():
  p=ROOT/f
  if not p.is_file() or sha(p)!=h:raise RuntimeError('Frozen export changed or missing: '+f)
 return len(manifest['files'])
def source_manifest():
 folders=['portable','scripts','vendor','data','reports','references','tests']
 paths=[p for folder in folders for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc']]
 paths += [ROOT/f for f in ['run_k4.py','requirements.txt','requirements-torch.txt','README.md','AGENTS.md']]
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def verify_run(run):
 run=Path(run);manifest=read(run/'source_manifest.json')
 for f,h in manifest.items():
  if sha(ROOT/f)!=h:raise RuntimeError('Source changed after run freeze: '+f)
 original=ROOT/'data'/TAG/'protocol.json'
 if sha(run/'data/protocol.json')!=sha(original):raise RuntimeError('Run protocol changed')
 for m,spec in read(original)['models'].items():
  if sha(run/'data'/f'{m}_samples.jsonl')!=spec['sha256']:raise RuntimeError('Run samples changed: '+m)
 return manifest

def parse_gpus(value):
 try:g=[int(x) for x in value.split(',')]
 except ValueError:raise ValueError('--gpus must contain four comma-separated integer indices')
 if len(g)!=4 or len(set(g))!=4 or min(g)<0:raise ValueError('Exactly four DISTINCT nonnegative GPU indices are required')
 return g

def gpu_snapshot():
 def query(fields):return subprocess.check_output(['nvidia-smi',*fields,'--format=csv,noheader,nounits'],text=True,timeout=15).strip()
 devices=[]
 for line in query(['--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu']).splitlines():
  i,u,n,total,used,util=[x.strip() for x in line.split(',')]
  devices.append({'index':int(i),'uuid':u,'name':n,'total_mib':int(total),'used_mib':int(used),'utilization':int(util)})
 procs=query(['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'])
 return {'devices':devices,'processes':procs,'time_unix':time.time()}

def gpu_ready(snapshot,gpus):
 by={d['index']:d for d in snapshot['devices']}
 if any(g not in by for g in gpus):raise RuntimeError('Requested GPU index does not exist')
 if any(by[g]['total_mib']<24000 for g in gpus):raise RuntimeError('This packaged FP16 study requires at least 24 GB per GPU; 32 GB recommended')
 return all(by[g]['used_mib']<=1000 and by[g]['utilization']<=5 and by[g]['uuid'] not in snapshot['processes'] for g in gpus)

def configure_models(paths):
 code_path()
 import discovery_common,kv_generalization_core
 discovery_common.MODEL=str(Path(paths['llada']).resolve())
 kv_generalization_core.MODEL_PATHS.update({k:str(Path(v).resolve()) for k,v in paths.items()})
