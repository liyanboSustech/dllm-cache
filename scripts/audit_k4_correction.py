"""Independent saved-artifact audit; no model/metric selection."""
import collections,hashlib,json,subprocess
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];T='k4_correction_20261107';D,P,O=[R/f/T for f in ['data','reports','results']]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
p=read(D/'protocol.json');sources=read(P/'sources_frozen.json');count=collections.Counter();warm={};uuids=[]
assert sha(D/'protocol.json')==read(P/'protocol_frozen.json')['sha256']
for f,h in sources.items():assert sha(R/f)==h,f
assert all(c==0 for c in read(P/'exit_codes.json').values())
for gpu,model,shard in [(0,'llada',0),(1,'llada',1),(2,'dream',0),(3,'dream',1)]:
 root=O/model/f'gpu{gpu}';env=read(root/'environment.json');uuid=env['gpu_uuid'];uuids.append(uuid)
 assert env['sources']==sources and env['protocol_sha256']==sha(D/'protocol.json')
 assert read(root/'completed.json')=={'samples':16,'outputs':80}
 w=read(root/'warmup.json')
 if model in warm:assert warm[model]==w
 warm[model]=w
 mon=rows(root/'gpu_monitor.jsonl');own={x.split(',')[1].strip() for x in mon[0]['processes'].splitlines() if uuid in x};assert len(own)==1
 for m in mon:
  assert not {x.split(',')[1].strip() for x in m['processes'].splitlines() if uuid in x}-own;count['clean_monitor_records']+=1
 spec=p['models'][model];assert sha(R/spec['samples'])==spec['sha256']
 mask=126336 if model=='llada' else 151666;nl=32 if model=='llada' else 28;heads=32 if model=='llada' else 4
 for sample in rows(R/spec['samples'])[shard::2]:
  case=root/sample['sample_id'];rr=rows(case/'outputs.jsonl');probes=rows(case/'probes.jsonl')
  assert read(case/'completed.json')=={'outputs':5,'checkpoints':7}
  assert {r['condition'] for r in rr}=={'S25','K4','R1','R2','R3'} and len(rr)==5
  assert sha(case/'probes.jsonl')==read(case/'scores_frozen.json')['probes_sha256']
  plen=sample['prompt_length'];ext=plen+224;ret=int(.25*ext);base=next(r for r in rr if r['condition']=='S25')
  old=read(R/'results/early_sources_20261106'/model/f'gpu{gpu}'/sample['sample_id']/'native_S25.json')
  assert old['generated_token_ids']==base['generated_token_ids'] and old['gpu_uuid']==uuid
  count['historical_native_exact']+=1
  state=sample['prompt_ids']+[mask]*256;byblock={cp['block']:cp for cp in probes};assert set(byblock)==set(range(7))
  for c in base['commits']:
   block,step=c['block'],c['step'];start=plen+block*32
   if block<7 and step==1:assert state==byblock[block]['anchor_input_ids']
   if block<7 and step==2:
    cp=byblock[block];assert cp['x_ids']==state and cp['start']==start and cp['step']==2
    active=[i for i in range(32) if state[start+i]==mask];assert cp['active_positions']==active
    assert cp['native_cache_exact'] and cp['anchor_values_unchanged']
    absolute=list(range(start))+list(range(start+32,plen+256));assert cp['external_absolute_positions']==absolute
    assert set(cp['plans'])==set(map(str,range(nl)))
    for li,plan in cp['plans'].items():
     for key in ['base','next']:assert len(plan[key])==len(set(plan[key]))==ret and all(0<=i<ext for i in plan[key])
     assert plan['next_available']==32 and plan['next_retained']==sum(start<=i<start+32 for i in plan['next'])
     count['layer_plans']+=1
    cr={r['condition']:r for r in cp['rows']};assert len(cr)==nl+9
    expected={'S25':[],'K4':spec['K4'],**{f'R{i+1}':ls for i,ls in enumerate(spec['random_four'])},**{f'L{i:02d}':[i] for i in range(nl)},**{f'K4_MINUS_{li:02d}':[i for i in spec['K4'] if i!=li] for li in spec['K4']}}
    assert set(cr)==set(expected)
    for name,r in cr.items():
     assert r['layers']==expected[name] and r['external_counts']==[ret]*nl and r['kv_bytes']==ret*nl*2*heads*128*2
     vals=np.asarray(r['js_per_position']);assert len(vals)==len(active) and np.isfinite(vals).all() and vals.min()>=-1e-6 and vals.max()<np.log(2)+1e-6
     assert abs(float(vals.mean())-r['js'])<1e-7 and abs((cr['S25']['js']-r['js'])-r['gain'])<1e-12
     count['probe_rows']+=1
    count['checkpoints']+=1
   for pos,tok in zip(c['positions'],c['tokens']):assert state[pos]==mask;state[pos]=tok
  for r in rr:
   assert r['gpu_uuid']==uuid and r['sample_id']==sample['sample_id'] and r['model']==model
   state=sample['prompt_ids']+[mask]*256;assert [(c['block'],c['step']) for c in r['commits']]==[(b,t) for b in range(8) for t in range(32)]
   for c in r['commits']:
    start=plen+c['block']*32;step=c['step'];rem=state[start:start+32].count(mask)
    if model=='llada' or step==0:n=1
    elif step==31:n=rem
    else:
     times=np.linspace(1,.001,33,dtype=np.float32);n=int(np.float32(rem)*np.float32(1-times[step+1]/times[step]))
    assert len(c['positions'])==len(c['tokens'])==len(set(c['positions']))==n
    for pos,tok in zip(c['positions'],c['tokens']):assert start<=pos<start+32 and state[pos]==mask and tok!=mask;state[pos]=tok
    count['commit_steps']+=1
   assert state[plen:]==r['generated_token_ids'] and r['remaining_mask_tokens']==0;count['outputs']+=1
assert len(set(uuids))==4 and count['outputs']==320 and count['checkpoints']==448 and count['probe_rows']==17472
excluded={'AGENTS.md','RESEARCH_CONTEXT.txt','METHOD_AND_EXPERIMENTS.md','status.json'}
old=read(R/'reports/matched_sources_20261106/final_manifest.json');old_n=0
for f,item in old['files'].items():
 if f not in excluded:assert sha(R/f)==item['sha256'],f;old_n+=1
result={'passed':True,'counts':dict(count),'old_immutable_files_unchanged':old_n,'sources_unchanged':len(sources),'four_GPU_UUIDs':uuids,'same_model_warmups_exact':True,'all_actual_budgets_matched':True,'reserved_sets_untouched':True}
(P/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
