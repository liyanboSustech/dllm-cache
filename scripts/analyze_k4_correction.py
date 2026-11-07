"""Frozen first-step analysis; no selector construction or metric search."""
import argparse,collections,json,random
from pathlib import Path
import numpy as np
from analyze_k12_layer_validation import read,write,sha,rows,digest,holm
from k4_correction_core import TAG,LAYERS
ROOT=Path(__file__).resolve().parents[1];REPORT,RESULT,DATA=[ROOT/f/TAG for f in ['reports','results','data']]
SEED=2026110702

def stats(values):
 x=np.asarray(values,dtype=float);n=len(x)
 if not n:return {'n':0,'difference':None,'ci95':[None,None],'p':1.}
 rng=np.random.default_rng(SEED);boot=x[rng.integers(0,n,(10000,n))].mean(1)
 obs=abs(float(x.mean()));ge=0
 for _ in range(100):
  signs=rng.integers(0,2,(1000,n))*2-1;ge+=int((abs((signs*x).mean(1))>=obs-1e-15).sum())
 return {'n':n,'difference':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'p':float((ge+1)/100001),'positive_questions':int((x>0).sum()),'negative_questions':int((x<0).sum())}

def load():
 pp=sorted(RESULT.glob('*/gpu*/gsm*/outputs.jsonl'))
 return [r for p in pp for r in rows(p)],{p.parent.name:read(p.parent/'sample.json') for p in pp}

def prepare(full,samples):
 inherited={};provenance={}
 for tag in ['kv_generalization_20261104','kv_history_phase_20261104','decision_paths_20261106','early_sources_20261106','matched_sources_20261106']:
  for name in ['review_auto.json','review_previous.json','review_semantic.json']:
   p=ROOT/'reports'/tag/name
   if not p.exists():continue
   provenance[str(p.relative_to(ROOT))]=sha(p)
   for key,label in read(p).items():
    if key in inherited:assert all(label[f]==inherited[key][f] for f in ['complete','correct']),key
    inherited[key]={**label,'inherited_from':str(p.relative_to(ROOT))}
 groups=collections.defaultdict(list)
 for r in full:groups[digest(r)].append(r)
 auto={};pending=[]
 for key,rs in sorted(groups.items()):
  r=rs[0];s=samples[r['sample_id']]
  if key in inherited:auto[key]=inherited[key]
  elif r['parsed_answer'] is not None:auto[key]={'correct':r['correct'],'complete':True,'label_source':'Explicit parseable Final; numeric target match.'}
  else:pending.append({'review_id':key,'question':s['question'],'gold_answer':s['answer'],'answer':r['answer_before_stop']})
 random.Random(SEED).shuffle(pending)
 write(REPORT/'review_auto.json',auto);write(REPORT/'review_pending.json',pending);write(REPORT/'inherited_sources.json',provenance)
 manual=read(REPORT/'review_semantic.json') if (REPORT/'review_semantic.json').exists() else {}
 progress={'outputs':len(full),'unique_answers':len(groups),'automatic_or_inherited':len(auto),'manual_required':len(pending),'pending':sum(r['review_id'] not in manual for r in pending)}
 write(REPORT/'progress.json',progress);return progress

def score_table():
 subjects={};detail=[];coverage=collections.Counter();stops=collections.Counter()
 for p in sorted(RESULT.glob('*/gpu*/gsm*/probes.jsonl')):
  model=p.parents[2].name;sid=p.parent.name;rr=rows(p);assert len(rr)==7
  assert sha(p)==read(p.parent/'scores_frozen.json')['probes_sha256']
  by=collections.defaultdict(list)
  for cp in rr:
   stops[model]+=int(cp['prior_stop_present'])
   for li,plan in cp['plans'].items():
    coverage[model+':qk_nonfinite_layers']+=int(plan['qk_nonfinite']>0)
    coverage[model+':next_coverage_shortfall_layers']+=int(plan['next_retained']<min(len(plan['base']),plan['next_available']))
   for r in cp['rows']:
    by[r['condition']].append(r['gain']);detail.append({'model':model,'sample_id':sid,'block':cp['block'],'prior_stop':cp['prior_stop_present'],**r})
  assert all(len(v)==7 for v in by.values())
  subjects[model,sid]={k:float(np.mean(v)) for k,v in by.items()}
 return subjects,detail,dict(coverage),dict(stops)

def analyze():
 full,samples=load();progress=prepare(full,samples)
 assert progress['outputs']==320 and progress['pending']==0,progress
 assert sha(__file__)==read(REPORT/'analysis_frozen.json')['sha256']
 assert all(v==0 for v in read(REPORT/'exit_codes.json').values())
 labels={**read(REPORT/'review_auto.json'),**read(REPORT/'review_semantic.json')}
 for r in full:
  label=labels[digest(r)];r.update(semantic=label,semantic_correct=bool(label['correct']),complete=bool(label['complete']),strict_correct=bool(r['correct']),alternative_correct=bool(label.get('correct_if_ambiguous',label['correct'])))
 g={(r['model'],r['sample_id'],r['condition']):r for r in full};assert len(g)==320
 scores,detail,coverage,stops=score_table();assert len(scores)==64
 ids=sorted(samples);assert len(ids)==32
 primary={};layer_profiles={};quality={};links={};effects={}
 for model in LAYERS:
  s=[scores[model,sid] for sid in ids]
  primary[model+':K4_gain_positive']=stats([x['K4'] for x in s])
  primary[model+':K4_minus_random_mean']=stats([x['K4']-np.mean([x[r] for r in ['R1','R2','R3']]) for x in s])
  primary[model+':mean_K4_leave_one_out_positive']=stats([np.mean([x['K4']-x[f'K4_MINUS_{li:02d}'] for li in LAYERS[model]]) for x in s])
  layer_profiles[model]={'single':{str(li):stats([x[f'L{li:02d}'] for x in s]) for li in range(32 if model=='llada' else 28)},
   'conditional_K4':{str(li):stats([x['K4']-x[f'K4_MINUS_{li:02d}'] for x in s]) for li in LAYERS[model]},
   'K4_vs_each_random':{r:stats([x['K4']-x[r] for x in s]) for r in ['R1','R2','R3']}}
  quality[model]={c:{'n':32,'correct':sum(g[model,sid,c]['semantic_correct'] for sid in ids),'incomplete':sum(not g[model,sid,c]['complete'] for sid in ids),'complete_wrong':sum(g[model,sid,c]['complete'] and not g[model,sid,c]['semantic_correct'] for sid in ids),'strict_correct':sum(g[model,sid,c]['strict_correct'] for sid in ids),'alternative_correct':sum(g[model,sid,c]['alternative_correct'] for sid in ids)} for c in ['S25','K4','R1','R2','R3']}
  effects[model]=stats([int(g[model,sid,'K4']['semantic_correct'])-np.mean([int(g[model,sid,c]['semantic_correct']) for c in ['R1','R2','R3']]) for sid in ids])
  links[model]={}
  for metric in ['semantic_correct','strict_correct','alternative_correct']:
   auc=[];per=[]
   for sid in ids:
    cc=['K4','R1','R2','R3'];pos=[c for c in cc if g[model,sid,c][metric]];neg=[c for c in cc if not g[model,sid,c][metric]]
    if not pos or not neg:continue
    v=float(np.mean([float(scores[model,sid][a]>scores[model,sid][b])+.5*float(scores[model,sid][a]==scores[model,sid][b]) for a in pos for b in neg]))
    auc.append(v);per.append({'sample_id':sid,'auc':v})
   st=stats(auc);links[model][metric]={'n_mixed':len(auc),'auc':st['difference'],'ci95':st['ci95'],'per_question':per}
 adjusted=holm({k:v['p'] for k,v in primary.items()})
 for k,v in primary.items():v['holm_p']=adjusted[k]
 decisions={}
 for model in LAYERS:
  mechanism=all(v['difference']>0 and v['holm_p']<.05 for k,v in primary.items() if k.startswith(model+':'))
  a=links[model]['semantic_correct'];link=a['n_mixed']>=12 and a['auc']>=.60 and a['ci95'][0]>.5 and effects[model]['difference']>0
  decisions[model]={'mechanism_gate':bool(mechanism),'task_link_gate':bool(link),'supported':bool(mechanism and link)}
 final=REPORT/'final';final.mkdir(exist_ok=True)
 out={'progress':progress,'primary':primary,'layer_profiles':layer_profiles,'quality':quality,'task_links':links,'accuracy_K4_minus_random_mean':effects,'decision':decisions,'cross_model_supported':all(x['supported'] for x in decisions.values()),'native_QK_coverage':coverage,'prior_stop_checkpoints_of224':stops,'limits':['32 reused development questions, no independent validation.','Full same-age cached reference is not uncached base or gold answer.','Primary averages ALL seven blocks; many may be after first STOP. No posthoc checkpoint change.','Native Dream overflow/tie semantics unchanged; actual coverage reported.','Frozen mapped K4 Dream is not already proven effective.','Assistant semantic review not independent human.','No new layer selector, next stage, timing or novelty claim.']}
 write(final/'summary.json',out)
 for name,rr in [('semantic_outputs',full),('probe_details',detail),('subject_scores',[{'model':m,'sample_id':sid,'scores':s} for (m,sid),s in scores.items()])]:
  (final/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
 print(json.dumps({'primary':primary,'quality':quality,'decision':decisions},ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
 if a.prepare:print(json.dumps(prepare(*load())))
 else:analyze()
