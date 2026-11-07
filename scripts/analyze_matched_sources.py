"""Prespecified semantic review and independent paired regional test."""
import argparse,collections,json,random
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from analyze_k12_layer_validation import read,write,sha,rows,digest,holm
from matched_sources_core import TAG,conditions
ROOT=Path(__file__).resolve().parents[1];REPORT,RESULT,DATA=[ROOT/f/TAG for f in ['reports','results','data']];SEED=2026110634
def load():
 pp=sorted(RESULT.glob('*/gpu*/gsm*/outputs.jsonl'));return [r for p in pp for r in rows(p)],{p.parent.name:read(p.parent/'sample.json') for p in pp}
def prepare(full,samples):
 inherited={};provenance={}
 for tag in ['kv_generalization_20261104','kv_history_phase_20261104','decision_paths_20261106','early_sources_20261106']:
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
 random.Random(SEED).shuffle(pending);write(REPORT/'review_auto.json',auto);write(REPORT/'review_pending.json',pending);write(REPORT/'inherited_sources.json',provenance)
 manual=read(REPORT/'review_semantic.json') if (REPORT/'review_semantic.json').exists() else {}
 p={'outputs':len(full),'complete_model_questions':len(list(RESULT.glob('*/gpu*/gsm*/completed.json'))),'unique_answers':len(groups),'automatic_or_inherited':len(auto),'manual_required':len(pending),'manual_done':sum(r['review_id'] in manual for r in pending),'pending':sum(r['review_id'] not in manual for r in pending)};write(REPORT/'progress.json',p);return p
def stats(vals):
 x=np.asarray(vals,dtype=float);n=len(x);boot=x[np.random.default_rng(SEED).integers(0,n,(10000,n))].mean(1);w=int((x>0).sum());l=int((x<0).sum())
 return {'n':n,'difference':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'wins':w,'losses':l,'exact_p':float(binomtest(w,w+l,.5).pvalue) if w+l else 1.}
def analyze(full,samples):
 p=prepare(full,samples);assert p['outputs']==1536 and p['complete_model_questions']==320 and p['pending']==0,p
 assert sha(__file__)==read(REPORT/'analysis_frozen.json')['sha256'];assert all(v==0 for v in read(REPORT/'exit_codes.json').values());assert sha(DATA/'protocol.json')==read(REPORT/'protocol_frozen.json')['sha256']
 labels={**read(REPORT/'review_auto.json'),**read(REPORT/'review_semantic.json')}
 for r in full:
  l=labels[digest(r)];r.update(semantic=l,semantic_correct=bool(l['correct']),complete=bool(l['complete']),strict_correct=bool(r['correct']),alternative_correct=bool(l.get('correct_if_ambiguous',l['correct'])),alternative_complete=bool(l.get('complete_if_ambiguous',l['complete'])))
 g={(r['model'],r['sample_id'],r['condition']):r for r in full};assert len(g)==1536;quality={};effects={};primary={};sensitivity={};decision={};pairs=[]
 for cohort in ['development','confirmation']:
  ids=sorted(sid for sid,s in samples.items() if s['cohort']==cohort);assert len(ids)==(32 if cohort=='development' else 128);quality[cohort]={};effects[cohort]={}
  for model in ['llada','dream']:
   for sid in ids:assert len({g[model,sid,c]['gpu_uuid'] for c in conditions(cohort)})==1
   quality[cohort][model]={};effects[cohort][model]={}
   for c in conditions(cohort):
    rr=[g[model,sid,c] for sid in ids];quality[cohort][model][c]={'n':len(ids),'correct':sum(r['semantic_correct'] for r in rr),'incomplete':sum(not r['complete'] for r in rr),'complete_wrong':sum(r['complete'] and not r['semantic_correct'] for r in rr),'strict_correct':sum(r['strict_correct'] for r in rr),'alternative_correct':sum(r['alternative_correct'] for r in rr)}
   comparisons=[(c,'BASE') for c in conditions(cohort) if c!='BASE']+[(f'MASK_{i}',f'TEXT_{i}') for i in (range(3) if cohort=='development' else range(1))]
   for a,b in comparisons:
    name=a+'-'+b;effects[cohort][model][name]={}
    for metric in ['semantic_correct','complete','strict_correct','alternative_correct','alternative_complete']:
     v=stats([int(g[model,sid,a][metric])-int(g[model,sid,b][metric]) for sid in ids]);effects[cohort][model][name][metric]=v
     if cohort=='confirmation' and name in ['MASK_0-TEXT_0','MASK_0-BASE']:
      sensitivity.setdefault(metric,{})[model+':'+name]=v
      if metric=='semantic_correct':primary[model+':'+name]=v
    for sid in ids:
     ra,rb=g[model,sid,a],g[model,sid,b]
     if ra['answer_before_stop']!=rb['answer_before_stop']:pairs.append({'model':model,'cohort':cohort,'sample_id':sid,'comparison':name,'question':samples[sid]['question'],'gold':samples[sid]['answer'],'a':{k:ra[k] for k in ['answer_before_stop','semantic_correct','complete']},'b':{k:rb[k] for k in ['answer_before_stop','semantic_correct','complete']}})
 for metric,comparisons in sensitivity.items():
  adjusted=holm({k:v['exact_p'] for k,v in comparisons.items()});assert len(adjusted)==4
  for k,v in comparisons.items():v['holm_p']=adjusted[k]
 for model in ['llada','dream']:
  decision[model]=all(primary[model+':'+c]['difference']>0 and primary[model+':'+c]['holm_p']<.05 for c in ['MASK_0-TEXT_0','MASK_0-BASE'])
 out={'counts':p,'quality':quality,'effects':effects,'primary':primary,'sensitivity':sensitivity,'decision':decision,'cross_model_supported':all(decision.values()),'label_manifest':{'auto':sha(REPORT/'review_auto.json'),'manual':sha(REPORT/'review_semantic.json')},'limits':['Fresh128 confirmation separate from32 reused development; same GSM8K train task, not unseen task or official test.','Uniform equal-count regional subset, not every possible MASK selector; TEXT=prompt in first block.','Matched MASK/TEXT augmented budgets; BASE lower and ALL full, no fixed25% deployment claim.','Assistant review not independent human; inherited exact-text labels unchanged.','One block transient intervention then S25; no universal layer/stage claim or deployment speed/novelty proof.']}
 final=REPORT/'final';final.mkdir(exist_ok=True);write(final/'summary.json',out)
 for name,rs in [('semantic_outputs',full),('all_changed_answer_pairs',pairs)]: (final/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
 print(json.dumps({'quality':quality,'primary':primary,'decision':decision},ensure_ascii=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pending',type=int);a=p.parse_args();full,samples=load()
 if a.prepare or a.pending:
  print(json.dumps(prepare(full,samples)))
  if a.pending:
   done=read(REPORT/'review_semantic.json') if (REPORT/'review_semantic.json').exists() else {}
   for r in [x for x in read(REPORT/'review_pending.json') if x['review_id'] not in done][:a.pending]:print(json.dumps(r,ensure_ascii=False))
 else:analyze(full,samples)
if __name__=='__main__':main()
