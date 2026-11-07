"""Frozen method/model-hidden labels and question-paired decision-path analysis."""
import argparse,collections,json,random
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from analyze_k12_layer_validation import read,write,sha,rows,digest,holm
ROOT=Path(__file__).resolve().parents[1];TAG='early_sources_20261106';REPORT,RESULT,DATA=[ROOT/f/TAG for f in ['reports','results','data']];SEED=2026110621
CONDS=['BASE','ORDER','CONTENT','BOTH'];PHASES=['b0_r04','b1_r04']
EFFECTS={'ORDER_minus_BASE':{'ORDER':1,'BASE':-1},'CONTENT_minus_BASE':{'CONTENT':1,'BASE':-1},'BOTH_minus_BASE':{'BOTH':1,'BASE':-1},'ORDER_minus_CONTENT':{'ORDER':1,'CONTENT':-1},'interaction':{'BOTH':1,'ORDER':-1,'CONTENT':-1,'BASE':1},'order_recovery_over_half_both':{'ORDER':1,'BOTH':-.5,'BASE':-.5}}
PRIMARY=list(EFFECTS)[:4]
def load():
 pp=sorted(RESULT.glob('*/gpu*/gsm*/outputs.jsonl'));return [r for p in pp for r in rows(p)],{p.parent.name:read(p.parent/'sample.json') for p in pp}
def prepare(full,samples):
 inherited={};provenance={}
 for tag in ['kv_generalization_20261104','kv_history_phase_20261104','decision_paths_20261106']:
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
  elif r['parsed_answer'] is not None:auto[key]={'correct':r['correct'],'complete':True,'label_source':'Explicit parseable Final, frozen numeric target match.'}
  else:pending.append({'review_id':key,'question':s['question'],'gold_answer':s['answer'],'answer':r['answer_before_stop']})
 random.Random(SEED).shuffle(pending);write(REPORT/'review_auto.json',auto);write(REPORT/'review_pending.json',pending);write(REPORT/'inherited_sources.json',provenance)
 manual=read(REPORT/'review_semantic.json') if (REPORT/'review_semantic.json').exists() else {}
 p={'outputs':len(full),'by_model':dict(collections.Counter(r['model'] for r in full)),'complete_model_questions':len(list(RESULT.glob('*/gpu*/gsm*/completed.json'))),'unique_answers':len(groups),'automatic_or_inherited':len(auto),'manual_required':len(pending),'manual_done':sum(r['review_id'] in manual for r in pending),'pending':sum(r['review_id'] not in manual for r in pending)};write(REPORT/'progress.json',p);return p

def stats(values,binary_pair=False):
 x=np.asarray(values,dtype=float);n=len(x)
 if n==0:return {'n_questions':0,'difference':None,'ci95':None}
 boot=x[np.random.default_rng(SEED).integers(0,n,(10000,n))].mean(1)
 out={'n_questions':n,'difference':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'positive_questions':int((x>0).sum()),'negative_questions':int((x<0).sum())}
 if binary_pair:
  assert set(x)<=set([-1.,0.,1.]);w=int((x>0).sum());l=int((x<0).sum());out['exact_two_sided_p']=float(binomtest(w,w+l,.5).pvalue) if w+l else 1.
 return out


def analyze(full,samples):
 p=prepare(full,samples);assert p['outputs']==1664 and p['complete_model_questions']==64 and p['pending']==0,p
 assert sha(__file__)==read(REPORT/'analysis_frozen.json')['sha256'];assert sha(DATA/'protocol.json')==read(REPORT/'protocol_frozen.json')['sha256'];assert all(v==0 for v in read(REPORT/'exit_codes.json').values())
 protocol=read(DATA/'protocol.json');phases=protocol['phases'];conds={ph:[c['key'] for c in protocol['conditions'][ph]] for ph in phases}
 labels={**read(REPORT/'review_auto.json'),**read(REPORT/'review_semantic.json')}
 for r in full:
  l=labels[digest(r)];r.update(semantic=l,semantic_correct=bool(l['correct']),complete=bool(l['complete']),strict_correct=bool(r['correct']),alternative_correct=bool(l.get('correct_if_ambiguous',l['correct'])),alternative_complete=bool(l.get('complete_if_ambiguous',l['complete'])))
 g={(r['model'],r['sample_id'],r['phase'],r['condition']):r for r in full};assert len(g)==1664;ids=sorted(samples);assert len(ids)==32
 quality={};effects={};primary={};transitions={};decision={}
 for m in ['llada','dream']:
  for sid in ids:assert len({r['gpu_uuid'] for r in full if r['model']==m and r['sample_id']==sid})==1
  quality[m]={};effects[m]={};transitions[m]={}
  specs={}
  for ph in phases:
   quality[m][ph]={};transitions[m][ph]={}
   for c in conds[ph]:
    rr=[g[m,sid,ph,c] for sid in ids];quality[m][ph][c]={'n':32,'correct':sum(r['semantic_correct'] for r in rr),'incomplete':sum(not r['complete'] for r in rr),'complete_wrong':sum(r['complete'] and not r['semantic_correct'] for r in rr),'strict_correct':sum(r['strict_correct'] for r in rr),'alternative_correct':sum(r['alternative_correct'] for r in rr),'alternative_incomplete':sum(not r['alternative_complete'] for r in rr),'preexisting_STOP':sum(r['any_stop_already_present'] for r in rr)}
    if c!='BASE':specs[ph+':'+c+'_minus_BASE']={(ph,c):1,(ph,'BASE'):-1}
    counts=collections.Counter()
    for r in rr:
     b=g[m,r['sample_id'],ph,'BASE'];state=lambda x:'correct' if x['semantic_correct'] else 'complete_wrong' if x['complete'] else 'incomplete';counts[state(b)+' -> '+state(r)]+=1
    transitions[m][ph][c]=dict(counts)
   for src in ['ALL','MASK','TEXT']:
    if src+'_ORDER' in conds[ph]:
     specs[ph+':'+src+'_ORDER_minus_CONTENT']={(ph,src+'_ORDER'):1,(ph,src+'_CONTENT'):-1}
     specs[ph+':'+src+'_interaction']={(ph,src+'_BOTH'):1,(ph,src+'_ORDER'):-1,(ph,src+'_CONTENT'):-1,(ph,'BASE'):1}
   if 'MASK_BOTH' in conds[ph]:
    specs[ph+':MASK_BOTH_minus_TEXT_BOTH']={(ph,'MASK_BOTH'):1,(ph,'TEXT_BOTH'):-1}
    specs[ph+':region_interaction']={(ph,'ALL_BOTH'):1,(ph,'MASK_BOTH'):-1,(ph,'TEXT_BOTH'):-1,(ph,'BASE'):1}
  for arm in ['ORDER','CONTENT','BOTH']:specs['timing:ALL_'+arm+'_early_minus_late']={('b0_early','ALL_'+arm):1,('b0_late','ALL_'+arm):-1}
  specs['b1_clean_early:PROMPT_minus_HISTORY']={('b1_clean_early','PROMPT_BOTH'):1,('b1_clean_early','HISTORY_BOTH'):-1}
  main_names=['b0_early:ALL_BOTH_minus_BASE','timing:ALL_BOTH_early_minus_late','b0_early:MASK_BOTH_minus_BASE','b0_early:TEXT_BOTH_minus_BASE','b0_early:MASK_BOTH_minus_TEXT_BOTH','b1_clean_early:MASK_BOTH_minus_BASE','b1_clean_early:TEXT_BOTH_minus_BASE','b1_clean_early:MASK_BOTH_minus_TEXT_BOTH']
  for metric in ['semantic_correct','complete','strict_correct','alternative_correct','alternative_complete']:
   effects[m][metric]={}
   for name,coefs in specs.items():
    vals=[sum(w*g[m,sid,ph,c][metric] for (ph,c),w in coefs.items()) for sid in ids];main=metric=='semantic_correct' and name in main_names
    v=stats(vals,main);effects[m][metric][name]=v
    if main:primary[m+':'+name]=v
  # Pre-existing STOP strata remain descriptive, no outcome-based exclusion in main.
  effects[m]['pre_STOP_semantic_correct']={}
  for name,coefs in specs.items():
   selected=[sid for sid in ids if not any(g[m,sid,ph,'BASE']['any_stop_already_present'] for ph,c in coefs)]
   effects[m]['pre_STOP_semantic_correct'][name]=stats([sum(w*g[m,sid,ph,c]['semantic_correct'] for (ph,c),w in coefs.items()) for sid in selected])
 adj=holm({k:v['exact_two_sided_p'] for k,v in primary.items()});assert len(adj)==16
 for k,v in primary.items():v['holm_p']=adj[k]
 for m in ['llada','dream']:
  yes=lambda n:primary[m+':'+n]['difference']>0 and primary[m+':'+n]['holm_p']<.05
  decision[m]={'early_ALL_recovery':yes('b0_early:ALL_BOTH_minus_BASE'),'earlier_better_than_late':yes('timing:ALL_BOTH_early_minus_late'),'b0_MASK_recovery':yes('b0_early:MASK_BOTH_minus_BASE'),'b0_TEXT_recovery':yes('b0_early:TEXT_BOTH_minus_BASE'),'b0_MASK_over_TEXT':yes('b0_early:MASK_BOTH_minus_TEXT_BOTH'),'b1_MASK_recovery':yes('b1_clean_early:MASK_BOTH_minus_BASE'),'b1_TEXT_recovery':yes('b1_clean_early:TEXT_BOTH_minus_BASE'),'b1_MASK_over_TEXT':yes('b1_clean_early:MASK_BOTH_minus_TEXT_BOTH')}
 final=REPORT/'final';final.mkdir(exist_ok=True);out={'counts':p,'quality':quality,'primary':primary,'effects':effects,'completion_accuracy_transitions':transitions,'decision':decision,'label_manifest':{'auto':sha(REPORT/'review_auto.json'),'manual':sha(REPORT/'review_semantic.json')},'limits':['Reused32 development questions; not independent confirmation.','Regional restoration adds differing KV counts and extra reference compute; not equal-budget importance or speed.','b1 uses clean C100 prefix, not same upstream as b0; no crossphase timing-only claim.','Same layout late ALL rerun isolates timing within block0; old canonical-order C100 reference not assumed numerically identical.','MASK/TEXT restoration on own-branch state; factorial interactions not unique additive causal shares.','Assistant condition-hidden semantic review, not independent human evaluation.']}
 write(final/'summary.json',out);(final/'semantic_outputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in full));print(json.dumps({'quality':quality,'primary':primary,'decision':decision},ensure_ascii=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pending',type=int);a=p.parse_args();full,samples=load()
 if a.prepare or a.pending:
  print(json.dumps(prepare(full,samples)))
  if a.pending:
   done=read(REPORT/'review_semantic.json') if (REPORT/'review_semantic.json').exists() else {}
   for r in [x for x in read(REPORT/'review_pending.json') if x['review_id'] not in done][:a.pending]:print(json.dumps(r,ensure_ascii=False))
 else:analyze(full,samples)
if __name__=='__main__':main()
