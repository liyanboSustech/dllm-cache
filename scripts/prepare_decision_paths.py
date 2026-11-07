import json,shutil
from datetime import datetime,timezone
from discovery_common import ROOT,sha,write_json,read_jsonl
TAG='decision_paths_20261106'
def main():
 d=ROOT/'data'/TAG;r=ROOT/'reports'/TAG;d.mkdir();r.mkdir();models={}
 for model in ['llada','dream']:
  src=ROOT/f'data/kv_history_phase_20261104/{model}_samples.jsonl';dst=d/f'{model}_samples.jsonl';shutil.copyfile(src,dst)
  models[model]={'samples':str(dst.relative_to(ROOT)),'sha256':sha(dst),'source':str(src.relative_to(ROOT)),'n':32}
 assert [s['sample_id'] for s in read_jsonl(d/'llada_samples.jsonl')]==[s['sample_id'] for s in read_jsonl(d/'dream_samples.jsonl')]
 p={'tag':TAG,'frozen_utc':datetime.now(timezone.utc).isoformat(),'authorization':'User explicitly requests BASE/order-only/content-only/both mechanism comparison on LLaDA and Dream, four GPUs.',
 'models':models,'sample_role':'Reuse exact32 paired prior development questions, no outcome filtering, original modulo2 GPU assignments. Old reserved160+64 and sealed holdouts untouched.',
 'conditions':['BASE','ORDER','CONTENT','BOTH'],'phases':['b0_r04','b1_r04'],'reference_trajectory':'Native S25; first pre-commit cache-active state with exactly4 revealed positions in block0 or1. Include all states, record existing STOP.',
 'primary_phase':'b0_r04','secondary_phase':'b1_r04','reference':'C100 all external KV from exact same native step1 capture, frozen age and RoPE. Evaluate sparse and full on the CURRENT STATE OF EACH BRANCH, every intervention step. No gold token or future trajectory used.',
 'decision_sources':{'BASE':['sparse positions','sparse words at those positions'],'ORDER':['full positions','sparse words at those positions'],'CONTENT':['sparse positions','full words at those positions'],'BOTH':['full positions','full words at those positions']},
 'scope':'Only remaining steps of fork block. Every subsequent block uses native S25. Original native transfer count, sampler/tie semantics,256step budget,no EOS suppression, no early exit.',
 'interpretation':'Full reference restores all pruned external KV, including prompt/history/MASK. This diagnoses overall cache-compression decision paths, NOT MASK-specific causality by itself. Full oracle reference changes auxiliary memory/compute; not a deployable equal-budget/speed method. No refresh, trained planner, selector, or retired residual/EOS gate.',
 'generation':{'G':256,'T':256,'block':32,'temperature':0,'seed':2025,'dtype':'float16','attention':'efficient SDPA','llada':'native confidence argmax','dream':'native maskgit_plus eps=.001 with official logit alignment and transfer schedule'},
 'counts':{'models':2,'questions_per_model':32,'phases':2,'conditions':4,'continuations':512,'baseline_captures':64},
 'analysis':{'seed':2026110611,'primary_metric':'Semantic task accuracy; completion secondary; strictFinal and ambiguity sensitivities.',
 'primary_comparisons':[['ORDER','BASE'],['CONTENT','BASE'],['BOTH','BASE'],['ORDER','CONTENT']],
 'primary_family':'Eight paired accuracy contrasts at b0_r04: four per model; exact two-sided McNemar + Holm8;10000 question-bootstrap95%CI. No comparison may treat two phases as independent questions.',
 'secondary':'All same comparisons at b1_r04, question-mean pooled2phases, pre-STOP subsets, completion/strict/ambiguity; descriptive clustered CIs. Factorial interaction BOTH-ORDER-CONTENT+BASE and recovery contrast (ORDER-BASE)-.5*(BOTH-BASE), not additive causal shares.',
 'decision':'Order contribution supported only if ORDER>BASE with Holm8 p<.05 at primary phase. Order predominance requires additionally ORDER>CONTENT with corrected p<.05. BOTH>BASE checks whether full-reference correction helps at all. Failure with32 development questions is inconclusive, no automatic extra samples or score design.',
 'review':'Method/model-hidden assistant semantic review, exact-question exact-text prior labels inherited; not independent human evaluation. Keep original gold, separately document errors.'},
 'validation':['CPU source selection and tie/empty/multi-position cases','all four arms null when reference=sparse','BASE equals native full256tokens at each phase','BOTH equals independent full-reference continuation on warmup','current-state and cache-age invariants','original S25 baseline exact sameGPU historic replay','two-GPU per-model warmups identical','source hashes and foreign-process monitoring'],
 'hardware':'GPU0/1 LLaDA, GPU2/3 Dream, shard0/1. All arms/phases of one question/model on one physicalGPU; original assignments retained.'}
 write_json(d/'protocol.json',p);write_json(r/'protocol_frozen.json',{'sha256':sha(d/'protocol.json')});print(json.dumps(p['counts']))
if __name__=='__main__':main()
