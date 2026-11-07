import json,random
from datetime import datetime,timezone
from discovery_common import ROOT,sha,write_json,read_jsonl
from k4_correction_core import TAG,LAYERS
D=ROOT/'data'/TAG;P=ROOT/'reports'/TAG
D.mkdir(exist_ok=False);P.mkdir(exist_ok=False)
rng=random.Random(2026110701)
randoms={'llada':[[10,11,15,30],[9,20,27,30],[3,4,14,19]],'dream':[sorted(rng.sample(range(28),4)) for _ in range(3)]}
models={}
for name in LAYERS:
 src=ROOT/f'data/early_sources_20261106/{name}_samples.jsonl';ss=read_jsonl(src)
 assert len(ss)==32 and {s['source_row_idx'] for s in ss}==set(range(49,81))
 dst=D/f'{name}_samples.jsonl';dst.write_bytes(src.read_bytes())
 models[name]={'samples':str(dst.relative_to(ROOT)),'sha256':sha(dst),'source':str(src.relative_to(ROOT)),
               'source_sha256':sha(src),'K4':LAYERS[name],'random_four':randoms[name]}
p={'tag':TAG,'frozen_utc':datetime.now(timezone.utc).isoformat(),'authorization':'Only step1: explain native K4 versus random four via fixed same-budget JS correction, both models four GPUs. Stop if unsupported; no metric retuning, selector or fresh-sample phase.',
 'models':models,'samples':'32 previous development questions rows49..80, no outcome filtering; no independent confirmation claim; all reserved sets untouched.',
 'signal':'G(C)=mean_active JS(P_full_same_age,P_S25)-mean_active JS(P_full_same_age,P_C). Full vocabulary FP32 log_softmax; natural log; temperature1; no clipping/top-k/position/EOS weighting. Each subject averages all seven blocks equally, positions equally within block.',
 'checkpoints':'On common native S25 trajectory, blocks0..6 step2 BEFORE first sparse read, own original block step1 K/V and unchanged RoPE. All branches share input; no writes. Include all seven blocks regardless STOP, record prior STOP; pre-STOP subset descriptive only, never replaces primary.',
 'intervention':'Exact frozen PriorityCache NEXT rule incl native FP16 QK and topk order/ties; floor25% external KV EACH layer. Changed layers prioritize next32 then native QK remainder; all others native S25. Log native nonfinite QK and actual next coverage; do not silently change selection to FP32. Full same-age reference has100% external and is diagnostic only, not uncached base.',
 'conditions':'Per checkpoint: FULL,S25,each single NEXT layer,K4,K4-minus-each-of-four,three frozen random4. LLaDA K4 validated historical set; Dream previously depth-mapped set, effectiveness not assumed. No greedy selection.',
 'counts':{'questions_per_model':32,'models':2,'checkpoints':448,'diagnostic_forwards_excluding_null':17920,'extra_native_null_forwards':448,'full_generations':320},
 'generation':'After all four shards finish all diagnostic scores, run full native K4 and three random4 generations; S25 already captured. Five full trajectories per question/model, no single-layer or drop-one complete continuation. G=T256,block32,temp0,seed2025,FP16 EFFICIENT_ATTENTION; native Dream shift and transfer counts.',
 'analysis':{'unit':'question, never layer/block as independent replicate','bootstrap':10000,'permutation_draws':100000,'seed':2026110702,
  'primary_family':[m+':'+c for m in LAYERS for c in ['K4_gain_positive','K4_minus_random_mean','mean_K4_leave_one_out_positive']],
  'test':'Per-subject seven-block averages. Two-sided paired sign-flip Monte Carlo +1 correction; Holm across6. Question bootstrap95% intervals. Report each of3 random contrasts and all layers descriptively, no best random or layer choice.',
  'mechanism_gate':'Per model all3 primary effects positive with Holm p<.05. Report rank of each original K4 layer and each conditional contribution, no requirement they are globally optimal. Both models required for crossmodel support.',
  'outcome_link':'Semantic final correctness reviewed condition/model-hidden. Across K4 and3random4 within each question, AUC ranks correct above incorrect using mean seven-block G; ties .5. Need >=12 mixed questions, AUC>=.60 and question-bootstrap lower95%>.5, plus positive K4 accuracy minus mean random accuracy. Else signal has not explained task advantage, including insufficient evidence. StrictFinal and ambiguity alternate descriptive.',
  'stop':'Any failed support gate means no automatic next stage, no extra samples or altered score/weights/checkpoints. Positive JS alone is not task correctness or full mechanistic explanation.',
  'review':'Existing exact sample+answer labels reused with provenance; new answers model/condition-hidden assistant review, not independent human.'},
 'hardware':'GPUs0/1 LLaDA;2/3Dream. Samequestion sameUUID, native warmup exact old outputs, sourcehashes, actual counts/values/cache age, monitoring foreignprocesses. Wait for idle, never kill foreign jobs. All timing diagnostic only.'}
write_json(D/'protocol.json',p);write_json(P/'protocol_frozen.json',{'sha256':sha(D/'protocol.json')})
print(json.dumps({'random_four':randoms,'protocol_sha256':sha(D/'protocol.json')},ensure_ascii=False))
