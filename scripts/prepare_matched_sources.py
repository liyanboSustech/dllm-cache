import json,random,hashlib,re,unicodedata
from datetime import datetime,timezone
from transformers import AutoTokenizer
from discovery_common import ROOT,sha,write_json,read_jsonl
from matched_sources_core import TAG,SEEDS,conditions
D=ROOT/'data'/TAG;P=ROOT/'reports'/TAG;D.mkdir();P.mkdir()
def norm(q):return ' '.join(unicodedata.normalize('NFKC',q).split()).casefold()
used=set(range(81));questions=set();old={}
def visit(v):
 if isinstance(v,dict):
  sid=str(v.get('sample_id',''));match=re.fullmatch(r'gsm8k_train_(\d+)',sid)
  if match:used.add(int(match[1]))
  if match or v.get('task')=='gsm8k':
   if 'source_row_idx' in v:used.add(int(v['source_row_idx']))
   if 'question' in v:questions.add(norm(v['question']))
  for x in v.values():visit(x)
 elif isinstance(v,list):
  for x in v:visit(x)
for path in sorted((ROOT/'data').rglob('*.jsonl')):
 if D in path.parents:continue
 old[str(path.relative_to(ROOT))]=sha(path)
 for line in path.read_text().splitlines():
  if line.strip():visit(json.loads(line))
pool={};sources={}
for path in sorted((ROOT/'data/kv_generalization_20261104/source').glob('gsm8k_rows*.json')):
 sources[str(path.relative_to(ROOT))]=sha(path)
 for r in json.loads(path.read_text())['rows']:
  assert not r['truncated_cells'];pool[r['row_idx']]=r['row']
eligible=[];seen=set(questions)
for i,r in sorted(pool.items()):
 q=norm(r['question'])
 if i not in used and q not in seen:eligible.append(i);seen.add(q)
random.Random(2026110630).shuffle(eligible);chosen=eligible[:128];assert len(chosen)==128,(len(eligible),len(used))
write_json(D/'exclusions.json',{'old_files':old,'source_files':sources,'excluded_row_ids':sorted(used),'excluded_normalized_question_hashes':sorted(hashlib.sha256(q.encode()).hexdigest() for q in questions),'eligible_count':len(eligible),'chosen_rows':chosen,'reserved160_plus64_untouched':True,'scope':'All locally allocated data jsonl recursively; legacy sealed remote holdouts absent here, no global deduplication claim.'})
models={}
for model,path in [('llada','LLaDA-8B-Instruct'),('dream','Dream-v0-Instruct-7B')]:
 tok=AutoTokenizer.from_pretrained('/dataset/money-shredder-group/tv/'+path,trust_remote_code=True,local_files_only=True)
 samples=[{**s,'cohort':'development'} for s in read_jsonl(ROOT/f'data/early_sources_20261106/{model}_samples.jsonl')]
 for j,i in enumerate(chosen):
  r=pool[i];prompt=r['question']+'\n\nGive a concise calculation. End with a separate line: Final answer: <number>.'
  rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True);ids=tok(rendered)['input_ids']
  samples.append({'sample_id':f'gsm8k_train_{i:04d}','source_row_idx':i,'task':'gsm8k','cohort':'confirmation','phase_index':j,'question':r['question'],'answer':r['answer'].split('####')[-1].strip(),'prompt':prompt,'rendered_prompt':rendered,'prompt_ids':ids,'prompt_length':len(ids)})
 dest=D/f'{model}_samples.jsonl';dest.write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in samples));models[model]={'samples':str(dest.relative_to(ROOT)),'sha256':sha(dest),'n':160}
p={'tag':TAG,'frozen_utc':datetime.now(timezone.utc).isoformat(),'authorization':'User requests same-count same-layer MASK/text restoration then independent samples, both models and four GPUs.','models':models,'conditions':{c:conditions(c) for c in ['development','confirmation']},'counts':{'development_questions':32,'confirmation_questions':128,'models':2,'outputs':1536,'native_S25_captures':320},'selection':{'seed':2026110630,'exclusions_sha256':sha(D/'exclusions.json'),'subset_seeds':SEEDS,'rule':'Per layer k=min(32,missing_MASK,missing_TEXT). Uniform subset without replacement, independent deterministic SHA seed per question/model/layer/region/draw. Original S25 slots unchanged; append chosen restored original slots sorted by absolute position. All layers used, no score or layer tuning. Seed0 primary fixed before all outputs; seeds1/2 development sensitivity only.'},'intervention':'First block step2 immediately before first sparse read. Same original step1 KV age/RoPE/K/V; joint actual forward affects content and positions. Intervention to block0 end only, then nativeS25. TEXT is prompt here; no external generated history. No displacement; matched augmented budgets betweenMASK/TEXT, not same as S25. ALL all deletedKV restored is unmatched positivecontrol. No new deployment method or speed claim.','generation':{'G':256,'T':256,'block':32,'temperature':0,'seed':2025,'dtype':'float16','attention':'EFFICIENT_ATTENTION','sampler':'unchanged per-model native transfer and Dream alignment'},'analysis':{'primary_family':['llada:MASK_0-TEXT_0','llada:MASK_0-BASE','dream:MASK_0-TEXT_0','dream:MASK_0-BASE'],'test':'Independent128 only: two-sided exact paired McNemar, Holm4; 10000 paired question bootstrap95%CI seed2026110634. Development32 separate, no pooling or bestseed. Semantic target correctness primary; complete wrong and incomplete separate. StrictFinal and ambiguity sensitivity.','gate':'Per model support requires positive accuracy and Holm p<.05 for BOTH MASK-TEXT and MASK-BASE; crossmodel claim requires both models. Nonsignificance is inconclusive, not equivalence. Report wins/losses, bounds and ALL positivecontrol. No retuning or extra samples after results.','review':'Exact question+answer inherited labels, remaining unique answers condition/model-hidden assistant semantic review, not independent human.'},'hardware':'GPUs0/1 LLaDA,2/3 Dream, samequestion sameUUID. Preflight barrier and development cohort barrier. Independent cohort runs regardless development results; all rules frozen upfront.'}
write_json(D/'protocol.json',p);write_json(P/'protocol_frozen.json',{'sha256':sha(D/'protocol.json')});print({'counts':p['counts'],'eligible':len(eligible),'excluded':len(used)})
