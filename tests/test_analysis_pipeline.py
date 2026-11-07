"""Synthetic artifacts exercise audit -> blinded queue -> frozen stopping rule.
All artifacts are temporary and are never scientific results.
"""
import contextlib,io,tempfile,unittest
from pathlib import Path
import numpy as np
from portable.common import ROOT,TAG,read,write,rows,append,sha
from portable.runner import freeze_run,analyze
from portable.audit import audit

class PipelineTest(unittest.TestCase):
 def test_complete_artifacts_reach_failed_gate_without_gpu_or_manual_labels(self):
  with tempfile.TemporaryDirectory() as tmp:
   run=freeze_run(Path(tmp)/'synthetic',{'llada':'/synthetic','dream':'/synthetic'},[4,5,6,7],{'synthetic':True})
   p=read(run/'data/protocol.json');manifest=read(run/'source_manifest.json')
   write(run/'reports/exit_codes.json',{str(g):0 for g in [4,5,6,7]})
   for gpu,model,shard in [(4,'llada',0),(5,'llada',1),(6,'dream',0),(7,'dream',1)]:
    out=run/'results'/model/f'gpu{gpu}';out.mkdir(parents=True)
    uuid=f'GPU-synthetic-{gpu}';nl=32 if model=='llada' else 28;heads=32 if model=='llada' else 4;mask=126336 if model=='llada' else 151666
    write(out/'environment.json',{'gpu_uuid':uuid,'sources':manifest,'protocol_sha256':sha(run/'data/protocol.json')})
    write(out/'completed.json',{'samples':16,'outputs':80});write(out/'warmup.json',{'S25':[1]*256,'K4':[1]*256})
    write(run/'reports'/f'preflight_gpu{gpu}.json',{'S25_same_device_official_exact':True,'K4_same_device_checked':True})
    append(out/'gpu_monitor.jsonl',{'processes':f'{uuid}, 1234, synthetic, 0'})
    spec=p['models'][model]
    for sample in rows(run/'data'/f'{model}_samples.jsonl')[shard::2]:
     case=out/sample['sample_id'];case.mkdir();write(case/'sample.json',sample)
     plen=sample['prompt_length'];ext=plen+224;ret=int(.25*ext);state=sample['prompt_ids']+[mask]*256;commits=[];anchor=None
     conds={'S25':[],'K4':spec['K4'],**{f'R{i+1}':ls for i,ls in enumerate(spec['random_four'])},**{f'L{i:02d}':[i] for i in range(nl)},**{f'K4_MINUS_{li:02d}':[i for i in spec['K4'] if i!=li] for li in spec['K4']}}
     for block in range(8):
      start=plen+block*32
      for step in range(32):
       if step==1:anchor=state.copy()
       if step==2 and block<7:
        active=[i for i in range(32) if state[start+i]==mask];plan={'base':list(range(ret)),'next':list(range(ret)),'next_available':32,'next_retained':sum(start<=i<start+32 for i in range(ret)),'qk_nonfinite':0}
        cp={'block':block,'step':2,'start':start,'x_ids':state.copy(),'anchor_input_ids':anchor,'active_positions':active,'prior_stop_present':False,'native_cache_exact':True,'anchor_values_unchanged':True,'external_absolute_positions':list(range(start))+list(range(start+32,plen+256)),'plans':{str(li):plan for li in range(nl)},'rows':[{'condition':c,'layers':ls,'js_per_position':[.05]*len(active),'js':.05,'gain':0.,'external_counts':[ret]*nl,'kv_bytes':ret*nl*2*heads*128*2} for c,ls in conds.items()]}
        append(case/'probes.jsonl',cp)
       active=[start+i for i in range(32) if state[start+i]==mask]
       if model=='llada' or step==0:n=1
       elif step==31:n=len(active)
       else:
        times=np.linspace(1,.001,33,dtype=np.float32);n=int(np.float32(len(active))*np.float32(1-times[step+1]/times[step]))
       positions=active[:n];commits.append({'block':block,'step':step,'positions':positions,'tokens':[1]*n,'confidence':[1.]*n})
       for pos in positions:state[pos]=1
     write(case/'scores_frozen.json',{'probes_sha256':sha(case/'probes.jsonl'),'historical_V100_S25_exact':False})
     write(case/'completed.json',{'outputs':5,'checkpoints':7})
     for c in ['S25','K4','R1','R2','R3']:
      append(case/'outputs.jsonl',{'condition':c,'model':model,'sample_id':sample['sample_id'],'gpu_uuid':uuid,'generated_token_ids':[1]*256,'remaining_mask_tokens':0,'commits':commits,'answer_before_stop':'SYNTHETIC TEST FIXTURE ONLY. Final answer: 0','parsed_answer':'0','correct':False})
   report=audit(run);self.assertTrue(report['passed']);self.assertEqual(report['counts']['outputs'],320)
   review=analyze(run,prepare=True);self.assertEqual(review['pending'],0)
   with contextlib.redirect_stdout(io.StringIO()):decision=analyze(run)
   self.assertFalse(any(d['supported'] for d in decision.values()))
   self.assertTrue(read(run/'status.json')['no_next_stage_started'])
   # Corrupt a retained budget while preserving JSON validity: audit must fail.
   file=next((run/'results').glob('llada/gpu*/gsm*/probes.jsonl'));data=rows(file);data[0]['rows'][0]['external_counts'][0]+=1
   file.write_text(''.join(__import__('json').dumps(r)+'\n' for r in data))
   write(file.parent/'scores_frozen.json',{'probes_sha256':sha(file),'historical_V100_S25_exact':False})
   with self.assertRaises(AssertionError):audit(run)

if __name__=='__main__':unittest.main()
