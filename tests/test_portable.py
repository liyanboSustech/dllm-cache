import json,tempfile,unittest
from pathlib import Path
from portable.common import ROOT,parse_gpus,gpu_ready,verify_export,write,sha,code_path

class PortableTests(unittest.TestCase):
 def test_four_unique_gpus(self):
  self.assertEqual(parse_gpus('4,5,6,7'),[4,5,6,7])
  for s in ['0,0,1,2','0,1','-1,0,1,2','0,x,2,3']:
   with self.assertRaises(ValueError):parse_gpus(s)
 def test_memory_and_process_not_utilization_alone(self):
  s={'devices':[{'index':i,'uuid':f'GPU-{i}','total_mib':32768,'used_mib':0,'utilization':0} for i in range(8)],'processes':''}
  self.assertTrue(gpu_ready(s,[4,5,6,7]));s['processes']='GPU-5, 1234, another_job, 32'
  self.assertFalse(gpu_ready(s,[4,5,6,7]));s['processes']='';s['devices'][4]['used_mib']=4834
  self.assertFalse(gpu_ready(s,[4,5,6,7]))
 def test_immutable_export_and_subjects(self):
  self.assertGreater(verify_export(),60)
  data=ROOT/'data/k4_correction_20261107'
  protocol=json.loads((data/'protocol.json').read_text())
  for model,spec in protocol['models'].items():
   p=data/f'{model}_samples.jsonl';self.assertEqual(sha(p),spec['sha256'])
   rows=[json.loads(x) for x in p.read_text().splitlines()]
   self.assertEqual({r['source_row_idx'] for r in rows},set(range(49,81)))
 def test_score_aggregation_keeps_seven_equal_blocks(self):
  code_path();import analyze_k4_correction as a
  old=a.RESULT
  try:
   with tempfile.TemporaryDirectory() as d:
    a.RESULT=Path(d);case=Path(d)/'llada/gpu6/gsm8k_train_0049';case.mkdir(parents=True)
    cp=[{'block':i,'prior_stop_present':i>0,'plans':{},'rows':[{'condition':'K4','gain':float(i)}]} for i in range(7)]
    p=case/'probes.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in cp));write(case/'scores_frozen.json',{'probes_sha256':sha(p)})
    subjects,detail,_,stops=a.score_table()
    self.assertEqual(subjects['llada','gsm8k_train_0049']['K4'],3.)
    self.assertEqual(stops['llada'],6)
  finally:a.RESULT=old

if __name__=='__main__':unittest.main()
