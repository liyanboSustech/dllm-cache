"""Correct a scorer implementation mismatch without changing frozen protocol.

The protocol says first numeric value WITHIN the final explicit answer field.
The initial runner mistakenly required that value to immediately follow the
label/formatting. Preserve every original result and write a separate scored copy.
"""
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


def parse_final_field(text):
    fields=re.findall(r'Final answer:\s*([^\n]*)',text,re.I)
    number=re.search(r'[-+]?\d[\d,]*(?:\.\d+)?',fields[-1]) if fields else None
    return number.group(0).replace(',','') if number else None


def rescore(path):
    path=Path(path)
    assert (path/'completed.json').exists()
    raw=(path/'results.jsonl').read_bytes()
    rows=[json.loads(s) for s in raw.decode().splitlines()]
    changes=[]
    for r in rows:
        if r['task']!='gsm8k':continue
        original={k:r[k] for k in ['parsed_answer','correct','em','missing_final_answer']}
        parsed=parse_final_field(r['answer_before_stop'])
        try:
            correct=parsed is not None and Decimal(parsed)==Decimal(r['gold_answer'].replace(',',''))
        except InvalidOperation:
            correct=False
        r.update({'parsed_answer':parsed,'correct':bool(correct),'em':float(bool(correct)),
                  'missing_final_answer':parsed is None,'original_scoring':original})
        if parsed!=original['parsed_answer'] or bool(correct)!=original['correct']:
            changes.append({'sample_id':r['sample_id'],'mode':r['mode'],'before':original,
                            'after':{k:r[k] for k in original}})
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    destination=path/'results_rescored.jsonl'
    if destination.exists():
        assert destination.read_text()==payload,'Refusing to change an existing rescoring'
    else:
        destination.write_text(payload)
    record={'type':'implementation correction under the already frozen protocol; no generation rerun or sample change',
        'reason':'Protocol specifies first numeric value within Final answer field; initial implementation used re.match instead of re.search within that field.',
        'raw_results_sha256':hashlib.sha256(raw).hexdigest(),'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'guard':'No extraction fallback outside an explicit Final answer: field. Correct numbers in reasoning alone do not pass.',
        'changes':changes}
    (path/'scoring_correction.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return rows


if __name__=='__main__':
    import sys
    rescore(sys.argv[1])
