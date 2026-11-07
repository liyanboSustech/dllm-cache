"""Shared, frozen model/scoring helpers for discovery pilot."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from run_baseline_smoke import (Cache100, CustomCache, LLaDAConfig, LLaDAModelLM,
                                emit, generate, gpu_snapshot, wait_idle)

DATA = ROOT / 'data/discovery_20261103'
MODEL = '/dataset/money-shredder-group/tv/LLaDA-8B-Instruct'
MODES = ['D', 'C100', 'S50', 'S25']
spec = importlib.util.spec_from_file_location('hotpot_eval', DATA/'source/hotpot_evaluate_v1.py')
hotpot = importlib.util.module_from_spec(spec)
# Official evaluator uses ujson only for its CLI file I/O. Keep all scoring
# functions unchanged and use the standard JSON module for that import.
evaluator_source = (DATA/'source/hotpot_evaluate_v1.py').read_text()
exec(compile(evaluator_source.replace('import ujson as json', 'import json', 1),
             str(DATA/'source/hotpot_evaluate_v1.py'), 'exec'), hotpot.__dict__)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush()


def write_json(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_model():
    torch.set_num_threads(4)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    cfg = LLaDAConfig.from_pretrained(MODEL, local_files_only=True)
    cfg.block_len = 32
    cfg.flash_attention = False
    cfg.use_cache = False
    model, info = LLaDAModelLM.from_pretrained(MODEL, config=cfg, torch_dtype=torch.float16,
        device_map={'':'cuda:0'}, local_files_only=True, output_loading_info=True)
    assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model.eval()
    assert all(p.device.type == 'cuda' and p.dtype == torch.float16 for p in model.parameters())
    return model, tokenizer, info


def score(tokenizer, sample, ids):
    stop_ids = {126081,126348}
    positions = [i for i,t in enumerate(ids) if t in stop_ids]
    end = min(positions) if positions else len(ids)
    text = tokenizer.decode(ids[:end], skip_special_tokens=True).strip()
    out = {'raw_output':tokenizer.decode(ids, skip_special_tokens=False),
        'answer_before_stop':text, 'first_stop_position':positions[0] if positions else None,
        'no_stop_within_budget':not positions, 'remaining_mask_tokens':ids.count(126336),
        'generated_token_ids':ids, 'gold_answer':sample['answer']}
    grams = list(zip(*[text.lower().split()[i:] for i in range(4)]))
    out['repeated_4gram_fraction'] = (len(grams)-len(set(grams)))/max(1,len(grams))
    if sample['task'] == 'gsm8k':
        fields = re.findall(r'Final answer:\s*([^\n]*)', text, re.I)
        number = re.match(r'[\s*$]*([-+]?(?:\d[\d,]*)(?:\.\d+)?)', fields[-1]) if fields else None
        parsed = number.group(1).replace(',','') if number else None
        try:
            correct = parsed is not None and Decimal(parsed) == Decimal(sample['answer'].replace(',',''))
        except InvalidOperation:
            correct = False
        out.update({'parsed_answer':parsed,'correct':bool(correct),'em':float(bool(correct)),
                    'f1':None,'missing_final_answer':parsed is None})
    else:
        em = hotpot.exact_match_score(text,sample['answer'])
        f1, precision, recall = hotpot.f1_score(text,sample['answer'])
        out.update({'parsed_answer':text,'correct':bool(em),'em':float(em),'f1':f1,
                    'precision':precision,'recall':recall,'missing_final_answer':not bool(text)})
    return out


class Monitor:
    def __init__(self, uuid, path):
        self.uuid, self.path = uuid, path
        initial = gpu_snapshot()
        self.own = {x.split(',')[1].strip() for x in initial['processes'].splitlines() if uuid in x}
        assert len(self.own) == 1, f'Unexpected GPU process list: {initial}'
        self.stop = threading.Event()
        self.interference = False
        self.error = None
        self.thread = threading.Thread(target=self.loop,daemon=True)
        self.thread.start()

    def loop(self):
        while not self.stop.is_set():
            try:
                snap = gpu_snapshot()
                pids = {x.split(',')[1].strip() for x in snap['processes'].splitlines() if self.uuid in x}
                if pids-self.own:
                    self.interference = True
                append(self.path,snap)
            except Exception as exc:
                self.error = repr(exc)
            self.stop.wait(5)

    def check(self):
        if self.interference or self.error:
            raise RuntimeError(f'GPU interference/monitor error; timing invalid: {self.error}')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)


def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
