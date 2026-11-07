"""Frozen method-hidden review and paired analysis of fixed layer controls."""
import argparse
import collections
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
TAG = 'k12_layer_validation_20261105'
REPORT, RESULT, DATA = [ROOT / folder / TAG for folder in ['reports', 'results', 'data']]
SEED = 2026110593


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    text = Path(path).read_text()
    lines = text.splitlines()
    if not text.endswith('\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def digest(row):
    return row['sample_id'] + '_' + hashlib.sha256(row['answer_before_stop'].encode()).hexdigest()[:14]


def load():
    paths = sorted(RESULT.glob('gpu*/gsm*/outputs.jsonl'))
    return [r for p in paths for r in rows(p)], {p.parent.name: read(p.parent / 'sample.json') for p in paths}


def prepare(full, samples):
    groups = collections.defaultdict(list)
    for row in full:
        if not row.get('numerical_failure'):
            groups[digest(row)].append(row)
    automatic, pending = {}, []
    for key, group in sorted(groups.items()):
        row = group[0]
        if row['parsed_answer'] is not None:
            automatic[key] = {'complete': True, 'correct': row['correct'], 'label_source': 'Explicit parseable Final answer, numeric target match.'}
        else:
            sample = samples[row['sample_id']]
            pending.append({'review_id': key, 'question': sample['question'], 'gold_answer': sample['answer'], 'answer': row['answer_before_stop']})
    random.Random(SEED).shuffle(pending)
    write(REPORT / 'review_auto.json', automatic)
    write(REPORT / 'review_pending.json', pending)
    manual = read(REPORT / 'review_semantic.json') if (REPORT / 'review_semantic.json').exists() else {}
    progress = {'outputs': len(full), 'complete_cases': len(list(RESULT.glob('gpu*/gsm*/completed.json'))),
                'unique_answers': len(groups), 'automatic_final': len(automatic), 'manual_required': len(pending),
                'manual_done': sum(r['review_id'] in manual for r in pending),
                'pending': sum(r['review_id'] not in manual for r in pending),
                'numerical_failures': sum(r.get('numerical_failure', False) for r in full)}
    write(REPORT / 'progress.json', progress)
    return progress


def paired(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    bootstrap = diff[np.random.default_rng(SEED).integers(0, n, (10000, n))].mean(1)
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    p = float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.
    return {'n': n, 'difference': float(diff.mean()), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'wins': wins, 'regressions': losses, 'ties': n - wins - losses, 'exact_two_sided_p': p}


def holm(values):
    ordered = sorted(values, key=values.get)
    result, running = {}, 0.
    for index, name in enumerate(ordered):
        running = max(running, min(1., values[name] * (len(ordered) - index)))
        result[name] = running
    return result


def median_summary(values):
    values = np.asarray(values, dtype=float)
    bootstrap = np.median(values[np.random.default_rng(SEED).integers(0, len(values), (10000, len(values)))], axis=1)
    return {'median': float(np.median(values)), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist()}


def analyze(full, samples):
    progress = prepare(full, samples)
    protocol = read(DATA / 'protocol.json')
    conditions, n = protocol['conditions'], protocol['data']['n']
    assert progress['outputs'] == n * len(conditions) and progress['complete_cases'] == n and not progress['pending'], progress
    assert all(code == 0 for code in read(REPORT / 'exit_codes.json').values())
    assert sha(__file__) == read(REPORT / 'analysis_frozen.json')['sha256']
    assert sha(DATA / 'protocol.json') == read(REPORT / 'protocol_frozen.json')['sha256']
    labels = {**read(REPORT / 'review_auto.json'), **read(REPORT / 'review_semantic.json')}
    by = collections.defaultdict(dict)
    for row in full:
        row['semantic'] = labels[digest(row)] if not row.get('numerical_failure') else {'complete': False, 'correct': False, 'label_source': 'Numerical failure'}
        assert row['condition'] not in by[row['sample_id']]
        by[row['sample_id']][row['condition']] = row
    ids = sorted(by)
    assert len(ids) == n and all(set(v) == {c['name'] for c in conditions} for v in by.values())
    assert all(len({r['gpu_uuid'] for r in v.values()}) == 1 for v in by.values())

    def vector(name, field='correct', alternative=False):
        return [by[sid][name]['semantic'].get(field + '_if_ambiguous', by[sid][name]['semantic'][field]) if alternative else by[sid][name]['semantic'][field] for sid in ids]

    quality, versus_dense, accuracy, completion, strict, alternative = {}, {}, {}, {}, {}, {}
    for condition in conditions:
        name = condition['name']
        records = [by[sid][name] for sid in ids]
        correct, complete = vector(name), vector(name, 'complete')
        quality[name] = {
            **condition, 'n': n, 'correct': sum(correct), 'accuracy': sum(correct) / n,
            'incomplete': n - sum(complete),
            'strict_correct': sum(r.get('correct', False) for r in records),
            'alternative_correct': sum(vector(name, alternative=True)),
            'alternative_incomplete': n - sum(vector(name, 'complete', alternative=True)),
            'ambiguous_outputs': sum(r['semantic'].get('ambiguous', False) for r in records),
            'incomplete_with_stop': sum(not r['semantic']['complete'] and r.get('first_stop_position') is not None for r in records),
            'incomplete_without_stop': sum(not r['semantic']['complete'] and r.get('no_stop_within_budget', False) for r in records),
            'numerical_failures': sum(r.get('numerical_failure', False) for r in records),
        }
        versus_dense[name] = paired(correct, vector('DENSE'))
        versus_dense[name]['empirical_within_2pp'] = versus_dense[name]['difference'] >= -.02 - 1e-12
    controls = protocol['analysis']['confirmatory_family']
    for name in controls:
        accuracy[name] = paired(vector('K12_NEXT'), vector(name))
        completion[name] = paired(vector('K12_NEXT', 'complete'), vector(name, 'complete'))
        strict[name] = paired([by[s]['K12_NEXT'].get('correct', False) for s in ids], [by[s][name].get('correct', False) for s in ids])
        alternative[name] = paired(vector('K12_NEXT', alternative=True), vector(name, alternative=True))
    for table in [accuracy, strict, alternative]:
        adjusted = holm({name: value['exact_two_sided_p'] for name, value in table.items()})
        for name, value in table.items():
            value['holm_p'] = adjusted[name]
            value['positive_supported'] = value['difference'] > 0 and value['ci95'][0] > 0 and adjusted[name] < .05
    timings = [r for path in RESULT.glob('gpu*/gsm*/timings.jsonl') for r in rows(path)]
    by_time = collections.defaultdict(list)
    for row in timings:
        by_time[(row['sample_id'], row['condition'])].append(row)
    time_summary = {}
    for condition in conditions:
        name = condition['name']
        valid = [sid for sid in ids if len(by_time[(sid, name)]) == 1]
        measurements = [by_time[(sid, name)][0] for sid in valid]
        if not measurements:
            continue
        info = {'n': len(valid), 'repeats': 1, 'median_seconds': float(np.median([r['wall_seconds'] for r in measurements])),
                'median_peak_allocated_bytes': float(np.median([r['peak_allocated_bytes'] for r in measurements])),
                'median_external_cache_bytes': float(np.median([by[sid][name]['external_cache_bytes'] for sid in valid])), 'paired_ratios': {}}
        for reference in ['S25', 'C100', 'DENSE']:
            pairs = [sid for sid in valid if len(by_time[(sid, reference)]) == 1]
            ratios = [by_time[(sid, name)][0]['wall_seconds'] / by_time[(sid, reference)][0]['wall_seconds'] for sid in pairs]
            if ratios:
                info['paired_ratios'][reference] = {'n': len(pairs), **median_summary(ratios)}
        time_summary[name] = info
    random_names = ['R12_1_NEXT', 'R12_2_NEXT', 'R12_3_NEXT']
    random_mean = np.mean(np.asarray([vector(name) for name in random_names], dtype=float), axis=0)
    random_diff = np.asarray(vector('K12_NEXT'), dtype=float) - random_mean
    random_boot = random_diff[np.random.default_rng(SEED).integers(0, n, (10000, n))].mean(1)
    out = {
        'counts': progress, 'quality': quality, 'paired_K12_accuracy': accuracy, 'paired_K12_completion': completion,
        'paired_vs_dense': versus_dense, 'strict_sensitivity': strict, 'ambiguity_sensitivity': alternative,
        'random_mean': {'accuracy': float(random_mean.mean()), 'K12_minus_mean': float(random_diff.mean()), 'ci95': np.quantile(random_boot, [.025, .975]).tolist(), 'note': 'Question bootstrap over the three frozen sets; not uncertainty over all possible layer sets.'},
        'timing': time_summary,
        'decision': {'primary_all_three_random12_supported': all(accuracy[n]['positive_supported'] for n in random_names), 'K12_vs_ALL_supported': accuracy['ALL_NEXT']['positive_supported'], 'K12_vs_ALL_direction_positive': accuracy['ALL_NEXT']['difference'] > 0,
                     'broad_control_superiority_supported': all(value['positive_supported'] for value in accuracy.values()),
                     'interpretation_rule': 'Primary requires superiority to all3 preselected random12 sets. Positive uncertain comparisons are inconclusive, not equivalence. No retuning or extra samples.'},
        'label_manifest': {'auto': sha(REPORT / 'review_auto.json'), 'manual': sha(REPORT / 'review_semantic.json')},
        'limitations': ['LLaDA and one locally fresh train subset only.', 'Single-pass exploratory timing.', 'Assistant semantic review, not independent human review.', 'ALL_NEXT is not official DPad; no official DPad or MaskKV benchmark.', 'No near-base equivalence, optimal12 or all-random-layer-set claim.'],
    }
    final = REPORT / 'final'
    final.mkdir(exist_ok=True)
    write(final / 'summary.json', out)
    (final / 'semantic_outputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in full))
    changed = []
    for sid in ids:
        for name in controls:
            a, b = by[sid]['K12_NEXT'], by[sid][name]
            if (a['semantic']['correct'], a['semantic']['complete']) != (b['semantic']['correct'], b['semantic']['complete']):
                changed.append({'sample_id': sid, 'question': samples[sid]['question'], 'gold': samples[sid]['answer'], 'control': name,
                                'K12': {'label': a['semantic'], 'answer': a.get('answer_before_stop')}, 'other': {'label': b['semantic'], 'answer': b.get('answer_before_stop')}})
    (final / 'changed_answer_pairs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in changed))
    with (final / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['condition', 'correct', 'n', 'accuracy', 'incomplete', 'strict_correct', 'median_seconds', 'external_cache_MiB'])
        writer.writeheader()
        for name, row in quality.items():
            timing = time_summary.get(name, {})
            writer.writerow({'condition': name, **{key: row[key] for key in ['correct', 'n', 'accuracy', 'incomplete', 'strict_correct']},
                             'median_seconds': timing.get('median_seconds'), 'external_cache_MiB': timing.get('median_external_cache_bytes', 0) / 2**20})
    print(json.dumps({'counts': progress, 'quality': quality, 'paired_K12_accuracy': accuracy, 'decision': out['decision']}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--pending', type=int)
    args = parser.parse_args()
    full, samples = load()
    if args.prepare or args.pending:
        print(json.dumps(prepare(full, samples)))
        if args.pending:
            done = read(REPORT / 'review_semantic.json') if (REPORT / 'review_semantic.json').exists() else {}
            grouped = collections.OrderedDict()
            for row in [r for r in read(REPORT / 'review_pending.json') if r['review_id'] not in done][:args.pending]:
                sid, answer_id = row['review_id'].rsplit('_', 1)
                group = grouped.setdefault(sid, {'sample': sid, 'question': row['question'], 'gold': row['gold_answer'], 'answers': []})
                group['answers'].append({'id': answer_id, 'text': row['answer']})
            for group in grouped.values():
                print(json.dumps(group, ensure_ascii=False))
        return
    analyze(full, samples)


if __name__ == '__main__':
    main()
