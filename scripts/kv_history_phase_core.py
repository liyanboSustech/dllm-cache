"""Prompt/history factorials on matched reveal schedules and cache ages.

The old experiment modules are imported without modification. These interventions
are diagnostics, not online cache policies or speed measurements.
"""
from contextlib import contextmanager
import math
import torch
import torch.nn.functional as F

from kv_generalization_core import Backend, cache_new, context_forward, local_metrics
from kv_generalization_core import cache_from_sources
import kv_factor_core as factor

BLOCKS = (0, 1)
REVEALED = (4, 12, 20, 28)
LOCAL_CONDITIONS = ['KV_NEXT', 'RR_CURRENT', 'S25', 'C100', 'K_NEXT', 'V_NEXT',
                    'KV_ALL', 'K_ALL', 'V_ALL'] + [
    f'{donor}_{age}_{source}_{component}'
    for donor in ['A', 'B'] for age in ['ANCHOR', 'CURRENT']
    for source in ['P', 'H', 'PH'] for component in ['K', 'V', 'KV']]
FULL_CONDITIONS = ['C100', 'S25', 'KV_NEXT', 'A_ANCHOR_P_KV', 'A_ANCHOR_H_KV',
                   'A_ANCHOR_PH_KV', 'RR_CURRENT', 'A_CURRENT_P_KV',
                   'A_CURRENT_H_KV', 'A_CURRENT_PH_KV']


def phase_key(block, revealed):
    return f'b{block}_r{revealed:02d}'


def mix_input(recipient, donor, plen, source):
    """Only token identities change; the reveal mask and lengths are fixed."""
    assert recipient.shape == donor.shape
    x = recipient.clone()
    if source in ['P', 'PH']:
        x[:, :plen] = donor[:, :plen]
    if source in ['H', 'PH']:
        x[:, plen:] = donor[:, plen:]
    assert source in ['RR', 'P', 'H', 'PH']
    return x


def native_forced_tokens(b, logits, start, full, positions):
    z = b.aligned_logits(logits, start, full)
    if not positions:
        return []
    z = z[0, torch.tensor(positions, device=z.device) - start]
    if b.name == 'dream':
        from vendor.sparse_dream.generation_utils import sample_tokens
        _, tokens = sample_tokens(z, **b.sampler_options)
    else:
        tokens = z.argmax(-1)
    return tokens.tolist()


@torch.inference_mode()
def trajectory(b, sample, policy, forced=None, check=None):
    """Native recipient, or donor with recipient's exact commit positions.

    Forced donors predict their own tokens online. No recipient/donor answer or
    future token is copied into donor history. Their reveal order is constrained.
    """
    assert policy in ['S25', 'C100']
    plen = sample['prompt_length']
    x = torch.full((1, plen + 256), b.mask, dtype=torch.long, device=b.device)
    x[:, :plen] = torch.tensor([sample['prompt_ids']], device=b.device)
    checkpoints, commits = {}, []
    first_stop = None
    finite = torch.ones((), device=b.device, dtype=torch.bool)
    for block in range(8):
        if check:
            check()
        start = plen + block * 32
        official = cache_new(b)
        anchor = anchor_input = None
        for step in range(32):
            if step == 0:
                logits = b.forward(x, start, 0, official)
            elif step == 1:
                anchor_input = x.clone()
                anchor = {}
                logits = b.forward(x, start, 1, official, anchor)
                for li, e in anchor.items():
                    for n in ['k', 'v']:
                        assert torch.equal(official.cache[li][n], e[n].index_select(-2, e['selection']))
                cache = cache_from_sources(b, anchor, {}, policy)
            else:
                logits = b.forward(x[:, start:start+32], start, 2, cache)
            finite &= torch.isfinite(logits).all()
            if forced is None:
                pos, tok, conf = b.choice(logits, x, start, step, step < 2)
            else:
                row = forced[block * 32 + step]
                assert (row['block'], row['step']) == (block, step)
                pos = row['positions']
                assert all(start <= p < start + 32 and int(x[0, p]) == b.mask for p in pos)
                tok = native_forced_tokens(b, logits, start, step < 2, pos)
                conf = []
            assert b.mask not in tok, 'Native donor emitted MASK; do not silently repair it.'
            row = {'block': block, 'step': step, 'positions': pos, 'tokens': tok, 'confidence': conf}
            if block in BLOCKS and step >= 2:
                count = int((x[:, start:start+32] != b.mask).sum())
                for target in REVEALED:
                    key = phase_key(block, target)
                    if count >= target and key not in checkpoints:
                        masked = torch.where(x[0, start:start+32] == b.mask)[0]
                        checkpoints[key] = {
                            'x': x.clone(), 'anchor': anchor, 'anchor_input': anchor_input,
                            'start': start, 'step': step, 'block': block, 'target_revealed': target,
                            'actual_revealed': count, 'event_position': pos[0] if pos else start + int(masked[0]),
                            'native': row, 'prior_stop_commit': first_stop,
                            'stop_in_previous_blocks': any(t in b.stops for t in x[0, plen:start].tolist()),
                            'any_stop_already_present': any(t in b.stops for t in x[0, plen:].tolist()),
                        }
            if first_stop is None and any(t in b.stops for t in tok):
                first_stop = {'block': block, 'step': step, 'position': min(p for p, t in zip(pos, tok) if t in b.stops)}
            commits.append(row)
            if pos:
                x[0, pos] = torch.tensor(tok, device=b.device)
            del logits
        del cache, official, anchor, anchor_input
    assert finite.item(), 'Nonfinite trajectory logits'
    ids = x[0, plen:].tolist()
    assert b.mask not in ids
    assert set(checkpoints) == {phase_key(bl, r) for bl in BLOCKS for r in REVEALED}
    return {'ids': ids, 'commits': commits, 'checkpoints': checkpoints, 'first_stop_commit': first_stop}


def source_statistics(reference, altered, indices):
    stats = {}
    for name in ['k', 'v']:
        a = reference[name].index_select(-2, indices).float()
        z = altered[name].index_select(-2, indices).float()
        stats[name] = {
            'cosine_mean': float(F.cosine_similarity(a, z, dim=-1).mean()),
            'norm_ratio_mean': float((z.norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)).mean()),
            'relative_l2': float((z-a).norm() / a.norm().clamp_min(1e-12)),
        }
    return stats


@torch.inference_mode()
def make_banks(b, sample, cp, donor_cps, age, anchor_bank=None):
    plen, start = sample['prompt_length'], cp['start']
    key = 'anchor_input' if age == 'ANCHOR' else 'x'
    rx = cp[key]
    mapping = factor.plans(b.model, cp['anchor'], start, start+32, rx.shape[1], b.layers)
    if age == 'ANCHOR' and anchor_bank is not None:
        return anchor_bank
    banks = {'RR': context_forward(b, rx, start)}
    audit = {'age': age, 'history_tokens': int((rx[0, plen:] != b.mask).sum()), 'sources': {}}
    if age == 'ANCHOR':
        ext = torch.cat((torch.arange(start, device=b.device), torch.arange(start+32, rx.shape[1], device=b.device)))
        for li in b.layers:
            for n in ['k', 'v']:
                assert torch.equal(banks['RR'][li][n].index_select(-2, ext), cp['anchor'][li][n])
        audit['self_reforward_exact'] = True
    for donor in ['A', 'B']:
        dx = donor_cps[donor][key]
        assert torch.equal(rx == b.mask, dx == b.mask)
        for source in ['P', 'H', 'PH']:
            name = f'{donor}_{source}'
            mixed = mix_input(rx, dx, plen, source)
            assert torch.equal(mixed == b.mask, rx == b.mask)
            banks[name] = context_forward(b, mixed, start)
            audit['sources'][name] = {
                'prompt_token_changes': int((mixed[:, :plen] != rx[:, :plen]).sum()),
                'history_token_changes': int((mixed[:, plen:] != rx[:, plen:]).sum()),
                'mask_pattern_exact': True,
                'layers': {str(li): source_statistics(banks['RR'][li], banks[name][li], p['absolute'][p['target'][p['slots']]])
                           for li, p in mapping.items() if len(p['slots'])},
            }
    return {'banks': banks, 'audit': audit}


def condition_info(condition):
    if condition == 'RR_CURRENT':
        return 'CURRENT', 'RR', 'KV'
    if condition.startswith(('A_', 'B_')):
        donor, age, source, component = condition.split('_')
        return age, f'{donor}_{source}', component
    return 'ANCHOR', 'RR', 'KV'


def intervention_cache(b, cp, condition, banks):
    start, length = cp['start'], cp['x'].shape[1]
    layers = range(b.n) if condition.endswith('_ALL') else b.layers
    mapping = factor.plans(b.model, cp['anchor'], start, start+32, length, layers)
    if condition in ['S25', 'C100', 'KV_NEXT', 'K_NEXT', 'V_NEXT', 'KV_ALL', 'K_ALL', 'V_ALL']:
        cache = cache_from_sources(b, cp['anchor'], mapping, condition)
    else:
        cache = factor.make_cache(b.model, cp['anchor'], mapping, 'KV_NEXT')
        age, source, component = condition_info(condition)
        bank = banks[age]['banks']
        for li, p in mapping.items():
            slots = p['slots']
            indices = p['absolute'][p['target'][slots]]
            for n in ['k', 'v']:
                # Both components have the same reference age; only provenance differs.
                chosen = source if n.upper() in component else 'RR'
                cache.cache[li][n][:, :, slots] = bank[chosen][li][n].index_select(-2, indices)
    expected = length - 32 if condition == 'C100' else int((length - 32) * .25)
    counts = [e['k'].shape[-2] for e in cache.cache.values()]
    assert len(counts) == b.n and all(n == expected for n in counts)
    assert all(e['k'].shape == e['v'].shape for e in cache.cache.values())
    audit = {'external_counts': counts, 'kv_bytes': sum(t.numel()*t.element_size() for e in cache.cache.values() for t in e.values()),
             'replaced_slots': {str(li): len(p['slots']) for li, p in mapping.items()}}
    return cache, mapping, audit


@contextmanager
def attention_trace(b, cp):
    """Outer SDPA hook sees Dream heads after the existing explicit GQA repeat."""
    original = F.scaled_dot_product_attention
    counter, captured = 0, {}
    active = torch.where(cp['x'][0, cp['start']:cp['start']+32] == b.mask)[0]
    if b.name == 'dream':
        active = (active-1).clamp_min(0)
    def wrapped(q, k, v, *args, **kwargs):
        nonlocal counter
        li = counter
        counter += 1
        out = original(q, k, v, *args, **kwargs)
        if li in b.layers:
            assert not args and kwargs.get('attn_mask') is None
            assert not kwargs.get('is_causal', False) and kwargs.get('dropout_p', 0.) == 0
            assert q.shape[1] == k.shape[1]
            captured[li] = {'q': q.index_select(-2, active).clone(), 'k': k.clone(), 'v': v.clone(),
                            'o': out.index_select(-2, active).clone()}
        return out
    F.scaled_dot_product_attention = wrapped
    try:
        yield captured
    finally:
        F.scaled_dot_product_attention = original
        assert counter == b.n


def relative(a, b):
    return float((a.float()-b.float()).norm() / b.float().norm().clamp_min(1e-12))


@torch.inference_mode()
def operator_metrics(reference, actual, mapping):
    stats = {}
    for li, r in reference.items():
        if li not in mapping:
            continue
        a = actual[li]
        if a['k'].shape != r['k'].shape:
            continue  # C100 has a different cache layout and budget.
        q = r['q'].float()
        ks, vs = r['k'].float(), r['v'].float()
        kt, vt = a['k'].float().clone(), a['v'].float().clone()
        slots = mapping[li]['slots']
        # Isolate the changed cached slots at fixed RR query, current-block KV,
        # and all other external KV. Whole-network current K/V changes are excluded.
        testk, testv = ks.clone(), vs.clone()
        testk[:, :, slots] = kt[:, :, slots]
        testv[:, :, slots] = vt[:, :, slots]
        scale = math.sqrt(q.shape[-1])
        p = (q @ ks.transpose(-1, -2) / scale).softmax(-1)
        pt = (q @ testk.transpose(-1, -2) / scale).softmax(-1)
        o = p @ vs
        ot = pt @ testv
        contribution = p.index_select(-1, slots) @ vs.index_select(-2, slots)
        stats[str(li)] = {
            'reference_replaced_attention_mass_mean': float(p.index_select(-1, slots).sum(-1).mean()),
            'reference_replaced_attention_mass_max': float(p.index_select(-1, slots).sum(-1).max()),
            'replacement_contribution_norm_over_output': float(contribution.norm()/o.norm().clamp_min(1e-12)),
            'fixed_query_output_relative_l2': relative(ot, o),
            'whole_network_query_relative_l2': relative(a['q'], r['q']),
            'whole_network_output_relative_l2': relative(a['o'], r['o']),
            'fixed_query_replaced_logit_change_rms': float(((q @ (testk-ks).transpose(-1,-2)/scale).index_select(-1,slots).square().mean()).sqrt()) if len(slots) else 0.,
        }
    return stats


@torch.inference_mode()
def probe(b, sample, cp, condition, banks, reference=None):
    cache, mapping, audit = intervention_cache(b, cp, condition, banks)
    with attention_trace(b, cp) as captured:
        logits = b.forward(cp['x'][:, cp['start']:cp['start']+32], cp['start'], 2, cache)
    result = {'condition': condition, 'cache_audit': audit, **local_metrics(b, cp, logits)}
    z = b.aligned_logits(logits, cp['start'], False)[0].float()
    active = cp['x'][0, cp['start']:cp['start']+32] == b.mask
    result['mean_stop_probability_over_masked'] = float(z[active].softmax(-1)[:, list(b.stops)].sum(-1).mean())
    if reference is not None:
        result['operator_metrics'] = operator_metrics(reference, captured, mapping)
    return result, captured


@torch.inference_mode()
def continue_fork(b, sample, cp, condition, banks, check=None):
    """Intervene until this block ends; subsequent blocks use S25 (C100 ref excepted)."""
    x = cp['x'].clone()
    plen, first = sample['prompt_length'], cp['block']
    cache, mapping, first_audit = intervention_cache(b, cp, condition, banks)
    audits, commits = [dict(block=first, **first_audit)], []
    finite = torch.ones((), dtype=torch.bool, device=b.device)
    for block in range(first, 8):
        if check:
            check()
        start = plen + block*32
        official = cache_new(b)
        initial = cp['step'] if block == first else 0
        for step in range(initial, 32):
            if step == 0:
                logits = b.forward(x, start, 0, official)
            elif step == 1:
                anchor = {}
                logits = b.forward(x, start, 1, official, anchor)
                policy = 'C100' if condition == 'C100' else 'S25'
                cache = cache_from_sources(b, anchor, {}, policy)
                audits.append({'block': block, 'external_counts': [e['k'].shape[-2] for e in cache.cache.values()],
                               'kv_bytes': sum(t.numel()*t.element_size() for e in cache.cache.values() for t in e.values())})
                del anchor
            else:
                logits = b.forward(x[:, start:start+32], start, 2, cache)
            finite &= torch.isfinite(logits).all()
            pos, tok, conf = b.choice(logits, x, start, step, step < 2)
            commits.append({'block': block, 'step': step, 'positions': pos, 'tokens': tok, 'confidence': conf})
            if pos:
                x[0, pos] = torch.tensor(tok, device=b.device)
            del logits
    if not finite.item():
        raise FloatingPointError('Nonfinite continuation logits')
    ids = x[0, plen:].tolist()
    assert b.mask not in ids
    return {'cache_audit': audits, 'commits': commits, **b.score(sample, ids)}
