# === Stage 10C: OOF-promoted multi-scale GR alignment on the ravaghi branch ===
import hashlib as _a_hashlib
import json as _a_json
from pathlib import Path as _APath

import numpy as _a_np
import pandas as _a_pd
from numba import njit as _a_njit

_A_BRANCH_WEIGHT = 0.20
_A_CORRECTION_CAP = 8.0
_A_MIN_PREFIX_CORRELATION = 0.30


@_a_njit(cache=False, nogil=True)
def _a_shape_emission(observed, expected, tracked, half_windows, scale_weights, sigma):
    n_states, n_rows = expected.shape
    emission = _a_np.empty((len(tracked), n_states), dtype=_a_np.float64)
    for ti in range(len(tracked)):
        center = tracked[ti]
        for state in range(n_states):
            cost = 0.0
            for si in range(len(half_windows)):
                half = half_windows[si]
                start = max(0, center - half)
                stop = min(n_rows, center + half + 1)
                count = stop - start
                om = 0.0
                em = 0.0
                for row in range(start, stop):
                    om += observed[row]
                    em += expected[state, row]
                om /= count
                em /= count
                cov = 0.0
                osq = 0.0
                esq = 0.0
                for row in range(start, stop):
                    left = observed[row] - om
                    right = expected[state, row] - em
                    cov += left * right
                    osq += left * left
                    esq += right * right
                denominator = _a_np.sqrt(osq * esq)
                corr = cov / denominator if denominator > 1e-12 else 0.0
                corr = min(1.0, max(-1.0, corr))
                cost += scale_weights[si] * (1.0 - corr)
            residual = (observed[center] - expected[state, center]) / sigma
            amplitude = min(residual * residual, 16.0)
            emission[ti, state] = cost + 0.08 * amplitude
    return emission


@_a_njit(cache=False, nogil=True)
def _a_viterbi(emission):
    n_steps, n_states = emission.shape
    center = (n_states - 1) / 2.0
    previous = _a_np.empty(n_states, dtype=_a_np.float64)
    back = _a_np.empty((n_steps, n_states), dtype=_a_np.int16)
    for state in range(n_states):
        distance = state - center
        previous[state] = emission[0, state] + 0.08 * distance * distance
        back[0, state] = -1
    for step in range(1, n_steps):
        current = _a_np.empty(n_states, dtype=_a_np.float64)
        for state in range(n_states):
            best_cost = 1e300
            best_previous = state
            lower = max(0, state - 3)
            upper = min(n_states - 1, state + 3)
            for source in range(lower, upper + 1):
                jump = state - source
                cost = previous[source] + 0.03 * jump * jump
                if cost < best_cost:
                    best_cost = cost
                    best_previous = source
            distance = state - center
            current[state] = best_cost + emission[step, state] + 0.0004 * distance * distance
            back[step, state] = best_previous
        previous = current
    states = _a_np.empty(n_steps, dtype=_a_np.int16)
    states[-1] = int(_a_np.argmin(previous))
    for step in range(n_steps - 1, 0, -1):
        states[step - 1] = back[step, states[step]]
    return states


def _a_prefix_correlation(horizontal, typewell_tvt, typewell_gr):
    known = horizontal[horizontal['TVT_input'].notna() & horizontal['GR'].notna()]
    if len(known) < 30:
        return 0.0
    observed = known['GR'].to_numpy(dtype=float)
    expected = _a_np.interp(known['TVT_input'].to_numpy(dtype=float), typewell_tvt, typewell_gr)
    observed -= _a_pd.Series(observed).rolling(31, center=True, min_periods=1).mean().to_numpy()
    expected -= _a_pd.Series(expected).rolling(31, center=True, min_periods=1).mean().to_numpy()
    denominator = float(_a_np.sqrt(_a_np.sum(observed**2) * _a_np.sum(expected**2)))
    return float(_a_np.sum(observed * expected) / denominator) if denominator > 1e-12 else 0.0


def _a_alignment_correction(horizontal, typewell, rows, base):
    gr_series = _a_pd.to_numeric(horizontal['GR'], errors='coerce').interpolate(limit_direction='both')
    if gr_series.isna().all() or len(rows) == 0:
        return _a_np.zeros(len(rows)), {'active': False, 'reason': 'missing_gr'}
    gr_series = gr_series.fillna(float(gr_series.mean()))
    observed = gr_series.iloc[rows].to_numpy(dtype=float)
    tw = typewell[['TVT', 'GR']].copy().sort_values('TVT')
    tw['GR'] = tw['GR'].interpolate(limit_direction='both')
    tw = tw.dropna(subset=['TVT', 'GR']).groupby('TVT', as_index=False)['GR'].mean()
    typewell_tvt = tw['TVT'].to_numpy(dtype=float)
    typewell_gr = tw['GR'].to_numpy(dtype=float)
    if len(typewell_tvt) < 10:
        return _a_np.zeros(len(rows)), {'active': False, 'reason': 'short_typewell'}
    known = horizontal[horizontal['TVT_input'].notna() & horizontal['GR'].notna()]
    if len(known) >= 20:
        reference = _a_np.interp(known['TVT_input'].to_numpy(dtype=float), typewell_tvt, typewell_gr)
        actual = known['GR'].to_numpy(dtype=float)
        design = _a_np.column_stack([reference, _a_np.ones(len(reference))])
        gain, offset = _a_np.linalg.lstsq(design, actual, rcond=None)[0]
        gain = float(_a_np.clip(gain, 0.5, 1.5))
        offset = float(_a_np.clip(_a_np.median(actual - gain * reference), -50.0, 50.0))
        residual = actual - (gain * reference + offset)
        sigma = float(_a_np.clip(1.4826 * _a_np.median(_a_np.abs(residual - _a_np.median(residual))), 8.0, 60.0))
    else:
        gain, offset, sigma = 1.0, 0.0, 30.0
    offsets = _a_np.arange(-20.0, 20.5, 1.0)
    expected = _a_np.vstack([
        offset + gain * _a_np.interp(base + candidate_offset, typewell_tvt, typewell_gr)
        for candidate_offset in offsets
    ])
    tracked = _a_np.arange(0, len(base), 8, dtype=_a_np.int64)
    if tracked[-1] != len(base) - 1:
        tracked = _a_np.append(tracked, len(base) - 1)
    emission = _a_shape_emission(
        observed, expected, tracked,
        _a_np.asarray([3, 8, 20], dtype=_a_np.int64),
        _a_np.asarray([0.30, 0.40, 0.30], dtype=float), sigma,
    )
    states = _a_viterbi(emission)
    tracked_correction = offsets[states]
    correction = _a_np.interp(_a_np.arange(len(base)), tracked, tracked_correction)
    center = int(_a_np.argmin(_a_np.abs(offsets)))
    zero_cost = float(emission[:, center].sum())
    path_cost = float(emission[_a_np.arange(len(tracked)), states].sum())
    state_center = 0.5 * (len(offsets) - 1)
    path_cost += 0.08 * float(states[0] - state_center) ** 2
    path_cost += 0.0004 * float(_a_np.sum(_a_np.square(states[1:] - state_center)))
    for index in range(1, len(states)):
        jump = float(states[index] - states[index - 1])
        path_cost += 0.03 * jump * jump
    cost_gain = (zero_cost - path_cost) / max(len(tracked), 1)
    prefix = _a_prefix_correlation(horizontal, typewell_tvt, typewell_gr)
    active = cost_gain >= 0.03 and prefix >= _A_MIN_PREFIX_CORRELATION
    correction = _a_np.clip(correction, -_A_CORRECTION_CAP, _A_CORRECTION_CAP)
    if not active:
        correction[:] = 0.0
    return correction, {
        'active': bool(active), 'prefix_shape_corr': prefix,
        'cost_gain_per_step': float(cost_gain),
        'mean_abs_correction': float(_a_np.mean(_a_np.abs(correction))),
        'max_abs_correction': float(_a_np.max(_a_np.abs(correction))),
    }


_A_WORK = _APath('/kaggle/working') if _APath('/kaggle/working').exists() else _APath('.')
_a_sample = _a_pd.read_csv(CFG.dataset_path / 'sample_submission.csv')[['id']].copy()
_a_sample['id'] = _a_sample['id'].astype(str)
_a_base = sub_1[['id', 'tvt']].copy()
_a_base['id'] = _a_base['id'].astype(str)
if not _a_base['id'].equals(_a_sample['id']):
    raise RuntimeError('Stage 10C base IDs do not match sample submission order')
_a_output = dict(zip(_a_base['id'], _a_base['tvt'].astype(float)))
_a_split = _a_base.copy()
_a_split['well'] = _a_split['id'].str[:8]
_a_split['row_idx'] = _a_split['id'].str[9:].astype(int)
_a_reports = []
for _a_well_index, (_a_wid, _a_group) in enumerate(_a_split.groupby('well', sort=False), 1):
    try:
        _a_horizontal = _a_pd.read_csv(CFG.dataset_path / 'test' / f'{_a_wid}__horizontal_well.csv')
        _a_typewell = _a_pd.read_csv(CFG.dataset_path / 'test' / f'{_a_wid}__typewell.csv')
        _a_ordered = _a_group.sort_values('row_idx')
        _a_rows = _a_ordered['row_idx'].to_numpy(dtype=int)
        _a_values = _a_ordered['tvt'].to_numpy(dtype=float)
        _a_correction, _a_report = _a_alignment_correction(_a_horizontal, _a_typewell, _a_rows, _a_values)
        _a_moved = _A_BRANCH_WEIGHT * _a_correction
        for _a_id, _a_value, _a_delta in zip(_a_ordered['id'], _a_values, _a_moved):
            _a_output[str(_a_id)] = float(_a_value + _a_delta)
        _a_report.update({'well': _a_wid, 'rows': int(len(_a_rows)), 'mean_abs_move': float(_a_np.mean(_a_np.abs(_a_moved))), 'max_abs_move': float(_a_np.max(_a_np.abs(_a_moved)))})
    except Exception as _a_error:
        _a_report = {'well': _a_wid, 'active': False, 'reason': 'error', 'error': repr(_a_error), 'rows': int(len(_a_group)), 'mean_abs_move': 0.0, 'max_abs_move': 0.0}
    _a_reports.append(_a_report)
    print('[Stage10C %d] %s active=%s prefix=%.3f move=%.3f' % (_a_well_index, _a_wid, _a_report.get('active'), _a_report.get('prefix_shape_corr', float('nan')), _a_report.get('mean_abs_move', 0.0)), flush=True)

sub_1 = _a_sample.copy()
sub_1['tvt'] = sub_1['id'].map(_a_output).astype(float)
if not _a_np.isfinite(sub_1['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('Stage 10C produced non-finite ravaghi values')
_a_report_frame = _a_pd.DataFrame(_a_reports)
_a_report_frame.to_csv(_A_WORK / 'stage10c_alignment_well_audit.csv', index=False)
_a_audit = {
    'profile': 'prefix030_cap8', 'branch': 'ravaghi',
    'branch_weight': _A_BRANCH_WEIGHT, 'correction_cap': _A_CORRECTION_CAP,
    'minimum_prefix_correlation': _A_MIN_PREFIX_CORRELATION,
    'wells': int(len(_a_report_frame)),
    'active_wells': int(_a_report_frame.get('active', _a_pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
    'rows': int(len(sub_1)),
    'mean_abs_branch_move': float(_a_np.mean(_a_np.abs(sub_1['tvt'].to_numpy(dtype=float) - _a_base['tvt'].to_numpy(dtype=float)))),
    'max_abs_branch_move': float(_a_np.max(_a_np.abs(sub_1['tvt'].to_numpy(dtype=float) - _a_base['tvt'].to_numpy(dtype=float)))),
    'effective_linear_weight_before_projection': float(_A_BRANCH_WEIGHT * 0.3 * 0.55),
}
(_A_WORK / 'stage10c_alignment_audit.json').write_text(_a_json.dumps(_a_audit, indent=2, sort_keys=True), encoding='utf-8')
print('STAGE10C_ALIGNMENT_AUDIT =', _a_audit, flush=True)
