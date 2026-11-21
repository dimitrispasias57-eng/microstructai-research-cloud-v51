"""Explicit foreground semantics and opt-in adapters to unchanged generic metrics.

These supplements never insert fields into the frozen structural metric dictionary.
No RBC score, segmentation function, normalization formula or tolerance is edited.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from full_product import bootstrap
import service
from review_core import sha
from modules import metrics as frozen

VERSION = 'rc2-general-gap-supplement-1'
ROLES = ('Custom / other foreground', 'Gaps between structures', 'Pores / voids in material')
GAP_HELP = ('Gaps are spaces between structures; pores are voids in a material and can be open or interconnected, '
            'not only closed holes. Here both are measured as the explicitly selected 2-D foreground phase. '
            'Connected regions are not individual 3-D pores. Enclosed background holes are a different topology metric. '
            'Exclude outside-specimen background with an appropriate ROI; it is not automatically pore space. '
            'Area fraction is 2-D, not automatically bulk porosity. Equivalent-circle diameter is not channel width or throat size.')
SLOPE_HELP = ('Slope of log10(component count) against log10(component area) using the unchanged 12-log-bin fit. '
              'Dimensionless spatial area-distribution descriptor, NOT a temporal rate or proof of a power law. '
              'Requires at least three components, differing areas and three occupied bins; report R² and occupied bins. '
              'ROI-edge components remain included and may be truncated. No new fit-quality cutoff is imposed.')
SCORE_HELP = ('EXPERIMENTAL generic structure scores: gSASI and gAggIndex, not RBC SASI, GapScore, PTI or AggIndex. '
              'These are relative to the explicitly declared comparable group and are not validated aggregation measures. '
              'Adding/removing cases can change every score. Each score uses its own disclosed complete-case reference group; '
              'missing values are never imputed. At least three eligible, analyzed, nonuniform masks and variation in required '
              'features are required here as a technical guard, not a validation threshold. A single mask has no standalone score.')
SPECS = {
    'gSASI_0_100': ('gSASI', ('correlationLength_px', 'meanDiam_px', 'objectCount')),
    'gAggIndex_0_100': ('gAggIndex', ('meanDiam_px', 'areaFrac', 'LF_log10_LFoverHF_generic',
                                  'correlationLength_px', 'objectCount', 'sizeDistributionSlope', 'textureHeterogeneity_tileCV')),
}
FEATURES = tuple(dict.fromkeys(k for _, fields in SPECS.values() for k in fields))


def meaning(role, target):
    if role not in ROLES: raise ValueError('Choose an explicit foreground interpretation.')
    return target if role == ROLES[0] else role + ' — ' + target


def slope_name(role):
    return ('Gap' if role == ROLES[1] else 'Pore' if role == ROLES[2] else 'Component') + ' area-distribution slope'


def prepared_record(record):
    if not record.eligible or record.result is None or record.result_token != record.token():
        raise ValueError('Analyze the current supplied/reviewed mask before measuring supplements.')
    if record.context.mode != 'structural':
        raise ValueError('Generic supplements require a reviewed foreground phase, not instance morphology.')
    settings = service.Settings(**record.result['provenance']['settings'])
    return service.prepare(record.mask_bytes, record.name, settings)


def supplement(record, role=None):
    role = role or record.algorithm.get('foreground_role', ROLES[0])
    if role not in ROLES: raise ValueError('Unknown foreground interpretation.')
    prepared = prepared_record(record)
    mask = prepared['mask']
    slope, r2, bins = frozen.compute_gap_area_slope_matlab_like(mask)
    values = record.result['metrics']
    rows = [dict(key='componentAreaDistributionSlope', name=slope_name(role), value=slope, units='1'),
            dict(key='componentAreaDistributionSlope_R2', name='Area-distribution fit R²', value=r2, units='1'),
            dict(key='componentAreaDistributionSlope_nBins', name='Occupied fit bins', value=int(bins), units='count')]
    if role != ROLES[0]:
        phase = 'Gap' if role == ROLES[1] else 'Pore / void'
        for key, label, units in [('areaFrac', 'area fraction (2-D)', '1'), ('nObjects', 'connected-region count (2-D)', 'count'),
                                  ('meanArea', 'mean connected-region area', 'px²'), ('medianArea', 'median connected-region area', 'px²'),
                                  ('meanDiam', 'mean equivalent-circle diameter', 'px'), ('medianDiam', 'median equivalent-circle diameter', 'px')]:
            rows.append(dict(key='selected_phase__'+key, name=phase+' '+label, value=values[key], units=units,
                             source='Exact alias of frozen structural '+key))
    return dict(version=VERSION, foreground_role=role, foreground_meaning=record.context.meaning, rows=rows,
                slope_status='AVAILABLE' if np.isfinite(slope) else 'NOT EVALUABLE — insufficient components / area variation / occupied bins',
                definitions=dict(foreground=GAP_HELP, area_distribution=SLOPE_HELP),
                implementation_sha256=sha(Path(frozen.__file__).read_bytes()), adapter_sha256=sha(Path(__file__).read_bytes()),
                mask_file_sha256=sha(record.mask_bytes), review_token=record.token(),
                roi_1based_xywh=record.result['provenance']['roi_1based_xywh'],
                frozen_metrics_unchanged=True, reference_quality='Supplied or user-reviewed; not ground truth')


def feature_row(record):
    """Only existing mask-derived generic inputs; no optical/RBC outputs are used."""
    mask = prepared_record(record)['mask']
    native = frozen.compute_object_mode_metrics(mask.astype(float), mask)
    return {k: native[k] for k in FEATURES}


def profile(record):
    c = record.context
    x, y, w, h = service.validate_roi(c.roi, record.mask.shape)
    return dict(meaning=c.meaning, polarity=c.polarity, roi=(x,y,w,h), scale=c.um_per_pixel,
                scale_source=c.calibration_source, kind=record.kind, union=c.label_union_confirmed,
                channel=record.algorithm.get('channel'), parameters=record.algorithm.get('parameters'),
                algorithm=record.algorithm.get('name'), algorithm_version=record.algorithm.get('version'))


def score_group(frame, *, enabled=False, comparable=False, group_id=''):
    """Read the frozen score output only on eligible complete reference cohorts.

    The native function neutral-imputes NaN. Excluding incomplete rows BEFORE
    calling it prevents a missing descriptor turning into an invented score.
    """
    out = frame.copy(deep=True).reset_index(drop=True)
    refs = {}
    valid = out.get('valid', pd.Series(False, index=out.index)).fillna(False).astype(bool)
    features = out.reindex(columns=FEATURES).apply(pd.to_numeric, errors='coerce')
    occupancy_ok = (features['objectCount'] > 0) & (features['areaFrac'] > 0) & (features['areaFrac'] < 1)
    for column, (name, required) in SPECS.items():
        out[column] = np.nan
        out[name+'_available'] = False
        reason = ('DISABLED' if not enabled else 'COMPARABLE_GROUP_NOT_CONFIRMED' if not comparable or not group_id.strip() else '')
        out[name+'_reason'] = reason or 'EXCLUDED — unreviewed, unanalyzed, empty/uniform or missing required features'
        complete = valid & occupancy_ok & np.isfinite(features[list(required)]).all(axis=1)
        positions = list(out.index[complete])
        reference = features.loc[complete].copy()
        refs[name] = dict(required_features=list(required), positions_0based=positions, n=len(positions),
                          values=reference[list(required)].to_dict('records'), group_id=group_id,
                          normalization='Frozen robust_z: median / (1.4826 MAD); native standard-deviation fallback. Per-score complete-case group.',
                          minimum_n=3, minimum_n_scope='Technical degeneracy guard only, not scientific validation')
        if not reason and len(positions) < 3: reason = 'NOT EVALUABLE — fewer than three complete eligible reference masks'
        if not reason and not any(np.any(frozen.robust_z(reference[k]).to_numpy() != 0) for k in required):
            reason = 'NOT EVALUABLE — no usable variation in required reference features'
        if reason:
            out.loc[complete, name+'_reason'] = reason
        else:
            native = frozen.compute_generic_structure_scores(reference, enabled=True)
            values = native[column]
            finite = np.isfinite(values)
            out.loc[values.index[finite], column] = values[finite]
            out.loc[values.index[finite], name+'_available'] = True
            out.loc[values.index[finite], name+'_reason'] = 'EXPERIMENTAL — relative to disclosed reference group'
            out.loc[values.index[~finite], name+'_reason'] = 'NOT EVALUABLE — nonfinite native output'
        refs[name]['reference_sha256'] = service.digest(refs[name])
    return dict(frame=out, provenance=dict(version=VERSION, enabled=enabled, comparable_group_confirmed=comparable,
                    group_id=group_id, references=refs, scope=SCORE_HELP,
                    implementation_sha256=sha(Path(frozen.__file__).read_bytes()), adapter_sha256=sha(Path(__file__).read_bytes()),
                    feature_source='Frozen compute_object_mode_metrics on the exact ROI binary mask. Only mask-derived features retained. '
                                   'correlationLength_px is its native definition, NOT substituted General Xi_horizontal/radial.',
                    excluded_rows_retained=True, missing_values_imputed=False))


def measure_group(records, *, group_id, comparable):
    rows=[]; audit=[]; profiles=[]; start=time.perf_counter()
    for i,r in enumerate(records):
        if time.perf_counter()-start>60: raise ValueError('General score group exceeded the 60-second budget; no partial score group retained.')
        valid = r.eligible and r.context.mode=='structural' and r.result is not None and r.result_token==r.token()
        row=dict(position=i, filename=r.name, valid=valid)
        if valid:
            row.update(feature_row(r)); profiles.append(service.digest(profile(r)))
            audit.append(r.provenance())
        else: audit.append(dict(filename=r.name, measured=False, decision=r.decision, token=r.token()))
        rows.append(row)
    if len(set(profiles))>1:
        raise ValueError('Reference-group settings differ (target, ROI, calibration, channel or proposal method). No scores were calculated. Use a genuinely comparable group; do not change valid settings just to force agreement.')
    bundle=score_group(pd.DataFrame(rows),enabled=True,comparable=comparable,group_id=group_id)
    bundle['provenance']['review_records']=audit
    bundle['provenance']['processing_profiles']=[profile(r) for r in records if r.eligible and r.context.mode=='structural' and r.result is not None and r.result_token==r.token()]
    return bundle


def display_supplement(data):
    import streamlit as st
    from researcher_ui.common import table
    with st.expander('Gap / pore phase and area-distribution measurements', expanded=data['foreground_role']!=ROLES[0]):
        st.write(GAP_HELP)
        st.caption('Foreground interpretation: '+data['foreground_role']+'. No complement is inferred; measure the selected/reviewed phase.')
        table(data['rows']); st.caption(SLOPE_HELP)
        if data['slope_status'].startswith('NOT EVALUABLE'):st.info(data['slope_status'])


def experimental_ui(records, prefix):
    import streamlit as st
    from researcher_ui.common import key, table, exports, metadata
    with st.expander('Optional experimental General scores — gSASI / gAggIndex'):
        st.warning(SCORE_HELP)
        enabled=st.checkbox('Enable experimental General scores',key=key(prefix+'_enabled'))
        if not enabled:return
        eligible=[r for r in records if r.eligible and r.context.mode=='structural' and r.result is not None and r.result_token==r.token()]
        if len(eligible)<3:
            st.info('NOT EVALUABLE — analyze at least three comparable supplied/reviewed phase masks. For original images use Temporal structure for one recording; for independent static cases use the supplied-mask batch workflow. Do not duplicate one image to create a reference group.');return
        group_id=st.text_input('Reference group ID / description',key=key(prefix+'_group'))
        comparable=st.checkbox('These masks have comparable target, sampling scale, ROI and processing; I accept group-dependent exploratory scores',key=key(prefix+'_comparable'))
        signature=service.digest([[(r.token(),r.decision,r.result_token) for r in records],group_id,comparable,VERSION,sha(Path(__file__).read_bytes())])
        slot=key(prefix+'_scores')
        if st.button('Calculate experimental General scores',disabled=not comparable or not group_id.strip(),key=key(prefix+'_calculate')):
            st.session_state.pop(slot,None)
            if not service.ANALYSIS_SLOT.acquire(blocking=False):st.info('Server busy; retry shortly.');return
            try:st.session_state[slot]=(signature,measure_group(records,group_id=group_id,comparable=comparable))
            finally:service.ANALYSIS_SLOT.release()
        prior=st.session_state.get(slot)
        if prior and prior[0]==signature:
            bundle=prior[1]; frame=bundle['frame']; provenance=bundle['provenance']
            table(frame[[c for c in frame if c not in FEATURES]].to_dict('records'))
            st.caption('No absolute aggregation scale; no biological or clinical interpretation. Reference membership and unrounded input features are included in export. RBC GapScore is not a General gap measurement.')
            for name,ref in provenance['references'].items():
                st.caption(f'{name}: {ref["n"]} complete reference masks; original positions (zero-based): {ref["positions_0based"]}.')
            result=dict(general_experimental_scores=provenance, rows=frame.to_dict('records'))
            exports('Experimental General structure scores',result,frame[[c for c in frame if c not in FEATURES]].to_dict('records'),
                    frame.to_dict('records'),{},metadata(prefix+'_metadata'),SCORE_HELP,domain_neutral=True)
