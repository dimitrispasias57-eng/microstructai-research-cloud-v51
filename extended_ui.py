"""Additional v5.1 pages; no independent scientific algorithms."""
import io, json, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import backend as b
import extensions as e
import reporting as rpt
from contextlib import contextmanager

@contextmanager
def observe_host_resources():
    """Observe adapter execution; never alter inputs, results or frozen engines."""
    import threading, time, psutil
    process=psutil.Process();stop=threading.Event()
    samples=[process.memory_info().rss];started=time.perf_counter();cpu=time.process_time()
    measurement={'sampling_interval_ms':20,'scope':'Whole host process RSS/CPU during this operation; includes other sessions and lock wait, not per-session attribution.'}
    def sample():
        while not stop.wait(.02):samples.append(process.memory_info().rss)
    sampler=threading.Thread(target=sample,name='host-resource-observer',daemon=True);sampler.start()
    try:yield measurement
    finally:
        stop.set();sampler.join(timeout=1);samples.append(process.memory_info().rss)
        measurement.update(wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu,rss_start_mib=samples[0]/1024**2,peak_sampled_process_rss_mib=max(samples)/1024**2,rss_end_mib=samples[-1]/1024**2,samples=len(samples))

NAMES=['Project / Dataset','Advanced M2 / Reference Workflow','Supplied-instance Morphology','Fibrin / Clot Structure','PCA / Multivariate Analysis','Structural Detail / QC / Robustness','Supplied-track Flow Analysis']
NAMES+=['Descriptive / Reference-table Comparison','Device-distance Structure']
NAMES+=['Experimental Proposals / Mask Review']
NAMES+=['Multi-sample Comparison']
NAMES+=['Independent-recording Batch']
NAMES+=['Historical Object Recording']
BADGES={NAMES[0]:'SESSION-OWNED RESEARCH WORKSPACE',NAMES[1]:'EXTERNALLY EVALUATED REFERENCE WORKFLOW — ADVANCED / LOCAL ONLY',
 NAMES[2]:'RESEARCH USE WITH LIMITATIONS',NAMES[3]:'EXPLORATORY',NAMES[4]:'EXPLORATORY',NAMES[5]:'EXPLORATORY',NAMES[6]:'RESEARCH USE WITH LIMITATIONS',NAMES[7]:'EXPLORATORY',NAMES[8]:'EXPLORATORY',NAMES[9]:'RESEARCH USE WITH LIMITATIONS — EXPERIMENTAL PROPOSALS'}
LIMITS={NAMES[0]:'Ephemeral session, not permanent storage. No PHI or confidential data.',
 NAMES[1]:'ADVANCED / SUPERVISED SERVICE — NOT LIVE IN FREE CLOUD. No external AI request is made by this application.',
 NAMES[2]:'User-supplied instances; identity is asserted, not inferred. Projected morphology is not diagnosis, mechanical deformability or aggregation validation.',
 NAMES[3]:'User-declared distinguishable clot phase; generic experimental proposals, not validated fibrin recognition. No clotting/lysis time, TEG/ROTEM equivalence or clinical validity.',
 NAMES[4]:'Standardised complete-case descriptive PCA. Axes do not establish biological classes; observations need not be independent.',
 NAMES[5]:'Existing descriptive geometry and perturbation heuristics, not segmentation accuracy or calibrated confidence. Gap tracks are NOT RBC identities.',
 NAMES[6]:'Supplied bounding-box-centre tracks, not detected cells or membrane motion. Missing-frame intervals are excluded from rate calculations; no invented interpolation.',
 NAMES[7]:'Descriptive association and implementation comparison only. Repeated frames are not independent replicates; correlations do not establish rheological equivalence, causality or clinical validity.',
 NAMES[8]:'User-supplied device geometry. Distance-band boundaries clip structures; spatial association is not a causal device effect or a mechanics measurement.',
 NAMES[9]:'Historical fixed, domain-dependent classical proposals; no live M2 or retraining. Explicit review/replacement is required. Target-object masks are not automatically individual RBCs or independent ground truth.'}
BADGES[NAMES[10]]='EXPLORATORY'
LIMITS[NAMES[10]]='Stored results only; no re-segmentation. Samples must be comparable and independently defined. Frames are not independent replicates. No instrument, clinical or mechanistic equivalence.'
BADGES[NAMES[11]]='RESEARCH / EXPLORATORY STRUCTURAL ANALYSIS'
LIMITS[NAMES[11]]='Separate recordings retain their frozen start/QC/reference normalization. No independent image-rheology validation; never split one recording into folders to evade limits. Unreviewed automated outputs are not accepted masks or ground truth.'
BADGES[NAMES[12]]='EXPLORATORY — HISTORICAL OBJECT MODE'
LIMITS[NAMES[12]]='Generic detected structures, not RBC plasma gaps or identified cells. The historical recording scores, optical/POD preprocessing and conventions are distinct from General Structural Analysis and RBC scores. No clinical, rheological or mechanical validation.'

def project(api):
    api['header'](NAMES[0])
    st.text_input('Project label (no personal identifiers)',key='project_label')
    st.text_area('Research context / dataset provenance',key='project_notes')
    inventory=[]
    for prefix in ('rbc','general','clot'):
        slot=st.session_state.get(prefix+'_result')
        if slot:inventory.append({'module':prefix,'configuration_token':slot['signature'],'reviewed_observations':len(slot['review'])})
    inventory.append({'module':'native','uploaded_observations':len(st.session_state.get('native_records',[]))})
    st.dataframe(pd.DataFrame(inventory),hide_index=True)
    packages=st.session_state.exports
    st.caption('The project download includes only this session’s exported results. Raw inputs are not silently uploaded to a shared archive; keep your original files locally.')
    if packages:
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('project.json',json.dumps({'version':b.VERSION,'project':st.session_state.get('project_label',''),'notes':st.session_state.get('project_notes',''),'session_id':st.session_state.session_id,'inventory':inventory},indent=2))
            for i,(name,p) in enumerate(packages.items()):z.writestr(f'module_{i+1:02d}.zip',p['zip'])
        st.download_button('Download this project (ZIP)',buf.getvalue(),'microstructai_project.zip','application/zip')
        selected=st.multiselect('Sections for one explicitly linked sample',list(packages),key='project_sections')
        confirmed=st.checkbox('The selected sections describe the same stated sample; no repeated-cell pooling or cross-module biological score is inferred.',key='project_same_sample')
        if selected and confirmed and st.session_state.get('project_label','').strip():
            import html
            parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font:16px Arial;max-width:1000px;margin:24px auto;padding:16px}a,p{overflow-wrap:anywhere}</style><h1>MicroStructAI sectioned research report</h1>',
                '<p>Sample: '+html.escape(st.session_state.project_label)+'</p><p>Research Use Only. Separate measurements, no composite phenotype score. Unselected modules are not assessed, not zero. Identity linkage is user-declared.</p>']
            archive=io.BytesIO()
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for i,name in enumerate(selected):
                    directory=f'section_{i+1:02d}';p=packages[name]
                    z.writestr(directory+'/report.html',p['html']);z.writestr(directory+'/results.csv',p['csv']);z.writestr(directory+'/complete_bundle.zip',p['zip'])
                    parts.append(f'<h2>{html.escape(name)}</h2><p>{html.escape(str(p["metadata"].get("status","Research use with limitations")))}</p><a href="{directory}/report.html">Open section report</a>')
                z.writestr('index.html',''.join(parts))
            st.download_button('Sectioned same-sample report (ZIP)',archive.getvalue(),'microstructai_sample_report.zip','application/zip')

def advanced(api):
    api['header'](NAMES[1])
    st.write('The frozen Astra-M2 workflow proposes whole-cell boundaries using a multimodal model, applies deterministic refinement and QC, and routes results for review. It has external erythroSight evidence, unlike the live native model’s internal-only evidence.')
    st.write('External cohort: 200 cases / 20 repository-identified participants. 106 masks; 94 abstentions. Coverage: 53%. Blinded audit of 50 available masks: 49 GOOD, 1 MINOR_ERROR, 0 MAJOR_ERROR. Mean Dice 0.949934 and IoU 0.905625 among available masks.')
    st.warning('This is a stronger type of evidence (external evaluation), not proof of greater accuracy than Native U-Net: the cohorts and selections are not paired. Selective coverage remains a major limitation; abstentions are not successful segmentations.')
    st.info('The existing authenticated external-model runtime is not a self-contained €0 Streamlit dependency. Use the preserved supervised local/reference workflow. This page neither starts inference nor substitutes a different model.')
    st.write('To compare independently produced masks here, use Ground-Truth / Validation. Do not label computer-assisted reviewed masks as independent manual ground truth.')

def instances(api):
    module=NAMES[2];api['header'](module)
    demo=st.checkbox('Synthetic labelled instances — software demonstration',key='inst_demo')
    supplied=api['upload']('Label image: 0 background, positive integer instance IDs','inst_upload',False)
    stack=st.checkbox('Declared separate-cell TIFF stack (one binary cell per plane), not a z/time volume',key='inst_stack')
    if demo:
        y,x=np.mgrid[:128,:160];labels=np.zeros((128,160),np.uint16)
        for i in range(5):labels[((x-(20+30*i))/(6+i))**2+((y-55)/(9+i))**2<1]=i+1
        name='synthetic_instances.tif';digest=b.native.sha_bytes(labels.tobytes())
    elif supplied:
        from legacy.RBC_MORPHOLOGY_DEV.morphology import io as morphology_io
        labels,input_metadata=morphology_io.load_mask(supplied.getvalue(),supplied.name,stack=stack);name=supplied.name;digest=b.native.sha_bytes(supplied.getvalue())
    else:return
    components=st.checkbox('Measure connected regions instead of supplied instance IDs',key='inst_components')
    convention=st.selectbox('Existing measurement convention',['Frozen morphology engine (default)','Legacy regionprops instance measurements'],key='inst_convention')
    if convention.startswith('Legacy'):st.warning('Historical 4-neighbour perimeter and row-axis orientation. These values are not interchangeable with the default contour-based perimeter/angle conventions. No formula is substituted or harmonised.')
    confirmed=st.checkbox('I reviewed this label map and confirm each positive ID is the intended instance (or selected connected-region semantics).',key='inst_confirm')
    sample=st.text_input('Sample ID (no identifiers)',key='inst_sample')
    scale=None;scale_source=''
    if st.checkbox('Use measured isotropic spatial calibration',key='inst_cal'):
        scale=st.number_input('Micrometres / pixel',min_value=.000001,value=1.,key='inst_scale')
        scale_source=st.text_input('Calibration evidence',key='inst_cal_source')
    exclude=st.checkbox('Explicitly exclude ROI-border instances',key='inst_exclude')
    token=api['signature']([digest,components,confirmed,scale,scale_source,exclude,sample,stack,convention])
    saved=st.session_state.get('inst_result')
    if saved and saved['token']!=token:st.session_state.pop('inst_result',None);st.session_state.exports.pop(module,None);saved=None
    if st.button('Measure reviewed instances',disabled=not confirmed):
        if convention.startswith('Legacy'):
            if components or exclude:raise ValueError('Legacy mode requires supplied IDs and retains border QC flags; automatic component IDs and border filtering are unavailable in this historical contract.')
            e.review_engine()
            import rbc_morphology as legacy_instances
            with e.capacity():result=legacy_instances.analyze_instances(labels,identity_confirmed=confirmed,um_per_pixel=scale,calibration_source=scale_source,sample_id=sample,source={'filename':name,'sha256':digest})
        else:result=e.supplied_instances(labels,confirmed=confirmed,components=components,scale=scale,scale_source=scale_source,image_id=name,sample_id=sample,exclude_border=exclude)
        saved={'token':token,'result':result};st.session_state.inst_result=saved
    if not saved:return
    result=saved['result'];table=pd.DataFrame(result['instances'])
    if convention.startswith('Legacy'):
        api['show'](module,table,{'input_count':len(table),'analysed_count':len(table),'human_review_status':'USER_CONFIRMED','source_sha256':digest,'instance_provenance':result['provenance']},extras={'distributions.csv':rpt.csv_bytes(pd.DataFrame(result['distributions']))});return
    included=table.loc[table.included]
    st.write(f'{len(included)} / {len(table)} instances included. Exclusions remain in results.')
    st.dataframe(pd.DataFrame(result['summaries']),hide_index=True)
    metric=st.selectbox('Distribution descriptor',list(e.morphology.METRICS),key='inst_metric')
    figure=rpt.distribution_bytes(included,metric);st.image(figure)
    api['show'](module,table,{'input_count':len(table),'analysed_count':len(included),'human_review_status':'USER_CONFIRMED','source_sha256':digest,'instance_provenance':result['provenance']},{'morphology_distribution.png':figure},{'summaries.csv':rpt.csv_bytes(pd.DataFrame(result['summaries']))})
    with st.expander('Nonoverlapping field pooling — explicit same-sample linkage'):
        if st.button('Retain this field for sample summaries',disabled=not sample.strip(),key='inst_pool_add'):
            st.session_state.setdefault('instance_fields',{})[token]=result
        fields=list(st.session_state.get('instance_fields',{}).values());st.caption(f'{len(fields)} retained field records. Original rows/QC are retained, not averaged into biological scores.')
        nonoverlap=st.checkbox('These fields do not overlap and repeated cells are not counted twice.',key='inst_pool_ack')
        if fields and nonoverlap:
            groups=e.morphology.combine_samples(fields,nonoverlapping_fields_confirmed=True)
            summaries=pd.DataFrame([{'sample_id':g['sample_id'],**row} for g in groups for row in g['summaries']]);st.dataframe(summaries)
            st.download_button('Pooled descriptive summaries CSV',rpt.csv_bytes(summaries),'sample_instance_summaries.csv','text/csv')

def pca(api):
    module=NAMES[4];api['header'](module)
    choices=['Uploaded CSV','Synthetic software fixture']+[p+' results' for p in ('rbc','general','clot') if p+'_table' in st.session_state]
    source=st.selectbox('Feature table',choices,key='pca_source')
    if source=='Uploaded CSV':
        f=api['upload']('Numeric observations table','pca_upload',False,['csv'])
        if not f:return
        if f.size>8*1024**2:raise ValueError('PCA CSV limit: 8 MiB.')
        data=pd.read_csv(io.BytesIO(f.getvalue()))
    elif source.startswith('Synthetic'):
        x=np.arange(10,dtype=float);data=pd.DataFrame({'feature_a':x,'feature_b':x*x,'feature_c':np.sin(x)})
    else:data=st.session_state[source.split()[0]+'_table'].copy()
    if data.shape[0]>10000 or data.shape[1]>1000:raise ValueError('PCA limit: 10,000 rows and 1,000 columns; no rows/columns removed.')
    choices=b.finite_numeric_columns(data);columns=st.multiselect('Features to standardise',choices,default=choices[:3],key='pca_features')
    if len(columns)>50:raise ValueError('Use at most 50 selected PCA features for cloud resources.')
    token=api['signature']([data.to_json(),columns]);saved=st.session_state.get('pca_result')
    if saved and saved['token']!=token:st.session_state.pop('pca_result',None);st.session_state.exports.pop(module,None);saved=None
    if st.button('Run existing PCA'):
        with e.capacity():result=b.run_pca(data,columns)
        saved={'token':token,'result':result};st.session_state.pca_result=saved
    if not saved:return
    p=saved['result']
    if p.warning:st.warning(p.warning)
    if p.scores.empty:return
    st.write(f'{len(p.scores)} / {len(data)} complete observations used; missing values are not imputed.')
    st.dataframe(p.scores);st.dataframe(p.loadings);st.write('Explained variance',p.explained_variance.to_dict())
    figures={'pca_scores.png':rpt.pca_figure_bytes(p.scores),'pca_loadings.png':rpt.loadings_figure_bytes(p.loadings)}
    for image in figures.values():st.image(image)
    api['export'](module,p.scores,{'input_count':len(data),'analysed_count':len(p.scores),'features':columns,'standardisation':'Frozen StandardScaler then PCA','explained_variance':p.explained_variance.to_dict()},figures,{'loadings.csv':rpt.pca_loadings_bytes(p.loadings),'explained_variance.csv':rpt.csv_bytes(p.explained_variance.rename('explained_variance').rename_axis('component').reset_index())})

def session_records(prefix):
    slot=st.session_state[prefix+'_result']
    if prefix=='rbc':
        records=[dict(name=p['filename'],mask=p['mask'],image=p.get('camera_crop_gray',p['image']),review='HUMAN_REVIEWED' if i in slot['review'] else 'UNREVIEWED_EXPLORATORY') for i,p in enumerate(slot['result']['previews'])]
    else:records=slot['result']
    return records,slot

def structures(api):
    module=NAMES[5];api['header'](module)
    sources=['Synthetic software fixture']+[p+' masks' for p in ('rbc','general','clot') if p+'_result' in st.session_state]
    source=st.selectbox('Mask source',sources,key='detail_source')
    if source.startswith('Synthetic'):
        records=b.propose_general(b.decode_batch(b.fixture_frames()),{'channel':'Grayscale / luminance','min_area':40})
        for r in records:r['review']='SYNTHETIC_FIXTURE'
        table,_=b.measure_general(records);params={};base='synthetic'
    else:
        prefix=source.split()[0];records,slot=session_records(prefix);base=slot['signature']
        table=st.session_state.get(prefix+'_table');params=slot['result'].get('params',{}) if prefix=='rbc' else {}
    operation=st.radio('Existing analysis',['Static structural detail','Boundary / image QC','Full RBC structural sequence / passport'],key='detail_operation')
    index=st.selectbox('Observation',range(len(records)),format_func=lambda i:records[i]['name'],key='detail_case')
    r=records[index];st.image(rpt.panel(r.get('original_roi',r.get('image')),r['mask']),caption='Original · mask · overlay · contour')
    dark=st.checkbox('Also inspect dark-domain threshold/blur sensitivity (not individual RBCs)',key='detail_dark') if operation=='Boundary / image QC' else False
    if operation.startswith('Full') and not source.startswith('rbc'):st.info('Full RBC passport requires a completed RBC recording in this session; no RBC meaning is inferred for generic/clot masks.');return
    token=api['signature']([base,operation,index,dark]);saved=st.session_state.get('detail_result')
    if saved and saved['token']!=token:st.session_state.pop('detail_result',None);st.session_state.exports.pop(module,None);saved=None
    st.caption('Resource limits: static/QC ≤250,000 pixels and 500 regions; full sequence ≤12 frames / 1 million total pixels. Larger complete runs are local-only. Limits never resize, filter or subsample your data.')
    if st.button('Run existing structural tools'):
        if operation.startswith('Static'):result=e.static_structure(r)
        elif operation.startswith('Boundary'):result=e.robustness(r,dark)
        else:
            if table is None:raise ValueError('Finish the RBC review/exploratory workflow first.')
            result=e.structural_sequence(records,table,params)
        saved={'token':token,'result':result};st.session_state.detail_result=saved
    if not saved:return
    result=saved['result'];tables=e.tables_in(result)
    summary=result.get('summary',{})
    st.json(b.json_safe(summary))
    for warning in result.get('warnings',[]):st.warning(warning)
    selected=st.selectbox('Detailed output table',list(tables),key='detail_table') if tables else None
    output=tables[selected] if selected else pd.DataFrame([summary])
    st.dataframe(output,hide_index=True)
    extras={name+'.csv':rpt.csv_bytes(t) for name,t in tables.items()}
    if 'passport' in result:extras['structural_passport.md']=e.si.passport_markdown(result['passport']).encode()
    if 'dark_domain' in result:st.image(rpt.panel(r.get('original_roi',r.get('image')),result['dark_domain']['mask']),caption='Separate dark-domain diagnostic — not the primary mask')
    api['export'](module,output,{'input_count':len(records) if operation.startswith('Full') else 1,'analysed_count':len(records) if operation.startswith('Full') else 1,'source_signature':base,'operation':operation,'summary':summary,'review_status':[q.get('review','UNREVIEWED') for q in records]},extras=extras)

def flow(api):
    module=NAMES[6];api['header'](module)
    demo=st.checkbox('Synthetic supplied-track XML — software fixture',key='flow_demo')
    f=api['upload']('Cimrak-style tracked bounding boxes (XML)','flow_xml',False,['xml'])
    if demo:
        payload=('<data><images>'+''.join(f'<image><src>image{i}.png</src><boundingboxes><boundingbox><x_left_top>{10+i}</x_left_top><y_left_top>10</y_left_top><width>8</width><height>8</height><class_name><track_id>0</track_id></class_name></boundingbox></boundingboxes></image>' for i in range(1,6))+'</images></data>').encode()
    elif f:payload=f.getvalue()
    else:return
    scale=st.number_input('Spatial calibration (µm / pixel)',min_value=.000001,value=1.,key='flow_scale')
    step=st.number_input('Time per frame (seconds)',min_value=.000001,value=1.,key='flow_step')
    spatial=st.text_input('Spatial calibration evidence',key='flow_spatial')
    temporal=st.text_input('Frame timing evidence',key='flow_temporal')
    sequence_id=st.text_input('Sequence / experiment ID',key='flow_sequence')
    token=api['signature']([b.native.sha_bytes(payload),scale,step,spatial,temporal,sequence_id]);saved=st.session_state.get('flow_result')
    if saved and saved['token']!=token:st.session_state.pop('flow_result',None);st.session_state.exports.pop(module,None);saved=None
    if st.button('Measure supplied tracks',disabled=not(spatial.strip() and temporal.strip() and sequence_id.strip())):
        result=e.supplied_tracks(payload,scale,step,spatial,temporal,sequence_id);saved={'token':token,'result':result};st.session_state.flow_result=saved
    if saved:
        result=saved['result'];frame=pd.DataFrame(result['tracks']);st.json(b.json_safe(result['counts']))
        api['show'](module,frame,{'input_count':result['counts'].get('boxes',len(result['observations'])),'analysed_count':len(frame),'tracking_provenance':result},extras={'observations.csv':rpt.csv_bytes(pd.DataFrame(result['observations'])),'steps.csv':rpt.csv_bytes(pd.DataFrame(result['steps']))})

def read_csv(upload):
    if upload.size>8*1024**2:raise ValueError('Table upload limit: 8 MiB.')
    table=pd.read_csv(io.BytesIO(upload.getvalue()))
    if len(table)>10000 or len(table.columns)>1000:raise ValueError('Table limit: 10,000 rows and 1,000 columns. No truncation.')
    return table

def comparison(api):
    module=NAMES[7];api['header'](module)
    operation=st.radio('Existing comparison tool',['Paired sample / bulk association','Historical reference CSV comparison','Similar scores / different structure'],key='compare_tool')
    demo=st.checkbox('Synthetic comparison tables — software fixture',key='compare_demo')
    f=api['upload']('Structural / app-results CSV','compare_first',False,['csv'])
    if demo:
        x=np.arange(6,dtype=float);first=pd.DataFrame({'sampleID':['S'+str(i) for i in range(6)],'time_s':x,'gapAreaFrac':.1+x*.02,'gapNtotal':10+x,'score':x,'structure':x*x})
    elif f:first=read_csv(f)
    else:return
    args={};second=None
    if operation=='Similar scores / different structure':
        if len(first)>200:raise ValueError('Pair-discovery limit: 200 rows; all pairs retained. No silent sampling.')
        numeric=list(first.select_dtypes(include='number'))
        args['score_columns']=st.multiselect('Existing score columns',numeric,default=['score'] if 'score' in numeric else numeric[:1],key='compare_scores')
        args['structural_columns']=st.multiselect('Structural descriptors',numeric,default=['structure'] if 'structure' in numeric else numeric[-1:],key='compare_structures')
    else:
        f2=api['upload']('Paired bulk / reference CSV','compare_second',False,['csv'])
        if demo:second=first.copy();second['gapAreaFrac']+=.001
        elif f2:second=read_csv(f2)
        else:return
        if operation.startswith('Paired'):
            args['sample_column_structural']=st.selectbox('Structural sample ID',list(first),key='compare_id1')
            args['sample_column_bulk']=st.selectbox('Bulk sample ID',list(second),key='compare_id2')
            if len(first.select_dtypes(include='number').columns)+len(second.select_dtypes(include='number').columns)>50:raise ValueError('Select/export at most 50 numerical columns combined for this cloud correlation run.')
        else:
            from modules import matlab_comparison as mc
            args['match_method']=st.selectbox('Explicit row matching rule',mc.MATCH_METHODS+[mc.ORIGINAL_INDEX_MATCH,mc.ANALYSIS_INDEX_MATCH],key='compare_matching')
            st.caption('Uses the existing reference-comparison implementation (historically MATLAB CSV). Missing SASI may be derived by that frozen helper and is explicitly reported; derived values are not an independent reference.')
    confirmed=st.checkbox('I confirm the stated sample/row pairing and comparable metric definitions and units; this is descriptive research.',key='compare_confirm')
    token=api['signature']([first.to_json(),None if second is None else second.to_json(),operation,args,confirmed]);saved=st.session_state.get('compare_result')
    if saved and saved['token']!=token:st.session_state.pop('compare_result',None);st.session_state.exports.pop(module,None);saved=None
    if st.button('Run existing comparison',disabled=not confirmed):
        with e.capacity():
            if operation.startswith('Paired'):
                for table,key in ((first,args['sample_column_structural']),(second,args['sample_column_bulk'])):
                    if table[key].isna().any() or table[key].duplicated().any():raise ValueError('Unique nonmissing sample IDs required; no many-to-many multiplication or automatic pooling.')
                result=e.si.compare_bulk_measurements(first,second,**args)
            elif operation.startswith('Historical'):
                from modules import matlab_comparison as mc
                result=mc.compare_matlab_metrics(first,second,**args)
            else:result={'ranked_pairs':e.si.find_similar_score_different_structure(first,**args)}
        saved={'token':token,'result':result};st.session_state.compare_result=saved
    if not saved:return
    result=saved['result'];tables=e.tables_in(result)
    for warning in result.get('warnings',[])+result.get('derivedMetricNotes',[]):st.warning(warning)
    name=st.selectbox('Comparison output',list(tables),key='compare_output');table=tables[name];st.dataframe(table)
    figures={};extras={n+'.csv':rpt.csv_bytes(t) for n,t in tables.items()}
    if operation.startswith('Historical'):
        from modules import plotting as plots
        comparison_long=result.get('long',pd.DataFrame())
        if not comparison_long.empty:
            plot_metric=st.selectbox('Reference-comparison plot metric',['None']+list(comparison_long.metric.dropna().unique()),key='compare_plot_metric')
            if plot_metric!='None':
                st.caption('Historical reference labels are retained inside these figures for traceability. These are comparisons to the supplied reference, not independent ground truth; derived reference values remain disclosed above.')
                for suffix,factory in [('scatter',plots.plot_matlab_scatter),('difference',plots.plot_matlab_difference),('trend',plots.plot_matlab_trend)]:
                    figures['reference_'+suffix+'.png']=rpt.existing_figure_bytes(factory,comparison_long,plot_metric);st.image(figures['reference_'+suffix+'.png'])
                    extras['reference_'+suffix+'.svg']=rpt.existing_figure_bytes(factory,comparison_long,plot_metric,format='svg')
        paired=b.rbc_engine().paired_frame_export_df(comparison_long)
        if not paired.empty:extras['paired_frame_comparison.csv']=rpt.csv_bytes(paired)
        if not paired.empty and st.checkbox('Show existing paired correlation matrices',key='compare_matrix'):
            figures['paired_correlations.png']=rpt.existing_figure_bytes(plots.plot_matlab_spearman_matrix_comparison,paired);st.image(figures['paired_correlations.png'])
    api['export'](module,table,{'input_count':len(first),'analysed_count':result.get('matchedRows',len(tables.get('matched',table))),'operation':operation,'settings':args,'source_token':token,'diagnostics':{k:v for k,v in result.items() if not isinstance(v,pd.DataFrame)}},figures=figures,extras=extras)

def device(api):
    module=NAMES[8];api['header'](module)
    demo=st.checkbox('Synthetic device / gap masks — software fixture',key='device_demo')
    gap=api['upload']('Reviewed foreground / gap mask','device_gap',False);geometry=api['upload']('Device geometry binary mask','device_geometry',False)
    if demo:
        records=b.propose_general(b.decode_batch(b.fixture_frames(1)),{'channel':'Grayscale / luminance'});m=records[0]['mask'];g=np.zeros_like(m);g[:,:5]=True
    elif gap and geometry:
        m=b.binary(b.decode_batch([(gap.name,gap.getvalue())])[0]['image']);g=b.binary(b.decode_batch([(geometry.name,geometry.getvalue())])[0]['image'])
    else:return
    confirmed=st.checkbox('I reviewed the foreground and device masks on the same native pixel grid.',key='device_confirm')
    band_text=st.text_input('Distance-band edges (pixels, comma separated)',value='0, 10, 20, 40, 80',key='device_bands')
    edges=[float(x.strip()) for x in band_text.split(',')]
    if len(edges)>20 or not edges or not all(np.isfinite(edges)) or min(edges)<0 or np.any(np.diff(edges)<=0):raise ValueError('Use at most 20 finite, nonnegative, strictly increasing band edges. The unchanged engine includes the remaining far field.')
    token=api['signature']([b.native.sha_bytes(m.tobytes()),b.native.sha_bytes(g.tobytes()),edges,confirmed]);saved=st.session_state.get('device_result')
    if saved and saved['token']!=token:st.session_state.pop('device_result',None);st.session_state.exports.pop(module,None);saved=None
    if st.button('Measure distance bands',disabled=not confirmed):
        e.check_heavy([{'mask':m}],max_frames=1,max_pixels=250000)
        with e.capacity():table=e.si.device_distance_features(m,g,edges)
        saved={'token':token,'table':table};st.session_state.device_result=saved
    if saved:
        table=saved['table'];api['show'](module,table,{'input_count':1,'analysed_count':1,'human_review_status':'USER_CONFIRMED','mask_and_geometry_token':token,'distance_edges_px':edges})
    with st.expander('Existing repeated-frame influence profile — separate input'):
        f=api['upload']('Repeated distance-band results CSV','device_profile',False,['csv'])
        if f:
            data=read_csv(f);metric=st.selectbox('Distance-profile descriptor',list(data),key='device_profile_metric')
            profile_token=api['signature']([b.native.sha_bytes(f.getvalue()),metric])
            profile_saved=st.session_state.get('device_profile_result')
            if profile_saved and profile_saved['token']!=profile_token:
                st.session_state.pop('device_profile_result',None);profile_saved=None
            if st.button('Estimate existing exploratory profile'):
                result=e.si.estimate_device_influence_profile(data,metric)
                profile_saved={'token':profile_token,'result':result};st.session_state.device_profile_result=profile_saved
            if profile_saved:
                result=profile_saved['result'];st.json(b.json_safe({k:v for k,v in result.items() if k!='profile'}));st.dataframe(result['profile'])
                st.download_button('Profile CSV',rpt.csv_bytes(result['profile']),'device_profile.csv','text/csv')
        else:st.session_state.pop('device_profile_result',None)

def review(api):
    module=NAMES[9];api['header'](module);rc=e.review_engine()
    adapter=st.radio('Start with',['PathOlOgics v0.1 original target crops','RedTell v0.2 original grayscale images','Supplied masks for review / replacement'],key='review_adapter')
    same_cell=st.checkbox('One ordered SAME-CELL sequence (exactly one instance per observation)',key='review_same_cell')
    st.caption('PathOlOgics: native 80×80 RGB target-object crops only; mixed patient-disjoint evidence, not RBC identity. RedTell: frozen classical grayscale instance proposal; development-only, all 92 historical images consumed, no donor-independent validation. Neither replaces the 96×96 native model.')
    demo=st.checkbox('Synthetic target crop — software fixture only',key='review_demo')
    uploads=api['upload']('Original images or supplied masks, according to the selected route','review_files')
    if demo:
        yy,xx=np.mgrid[:80,:80];a=np.full((80,80,3),230,np.uint8);a[(xx-40)**2+(yy-40)**2<17**2]=[100,80,90];values=[('synthetic_target.png',b.native.png_bytes(a))]
    else:values=[(f.name,f.getvalue()) for f in uploads or []]
    if not values:return
    if len(values)>24 or sum(len(p) for _,p in values)>32*1024**2:raise ValueError('Review queue: ≤24 images and 32 MiB source bytes. No truncation.')
    confirmed=st.checkbox('The selected method fits this domain and boundaries are resolvable; I will review every proposal.',key='review_domain')
    token=api['signature']([[(n,b.native.sha_bytes(p)) for n,p in values],adapter,same_cell,confirmed]);bundle=st.session_state.get('review_queue')
    if bundle and bundle['token']!=token:st.session_state.pop('review_queue',None);st.session_state.exports.pop(module,None);bundle=None
    if st.button('Load / propose review masks',disabled=not confirmed):
        queue=rc.ReviewSession()
        with e.capacity():
            for name,payload in values:
                if adapter.startswith('Supplied'):r=rc.make_record(name,payload)
                else:r=rc.propose(payload,name,adapter.split(' original')[0],rc.Context(mode='morphology',meaning='User-reviewed instances',identity_confirmed=True) if adapter.startswith('RedTell') else rc.Context(meaning='Target-object foreground, identity unestablished'))
                if same_cell and len(np.unique(r.mask[r.mask>0]))!=1:raise ValueError(name+': exactly one proposed foreground instance required. No target was selected automatically.')
                queue.add(name,r)
        bundle={'token':token,'records':queue.records,'queue':queue,'same_cell':same_cell,'reviewer_events':[]};st.session_state.review_queue=bundle
    if not bundle:return
    names=list(bundle['records']);name=st.selectbox('Case to review',names,key='review_case');r=bundle['records'][name]
    original_only=st.checkbox('Inspect original only',key='review_original')
    for view in (['original'] if original_only else ['original','mask','overlay','boundary']):st.image(rc.render(r,view,ids=r.kind=='instances'),caption=view,width=320)
    st.write('State:',r.queue_status);st.write('Observable cautions:',rc.warnings(r))
    reviewer=st.text_input('Reviewer identity',key='review_identity');notes=st.text_area('Review notes',key='review_notes')
    replacement=api['upload']('Optional corrected mask: exact dimensions and same label semantics','review_replacement',False,['png','tif','tiff','bmp'])
    if replacement and st.button('Replace mask and invalidate prior acceptance'):
        bundle['queue'].replace(r,replacement.getvalue(),replacement.name);st.session_state.exports.pop(module,None);st.rerun()
    inspection=st.checkbox('I personally inspected this original/mask and the full target boundary.',key='review_inspected_'+name)
    for col,action in zip(st.columns(3),('ACCEPT','REJECT','REVIEW')):
        if col.button(action,key='review_'+action,disabled=not reviewer.strip() or (action=='ACCEPT' and not inspection)):
            r.decide(action,confirmed=inspection);bundle['reviewer_events'].append(dict(case=name,action=action,reviewer=reviewer,notes=notes,token=r.token(),utc=api['now']()));st.session_state.exports.pop(module,None);st.rerun()
    if not r.eligible or r.decision!='ACCEPT':st.info('Fresh explicit acceptance required; supplied status alone is not counted as human review.');return
    if same_cell and len(np.unique(r.mask[r.mask>0]))!=1:st.warning('Replacement no longer has exactly one instance; dynamic handoff remains blocked.');return
    if st.button('Measure current accepted mask'):
        with e.capacity():rc.analyze(r)
    if r.result is not None and r.result_token==r.token():
        result=rc.current_result(r);table=pd.DataFrame(result['instances']) if r.context.mode=='morphology' else pd.DataFrame([result['metrics']])
        api['show'](module,table,{'input_count':len(names),'analysed_count':1,'human_review_status':'EXPLICITLY_ACCEPTED','review_provenance':r.provenance(),'reviewer_events':bundle['reviewer_events']},{'reviewed_overlay.png':rc.render(r,'overlay')},{'original_review_bundle.zip':rc.bundle(r)})
    if same_cell:st.caption('Once every case is accepted, choose Reviewed classical same-cell proposals in Dynamic Morphology, then provide acquisition times there.')

def multi_sample(api):
    from modules import multi_sample as ms
    module=NAMES[10];api['header'](module)
    demo=st.checkbox('Synthetic sample tables — software fixture only',key='multi_demo')
    summary_file=api['upload']('Sample summary CSV (explicit sampleName, donor, flowHistory, stentStatus)','multi_summary',False,['csv'])
    frames_file=api['upload']('Optional frame metrics CSV (explicit sampleName)','multi_frames',False,['csv'])
    if demo:
        summary=pd.DataFrame({'sampleName':['synthetic_A','synthetic_B'],'donor':['anonymous_A','anonymous_B'],'flowHistory':['fixture','fixture'],'stentStatus':['none','none'],'median_SASI_0_100':[35.,50.],'median_AggIndex_0_100':[40.,55.]})
        frames=pd.DataFrame({'sampleName':['synthetic_A']*6+['synthetic_B']*6,'analysisIndex_1based':list(range(1,7))*2,'time_s':list(range(6))*2,'SASI_0_100':list(range(30,36))+list(range(45,51))})
    elif summary_file:
        summary=read_csv(summary_file);frames=read_csv(frames_file) if frames_file else pd.DataFrame()
    else:return
    required=['sampleName','donor','flowHistory','stentStatus']
    if any(c not in summary for c in required):raise ValueError('Provide explicit '+', '.join(required)+'. Use unknown for genuinely unknown labels; do not rely on filename inference.')
    if summary[required].isna().any().any() or (summary[required].astype(str).apply(lambda s:s.str.strip())=='').any().any() or summary.sampleName.duplicated().any():raise ValueError('Sample IDs must be unique and explicit; required labels cannot be blank.')
    if len(summary)>50:raise ValueError('Cloud comparison is limited to 50 explicitly identified samples.')
    if not frames.empty and ('sampleName' not in frames or frames.sampleName.isna().any() or not set(frames.sampleName).issubset(set(summary.sampleName))):raise ValueError('Every frame needs an explicit sampleName present in the summary.')
    confirm=st.checkbox('These are comparable, explicitly defined samples; I verified the labels, units and processing settings.',key='multi_confirm')
    token=api['signature']([summary.to_json(),frames.to_json(),confirm]);saved=st.session_state.get('multi_result')
    if saved and saved['token']!=token:st.session_state.pop('multi_result',None);st.session_state.exports.pop(module,None);saved=None
    if st.button('Compare stored sample results',disabled=not confirm):
        with e.capacity():result=ms.prepare_multi_sample_comparison(summary,frames)
        if result['errors']:raise ValueError('; '.join(result['errors']))
        saved={'token':token,'result':result};st.session_state.multi_result=saved
    if not saved:return
    result=saved['result'];tables=e.tables_in(result)
    for warning in result['warnings']:st.warning(warning)
    output=st.selectbox('Sample comparison table',list(tables),key='multi_table');st.dataframe(tables[output])
    figures={};metric=st.selectbox('Sample summary metric',['None']+result.get('metricsAvailable',[]),key='multi_metric')
    if metric!='None':
        statistic=st.selectbox('Summary statistic',['median','mean'],key='multi_statistic')
        group=st.selectbox('Bar grouping',['Sample','flowHistory','stentStatus','flowHistory x stentStatus'],key='multi_group')
        figures['sample_bars.png']=rpt.existing_figure_bytes(ms.plot_multi_sample_bar,result['summary'],metric,statistic=statistic,group_by=group);st.image(figures['sample_bars.png'])
        if not result['frames'].empty:
            selected=st.multiselect('Samples in trajectory',list(result['summary'].sampleName),default=list(result['summary'].sampleName)[:5],key='multi_selected')
            axes=[c for c in ('analysisIndex_1based','time_s','matlabTime_s','originalImageIndex_1based') if c in result['frames']]
            if selected and axes:
                axis=st.selectbox('Stored trajectory axis',axes,key='multi_axis')
                curve_warnings=[]
                def curve():
                    fig,warnings=ms.plot_multi_sample_time_curves(result['frames'],metric,selected,axis);curve_warnings.extend(warnings);return fig
                figures['sample_trajectories.png']=rpt.existing_figure_bytes(curve);st.image(figures['sample_trajectories.png'])
                for warning in curve_warnings:st.warning(warning)
    api['export'](module,result['summary'],{'input_count':len(summary),'analysed_count':len(result['summary']),'human_review_status':'LABELS_AND_COMPARABILITY_CONFIRMED_NOT_MASK_REVIEW','source_token':token,'warnings':result['warnings']},figures,{n+'.csv':rpt.csv_bytes(t) for n,t in tables.items()})

def batch(api):
    module=NAMES[11];api['header'](module)
    demo=st.checkbox('Two synthetic recordings — software fixture only',key='batch_demo')
    source=api['upload']('ZIP with one independent recording per folder','batch_archive',False,['zip'])
    if demo:
        from modules.web_uploads import MemoryImage
        groups={name:[MemoryImage(n,p) for n,p in b.fixture_frames()] for name in ('synthetic_A','synthetic_B')}
    elif source:groups=e.archive_groups(source.getvalue())
    else:return
    if len(groups)>4:raise ValueError('Cloud batch limit: four complete independent recordings.')
    references=api['upload']('Optional historical reference CSVs (explicit pairing below)','batch_references',True,['csv']) or []
    reference_payloads={f.name:f.getvalue() for f in references}
    if len(reference_payloads)!=len(references):raise ValueError('Reference filenames must be unique.')
    for f in references:read_csv(f)
    rows=[]
    for name in groups:
        selected=st.selectbox('Reference for '+name,['Skip MATLAB comparison']+['Uploaded: '+f.name for f in references],key='batch_ref_'+name)
        rows.append(dict(sampleName=name,imageFolder=name,donor='unknown',flowGroup='unknown',stentCondition='unknown',matlabCSVSelection=selected))
    labels=pd.DataFrame(rows)
    st.dataframe(labels);st.caption('Original frozen CSV comparator uses its documented automatic matching rules; matching diagnostics and derived-reference notes remain in the exports. Unknown biological labels are not inferred from filenames.')
    fps=st.number_input('Documented frame rate shared by these recordings (frames/s)',min_value=.001,value=1.,key='batch_fps')
    st.caption('Acquisition preset: unchanged USB camera RBC default. Use the individual-recording interface for a different documented acquisition configuration.')
    confirm=st.checkbox('Folders represent separate recordings; uniform timing and the stated acquisition preset are appropriate for all. I understand these masks are unreviewed.',key='batch_confirm')
    token=api['signature']([[(name,[(f.name,b.native.sha_bytes(f.getvalue())) for f in files]) for name,files in groups.items()],labels.to_json(),[(n,b.native.sha_bytes(p)) for n,p in reference_payloads.items()],fps,confirm])
    saved=st.session_state.get('batch_result')
    if saved and saved['token']!=token:st.session_state.pop('batch_result',None);st.session_state.exports.pop(module,None);saved=None
    if st.button('Run independent-recording batch',disabled=not confirm):
        result=e.recording_batch(groups,labels,reference_payloads,fps);saved={'token':token,'result':result};st.session_state.batch_result=saved
    if not saved:return
    result=saved['result'];st.write('Completed recordings:',result['success_count'],'Failed recordings:',result['failure_count'])
    for warning in result['warnings']:st.warning(warning)
    tables=e.tables_in(result);output=st.selectbox('Batch output',list(tables),key='batch_output');st.dataframe(tables[output])
    figures={};summary=result['sample_summary'];engine=b.rbc_engine()
    display=st.selectbox('Existing batch plot',['None','QC counts','Condition summary','Reference Pearson heatmap','Sample metric bars'],key='batch_plot')
    if display!='None':
        if display=='QC counts':factory=engine.plot_batch_qc_counts;args=(summary,)
        elif display=='Condition summary':factory=engine.plot_batch_condition_summary;args=(summary,)
        elif display=='Reference Pearson heatmap':factory=engine.plot_batch_pearson_heatmap;args=(result.get('matlab_summary',pd.DataFrame()),)
        else:
            columns=list(summary.select_dtypes(include='number'))
            if not columns:st.info('No numeric sample summaries available.');return
            column=st.selectbox('Batch summary column',columns,key='batch_plot_column');factory=engine.plot_batch_bar;args=(summary,column,'Stored sample summaries',column)
        figures['batch_diagnostic.png']=rpt.existing_figure_bytes(factory,*args);st.image(figures['batch_diagnostic.png'])
        st.caption('Stored results only. Unknown condition labels remain unknown; recordings are not automatically independent donors. Reference correlation does not establish physical validity.')
    api['export'](module,summary,{'input_count':len(groups),'analysed_count':result['success_count'],'failed_recordings':result['failure_count'],'human_review_status':'UNREVIEWED_EXPLORATORY','warnings':result['warnings'],'source_token':token},figures=figures,extras={**{n+'.csv':rpt.csv_bytes(t) for n,t in tables.items()},'historical_batch_report.txt':result['report_md'].encode()})

def legacy_objects(api):
    module=NAMES[12];api['header'](module);engine=b.rbc_engine()
    st.info('This restores the original Object Mode recording pipeline. The main General Structural Analysis route remains available. All historical settings are explicit in the sidebar; no calibration search is performed.')
    demo=st.checkbox('Synthetic object recording — software fixture',key='object_demo')
    uploads=api['upload']('Original images of one recording','object_files') or []
    archive=api['upload']('Or one complete recording from ZIP','object_archive',False,['zip'])
    if demo:values=b.fixture_frames()
    elif archive:
        if uploads:raise ValueError('Use original images OR ZIP, not both.')
        groups=e.archive_groups(archive.getvalue());group=st.selectbox('Recording folder',list(groups),key='object_folder')
        values=[(f.name,f.getvalue()) for f in groups[group]]
        st.caption('Only this complete folder is analysed; other folders are not silently pooled.')
    else:values=[(f.name,f.getvalue()) for f in uploads]
    if not values:return
    frames=b.decode_batch(values);st.image(b.display_u8(frames[0]['image']),caption='Original source; native channels retained',width=500)
    method=st.selectbox('Historical segmentation workflow',engine.OBJECT_ANALYSIS_METHODS,key='object_method',format_func=lambda s:{'Standard Object Mode':'Threshold-based objects','Enhanced Object Mode':'Seed-and-grow structures','Conservative Object Mode':'Adaptive objects with optical diagnostics','Experimental POD-removed Object Mode':'Experimental POD-removed objects'}[s])
    params=engine.object_segmentation_controls(title='Historical Object Mode — original controls',force_object_method=method)
    params['matlab_fps']=st.number_input('Documented recording rate (frames/s)',min_value=.001,value=1.,key='object_fps')
    target=st.text_input('What do the selected structures represent?',key='object_target')
    acquisition=st.checkbox('I confirm this is one recording, native filename order and uniform timing; selected foreground meaning is appropriate.',key='object_acquisition')
    st.caption('No automatic sample recognition. Historical RGB-to-grayscale processing and all selected preprocessing are recorded. Cloud limits: 200 images, 32 MiB and 16 million channel-pixels; no resizing or automatic subsampling.')
    token=api['signature']([[(f['name'],f['sha256']) for f in frames],params,target,acquisition]);saved=st.session_state.get('object_result')
    if saved and saved['token']!=token:st.session_state.pop('object_result',None);saved=None
    if st.button('Run historical Object Mode',disabled=not acquisition or not target.strip()):
        with observe_host_resources() as host_resources:result=e.legacy_object_recording(frames,params)
        saved={'token':token,'result':result,'reviews':{},'host_resources':host_resources};st.session_state.object_result=saved
    if not saved:return
    with st.expander('Host resource observation — this run'):
        st.caption('Operational measurements only. Shared-process RSS is not isolated session memory or a worst-case capacity guarantee.')
        st.json(saved.get('host_resources',{}))
    result=saved['result'];previews=result['previews'];index=st.selectbox('Inspect object observation',range(len(previews)),format_func=lambda i:previews[i]['filename'],key='object_case')
    preview=previews[index];mask=np.asarray(preview['mask'],bool);original=preview.get('camera_crop_gray',preview['image'])
    if st.checkbox('Original only — hide object proposal',key='object_original'):st.image(b.display_u8(original),width=600)
    else:st.image(rpt.panel(original,mask),caption='Original crop · mask · overlay · contour')
    with st.expander('Unchanged preprocessing views'):
        for key in ('wz','pod_removed_image','conservative_no_pod_mask','pod_removed_mask','removed_structure_mask'):
            value=preview.get(key)
            if isinstance(value,np.ndarray):st.image(b.display_u8(value),caption=key,width=450)
    for warning in result.get('warnings',[]):st.warning(str(warning))
    review_mode=st.radio('Object review workflow',['Reviewed analysis','Unreviewed exploratory analysis'],key='object_review_mode')
    if review_mode=='Reviewed analysis':
        reviewer=st.text_input('Object reviewer identity',key='object_reviewer')
        inspected=st.checkbox('I personally inspected and accept this original and the complete proposed foreground.',key='object_inspected_'+str(index))
        if st.button('Accept this object mask',disabled=not reviewer.strip() or not inspected):saved['reviews'][index]={'reviewer':reviewer,'utc':api['now']()}
        all_inspected=st.checkbox('I personally inspected and accept every mask in this recording (not automatic acceptance).',key='object_all_inspected')
        if st.button('Record all-mask review',disabled=not reviewer.strip() or not all_inspected):saved['reviews']={i:{'reviewer':reviewer,'utc':api['now']()} for i in range(len(previews))}
        st.caption(f"Reviewed {len(saved['reviews'])} / {len(previews)}")
        if len(saved['reviews'])!=len(previews):return
    table=result['metrics'].copy();table['human_review_status']=['HUMAN_REVIEWED' if i in saved['reviews'] else 'UNREVIEWED_EXPLORATORY' for i in range(len(previews))]
    figures={'object_panels.png':b.native.png_bytes(rpt.panel(original,mask))};extras={'engine_parameters.json':json.dumps(b.json_safe(result['params']),indent=2).encode()}
    display=st.selectbox('Historical Object display',['None','Object mask and POD metrics','Optical versus mask curves'],key='object_diagnostic')
    if display!='None':
        axes=[c for c in ('analysisIndex_1based','time_s','matlabTime_s') if c in table]
        axis=st.selectbox('Object plot axis',axes,key='object_axis')
        factory=engine.plot_object_mask_metrics_pod if display.startswith('Object') else engine.plot_normalized_syllectograms
        figures['object_diagnostic.png']=rpt.existing_figure_bytes(factory,table,axis);st.image(figures['object_diagnostic.png'])
        extras['object_diagnostic.svg']=rpt.existing_figure_bytes(factory,table,axis,format='svg')
        extras['object_diagnostic.pdf']=rpt.existing_figure_bytes(factory,table,axis,format='pdf')
    api['show'](module,table,{'input_count':len(frames),'analysed_count':len(table),'configuration':result['params'],'target':target,'source_token':token,'sources':[f['metadata'] for f in frames],'reviews':saved['reviews'],'warnings':result.get('warnings',[]),'host_resources':saved.get('host_resources',{})},figures,extras)

PAGES={NAMES[0]:project,NAMES[1]:advanced,NAMES[2]:instances,NAMES[4]:pca,NAMES[5]:structures,NAMES[6]:flow,NAMES[7]:comparison,NAMES[8]:device,NAMES[9]:review,NAMES[10]:multi_sample,NAMES[11]:batch,NAMES[12]:legacy_objects}
