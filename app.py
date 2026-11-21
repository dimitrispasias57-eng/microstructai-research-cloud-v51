"""Unified, session-isolated UI. Frozen scientific modules are not edited."""
import datetime, io, json, uuid
from legacy.accepted_beta6.research.product import runtime_profile
runtime_profile.configure()
import numpy as np
import pandas as pd
import streamlit as st
import backend as b
import reporting as rpt
import extensions as ext
import extended_ui as xui

st.set_page_config(page_title='MicroStructAI · Research Cloud',page_icon='🔬',layout='wide')
MODES=['Home','RBC Morphology','RBC Aggregation / Spatial Structure','Temporal / Video Analysis','Deformation / Dynamic Morphology','General Structural Analysis','Ground-Truth / Validation','Reports / Exports','Scientific foundation','Evidence & limitations']
BADGES={MODES[1]:'SUPERVISED RUO — INTERNALLY VALIDATED',MODES[2]:'RESEARCH / EXPLORATORY STRUCTURAL ANALYSIS',MODES[3]:'RESEARCH USE — TEMPORAL STRUCTURAL ANALYSIS',MODES[4]:'IMAGE-DERIVED DEFORMATION / DYNAMIC MORPHOLOGY',MODES[5]:'GENERAL RESEARCH STRUCTURAL ANALYSIS',MODES[6]:'RESEARCH REFERENCE COMPARISON — REFERENCE QUALITY IS USER-SUPPLIED'}
LIMITS={MODES[1]:'Centred native 96 × 96 grayscale DPC single-cell crops only. Internal evidence is MIXED; mandatory human approval. Not universal segmentation.',MODES[2]:'Image-derived RBC aggregation/spatial-structure metrics. Independent rheological validation has not been established (NO_DEFENSIBLE_PAIRING). Not RheoScan equivalence, diagnosis or yield stress.',MODES[3]:'Recording-relative structural trajectories. Frames are not independent cells. Event-relative timing requires documented time, event and baseline support.',MODES[4]:'Not validated as direct membrane stiffness or mechanical deformability measurement. Supplied same-cell identity is required; no automatic tracking or mechanical inference.',MODES[5]:'Connected regions are not recognised physical objects. Generic experimental threshold proposals require review. Xi is not automatically particle or aggregate diameter.',MODES[6]:'Agreement with a supplied reference is not automatically independent segmentation ground truth. Native-pixel comparisons only; no silent registration or repair.'}
MODES+=xui.NAMES
BADGES.update(xui.BADGES);LIMITS.update(xui.LIMITS)
if 'session_id' not in st.session_state:st.session_state.session_id=str(uuid.uuid4())
if 'exports' not in st.session_state:st.session_state.exports={}
if 'epoch' not in st.session_state:st.session_state.epoch=0
st.sidebar.title('MicroStructAI')
st.sidebar.caption('Research Cloud v5.1-rc1 · Research Use Only')
mode=st.sidebar.radio('Analysis mode',MODES,key='navigation')
st.sidebar.caption('€0 native inference · no external AI API. Uploads go to the Streamlit host, not an inference provider. Use authorised, non-identifiable research data only.')
st.sidebar.warning('Research Use Only. Do not upload protected health information (PHI), patient identifiers or confidential data.')
if st.sidebar.button('Clear this session'):
    epoch=st.session_state.epoch+1;st.session_state.clear();st.session_state.epoch=epoch;st.rerun()
st.sidebar.caption('Download before refresh/closing. Sessions are not permanent storage.')

@st.cache_resource
def model():return b.native.load_model()

def signature(value):return b.native.sha_bytes(json.dumps(value,sort_keys=True,default=str).encode())
def payloads(uploads):return [(u.name,u.getvalue()) for u in uploads or []]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
@st.cache_data
def deployment_identity():
    """Public reproducibility metadata only; no environment or credentials."""
    import subprocess
    identity={'version':b.VERSION,'entry_point':'app.py','entry_point_sha256':b.native.sha_bytes((b.ROOT/'app.py').read_bytes())}
    try:
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=b.ROOT,capture_output=True,text=True,check=True,timeout=3).stdout.strip()
        if len(commit)==40 and all(c in '0123456789abcdef' for c in commit):identity['git_commit']=commit
    except (OSError,subprocess.SubprocessError):identity['git_commit']=None
    return identity
def header(name):
    st.header(name);st.caption(BADGES[name]);st.warning(LIMITS[name])
def upload(label,key,multiple=True,types=None):
    return st.file_uploader(label,type=types or ['png','jpg','jpeg','bmp','tif','tiff'],accept_multiple_files=multiple,key=key+str(st.session_state.epoch))
def export_view(module,frame,metadata,figures=None,extras=None):
    metadata={'session_id':st.session_state.session_id,'deployment':deployment_identity(),'status':BADGES[module],'project':st.session_state.get('project_label',''),'project_notes':st.session_state.get('project_notes',''),**metadata}
    package=rpt.bundle(module,frame,metadata,LIMITS[module],figures,extras)
    st.session_state.exports[module]=package
    st.subheader('Export')
    a,c,d=st.columns(3)
    a.download_button('Results CSV',package['csv'],module.split('/')[0].strip().replace(' ','_')+'.csv','text/csv',key=module+'_csv')
    c.download_button('Research report (HTML)',package['html'],'microstructai_report.html','text/html',key=module+'_html')
    d.download_button('Figures, results and provenance (ZIP)',package['zip'],'microstructai_bundle.zip','application/zip',key=module+'_zip')
def show_table_plot(module,frame,metadata,figures=None,extras=None,*,plot_x=None,plot_default=None,plot_labels=None):
    st.dataframe(frame,hide_index=True)
    numeric=[c for c in frame if c!=plot_x and pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    chosen=st.multiselect('Plot metrics (check units before combining)',numeric,default=plot_default if plot_default is not None else numeric[:1],key=module+'_plot'+('_time' if plot_x else ''))
    figures=dict(figures or {})
    if chosen:
        fig=rpt.chart_bytes(frame,chosen,x=plot_x,**(plot_labels or {}));st.image(fig);figures['metric_plot.png']=fig
    export_view(module,frame,metadata,figures,extras)

def morphology_page():
    header(MODES[1]);st.caption('Upload → generate → inspect each mask → approve/exclude → export. Maximum 50 crops, 1 MiB each, 16 MiB total.')
    st.info('Internal blinded audit: 43/50 GOOD · 6/50 MINOR · 1/50 MAJOR · 98% GOOD+MINOR. QC missed the major error. Human approval remains mandatory for every observation.')
    records=st.session_state.get('native_records',[])
    with st.expander('Synthetic software demonstration — not biological evidence'):
        from demo import synthetic_image
        st.download_button('Download native synthetic fixture',b.native.png_bytes(synthetic_image()),'native_synthetic.png','image/png')
        if st.button('Load native synthetic demo',disabled=bool(records)):
            image=synthetic_image();records=[{'id':str(uuid.uuid4()),'name':'SYNTHETIC native demo','source_sha256':b.native.sha_bytes(b.native.png_bytes(image)),'image':image,'result':b.native.infer(image,model()),'revisions':[]}];st.session_state.native_records=records
    files=upload('Original native 96 × 96 DPC crops','native_upload')
    auth=st.checkbox('These are centred native 96 × 96 DPC single-cell crops in the tested domain; I am authorised to upload them.',key='native_domain')
    if st.button('Generate native masks',disabled=not(files and auth) or bool(records),type='primary'):
        values=payloads(files)
        if len(values)>50 or sum(len(p) for _,p in values)>16*1024**2:raise ValueError('Maximum 50 crops and 16 MiB total.')
        decoded=[(n,p,b.native.decode_image(p)) for n,p in values]
        progress=st.progress(0);records=[]
        for i,(n,p,a) in enumerate(decoded):
            records.append({'id':str(uuid.uuid4()),'name':n,'source_sha256':b.native.sha_bytes(p),'image':a,'result':b.native.infer(a,model()),'revisions':[]});progress.progress((i+1)/len(decoded))
        st.session_state.native_records=records
    if not records:return
    st.info('Batch retained. Export then clear this session to load another native batch.')
    selected=st.selectbox('Observation',range(len(records)),format_func=lambda i:f'{i+1}. {records[i]["name"]}',key='native_case')
    r=records[selected]
    if st.checkbox('Show original only',key='native_original'):st.image(r['image'],width=384)
    else:st.image(b.native.panel(r['image'],r['result']['mask']),caption='Original · mask · overlay · contour',width=1000)
    st.write('QC / routing:',r['result']['details']['status'],r['result']['details']['qc_flags'])
    with st.form('native_review_'+r['id']):
        reviewer=st.text_input('Reviewer identity');quality=st.selectbox('Quality',['Choose explicitly','GOOD','MINOR_ERROR','MAJOR_ERROR','REVIEWER_UNCERTAIN'])
        notes=st.text_area('Observations (optional)');confirmed=st.checkbox('I personally inspected the original, mask and contour.')
        if st.form_submit_button('Save native review'):
            if not reviewer.strip() or quality=='Choose explicitly' or not confirmed:st.error('Reviewer, explicit quality and personal inspection are required.')
            elif quality in ('GOOD','MINOR_ERROR') and not r['result']['details']['mask_available']:st.error('A no-mask observation cannot be accepted.')
            else:
                review={'reviewer':reviewer.strip(),'quality':quality,'notes':notes,'personally_inspected':True,'saved_utc':now(),'revision_id':str(uuid.uuid4())}
                r['review']=review;r['revisions'].append(review);st.success('Review saved in this session.')
    frame=pd.DataFrame(b.native_exports.rows(records));accepted=int((frame.measurement_status=='HUMAN_REVIEWED_RESEARCH_MEASUREMENT').sum())
    st.metric('Human-approved usable / uploaded observations',f'{accepted} / {len(records)}')
    st.caption('Unreviewed, rejected and uncertain morphology remains withheld. Repeated observations are not independent cells.')
    distribution={}
    available=[c for c in frame if pd.api.types.is_numeric_dtype(frame[c]) and c not in ('position','index')]
    if accepted and available:
        metric=st.selectbox('Morphology distribution',available,key='native_distribution')
        distribution['morphology_distribution.png']=rpt.distribution_bytes(frame.loc[frame.measurement_status=='HUMAN_REVIEWED_RESEARCH_MEASUREMENT'],metric)
        st.image(distribution['morphology_distribution.png'])
    show_table_plot(MODES[1],frame,{'input_count':len(records),'analysed_count':accepted,'reviewed_coverage':accepted/len(records),'science':b.native.verify_science()},
                    {'selected_panels.png':b.native.png_bytes(b.native.panel(r['image'],r['result']['mask'])),**distribution},
                    {'native_review_backup_and_masks.zip':b.native_exports.zip_bytes(records)})

def spatial_page(kind):
    is_rbc=kind==MODES[2];is_clot=kind==xui.NAMES[3];prefix='rbc' if is_rbc else 'clot' if is_clot else 'general'
    header(kind)
    with st.expander('Synthetic upload fixtures — software testing only'):
        vals=b.fixture_frames()
        for i,(name,data) in enumerate(vals[:1]):st.download_button('Download one synthetic image',data,name,'image/png',key=prefix+'_fixture')
        demo=st.checkbox('Use eight synthetic images instead of uploads',key=prefix+'_demo')
    start=st.radio('Start with',['Original image / image sequence','Supplied binary masks — advanced/trusted'] if not is_rbc else ['Original RBC recording / image sequence'],key=prefix+'_start')
    transport=st.radio('Upload format',['Individual image files','ZIP of image files / recordings'],key=prefix+'_transport')
    vals=[];archive_identity=None
    if transport.startswith('Individual'):
        files=upload('Images (natural filename order; no resizing)',prefix+'_upload');vals=payloads(files)
    else:
        archive=upload('ZIP: one recording per folder; no video containers',prefix+'_archive',False,['zip'])
        if archive:
            groups=ext.archive_groups(archive.getvalue());group=st.selectbox('Recording to analyse (no merging across folders)',list(groups),key=prefix+'_archive_group')
            vals=[(f.name,f.getvalue()) for f in groups[group]];archive_identity={'sha256':b.native.sha_bytes(archive.getvalue()),'selected_group':group,'other_groups_not_analysed':[g for g in groups if g!=group]}
            st.caption('Only the explicitly selected complete recording is analysed. Other folders are not silently pooled or counted as processed.')
    if demo:vals=b.fixture_frames();archive_identity=None
    st.caption('200 files maximum; 32 MiB compressed and 16 million channel-pixels total. Larger complete recordings: LOCAL/ADVANCED ONLY.')
    if not vals:
        st.session_state.pop(prefix+'_result',None);st.session_state.pop(prefix+'_table',None);st.session_state.exports.pop(kind,None)
        return
    frames=b.decode_batch(vals);first=frames[0]['image']
    st.image(b.display_u8(first),caption='Original source; channels retained',width=500)
    config={'supplied':start.startswith('Supplied'),'archive':archive_identity}
    study=st.text_input('Study / recording label (no personal identifiers)',key=prefix+'_study')
    if is_clot:
        config['phase']=st.selectbox('User-declared clot phase',['Fibrin','Whole-clot region','Other clot phase','RBC-in-clot'],key='clot_phase')
        config['phase_evidence']=st.text_input('Acquisition / stain / channel evidence for this phase',key='clot_evidence')
        config['clot_context']=st.checkbox('This is an actual clot sample and the selected phase is distinguishable; colour alone is not evidence.',key='clot_context')
        if not config['phase_evidence'].strip() or not config['clot_context']:st.info('Confirm clot context and phase evidence before creating proposals. This does not count as mask review.');return
    if is_rbc:
        params=b.rbc_defaults()
        preset=st.selectbox('Acquisition preset',list(b.rbc_engine().DEFAULT_CAMERA_PRESETS),format_func=lambda s:s.replace('MATLAB-compatible RBC legacy','Historical reference acquisition'),key='rbc_preset')
        params.update(b.rbc_engine().DEFAULT_CAMERA_PRESETS[preset])
        with st.expander('Existing acquisition preset import / export'):
            from modules.calibration import load_camera_preset_json, camera_preset_json
            preset_file=upload('Existing camera / acquisition preset JSON','rbc_preset_json',False,['json'])
            if preset_file:
                if preset_file.size>65536:raise ValueError('Preset exceeds 64 KiB.')
                imported=load_camera_preset_json(preset_file.getvalue())
                allowed=set(params)
                if set(imported['settings'])-allowed:raise ValueError('Preset contains settings outside this frozen recording interface: '+', '.join(sorted(set(imported['settings'])-allowed)))
                if any(isinstance(v,(list,dict)) or v is None for v in imported['settings'].values()):raise ValueError('Preset values must be scalar settings, not paths or nested payloads.')
                if st.checkbox('Use this documented preset (invalidates previous analysis and review)',key='rbc_preset_apply'):params.update(imported['settings'])
                st.json(imported)
            st.caption('This imports existing settings only. It does not calibrate or tune the pipeline. Controls below remain explicit overrides; the final engine parameters are in each export.')
        method=st.selectbox('Gap proposal workflow',['Seed-and-grow gap detection','Adaptive gap detection — no POD removal'],key='rbc_method')
        params['rbcAnalysisMethod']='Standard RBC Gap Mode' if method.startswith('Seed') else 'Conservative RBC Gap Mode'
        params['trackDark']=st.checkbox('Target dark regions (not automatically plasma gaps)',key='rbc_dark')
        params['matlab_fps']=st.number_input('Recorded frame rate (frames/s)',min_value=.001,value=1.,key='rbc_fps')
        params['cropFrac']=st.slider('Centred field fraction',.1,1.,float(params['cropFrac']),key='rbc_crop')
        with st.expander('Existing preprocessing / POD diagnostics — advanced'):
            st.caption('These expose existing engine controls, not a new calibration. Experimental POD residuals can change masks; primary/no-POD results remain distinguishable in the exports.')
            params['removeModes']=st.number_input('Leading POD modes removed',min_value=0,max_value=10,value=int(params.get('removeModes',1)),key='rbc_pod_modes')
            if st.checkbox('Use experimental POD-removed primary branch',key='rbc_pod_primary'):
                params['rbcAnalysisMethod']='Experimental POD-removed RBC Mode'
        timing=st.checkbox('I confirm natural filename order and uniform timing for this recording.',key='rbc_time')
        st.caption('Frozen start detection, QC, score normalisation and formulas are retained. Original parameter names are preserved in exports for reproducibility.')
        config={'params':params,'timing_confirmed':timing,'archive':archive_identity}
        st.download_button('Export current acquisition settings',camera_preset_json('User-stated acquisition',params,'not recalibrated',False).encode(),'acquisition_preset.json','application/json')
    else:
        config['representation']=st.selectbox('Structural representation',['Phase / foreground','Objects / connected regions','Network'],key=prefix+'_rep')
        config['target']=st.selectbox('Target example',['Custom structure','Cells','Particles','Droplets','Pores / gaps','Bubbles','Fibres','Cracks','Food phases'],key=prefix+'_target')
        config['channel']=st.selectbox('Analysis channel',['Grayscale / luminance']+(['Red','Green','Blue'] if first.ndim==3 else []),key=prefix+'_channel')
        if first.ndim==3:
            cols=st.columns(3)
            for j,c in enumerate(cols):c.image(b.display_u8(first[...,j]),caption=['Red','Green','Blue'][j])
        st.image(b.display_u8(b.plane(first,config['channel'])),caption='Selected channel preview',width=500)
        config['polarity']=st.selectbox('Target intensity',['bright','dark'],key=prefix+'_polarity')
        config['threshold']=st.selectbox('Experimental threshold method',['otsu','adaptive','percentile'],key=prefix+'_threshold')
        config['min_area']=st.number_input('Minimum connected-region area (pixels²)',min_value=0,max_value=100000,value=40,key=prefix+'_min')
        config['fill_holes']=st.checkbox('Fill holes in proposal (changes the measured foreground)',key=prefix+'_fill')
        config['illumination']=st.checkbox('Correct uneven illumination — experimental',value=False,key=prefix+'_illum')
        config['background_scale']=st.number_input('Background scale (pixels)',min_value=1.,max_value=500.,value=25.,disabled=not config['illumination'],key=prefix+'_bg')
        st.caption('Background scale should exceed the structure of interest. Correction affects only the selected channel proposal; original RGB is retained.')
        if st.checkbox('Use rectangular ROI',key=prefix+'_roi'):
            if st.checkbox('Draw rectangle on original image',key=prefix+'_draw'):
                import base64
                from streamlit.components.v1 import declare_component
                component=declare_component('microstructai_rectangle_roi',path=str(b.ROOT/'roi_component'))
                identity=signature([frames[0]['sha256'],first.shape])
                value=component(image='data:image/png;base64,'+base64.b64encode(b.native.png_bytes(b.display_u8(first))).decode(),width=int(first.shape[1]),height=int(first.shape[0]),input_id=identity,roi=None,key=prefix+'_canvas',default=None)
                if value is not None and value!=st.session_state.get(prefix+'_last_draw'):
                    rectangle=ext.drawn_roi(value,first.shape,identity)
                    for name,v in zip(('x','y','w','h'),rectangle):st.session_state[prefix+'_'+name]=v
                    st.session_state[prefix+'_last_draw']=value
                st.caption('Drawing fills the numeric coordinates below in native source pixels. Changes invalidate the proposal and review. This v5 route crops before proposing; it is not the older measurement-only ROI workflow.')
            cs=st.columns(4);h,w=first.shape[:2]
            config['roi']=[cs[0].number_input('x (zero-based)',0,w-2,0,key=prefix+'_x'),cs[1].number_input('y (zero-based)',0,h-2,0,key=prefix+'_y'),cs[2].number_input('Width',2,w,w,key=prefix+'_w'),cs[3].number_input('Height',2,h,h,key=prefix+'_h')]
        if config['representation']=='Objects / connected regions':st.info('Connected components are not cell identities. Touching objects are not automatically separated.')
    sig=signature({'files':[(r['name'],r['sha256']) for r in frames],'config':config,'study':study})
    old=st.session_state.get(prefix+'_result')
    if old and old['signature']!=sig:
        st.session_state.pop(prefix+'_result',None);st.session_state.pop(prefix+'_table',None);st.session_state.exports.pop(kind,None);old=None
        st.info('Inputs/settings changed. Previous results and review were invalidated; run again.')
    if st.button('Run frozen RBC analysis' if is_rbc else 'Create experimental mask proposal',key=prefix+'_run',type='primary'):
        if is_rbc and not timing:raise ValueError('Confirm acquisition order and timing before recording analysis.')
        result=b.analyze_rbc(frames,params) if is_rbc else b.propose_general(frames,config)
        old={'signature':sig,'result':result,'review':{},'created_utc':now()};st.session_state[prefix+'_result']=old
    if not old:return
    if is_rbc:
        previews=old['result']['previews'];table=old['result']['metrics'].copy()
        records=[{'name':p['filename'],'mask':p['mask'],'image':p.get('camera_crop_gray',p['image'])} for p in previews]
        st.caption('Engine QC exclusions and trusted-score columns remain authoritative. Review below is additional provenance, never an automatic QC override.')
        for w in old['result'].get('warnings',[])[:8]:st.warning(str(w))
    else:records=old['result'];table=None
    idx=st.selectbox('Inspect observation',range(len(records)),format_func=lambda i:records[i]['name'],key=prefix+'_inspect')
    rec=records[idx];a=rec.get('original_roi',rec.get('image'));mask=np.asarray(rec['mask'],bool)
    if st.checkbox('Original only — hide proposal',key=prefix+'_original'):st.image(b.display_u8(a),width=600)
    else:st.image(rpt.panel(a,mask),caption='Original · proposed mask · overlay · contour',width=1100)
    if is_rbc:
        with st.expander('Existing preprocessing and removed-region diagnostics'):
            source_preview=previews[idx]
            for key in ('wz','removed_gap_mask','v4_diagnostic_image'):
                value=source_preview.get(key)
                if isinstance(value,np.ndarray) and value.ndim in (2,3):st.image(b.display_u8(value),caption=key+' — display only, no metric recalculation',width=600)
            st.json(b.json_safe(source_preview.get('provenance',{})))
    if not is_rbc and config['illumination']:st.image(b.display_u8(rec['corrected']),caption='Selected-channel preprocessing preview',width=500)
    review_mode=st.radio('Review workflow',['Inspect every mask','Batch-reviewed — all masks personally inspected']+(['Unreviewed exploratory analysis'] if is_rbc or is_clot else []),key=prefix+'_review_mode')
    reviewer=st.text_input('Reviewer identity (for reviewed records)',key=prefix+'_reviewer')
    if review_mode.startswith('Inspect'):
        confirmed=st.checkbox('I inspected this original and mask and accept the stated foreground.',key=prefix+'_confirm_'+str(idx))
        if st.button('Accept this mask',key=prefix+'_accept'):
            if not confirmed or not reviewer.strip():st.error('Confirm inspection and enter reviewer.')
            else:old['review'][idx]={'reviewer':reviewer,'saved_utc':now()};st.success('Accepted this observation.')
    elif review_mode.startswith('Batch'):
        confirmed=st.checkbox('I personally inspected every mask in this batch; this is not automatic acceptance.',key=prefix+'_allconfirm')
        if st.button('Record batch review',key=prefix+'_all'):
            if not confirmed or not reviewer.strip():st.error('Confirm all-mask inspection and enter reviewer.')
            else:old['review']={i:{'reviewer':reviewer,'saved_utc':now()} for i in range(len(records))};st.success('Batch review recorded.')
    st.caption(f'Reviewed {len(old["review"])} / {len(records)}. No manual correction tool is implied; upload corrected masks in General Structural Analysis if needed.')
    if not review_mode.startswith('Unreviewed') and len(old['review'])!=len(records):return
    if not is_rbc:
        for i,r in enumerate(records):r['review']='HUMAN_REVIEWED' if i in old['review'] else 'UNREVIEWED_EXPLORATORY'
        measurement_token=signature([sig,sorted(old['review'])])
        if old.get('measurement_token')!=measurement_token:
            old['measurements']=b.measure_general(records);old['measurement_token']=measurement_token
        table,objects=old['measurements'];table=table.copy()
    table['human_review_status']=['HUMAN_REVIEWED' if i in old['review'] else 'UNREVIEWED_EXPLORATORY' for i in range(len(records))]
    metadata={'study':study,'configuration':config,'input_count':len(frames),'analysed_count':len(table),'reviews':old['review'],'sources':[r['metadata'] for r in frames],'created_utc':old['created_utc']}
    extras={'connected_regions.csv':rpt.csv_bytes(objects)} if not is_rbc else {'engine_parameters.json':json.dumps(b.json_safe(old['result'].get('params',params)),indent=2).encode()}
    if not is_rbc:
        with st.expander('Gap / pore measurements and optional generic structure scores'):
            st.caption('Gaps are spaces between structures; pores are voids within material. Both require the explicitly selected 2-D foreground, not automatically the background or 3-D porosity.')
            slopes=ext.area_slopes(records);st.dataframe(slopes,hide_index=True);extras['area_distribution_slopes.csv']=rpt.csv_bytes(slopes)
            st.caption('Gap / pore area-distribution slope: unchanged 12-log-bin fit, not a temporal slope or proof of a power law. Report fit R² and occupied bins.')
            enable_scores=st.checkbox('Enable experimental gSASI / gAggIndex',key=prefix+'_generic_scores')
            if enable_scores:
                st.warning('Generic scores are group-relative, not RBC SASI / GapScore / PTI / AggIndex and not validated aggregation. Changing group membership can change all scores. Only reviewed complete cases are eligible.')
                group_id=st.text_input('Comparable reference group ID',key=prefix+'_score_group')
                comparable=st.checkbox('Target, sampling scale, ROI and processing are comparable across this group.',key=prefix+'_comparable')
                if comparable and group_id.strip():
                    scores=ext.generic_scores(records,group_id,comparable);st.dataframe(scores['frame']);extras['generic_scores.csv']=rpt.csv_bytes(scores['frame']);extras['generic_scores_provenance.json']=json.dumps(b.json_safe(scores['provenance']),indent=2).encode()
    figures={'selected_panels.png':b.native.png_bytes(rpt.panel(a,mask))}
    if is_rbc:
        with st.expander('Historical mask / optical / start diagnostics'):
            diagnostic=st.selectbox('Existing diagnostic display',['None','Mask and POD metrics','Optical versus mask curves','Automatic-start diagnostics'],key='rbc_diagnostic')
            if diagnostic!='None':
                axis=st.selectbox('Diagnostic axis',[c for c in ('analysisIndex_1based','time_s','matlabTime_s') if c in table],key='rbc_diagnostic_axis')
                engine=b.rbc_engine()
                factories={'Mask and POD metrics':engine.plot_rbc_mask_metrics_matlab_v5,'Optical versus mask curves':engine.plot_normalized_syllectograms,'Automatic-start diagnostics':engine.plot_auto_start_diagnostics}
                args=(table,) if diagnostic=='Automatic-start diagnostics' else (table,axis)
                st.caption('Unchanged historical display of stored results. Raw and QC-filtered curves remain distinct; displayed reliability lines are historical rules, not new validated cutoffs. Optical intensity is not interchangeable with gap area or mechanical aggregation. Start diagnostics use their original stored-axis selection.')
                for fmt in ('png','svg','pdf'):
                    blob=rpt.existing_figure_bytes(factories[diagnostic],*args,format=fmt)
                    if fmt=='png':figures['historical_diagnostic.png']=blob;st.image(blob)
                    else:extras['historical_diagnostic.'+fmt]=blob
        with st.expander('QC-marked descriptor trajectories — existing implementation'):
            from modules import qc_plotting as qp
            specs=qp.available_metrics(table)
            chosen_qc=st.selectbox('QC-marked metric',['None']+[s.key for s in specs],key='rbc_qc_metric')
            if chosen_qc!='None':
                axis=st.selectbox('Stored axis',[c for c in ('analysisIndex_1based','time_s','matlabTime_s') if c in table],key='rbc_qc_axis')
                excluded=st.checkbox('Display excluded observations separately (never reconnect gaps)',key='rbc_qc_excluded')
                figures['qc_marked_trajectory.png']=rpt.existing_figure_bytes(qp.plot_qc_metric,table,chosen_qc,axis,show_excluded=excluded)
                st.image(figures['qc_marked_trajectory.png']);detail=qp.qc_details(table,chosen_qc);st.dataframe(detail)
                extras['qc_marked_values.csv']=rpt.csv_bytes(detail)
        with st.expander('Existing RBC correlation matrix / distributions'):
            if st.checkbox('Show frozen Spearman correlation matrix',key='rbc_corr'):
                from modules.plotting import plot_rbc_spearman_correlation_matrix
                trusted=st.checkbox('Only frozen main-summary frames',value=True,key='rbc_corr_trusted')
                figures['rbc_spearman.png']=rpt.existing_figure_bytes(plot_rbc_spearman_correlation_matrix,table,only_main_summary_frames=trusted)
                st.image(figures['rbc_spearman.png']);st.caption('Pairwise-complete descriptive correlations. Adjacent frames are dependent; correlation is not rheological or clinical validation.')
                for fmt in ('svg','pdf'):
                    blob=rpt.existing_figure_bytes(plot_rbc_spearman_correlation_matrix,table,only_main_summary_frames=trusted,format=fmt)
                    extras['rbc_spearman.'+fmt]=blob
            hist_columns=[c for c in table if pd.api.types.is_numeric_dtype(table[c]) and not pd.api.types.is_bool_dtype(table[c])]
            descriptor=st.selectbox('Distribution descriptor',['None']+hist_columns,key='rbc_hist')
            if descriptor!='None':figures['rbc_distribution.png']=rpt.distribution_bytes(table,descriptor);st.image(figures['rbc_distribution.png'])
    show_table_plot(kind,table,metadata,figures,extras)
    st.session_state[prefix+'_table']=table.copy()
    st.caption('Additional structural detail, topology, networks, sensitivity and gap persistence are available in Structural Detail / QC / Robustness for this session’s masks.')
    with st.expander('Exploratory PCA — existing standardised implementation'):
        candidates=b.finite_numeric_columns(table)
        selected=st.multiselect('PCA descriptors',candidates,default=candidates[:3],key=prefix+'_pca_cols')
        if old.get('pca_columns')!=selected:old.pop('pca',None)
        if st.button('Run PCA',key=prefix+'_pca'):
            old['pca']=b.run_pca(table,selected);old['pca_columns']=selected.copy()
        if 'pca' in old:
            p=old['pca']
            if p.warning:st.warning(p.warning)
            if not p.scores.empty:
                st.dataframe(p.scores);st.dataframe(p.loadings);st.write('Explained variance',p.explained_variance.to_dict())
                st.caption(f'{len(p.scores)}/{len(table)} complete rows used. PCA is descriptive, not biological classification; signs are arbitrary.')
                st.scatter_chart(p.scores,x='PC1',y='PC2')
                st.download_button('PCA scores CSV',rpt.csv_bytes(p.scores),'pca_scores.csv','text/csv',key=prefix+'_pca_csv')
                pfig=rpt.pca_figure_bytes(p.scores)
                pp=rpt.bundle(kind,p.scores,{'analysis':'Exploratory PCA','descriptors':selected,'complete_rows':len(p.scores),'input_count':len(table),'analysed_count':len(p.scores),'session_id':st.session_state.session_id},LIMITS[kind],{'pca_scores.png':pfig},{'pca_loadings.csv':rpt.pca_loadings_bytes(p.loadings),'explained_variance.json':json.dumps(b.json_safe(p.explained_variance.to_dict())).encode()})
                st.download_button('PCA report, scores and loadings (ZIP)',pp['zip'],'pca_bundle.zip','application/zip',key=prefix+'_pca_zip')

def temporal_page():
    header(MODES[3])
    st.info('Image-sequence analysis is available in RBC Aggregation / Spatial Structure and General Structural Analysis. Analyse and review the sequence there, then select its results here. A reusable video-container uploader has not yet been established by the historical audit; dataset-specific research extractors are not a supported app workflow.')
    sources=['Uploaded metric CSV','Synthetic curve — software fixture']+[f'{p} sequence results' for p in ['rbc','general'] if p+'_table' in st.session_state]
    source=st.selectbox('Trajectory source',sources,key='temp_source')
    if source.startswith('Uploaded'):
        f=upload('CSV with a descriptor and timestamps','temp_csv',False,['csv'])
        if not f:
            st.session_state.pop('temp_result',None);st.session_state.exports.pop(MODES[3],None)
            return
        data=b.rbc_temporal.parse_csv(f.getvalue())
    elif source.startswith('Synthetic'):data=b.rbc_temporal.synthetic_curve()
    else:data=st.session_state[source.split()[0]+'_table'].copy()
    st.dataframe(data.head())
    column=st.selectbox('Descriptor',list(data),index=list(data).index('gap_area_fraction') if 'gap_area_fraction' in data else 0,key='temp_col')
    axis=st.radio('Time axis',['Explicit timestamp column','Uniform recorded FPS'],key='temp_axis')
    time_col=st.selectbox('Timestamp column (seconds)',list(data),key='temp_timecol') if axis.startswith('Explicit') else None
    fps=st.number_input('Frame rate (frames/s)',min_value=.001,value=1.,key='temp_fps') if not time_col else None
    timing=st.checkbox('The time axis is documented and applies to these observations.',key='temp_timing')
    qc=st.selectbox('QC-valid column',['Not supplied']+list(data),key='temp_qc')
    with st.expander('Optional event/baseline evidence for qualified timing'):
        event=st.checkbox('Experimental t0 is known',key='temp_event');offset=st.number_input('Event offset from first observation (s)',min_value=0.,value=0.,key='temp_offset')
        initial=st.checkbox('Initial baseline is appropriate for this event',key='temp_baseline');evidence=st.text_input('Timing/baseline evidence',key='temp_evidence')
    current_sig=signature([source,data.to_json(),column,time_col,fps,timing,qc,event,offset,initial,evidence])
    if st.session_state.get('temp_signature')!=current_sig:
        st.session_state.pop('temp_result',None);st.session_state.exports.pop(MODES[3],None)
        st.session_state.temp_signature=current_sig
    if st.button('Analyse frozen temporal curve',type='primary'):
        context=b.rbc_temporal.TimingContext(time_documented=timing,event_known=event,event_offset_s=offset,initial_state_appropriate=initial,evidence=evidence)
        result=b.rbc_temporal.analyze_curve(data,column,time_column=time_col,fps=fps,qc_column=None if qc=='Not supplied' else qc,context=context)
        st.session_state.temp_result=result
    result=st.session_state.get('temp_result')
    if result:
        st.json(b.json_safe(result['qualified_timing']));table=pd.DataFrame({'time_s':result['time_axis'],'raw_value':result['raw_values'],'analysis_value':result['analysis_values'],'qc_valid':result['qc_valid']})
        st.json(b.json_safe(result['recording_relative_computational']['metrics']))
        st.caption('Plot: analysed signal versus recording time. Missing/excluded analysis values remain gaps; raw values are available separately. No sorting, resampling or smoothing is added by the plot. Timing qualification is unchanged.')
        show_table_plot(MODES[3],table,{'input_count':len(data),'analysed_count':len(table),'temporal_result':result},
                        plot_x='time_s',plot_default=['analysis_value'],
                        plot_labels={'x_label':'Recording time (s)','y_label':str(result['descriptor'])+' ('+str(result['units'])+')','title':'Temporal structural trajectory'})

def dynamics_page():
    header(MODES[4]);st.info('For original same-cell DPC crops, first use RBC Morphology and approve each mask. Otherwise supply externally reviewed binary whole-cell masks. No cell identity is inferred.')
    route=st.radio('Tracked-mask source',['Supplied reviewed masks','Approved native observations','Reviewed classical same-cell proposals','Synthetic tracked masks — software fixture'],key='dyn_route')
    masks=[];names=[]
    def invalidate_dynamic():
        st.session_state.pop('dyn_result',None);st.session_state.exports.pop(MODES[4],None)
    if route.startswith('Supplied'):
        files=upload('Same-cell masks in natural filename order','dyn_files')
        if not files:
            st.session_state.pop('dyn_result',None);st.session_state.exports.pop(MODES[4],None)
            return
        frames=b.decode_batch(payloads(files));masks=[b.binary(f['image']) for f in frames];names=[f['name'] for f in frames]
    elif route.startswith('Approved'):
        records=st.session_state.get('native_records',[])
        if not records:invalidate_dynamic();st.info('No native observations in this session.');return
        if any(r.get('review',{}).get('quality') not in ('GOOD','MINOR_ERROR') or not r['result']['details']['mask_available'] for r in records):invalidate_dynamic();st.warning('All observations must be explicitly approved; none are silently dropped.');return
        masks=[r['result']['mask'] for r in records];names=[r['name'] for r in records]
    elif route.startswith('Reviewed classical'):
        bundle=st.session_state.get('review_queue')
        if not bundle or not bundle.get('same_cell'):invalidate_dynamic();st.info('First generate and review a same-cell sequence in Experimental Proposals / Mask Review.');return
        records=list(bundle['records'].values())
        if not records or any(not r.eligible or r.decision!='ACCEPT' or len(np.unique(r.mask[r.mask>0]))!=1 for r in records):invalidate_dynamic();st.warning('Every frame must be freshly accepted with exactly one foreground instance. No automatic target selection.');return
        masks=[r.mask>0 for r in records];names=list(bundle['records'])
    else:
        y,x=np.mgrid[:96,:96];masks=[((x-48)/(20+i*.4))**2+((y-48)/(20-i*.2))**2<1 for i in range(12)];names=[f'synthetic_{i}' for i in range(12)]
    track=st.text_input('Stable same-cell track ID',key='dyn_track')
    identity=st.checkbox('I confirm every mask belongs to the same tracked cell and have reviewed its boundary.',key='dyn_identity')
    ts=st.text_area('One explicit timestamp in seconds per mask (comma separated)',value=', '.join(str(i) for i in range(len(masks))),key='dyn_times')
    optional={}
    with st.expander('Existing event / calibration / orientation controls — optional'):
        st.caption('Supply only documented experimental facts. The unchanged engine withholds kinetics when event windows, sampling or tracking are insufficient. Times are seconds; angular-speed bound is rad/s. No membrane mechanics are inferred.')
        config_text=st.text_area('Documented settings (JSON object; leave {} if unknown)',value='{}',key='dyn_optional')
        optional=json.loads(config_text)
        if not isinstance(optional,dict) or any(k in ('track_id','identity_confirmed') or k not in b.dynamics.Config.__dataclass_fields__ for k in optional):raise ValueError('Use only the listed existing optional configuration fields; track identity stays in the explicit review controls.')
        st.caption('Available fields: '+', '.join(k for k in b.dynamics.Config.__dataclass_fields__ if k not in ('track_id','identity_confirmed')))
    config=b.dynamics.Config(track_id=track,identity_confirmed=identity,**optional)
    sig=signature([route,names,[b.native.sha_bytes(m.tobytes()) for m in masks],ts,track,identity,optional])
    if st.session_state.get('dyn_signature')!=sig:st.session_state.pop('dyn_result',None);st.session_state.exports.pop(MODES[4],None)
    if st.button('Analyse tracked morphology',disabled=not identity,type='primary'):
        times=[float(x.strip()) for x in ts.split(',')]
        if len(times)!=len(masks):raise ValueError('Timestamp count must equal mask count; none are discarded.')
        result=b.dynamics.analyze_masks(times,masks,config);st.session_state.dyn_result=result;st.session_state.dyn_signature=sig
    result=st.session_state.get('dyn_result')
    if result:
        st.json(b.json_safe(result['metrics']));st.write('QC',result['qc']);table=pd.DataFrame(result['rows'])
        show_table_plot(MODES[4],table,{'input_count':len(masks),'analysed_count':len(table),'dynamic_result':result}, {'first_mask.png':b.native.png_bytes(masks[0].astype('uint8')*255)},plot_x='time_s' if 'time_s' in table else None)

def validation_page():
    header(MODES[6]);pred=upload('Prediction mask','val_pred',False);ref=upload('Reference mask','val_ref',False)
    original=upload('Optional original image — same native geometry','val_original',False)
    evidence=st.text_input('Reference provenance / annotation method',key='val_evidence')
    if not(pred and ref):
        st.session_state.pop('val_result',None)
        st.session_state.exports.pop(MODES[6],None)
        return
    a=b.binary(b.decode_batch([(pred.name,pred.getvalue())])[0]['image']);c=b.binary(b.decode_batch([(ref.name,ref.getvalue())])[0]['image'])
    source=b.decode_batch([(original.name,original.getvalue())])[0]['image'] if original else None
    if source is not None and source.shape[:2]!=a.shape:raise ValueError('Original and masks must have identical geometry; no registration or resizing.')
    sig=signature([b.native.sha_bytes(pred.getvalue()),b.native.sha_bytes(ref.getvalue()),b.native.sha_bytes(original.getvalue()) if original else None,evidence])
    if st.session_state.get('val_signature')!=sig:
        st.session_state.pop('val_result',None);st.session_state.exports.pop(MODES[6],None);st.session_state.val_signature=sig
    if st.button('Compare supplied masks',disabled=not evidence.strip(),type='primary'):
        st.session_state.val_result=b.compare_masks(a,c)
    result=st.session_state.get('val_result')
    if result is not None:
        table=pd.DataFrame([result]);st.dataframe(table)
        image=b.native.png_bytes(rpt.panel(c.astype('uint8')*255,a));st.image(image,caption='Reference · prediction · overlap view · prediction contour')
        figures={'comparison.png':image}
        if source is not None:
            figures['prediction_on_original.png']=b.native.png_bytes(rpt.panel(source,a));figures['reference_on_original.png']=b.native.png_bytes(rpt.panel(source,c))
            st.image(figures['prediction_on_original.png']);st.image(figures['reference_on_original.png'])
        export_view(MODES[6],table,{'input_count':2+bool(original),'analysed_count':1,'reference_provenance':evidence,'prediction_sha256':b.native.sha_bytes(pred.getvalue()),'reference_sha256':b.native.sha_bytes(ref.getvalue()),'original_sha256':b.native.sha_bytes(original.getvalue()) if original else None},figures)

def home():
    st.title('MicroStructAI');st.subheader('Quantitative Microscopy & RBC Image Analysis');st.caption('Research Use Only · Research Cloud v5.1-rc1 candidate')
    st.write('A modular research platform for RBC morphology, spatial organisation, temporal structure and microscopy-derived structural descriptors.')
    for start in (1,4):
        columns=st.columns(3)
        for col,name in zip(columns,MODES[start:start+3]):
            with col.container(border=True):st.subheader(name);st.caption(BADGES[name]);st.write(LIMITS[name])
    st.info('Choose an analysis mode in the sidebar. The working Native RBC Cloud v1.1 remains a separate, preserved release.')
    for name in xui.NAMES[1:]:
        with st.expander(name):st.caption(BADGES[name]);st.write(LIMITS[name])

try:
    b.verify_vendors();ext.verify_helpers()
    # Re-publish only after this page's current inputs/review pass its gates.
    # Early returns (removed input, withdrawn consent) must not leave a stale export.
    if mode in MODES[1:7] or mode in xui.PAGES or mode==xui.NAMES[3]:
        st.session_state.exports.pop(mode,None)
    if mode==MODES[0]:home()
    elif mode==MODES[1]:morphology_page()
    elif mode in (MODES[2],MODES[5],xui.NAMES[3]):spatial_page(mode)
    elif mode in xui.PAGES:xui.PAGES[mode]({'header':header,'upload':upload,'signature':signature,'show':show_table_plot,'export':export_view,'now':now})
    elif mode==MODES[3]:temporal_page()
    elif mode==MODES[4]:dynamics_page()
    elif mode==MODES[6]:validation_page()
    elif mode==MODES[7]:
        st.header(mode);st.caption('Only packages created in this browser session are listed. No shared output directory.')
        if not st.session_state.exports:st.info('Run a module and complete its required review to create an export.')
        for name,package in st.session_state.exports.items():
            st.subheader(name);st.json(b.json_safe(package['metadata']));st.download_button('Download '+name,package['zip'],name.split('/')[0].strip().replace(' ','_')+'.zip','application/zip',key='hub_'+name)
    elif mode=='Scientific foundation':
        st.header('Do not extract more certainty from an image than the observation supports.')
        st.write('The observation-limits research distinguishes measured projected appearance from unobserved three-dimensional shape, pose and mechanics. A 2-D contour alone does not identify membrane stiffness or mechanical deformability.')
        st.write('QC detects specified computational problems, not every biological error. Abstentions and excluded observations remain visible. Human review is mandatory for native cell masks. Coverage uses all uploaded observations; repeated frames are not independent cells.')
        st.write('Source hashes, settings and review status travel with exports. Research comparisons do not create clinical evidence.')
    else:
        st.header('Evidence & limitations')
        st.subheader('Native model — live free inference')
        st.write('Internal test: 454 cells / 5 held-out participants from one previously used dataset. Final refined participant-macro Dice 0.94571; IoU 0.90175. Public algorithmic references, not manual ground truth. Blinded audit: 43 GOOD, 6 MINOR_ERROR, 1 MAJOR_ERROR (49/50 acceptable). QC missed the major error; every mask requires human approval. Internal evidence remains MIXED.')
        st.subheader('Astra-M2 — externally evaluated historical reference workflow')
        st.write('No live Astra inference here. External erythroSight: 200 cases / 20 repository-identified participants; 106 masks (53% coverage), 94 abstentions. Audited masks: 49 GOOD, 1 MINOR_ERROR, 0 MAJOR_ERROR of 50. Mean reference Dice 0.949934; IoU 0.905625 among available masks. MIXED due selective coverage. Different crops/selections prevent a paired native-versus-Astra superiority claim.')
        st.subheader('Other modules')
        st.write('Aggregation/rheology pairing: NO_DEFENSIBLE_PAIRING. Classical descriptors, temporal trajectories, dynamics and PCA have software/synthetic evidence, not equivalent clinical validation. Dynamic morphology is not direct mechanical measurement. Supplied reference quality is the user’s responsibility.')
        st.write('LOCAL/ADVANCED ONLY: oversized recordings, historical external-API M2 and authenticated research harnesses. Video-container transport and unsupported research branches are not claimed as working product features; supplied tracks do not imply automatic cell identification. No paid or external inference API is configured.')
        st.json({'version':b.VERSION,'vendor_files_verified':b.verify_vendors(),'native_model':b.native.verify_science()})
        st.caption('Running deployment — public reproducibility metadata')
        st.json(deployment_identity())
except Exception as exc:
    st.error(f'This module could not complete: {exc}')
    st.caption('Other modules remain available. No source data was silently resized, removed or repaired. Correct the input or use the preserved local workflow.')
