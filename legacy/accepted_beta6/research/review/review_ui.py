"""Separate local review candidate. Run via run_local.py, not the hosted entry."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import base64
import html
import json
import time
import streamlit as st
import streamlit.components.v1 as components
from review_core import *

EXPERIMENTAL_LABEL = 'EXPERIMENTAL — REVIEW REQUIRED'
EXPERIMENTAL_MESSAGE = ('Automatic segmentation is experimental and domain-dependent. '
                        'Review the proposed mask before using quantitative results.')
HOSTED_BATCH_GUIDANCE = ('Recommended starting batch: 10–25 small cases. '
                         '50–100 small cases are SUPPORTED WITH PERFORMANCE LIMITATIONS: '
                         'loading and review-page updates may be slower. '
                         'The 100-case ceiling is not a speed guarantee; byte/pixel caps may '
                         'be reached sooner. Analysis still runs in chunks of ten.')

def synthetic_inputs(kind='binary'):
    yy,xx=np.mgrid[:80,:80]
    labels=np.zeros((80,80),np.uint16)
    labels[(xx-30)**2+(yy-39)**2<12**2]=7
    labels[(xx-58)**2+(yy-52)**2<8**2]=900
    image=np.full((80,80,3),235,np.uint8);image[labels>0]=[100,55,75]
    mask=labels if kind=='instances' else (labels>0).astype(np.uint8)*255
    return png(image),png(mask)

def main(hosted=False, workflow=None):
    if not hosted: st.set_page_config(page_title='MicroStructAI — Mask Review',layout='wide')
    if hosted: st.session_state.review_generation=st.session_state.get('_hosted_widget_generation',0)
    if 'review_generation' not in st.session_state: st.session_state.review_generation=0
    if 'review_session' not in st.session_state: st.session_state.review_session=ReviewSession()
    if not hosted and st.sidebar.button('Reset this review session',key='reset_review'):
        generation=st.session_state.review_generation+1
        st.session_state.clear();st.session_state.review_generation=generation;st.rerun()
    g=st.session_state.review_generation
    key=lambda s:f'review_{g}_{s}'
    session=st.session_state.review_session
    st.title('MicroStructAI · Mask review')
    st.caption(PRODUCT_VERSION+' · '+VERSION+' · '+DELIVERY)
    st.info('Trusted workflow: USER-SUPPLIED MASK or USER-REVIEWED / ACCEPTED MASK → quantitative analysis.')
    st.caption('Acceptance records an internal user decision; it does not make a mask ground truth, independently validate it, or prove biological identity. Research use only.')
    st.sidebar.write('100 cases/queue · 10/page · 8 MiB/file · 1 million pixels/file · 32 MiB retained source bytes / 8 million combined decoded pixels. No resizing. Larger images may reach these caps before 100 cases.')
    st.sidebar.caption('Ten retained replacements/case maximum; replacements count toward the byte cap. Analysis runs in chunks of ten, with a 60-second budget checked between files. This is not a hard per-file timeout.')
    st.sidebar.caption('Uploads are processed on the hosting server. Do not upload confidential or identifiable data.' if hosted else 'Localhost only. Nothing is uploaded to live apps.')
    if hosted:
        with st.sidebar.expander('Hosted batch size and performance'):
            st.write(HOSTED_BATCH_GUIDANCE)
            st.caption('Counts of 10, 25, 50 and 100 small artificial cases completed in hosted beta.5 testing. This is a software capacity check, not a guarantee for research images or concurrent users. Review each case; do not bypass review to speed up a batch.')

    st.subheader('1 · Choose inputs')
    routes=['Automatic proposal','User-supplied mask','External automatic proposal']
    if workflow in ('SUPPLIED MASK','IMAGE + SUPPLIED MASK'): routes=['User-supplied mask']
    elif workflow=='AUTOMATIC SEGMENTATION + REVIEW':routes=['Automatic proposal','External automatic proposal']
    route=st.radio('Mask source',routes,horizontal=True,key=key('route'))
    if route in ('Automatic proposal','External automatic proposal'):
        st.warning('**'+EXPERIMENTAL_LABEL+'**\n\n'+EXPERIMENTAL_MESSAGE)
    adapter=st.selectbox('Fixed segmentation adapter',['PathOlOgics v0.1','RedTell v0.2'],key=key('adapter')) if route=='Automatic proposal' else None
    kind=st.selectbox('Mask interpretation',['binary','instances'],key=key('kind')) if route!='Automatic proposal' else ('binary' if adapter=='PathOlOgics v0.1' else 'instances')
    external_algorithm=None
    if route=='External automatic proposal':
        st.caption('Import an existing automatic mask as UNREVIEWED. This is the generic adapter boundary; no new segmentation algorithm is inferred.')
        alg_name=st.text_input('External algorithm name',key=key('external_name'))
        alg_version=st.text_input('External algorithm version',key=key('external_version'))
        external_algorithm={'name':alg_name,'version':alg_version,'scope':'User-declared external automatic proposal'}
    choices=['Upload files','First-party synthetic demo'] if hosted else ['First-party synthetic demo','Upload files']
    if (HERE/'demo/pathologics_manifest.json').exists() and adapter=='PathOlOgics v0.1':choices.append('Consumed PathOlOgics development demonstrations')
    source=st.radio('Input source',choices,key=key('source'))
    images={};masks={};demo_info=[]
    if source=='First-party synthetic demo':
        image,mask=synthetic_inputs(kind)
        n=st.number_input('Synthetic cases',1,MAX_BATCH,1,key=key('demo_count'))
        for i in range(n):
            images[f'synthetic_{i+1}.png']=image
            if route!='Automatic proposal':masks[f'synthetic_{i+1}_mask.png']=mask
        st.caption('First-party artificial shapes, not RBC evidence. Deliberately inspect which object the crop-target proposal selected.')
    elif source=='Consumed PathOlOgics development demonstrations':
        manifest=json.loads((HERE/'demo/pathologics_manifest.json').read_text())
        st.warning('Already-consumed stained-smear crops. These are development demonstrations, not usability validation or new holdout evidence. Reference masks are NOT auto-accepted.')
        selected=st.multiselect('Development examples',[x['case_id'] for x in manifest['examples']],key=key('path_choices'))
        for entry in manifest['examples']:
            if entry['case_id'] in selected:
                images[entry['image_file']]=(HERE/'demo'/entry['image_file']).read_bytes();demo_info.append(entry)
    else:
        uploads=st.file_uploader('Original images (required for automatic masks; optional for supplied masks)',type=['png','jpg','jpeg','bmp','tif','tiff'],accept_multiple_files=True,key=key('images'))
        mask_uploads=st.file_uploader('Supplied or externally proposed masks — lossless grayscale',type=['png','bmp','tif','tiff'],accept_multiple_files=True,key=key('masks')) if route!='Automatic proposal' else []
        all_files=uploads+mask_uploads
        if len(uploads)>MAX_BATCH or len(mask_uploads)>MAX_BATCH or sum(f.size for f in all_files)>MAX_BATCH_BYTES or any(f.size>MAX_BYTES for f in all_files):
            session.reset();st.error('Upload limits exceeded. No earlier results remain active.');st.stop()
        if len({f.name for f in uploads})!=len(uploads) or len({f.name for f in mask_uploads})!=len(mask_uploads):
            session.reset();st.error('Duplicate filename. Rename files before upload.');st.stop()
        images={f.name:f.getvalue() for f in uploads};masks={f.name:f.getvalue() for f in mask_uploads}

    # Pairing is explicit; exact stems are only suggested, never silently assumed.
    pairing={}
    if route!='Automatic proposal':
        for name in masks:
            stem=Path(name).stem.removesuffix('_mask')
            suggested=next((n for n in images if Path(n).stem==stem),None)
            options=['No image']+list(images)
            pairing[name]=st.selectbox('Image paired with '+name,options,index=options.index(suggested) if suggested else 0,key=key('pair_'+name))
    selection=service.digest(dict(route=route,source=source,adapter=adapter,kind=kind,external_algorithm=external_algorithm,
        images={n:sha(b) for n,b in images.items()},masks={n:sha(b) for n,b in masks.items()},pairing=pairing))
    session.sync(selection)  # A second upload/source switch cannot resurrect old results.

    st.subheader('2 · Analysis meaning, ROI and calibration')
    mode=st.radio('Measurement route',['structural','morphology'],horizontal=True,key=key('mode'))
    meaning=st.text_input('Foreground meaning',value='Target object / user-defined foreground',key=key('meaning'))
    polarity=st.selectbox('Foreground polarity',['nonzero/high'] if kind=='instances' or route=='Automatic proposal' else ['nonzero/high','zero/low'],key=key('polarity'))
    st.caption('Morphology is DEVELOPMENT: binary masks give connected-component / TARGET OBJECT geometry, not automatically individual RBCs.')
    union=st.checkbox('For structural analysis, explicitly measure the union of labels (retain original IDs in export)',key=key('union')) if kind=='instances' and mode=='structural' else False
    identity=st.checkbox('Each positive label represents one supplied instance (identity asserted, not validated)',key=key('identity')) if kind=='instances' and mode=='morphology' else False
    roi=None;scale=None;scale_source=''
    with st.expander('ROI / calibration (changes invalidate automatic acceptance)'):
        if st.checkbox('Use rectangular ROI',key=key('use_roi')):
            cols=st.columns(4)
            roi=tuple(c.number_input(label,1,None,value,key=key('roi_'+label)) for c,label,value in zip(cols,['x','y','width','height'],[1,1,64,64]))
            st.caption('ROI is x,y,width,height in 1-based pixels. No resizing.')
        if st.checkbox('Use spatial calibration',key=key('use_scale')):
            scale=st.number_input('Micrometres per pixel',min_value=.000001,value=1.,format='%.6f',key=key('scale'))
            scale_source=st.text_input('Calibration source',key=key('scale_source'))
    context=Context(mode,meaning,polarity,roi,scale,scale_source,union,identity)
    try:
        for r in session.records.values(): r.configure(context)
    except ValueError as exc:
        for r in session.records.values():r.invalidate()
        st.error(str(exc));st.stop()

    label='Generate proposed masks' if route=='Automatic proposal' else 'Load masks for review'
    if st.button(label,key=key('load')):
        session.records={}
        try:
            if route=='Automatic proposal':
                if not images: raise ValueError('Upload an image first.')
                for name,data in images.items():
                    proposed=propose(data,name,adapter,context)
                    demo_entry=next((entry for entry in demo_info if entry['image_file']==name),None)
                    if demo_entry:proposed.algorithm['development_example']=demo_entry
                    session.add(name,proposed)
            else:
                if not masks:raise ValueError('Upload a mask first.')
                if external_algorithm and not all(external_algorithm[k].strip() for k in ('name','version')):raise ValueError('External algorithm name and version are required.')
                for name,data in masks.items():
                    imname=pairing[name]
                    if (route=='External automatic proposal' or workflow=='IMAGE + SUPPLIED MASK') and imname=='No image':raise ValueError('This route requires a paired image for every mask.')
                    record=make_record(name,data,image_bytes=images.get(imname),image_name=imname if imname!='No image' else None,
                        kind=kind,context=context,algorithm=external_algorithm)
                    if len(masks)>1 or workflow=='BATCH REVIEW + ANALYSIS':record.decide('REVIEW')
                    session.add(name,record)
        except ValueError as exc:
            session.records={};st.error(str(exc))
    if not session.records:st.info('Load inputs to begin. No analysis or report is active.');return
    records=list(session.records.values())
    st.subheader('3 · Review gallery — decisions are per file')
    for col,(label,value) in zip(st.columns(5),session.summary().items()):col.metric(label,value)
    status=st.selectbox('Status filter',['ALL','UNREVIEWED','ACCEPTED','REJECTED','REPLACED'],key=key('filter'))
    pages=max(1,(len(session.filtered(status))+PAGE_SIZE-1)//PAGE_SIZE)
    page=st.selectbox('Review page',range(1,pages+1),key=key('page_'+status))
    visible=session.page(status,page)
    st.caption(f'Page {page}/{pages} · {len(visible)} shown. Decisions persist across pages and filters. REPLACED includes pending replacements; only accepted replacements are eligible.')
    if route in ('Automatic proposal','External automatic proposal'):
        st.warning('**'+EXPERIMENTAL_LABEL+'**\n\n'+EXPERIMENTAL_MESSAGE)
    st.caption('Touching cells may merge; cells may split or be missed. Unusual RBC shapes are not automatically rejected. No reliable merge/split detector is claimed; inspect the image, label IDs and boundaries.')
    if adapter=='PathOlOgics v0.1':st.info('TARGET OBJECT MASK: a crop-level target can contain multiple attached objects. Acceptance does not prove a single RBC.')
    for start in range(0,len(visible),3):
        for col,r in zip(st.columns(min(3,len(visible)-start)),visible[start:start+3]):
            with col:
                case_key=key('case_'+sha(r.name.encode())[:12]+'_'+str(r.revision))
                st.image(render(r,ids=r.kind=='instances'),caption=r.name)
                st.write(r.queue_status+' · '+r.state+' · '+r.decision)
                note=warnings(r)
                st.caption('; '.join(note) if note else 'No configured plausibility warning was triggered.')
                st.caption(QC_NOTE)
                reviewed=st.checkbox('I reviewed this overlay and its mask',key=case_key+'_confirm')
                if st.button('ACCEPT',key=case_key+'_accept',disabled=not reviewed):r.decide('ACCEPT',True);st.rerun()
                c1,c2=st.columns(2)
                if c1.button('REVIEW',key=case_key+'_review'):r.decide('REVIEW');st.rerun()
                if c2.button('REJECT',key=case_key+'_reject'):r.decide('REJECT');st.rerun()
    st.caption('No whole-batch automatic acceptance. Inspect and accept each case. Supplied multi-case batches also require per-case acceptance.')
    selected=st.selectbox('Open case for detailed review',range(len(records)),format_func=lambda i:records[i].name,key=key('detail'))
    r=records[selected];detail_key=key('detail_'+sha(r.name.encode())[:12]+'_'+str(r.revision))
    view=st.radio('Review view',['overlay','original','mask','boundary'],horizontal=True,key=key('view'))
    opacity=st.slider('Overlay opacity',0.,1.,.4,key=key('opacity'))
    show_ids=st.checkbox('Show instance label IDs',value=r.kind=='instances',key=key('show_ids'))
    pixels=render(r,view,opacity,show_ids)
    st.image(pixels,caption='Review display only. Analysis uses original pixels; preview interpolation does not change the mask. Yellow outline, when present, marks the analysis ROI.')
    with st.expander('Native-resolution review / zoom'):
        zoom=st.select_slider('Display zoom',options=[1,2,4,8],key=key('zoom'))
        w=r.mask.shape[1]*zoom
        components.html('<div style="height:470px;overflow:auto;border:1px solid #bbb"><img alt="Native-pixel review" style="image-rendering:pixelated;max-width:none;width:'+str(w)+'px" src="data:image/png;base64,'+base64.b64encode(pixels).decode()+'"></div>',height=490,scrolling=True)
        st.caption('1× = one source pixel per CSS pixel. Browser/device scaling can affect screen appearance. Full-resolution PNG download retains source dimensions.')
        st.download_button('Download full-resolution review preview',pixels,'review_preview.png','image/png',key=detail_key+'_preview')
    with st.expander('Reject / upload corrected mask'):
        st.write('Use an external editor, then upload a lossless replacement of identical dimensions. Instance IDs are retained exactly; no relabelling, merging, hole filling or contour smoothing is performed here.')
        correction=st.file_uploader('Corrected mask',type=['png','bmp','tif','tiff'],key=detail_key+'_correction')
        if correction is not None:
            st.warning('Replacement selected but not applied yet. Click Replace below; any current result still refers to the previous mask.')
        if st.button('Replace mask — requires a new ACCEPT',disabled=correction is None,key=detail_key+'_replace'):
            try:
                session.replace(r,correction.getvalue(),correction.name);st.rerun()
            except ValueError as exc:st.error(str(exc))
    if demo_info:
        with st.expander('Development example context (already-consumed evaluation)'):
            st.json(demo_info)
            st.caption('Historical agreement is disclosed to demonstrate good and failed proposals; it is not a new performance test or user-review validation.')
    st.subheader('4 · Measure eligible masks / export')
    st.write(f'{sum(q.eligible for q in records)} / {len(records)} eligible. REJECT and REVIEW cases cannot be measured or exported as results.')
    if st.button('Analyze eligible masks',disabled=not any(q.eligible for q in records),key=key('analyze')):
        started=time.monotonic();processed=0
        for q in records:
            if processed>=PAGE_SIZE or time.monotonic()-started>=60:break
            if q.eligible and (q.result is None or q.result_token!=q.token()):
                try:
                    slot=getattr(service,'ANALYSIS_SLOT',None) if hosted else None
                    if slot is not None and not slot.acquire(blocking=False):raise ValueError('Hosted analyzer busy. Your review is retained; try again shortly.')
                    try: analyze(q);processed+=1
                    finally:
                        if slot is not None:slot.release()
                except ValueError as exc:st.error(q.name+': '+str(exc))
        st.info(f'{processed} new analyses completed this click. Click again for remaining accepted cases; existing results are retained.')
    for q in visible:
        if q.eligible and q.result is not None and q.result_token==q.token():
            result=q.result
            with st.expander('Result · '+q.name,expanded=len(records)==1):
                st.success(ACCEPT_NOTE if q.state==ACCEPTED else 'User-supplied mask analyzed. Review/accuracy not inferred.')
                if q.context.mode=='structural':st.dataframe(service.metric_rows(result),hide_index=True)
                else:st.dataframe(result['instances'],hide_index=True)
                st.json(result['provenance']['mask_review'],expanded=False)
                filekey=key('export_'+sha(q.name.encode())[:12]+'_'+str(q.revision))
                st.download_button('Download reviewed report (HTML)',report(q),'reviewed_report.html','text/html',key=filekey+'_html')
                rows=[service.summary_row(result)] if q.context.mode=='structural' else result['instances']
                st.download_button('Download measurements CSV',service.csv_bytes(rows),'reviewed_measurements.csv','text/csv',key=filekey+'_csv')
                st.download_button('Download measurements + provenance + final mask (ZIP)',bundle(q),'reviewed_analysis.zip','application/zip',key=filekey+'_zip')
    if any(q.eligible and q.result is not None and q.result_token==q.token() for q in records):
        signature=service.digest([(q.token(),q.decision,q.result_token) for q in records])
        if st.button('Prepare eligible batch download',key=key('prepare_export')):
            st.session_state[key('batch_bytes')]=(signature,batch_bundle(records))
        prepared=st.session_state.get(key('batch_bytes'))
        if prepared and prepared[0]==signature:
            st.download_button('Download eligible batch only',prepared[1],'reviewed_batch.zip','application/zip',key=key('batch_export'))
        st.caption('Export includes only current eligible completed results. review_status.json lists every selected case, including rejected, unreviewed and not-yet-measured cases. Original masks and every replacement are retained in each exported case.')

if __name__=='__main__': main()
