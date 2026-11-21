"""Memory-only shared presentation and export helpers; no scientific recalculation."""
import base64, csv, html, io, json, zipfile, threading
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from backend import VERSION, json_safe, native, display_u8
PLOT_LOCK=threading.Lock()

def distribution_bytes(frame,column):
    with PLOT_LOCK:
        fig=Figure(figsize=(8,4),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
        values=pd.to_numeric(frame[column],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
        ax.hist(values,bins='auto');ax.set_xlabel(column);ax.set_ylabel('Included observations (not independent donors)')
        ax.set_title('Descriptive distribution');buf=io.BytesIO();fig.savefig(buf,format='png',dpi=130);return buf.getvalue()

def loadings_figure_bytes(loadings):
    with PLOT_LOCK:
        fig=Figure(figsize=(8,max(4,.3*len(loadings))),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
        y=np.arange(len(loadings));ax.barh(y-.18,loadings.PC1,.35,label='PC1');ax.barh(y+.18,loadings.PC2,.35,label='PC2')
        ax.set_yticks(y,loadings.index);ax.set_xlabel('PCA loading');ax.legend();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=130);return buf.getvalue()

def existing_figure_bytes(factory,*args,format='png',**kwargs):
    """Serialize an unchanged historical plotting function under the shared lock."""
    with PLOT_LOCK:
        import matplotlib.pyplot as plt
        fig=factory(*args,**kwargs)
        try:
            buf=io.BytesIO();fig.savefig(buf,format=format,dpi=130,bbox_inches='tight');return buf.getvalue()
        finally:plt.close(fig)

def csv_bytes(frame):
    # Neutralise spreadsheet formulas in user-controlled text, leaving numbers typed.
    safe=frame.copy()
    for c in safe.columns:
        safe[c]=safe[c].map(lambda x:"'"+x if isinstance(x,str) and x[:1] in ('=','+','-','@','\t','\r') else x)
    return safe.to_csv(index=False).encode('utf8')

def panel(image,mask):
    from scipy.ndimage import binary_erosion
    a=display_u8(image)
    rgb=np.repeat(a[...,None],3,axis=2) if a.ndim==2 else a[...,:3].copy()
    if rgb.shape[:2]!=mask.shape:raise ValueError('Panel geometry mismatch; refusing an unregistered overlay.')
    overlay=rgb.copy();overlay[mask]=(rgb[mask]*.55+np.array([0,210,255])*.45).astype('uint8')
    edge=rgb.copy();edge[mask&~binary_erosion(mask)]=[255,175,20]
    return np.concatenate([rgb,np.repeat((mask*255).astype('uint8')[...,None],3,2),overlay,edge],axis=1)

def chart_bytes(frame,columns,x=None,*,x_label=None,y_label=None,title=None):
    with PLOT_LOCK:
        fig=Figure(figsize=(9,4),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
        for c in columns:ax.plot(frame[x] if x else np.arange(1,len(frame)+1),pd.to_numeric(frame[c],errors='coerce'),marker='.',label=c)
        ax.set_xlabel(x_label or x or 'Observation (not independent cells)');ax.set_ylabel(y_label or 'Metric value (column units)')
        if title:ax.set_title(title)
        ax.grid(alpha=.2)
        if columns:ax.legend(fontsize=8)
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=130);return buf.getvalue()

def pca_figure_bytes(scores):
    """Export the same PC1-vs-PC2 view as the UI, never the numeric row ID."""
    with PLOT_LOCK:
        fig=Figure(figsize=(7,5),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
        ax.scatter(scores['PC1'],scores['PC2'])
        ax.set_xlabel('PC1 score');ax.set_ylabel('PC2 score')
        ax.set_title('Exploratory PCA — complete observations')
        ax.grid(alpha=.2)
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=130);return buf.getvalue()

def pca_loadings_bytes(loadings):
    """Retain descriptor identities in a flat CSV without changing loadings."""
    return csv_bytes(loadings.rename_axis('descriptor').reset_index())

def bundle(module,frame,metadata,limitations,figures=None,extras=None):
    figures=figures or {};extras=extras or {}
    meta={'software':'MicroStructAI Research Cloud','version':VERSION,'module':module,**metadata,'research_use_only':True}
    review_status=metadata.get('human_review_status',metadata.get('review_status','Not declared for this analysis'))
    if 'reviews' in metadata:review_status=str(len(metadata['reviews']))+' explicitly reviewed observations; see provenance for the remaining observations'
    fields={'Scientific status':metadata.get('status','Research output; see module limitations'),
        'Inputs':metadata.get('input_count','Not declared'),'Analysed':metadata.get('analysed_count','Not declared'),
        'Human review':review_status,'Coverage / exclusions':'Retained in result rows and provenance; absent values are not zero.'}
    summary='<dl>'+''.join('<dt>'+html.escape(k)+'</dt><dd>'+html.escape(str(v))+'</dd>' for k,v in fields.items())+'</dl>'
    pictures=''.join('<figure><figcaption>'+html.escape(n)+'</figcaption><img alt="'+html.escape(n,quote=True)+'" src="data:image/png;base64,'+base64.b64encode(b).decode()+'"></figure>' for n,b in figures.items())
    # All values remain in the report. Only the table scrolls, never the document.
    # Keep long machine-readable provenance accessible without burying the figures.
    style='''*{box-sizing:border-box}body{font:15px/1.5 Arial,sans-serif;margin:0;color:#183143;background:#f5f7f9}
main{width:100%;max-width:1100px;margin:24px auto;padding:24px;background:white;min-width:0}
h1{font-size:26px}h2{font-size:20px}h1,h2,p,summary,figcaption{overflow-wrap:anywhere}
.ruo{border-left:4px solid #24788a;padding:12px;background:#edf5f7}.muted{color:#526674}
figure{margin:24px 0;break-inside:avoid}figcaption{font-weight:bold;margin-bottom:8px}
img{display:block;max-width:100%;height:auto}details{margin:24px 0;border:1px solid #dde5e8;padding:12px}
summary{cursor:pointer;font-weight:bold}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}
.table-scroll{width:100%;max-height:32rem;overflow:auto;border:1px solid #b8c9d1;border-radius:4px}
.table-scroll:focus{outline:3px solid #24788a;outline-offset:2px}
table{border-collapse:separate;border-spacing:0;font-size:12px;text-align:left}
td,th{padding:8px;border-right:1px solid #dde5e8;border-bottom:1px solid #dde5e8;vertical-align:top;min-width:10ch;max-width:32ch;overflow-wrap:anywhere}
thead th{position:sticky;top:0;background:#edf5f7;z-index:1}tbody tr:nth-child(even){background:#f7f9fa}
.print-note{display:none}@media(max-width:600px){main{margin:0;padding:16px}h1{font-size:22px}}
@media print{body{background:white}main{max-width:none;margin:0;padding:0}.table-scroll{display:none}.print-note{display:block}details{break-inside:avoid}figure{max-width:100%}}'''
    report=('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MicroStructAI research report</title><style>'+style+'</style></head><body><main>'
        '<h1>MicroStructAI Research Cloud '+VERSION+'</h1><h2>'+html.escape(module)+'</h2>'
        '<p class="ruo">Research Use Only. No diagnostic or clinical interpretation.</p><p>'+html.escape(limitations)+'</p>'
        +summary+
        '<p class="muted">'+str(len(frame))+' result rows · '+str(len(frame.columns))+' columns. Observations are not necessarily independent samples.</p>'
        + ('<h2>Figures</h2>'+pictures if pictures else '')
        + '<h2 id="results-heading">Complete results</h2><p id="table-help">Scroll within this table horizontally and vertically to inspect all columns and rows. Keyboard: focus the table and use arrow keys. No result columns are omitted. The full table is also in results.csv in the ZIP download.</p>'
        '<div class="table-scroll" role="region" aria-labelledby="results-heading" aria-describedby="table-help" tabindex="0">'+frame.to_html(index=False,escape=True)+'</div>'
        '<p class="print-note">The complete wide results table is available in the electronic HTML report and results.csv; it is not truncated onto printed pages.</p>'
        '<details><summary>Full provenance and analysis metadata</summary><pre>'+html.escape(json.dumps(json_safe(meta),indent=2))+'</pre></details>'
        '</main></body></html>').encode('utf8')
    files={'results.csv':csv_bytes(frame),'report.html':report,'provenance.json':json.dumps(json_safe(meta),indent=2,allow_nan=False).encode(),**figures,**extras}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    return {'csv':files['results.csv'],'html':report,'zip':buf.getvalue(),'metadata':meta}
