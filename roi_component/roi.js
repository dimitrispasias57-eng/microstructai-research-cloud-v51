/* Local, dependency-free Streamlit component. No storage, network requests or masks. */
(() => {
  const canvas = document.getElementById('roi'), ctx = canvas.getContext('2d');
  const status = document.getElementById('status');
  let args = null, image = null, start = null, bounds = null, imageURL = null, appliedROI = null;
  const send = (type, extra) => parent.postMessage({isStreamlitMessage:true,type,...extra}, '*');
  const resize = () => send('streamlit:setFrameHeight', {height:Math.ceil(document.body.getBoundingClientRect().height)+4});
  function paint() {
    if (!image || !args) return;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(image,0,0,canvas.width,canvas.height);
    if (bounds) {
      const [x0,y0,x1,y1] = bounds;
      ctx.lineWidth=Math.max(1,args.width/600);ctx.strokeStyle='#ffcf00';
      ctx.strokeRect(x0,y0,x1-x0,y1-y0);
    }
    resize();
  }
  function point(e) {
    const r = canvas.getBoundingClientRect();
    return [Math.max(0,Math.min(args.width,(e.clientX-r.left)*args.width/r.width)),
            Math.max(0,Math.min(args.height,(e.clientY-r.top)*args.height/r.height))];
  }
  window.addEventListener('message', e => {
    if (e.source !== parent || e.data?.type !== 'streamlit:render') return;
    const next=e.data.args;
    if (!next || !Number.isInteger(next.width) || !Number.isInteger(next.height) || next.width<=0 || next.height<=0 || !String(next.image).startsWith('data:image/png;base64,')) return;
    const changed = !args || args.input_id!==next.input_id;
    args=next;
    if (changed) {start=null;bounds=null;appliedROI=null;}
    if (JSON.stringify(args.roi)!==appliedROI) {
      appliedROI=JSON.stringify(args.roi);
      if (args.roi) {const [x,y,w,h]=args.roi; bounds=[x-1,y-1,x-1+w,y-1+h];}
      else bounds=null;
    }
    canvas.width=args.width;canvas.height=args.height;
    if (imageURL!==args.image) {
      imageURL=args.image; const pending=new Image();
      pending.onload=()=>{if(pending.src===imageURL){image=pending;paint();}};
      pending.src=imageURL;
    } else paint();
  });
  canvas.addEventListener('pointerdown', e => {
    if (!args || !image || e.button!==0) return;
    start=point(e);bounds=[...start,...start];canvas.setPointerCapture(e.pointerId);e.preventDefault();paint();
  });
  canvas.addEventListener('pointermove', e => {if(start){bounds=[...start,...point(e)];paint();}});
  canvas.addEventListener('pointerup', e => {
    if (!start) return;
    bounds=[...start,...point(e)];start=null;canvas.releasePointerCapture(e.pointerId);paint();
    status.textContent='Rectangle selected. Check the pixel coordinates, then click Apply ROI below.';
    send('streamlit:setComponentValue',{value:{input_id:args.input_id,bounds,event_id:crypto.randomUUID()},dataType:'json'});
    resize();
  });
  canvas.addEventListener('pointercancel',()=>{start=null;bounds=null;paint();});
  new ResizeObserver(resize).observe(document.body);
  send('streamlit:componentReady',{apiVersion:1});resize();
})();
