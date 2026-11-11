let pingBusy=false;   // one ping at a time, each with a 0.7 s limit, so requests never pile up behind a slow link. (11-10: was 1.5 s = the gimbal's heartbeat limit, so one lost ping already ended the session; now the next ping still fits inside the limit)
function ping(){if(pingBusy)return;pingBusy=true;const c=new AbortController(),t=setTimeout(()=>c.abort(),700);
  fetch('/gimbal/ping',{method:'POST',headers:{'X-Gimbal':'1'},cache:'no-store',signal:c.signal}).then(r=>r.json()).then(show).catch(()=>{}).finally(()=>{clearTimeout(t);pingBusy=false;netChip();});}
setInterval(ping,400);ping();setInterval(netChip,500);
const pageT0=Date.now();
function wake(){grace=Date.now();ping();if(Date.now()-pageT0>3000)openStream();}   // screen back on / tab shown / network back: the picture stream may have died while the page slept
document.addEventListener('visibilitychange',()=>{if(!document.hidden)wake();});addEventListener('pageshow',wake);addEventListener('online',wake);
// v25: frame rate from /healthz (read only): frames_in difference / up difference over the last 2 s
function hztick(){if(document.hidden)return;fetch('/healthz',{cache:'no-store'}).then(r=>r.json()).then(h=>{lastOk=Date.now();
  if(hzPrev&&h.up>hzPrev.up){const f=(h.frames_in-hzPrev.frames_in)/(h.up-hzPrev.up);$('fps').textContent=f>0?f.toFixed(1):'0';}hzPrev=h;}).catch(()=>{});}
setInterval(hztick,2000);hztick();
// v25: aim error while following (read only): /gimbal/people gives the boxes; the bridge brings the point 0.3 of the box height below its top to the picture centre (gimbal_bridge_v25.cpp aimed(), --aim-y 0.3)
const AIM_Y=0.08,PX=x=>(x>0?'+':x<0?'−':'')+Math.abs(x).toFixed(0)+' px';
function etick(){if(document.hidden||!autoNow)return;fetch('/gimbal/people',{cache:'no-store'}).then(r=>r.json()).then(p=>{lastOk=Date.now();
  const id=p.state&&p.state.id,b=(p.people||[]).find(q=>q.id===id);if(!b||!autoNow){$('ex').textContent='–';$('ey').textContent='–';return;}
  $('ex').textContent=PX((b.x1+b.x2)/2-p.width/2);$('ey').textContent=PX(b.y1+AIM_Y*(b.y2-b.y1)-p.height/2);}).catch(()=>{});}
setInterval(etick,500);
