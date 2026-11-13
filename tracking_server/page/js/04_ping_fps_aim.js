let pingBusy=false;   // one ping at a time, each with a 0.7 s limit, so requests never pile up behind a slow link. (11-10: was 1.5 s = the gimbal's heartbeat limit, so one lost ping already ended the session; now the next ping still fits inside the limit)
function ping(){if(pingBusy)return;pingBusy=true;const c=new AbortController(),t=setTimeout(()=>c.abort(),700);
  fetch('/gimbal/ping',{method:'POST',headers:{'X-Gimbal':'1'},cache:'no-store',signal:c.signal}).then(r=>r.json()).then(show).catch(()=>{}).finally(()=>{clearTimeout(t);pingBusy=false;netChip();});}
setInterval(ping,400);ping();setInterval(netChip,500);
const pageT0=Date.now();
function wake(){grace=Date.now();ping();if(Date.now()-pageT0>3000)openStream();}   // screen back on / tab shown / network back: the picture stream may have died while the page slept
document.addEventListener('visibilitychange',()=>{if(!document.hidden)wake();});addEventListener('pageshow',wake);addEventListener('online',wake);
// v25: frame rate from /healthz (read only): frames_in difference / up difference over the last 2 s
function hztick(){if(document.hidden)return;fetch('/healthz',{cache:'no-store'}).then(r=>r.json()).then(h=>{lastOk=Date.now();
  const r0=v=>Number.isFinite(v)?Math.round(v):'–';$('fps').textContent='카메라 '+r0(h.cam_fps)+' · 추적 '+r0(h.track_fps);hzPrev=h;   // 11-13: the server's speeds
  const rs=(typeof rpLast!=='undefined'&&rpLast&&rpLast.state)||{},vc=$('vchip');   // 11-13: picture chip (was a text bar drawn into the picture)
  if(vc){const rp=!!rs.on;vc.className='vchip num'+(rp?' replay':'');vc.textContent=(rp?'녹화 재생 · '+(rs.name||''):'실시간')+' · 추적 '+r0(h.track_fps)+' fps'+(Number.isFinite(h.delay_p50)?' · 지연 '+Math.round(h.delay_p50)+' ms':'');vc.hidden=false;}
  }).catch(()=>{});}
setInterval(hztick,2000);hztick();
// Match the bridge's configured reference height when displaying framing error.
const PX=x=>(x>0?'+':x<0?'−':'')+Math.abs(x).toFixed(0)+' px';
function etick(){if(document.hidden||!autoNow)return;fetch('/gimbal/people',{cache:'no-store'}).then(r=>r.json()).then(p=>{lastOk=Date.now();
  const id=p.state&&p.state.id,b=(p.people||[]).find(q=>q.id===id);if(!b||!autoNow){$('ex').textContent='–';$('ey').textContent='–';return;}
  const cfg=rlast&&rlast.gimbal_cfg, value=cfg&&cfg.aim_y, aim=value==null?NaN:Number(value);
  $('ex').textContent=PX((b.x1+b.x2)/2-p.width/2);
  $('ey').textContent=Number.isFinite(aim)&&aim>=0&&aim<=1?PX(b.y1+aim*(b.y2-b.y1)-p.height/2):'–';}).catch(()=>{});}
setInterval(etick,500);
