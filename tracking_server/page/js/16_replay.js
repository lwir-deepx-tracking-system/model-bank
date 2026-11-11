// 11-12: 녹화 재생 — a recording of the label site instead of the camera (GET /replay, POST /replay/start, /replay/stop). The server stops the gimbal and refuses moves while it plays.
const RPSEC=$('rpsec'),RPSEL=$('rpsel'),RPFPS=$('rpfps'),RPLOOP=$('rploop'),RPST=$('rpstart'),RPSP=$('rpstop'),RPNOTE=$('rpnote'),RPMSG=$('rpmsg');
let rpKeys='',rpList=[],rpBusy=false,rpLast=null;
function rpItem(n){return rpList.find(x=>x.name===n)||null;}
function rpNote(){const it=rpItem(RPSEL.value),st=rpLast&&rpLast.state;RPNOTE.textContent=st&&st.on&&st.note?st.note:it&&it.note?it.note:'';}
function rpShow(r){rpLast=r;
  if(!r.available){RPSEL.innerHTML='<option value="">쓸 수 없음</option>';rpKeys='';rpList=[];RPST.disabled=RPSP.disabled=true;RPNOTE.textContent='';RPMSG.textContent=r.why||'쓸 수 없음';return;}
  const keys=r.list.map(x=>x.name+'|'+x.frames+'|'+x.bits).join('\n');
  if(keys!==rpKeys){const keep=RPSEL.value;rpKeys=keys;rpList=r.list;
    RPSEL.innerHTML=r.list.length?r.list.map(x=>'<option value="'+esc(x.name)+'"'+(x.usable?'':' disabled')+'>'+esc(x.name)+' · '+x.frames+'장 · '+(x.bits?x.bits+'비트':'?')+(x.usable?'':' (쓸 수 없음)')+'</option>').join(''):'<option value="">녹화가 없습니다</option>';
    if(keep&&rpItem(keep))RPSEL.value=keep;}
  const st=r.state;
  if(st.on&&document.activeElement!==RPSEL&&rpItem(st.name))RPSEL.value=st.name;
  RPST.disabled=rpBusy||st.busy||!r.list.length;RPSP.disabled=rpBusy||st.busy||!st.on;
  let t=st.busy?'바꾸는 중…':st.on?'재생 중: '+st.name+' · '+st.fps+' fps · '+(st.loop?'반복':'한 번')+(st.pos?' · '+st.pos+' / '+st.frames+'장째 (약)':' · 불러오는 중'):'꺼짐 (카메라 영상)';
  if(st.msg)t+='\n'+st.msg.at+' '+st.msg.text;
  RPMSG.style.whiteSpace='pre-line';RPMSG.textContent=t;rpNote();}
function rptick(force){if(!force&&(document.hidden||!RPSEC.open))return;fetch('/replay'+(force==='refresh'?'?refresh=1':''),{cache:'no-store'}).then(r=>r.json()).then(rpShow).catch(()=>{});}
function rppost(url){rpBusy=true;RPST.disabled=RPSP.disabled=true;
  return fetch(url,{method:'POST',headers:{'X-Gimbal':'1'}}).then(r=>r.json().catch(()=>({ok:false,text:r.status===403||r.status===404?'서버 재시작 후 쓸 수 있습니다':'서버 응답 '+r.status}))).then(r=>{RPMSG.textContent=r.text||(r.ok?'완료':'실패');}).catch(()=>{RPMSG.textContent='서버 연결 끊김';}).finally(()=>{rpBusy=false;setTimeout(()=>rptick(true),300);});}
RPSEL.addEventListener('change',rpNote);
RPST.addEventListener('click',()=>{const n=RPSEL.value;if(!n)return;const f=Math.round(Number(RPFPS.value));if(!(f>=5&&f<=30)){RPMSG.textContent='초당 장 수는 5~30';return;}
  if(!confirm('녹화 "'+n+'" 를 카메라 대신 재생할까요?\n카메라 영상은 멈추고, 재생 중에는 짐벌이 움직이지 않습니다.'))return;
  rppost('/replay/start?seq='+encodeURIComponent(n)+'&fps='+f+'&loop='+(RPLOOP.checked?1:0));});
RPSP.addEventListener('click',()=>rppost('/replay/stop'));
RPSEC.addEventListener('toggle',()=>{if(RPSEC.open)rptick('refresh');});
setInterval(rptick,1000);rptick(true);
