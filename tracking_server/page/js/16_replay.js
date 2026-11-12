// 11-12: 녹화 재생 — a recording of the label site instead of the camera (GET /replay, POST /replay/start, /replay/stop). The server stops the gimbal and refuses moves while it plays.
// 11-12 later: video-player controls under the picture (POST /replay/seek?pos=, /replay/pause, /replay/play, /replay/speed?fps=) and the comparison numbers under the picture, kept after the end.
const RPSEC=$('rpsec'),RPSEL=$('rpsel'),RPFPS=$('rpfps'),RPLOOP=$('rploop'),RPST=$('rpstart'),RPSP=$('rpstop'),RPNOTE=$('rpnote'),RPMSG=$('rpmsg');
let rpKeys='',rpList=[],rpBusy=false,rpLast=null;
function rpItem(n){return rpList.find(x=>x.name===n)||null;}
function rpNote(){const it=rpItem(RPSEL.value),st=rpLast&&rpLast.state;RPNOTE.textContent=st&&st.on&&st.note?st.note:it&&it.note?it.note:'';}
function rpShow(r){rpLast=r;
  if(!r.available){RPSEL.innerHTML='<option value="">쓸 수 없음</option>';rpKeys='';rpList=[];RPST.disabled=RPSP.disabled=true;RPNOTE.textContent='';RPMSG.textContent=r.why||'쓸 수 없음';barShow(r);return;}
  const keys=r.list.map(x=>x.name+'|'+x.frames+'|'+x.bits).join('\n');
  if(keys!==rpKeys){const keep=RPSEL.value;rpKeys=keys;rpList=r.list;
    RPSEL.innerHTML=r.list.length?r.list.map(x=>'<option value="'+esc(x.name)+'"'+(x.usable?'':' disabled')+'>'+esc(x.name)+' · '+x.frames+'장 · '+(x.bits?x.bits+'비트':'?')+(x.usable?'':' (쓸 수 없음)')+'</option>').join(''):'<option value="">녹화가 없습니다</option>';
    if(keep&&rpItem(keep))RPSEL.value=keep;}
  const st=r.state;
  if(st.on&&document.activeElement!==RPSEL&&rpItem(st.name))RPSEL.value=st.name;
  if(st.on&&document.activeElement!==RPFPS&&st.fps!=null)RPFPS.value=Math.round(st.fps);
  RPST.disabled=rpBusy||st.busy||!r.list.length;RPSP.disabled=rpBusy||st.busy||!st.on;
  let t=st.busy?'바꾸는 중…':st.on?'재생 중: '+st.name+' · '+st.fps+' fps · '+(st.loop?'반복':'한 번')+(st.pos!=null?(st.paused?' · 멈춤':'')+' · '+(st.pos+1)+' / '+st.total+'번째 장':' · 불러오는 중'):'꺼짐 (카메라 영상)';
  if(st.msg)t+='\n'+st.msg.at+' '+st.msg.text;
  RPMSG.style.whiteSpace='pre-line';RPMSG.textContent=t;rpNote();evShow(r.eval);barShow(r);}
// 11-12: 정답과 비교 — numbers of the two scorings (새 채점 = with ignore regions, 옛 채점 = without), false boxes first
const EVMSG=$('rpevmsg'),EVT=$('rpevt'),EVRS=$('rpevreset'),EVDR=$('rpevdraw');let evDrawT=0;
function evN(v,p){return v==null?'–':p?Number(v).toFixed(1)+'%':String(v);}
function evShow(e){if(!e)return;
  if(Date.now()-evDrawT>3000)EVDR.checked=!!e.draw;
  EVRS.disabled=!(e.on&&e.scoring);
  if(!e.name){EVMSG.textContent='재생을 시작하면, 검수한 정답 라벨이 있는 녹화는 여기에 1초마다 점수가 나옵니다.';EVT.innerHTML='';return;}
  const c=e.counts||{},head=(e.on?'':'끝난 재생: ')+e.name;
  if(!e.scoring){EVMSG.textContent=head+' · '+(e.why||'채점하지 않음');EVT.innerHTML='';return;}
  let m=head+' · 정답 '+(e.label_file||'')+'\n'+(e.review||'')+'\n채점한 장 '+c.scored+(e.scored_unique!=null&&e.scored_unique!==c.scored?' (서로 다른 장면 '+e.scored_unique+')':'')+(e.passes>1?' (구간 '+e.passes+')':'')+' · 멈춘 동안 되풀이 '+(c.repeats||0)+' (안 셈) · 재생 카메라가 건너뜀 '+c.cam_skipped+' · 서버가 건너뜀 '+c.server_skipped+(c.untagged?' · PNG 번호 모름 '+c.untagged:'')+'\n'+e.since+' 부터'+(e.file?'\n결과 파일: '+e.file:'');
  EVMSG.style.whiteSpace='pre-line';EVMSG.textContent=m;
  const n=e.new||{},o=e.old||{};
  const rows=[['fp','잘못 그린 상자','false',0],['','정답 사람','gt',0],['','찾음','found',0],['','놓침','missed',0],['sub','작은 사람 (<32px)','missed_small',0],['sub','그 밖','missed_other',0],
    ['','찾은 비율','recall',1],['','맞은 비율','precision',1],['','번호 바뀜','idsw',0],['','IDF1','idf1',1],['','MOTA','mota',1],['','무시 구역 상자','ignored_out',0]];
  EVT.innerHTML='<tr><th></th><th>새 채점</th><th>옛 채점</th></tr>'+rows.map(([k,l,f,p])=>'<tr'+(k==='fp'?' class="fp"':'')+'><td'+(k==='sub'?' class="sub"':'')+'>'+l+'</td><td>'+evN(n[f],p)+'</td><td>'+(f==='ignored_out'?'–':evN(o[f],p))+'</td></tr>').join('');}
EVRS.addEventListener('click',()=>{if(!confirm('정답과 비교 숫자를 지금부터 다시 셀까요?'))return;rppost('/replay/eval/reset');});
EVDR.addEventListener('change',()=>{evDrawT=Date.now();fetch('/replay/eval/draw?on='+(EVDR.checked?1:0),{method:'POST',headers:{'X-Gimbal':'1'}}).catch(()=>{});});
// ---- under the picture: seek bar (one copy) and the comparison strip (same data as the block above: GET /replay -> eval)
const RB=$('rpbar'),RSK=$('rpseek'),RPL=$('rpplay'),RM10=$('rpm10'),RP10=$('rpp10'),RSL=$('rpslider'),RTM=$('rptime'),RJN=$('rpjn'),RGO=$('rpgo'),RCM=$('rpcm'),RNUM=$('rpnum'),RNHD=$('rpnhd'),RNX=$('rpnx'),RMAIN=$('rpmain'),ROLD=$('rpold'),RFOOT=$('rpfoot');
let rpDrag=false,rpHold=null,rpHoldT=0,rpClosed='',rpCmT=0;
const REASON={ended:'끝까지 재생함',stopped:'사용자가 멈춤',switched:'다른 녹화로 바꿈','server stopped':'서버가 꺼짐'};
function fmtT(s){s=Math.max(0,Math.round(s));const m=Math.floor(s/60),x=s%60;return (m<10?'0':'')+m+':'+(x<10?'0':'')+x;}
function nchip(l,v,cls){return '<span class="rpc'+(cls?' '+cls:'')+'">'+l+' <b>'+v+'</b></span>';}
function barShow(r){
  const st=r.state||{},e=r.eval||{};
  // seek bar: only while a recording plays (one copy; the slider follows state.pos, which comes from the frame header)
  const on=!!(st.on&&st.total>0);RSK.hidden=!on;
  if(on){const tot=st.total,fps=st.fps||25,pos=st.pos;
    RSL.max=Math.max(0,tot-1);
    if(rpHold!=null&&(Date.now()-rpHoldT>5000||(!st.seeking&&st.pos!=null&&Date.now()-rpHoldT>1500)))rpHold=null;
    const shown=rpDrag?Number(RSL.value):rpHold!=null?rpHold:st.seeking&&st.seek_to!=null?st.seek_to:pos;
    if(!rpDrag&&shown!=null)RSL.value=shown;
    const dis=st.busy||st.pos==null;
    RSL.disabled=RM10.disabled=RP10.disabled=RPL.disabled=RGO.disabled=RJN.disabled=dis;
    RPL.textContent=st.paused?'재생':'일시정지';
    RJN.max=tot;if(document.activeElement!==RJN)RJN.placeholder=shown!=null?String(Number(shown)+1):'';
    RTM.textContent=shown==null?'불러오는 중…':fmtT(shown/fps)+' / '+fmtT((tot-1)/fps)+' · '+(Number(shown)+1)+' / '+tot+'번째 장'+(st.paused?' · 멈춤':'')+(st.seeking?' · 건너뛰는 중':'')+(st.pass>1?' · 구간 '+st.pass:'');
    if(rpCmT&&Date.now()>rpCmT){RCM.textContent='';rpCmT=0;}
  }
  // numbers: live while replaying, kept (labelled) after it ended, until the next replay or 닫기
  const key=e.name?e.name+'|'+(e.ended||'live'):'';
  if(!e.name||(!e.on&&key===rpClosed)){RNUM.hidden=true;}
  else{RNUM.hidden=false;
    const hd=e.on?'정답과 비교':'재생 끝, 최종 '+(e.ended?e.ended.slice(11,19):'')+' · '+(REASON[e.end_reason]||e.end_reason||'');
    RNHD.textContent=hd+' · '+e.name;RNX.hidden=!!e.on;
    if(!e.scoring){RMAIN.innerHTML='<span class="rpc">'+esc((e.why||'채점하지 않음').replace(/정답 라벨\([^)]*\)이/,'정답 라벨이'))+'</span>';ROLD.textContent='';RFOOT.textContent=e.on?'영상 아래 막대로 어느 곳이든 건너뛸 수 있습니다.':'';}
    else{const n=e.new||{},o=e.old||{},c=e.counts||{};
      const f=(x)=>x||{};
      RMAIN.innerHTML=nchip('오검출',evN(n.false),'fp'+(n.false?'':' zero'))+nchip('찾음',evN(n.found)+' / '+evN(n.gt)+' 정답 사람')+nchip('놓침',evN(n.missed)+' (작은 사람 '+evN(n.missed_small)+')')+nchip('recall',evN(n.recall,1))+nchip('precision',evN(n.precision,1))+nchip('번호 바뀜',evN(n.idsw))+nchip('IDF1',evN(n.idf1,1))+nchip('MOTA',evN(n.mota,1));
      ROLD.innerHTML='<b>옛 채점</b> · 오검출 '+evN(o.false)+' · 찾음 '+evN(o.found)+' / '+evN(o.gt)+' · 놓침 '+evN(o.missed)+' (작은 사람 '+evN(o.missed_small)+') · recall '+evN(o.recall,1)+' · precision '+evN(o.precision,1)+' · 번호 바뀜 '+evN(o.idsw)+' · IDF1 '+evN(o.idf1,1)+' · MOTA '+evN(o.mota,1);
      RFOOT.textContent='새 채점 (무시 구역 뺌) · 채점한 장 '+c.scored+(e.scored_unique!=null&&e.scored_unique!==c.scored?' (서로 다른 장면 '+e.scored_unique+')':'')+' · 구간 '+(e.passes||1)+(c.repeats?' · 멈춘 동안 되풀이 '+c.repeats+' (안 셈)':'')+((e.seeks||e.passes>1)?' · 건너뛰거나 되감은 뒤 다시 지난 장면은 새 구간으로 한 번 더 셈 (숫자를 처음부터 다시 세려면 옆 패널의 "숫자 처음부터")':'');}
  }
  RB.hidden=RSK.hidden&&RNUM.hidden;
}
RNX.addEventListener('click',()=>{const e=rpLast&&rpLast.eval;if(e&&e.name){rpClosed=e.name+'|'+(e.ended||'live');barShow(rpLast);}});
function rpMsgBar(t){RCM.textContent=t||'';rpCmT=t?Date.now()+6000:0;}
function rpctl(url){return fetch(url,{method:'POST',headers:{'X-Gimbal':'1'}}).then(r=>r.json().catch(()=>({ok:false,text:'서버 응답 '+r.status}))).then(r=>{if(!r.ok)rpMsgBar(r.text||'실패');return r;}).catch(()=>{rpMsgBar('서버 연결 끊김');return {ok:false};}).finally(()=>setTimeout(()=>rptick(true),250));}
function rpSeek(pos){const st=rpLast&&rpLast.state;if(!st||!st.on)return;pos=Math.min(Math.max(0,Math.round(pos)),st.total-1);rpHold=pos;rpHoldT=Date.now();RSL.value=pos;rpctl('/replay/seek?pos='+pos);}
RSL.addEventListener('pointerdown',()=>{rpDrag=true;});
RSL.addEventListener('input',()=>{const st=rpLast&&rpLast.state;if(st&&st.on)RTM.textContent=fmtT(Number(RSL.value)/(st.fps||25))+' / '+fmtT((st.total-1)/(st.fps||25))+' · '+(Number(RSL.value)+1)+' / '+st.total+'번째 장 (놓으면 건너뜀)';});
RSL.addEventListener('change',()=>{rpDrag=false;rpSeek(Number(RSL.value));});   // sent only when released
addEventListener('pointerup',()=>{if(rpDrag){rpDrag=false;}});
RPL.addEventListener('click',()=>{const st=rpLast&&rpLast.state;if(st&&st.on)rpctl(st.paused?'/replay/play':'/replay/pause');});
RM10.addEventListener('click',()=>{const st=rpLast&&rpLast.state;if(st&&st.on&&st.pos!=null)rpSeek((rpHold!=null?rpHold:st.pos)-10*(st.fps||25));});
RP10.addEventListener('click',()=>{const st=rpLast&&rpLast.state;if(st&&st.on&&st.pos!=null)rpSeek((rpHold!=null?rpHold:st.pos)+10*(st.fps||25));});
function rpGo(){const st=rpLast&&rpLast.state;if(!st||!st.on)return;const n=Math.round(Number(RJN.value));if(!(n>=1&&n<=st.total)){rpMsgBar('장 번호는 1 ~ '+st.total);return;}RJN.value='';rpSeek(n-1);}
RGO.addEventListener('click',rpGo);RJN.addEventListener('keydown',e=>{if(e.key==='Enter')rpGo();});
if(window.ResizeObserver)new ResizeObserver(()=>document.documentElement.style.setProperty('--rbh',(RB.hidden?0:Math.ceil(RB.getBoundingClientRect().height)+6)+'px')).observe(RB);
let rpTick=0;
function rptick(force){if(!force){const on=rpLast&&rpLast.state&&rpLast.state.on;if(document.hidden||(!on&&!RPSEC.open&&(++rpTick%8)))return;}fetch('/replay'+(force==='refresh'?'?refresh=1':''),{cache:'no-store'}).then(r=>r.json()).then(rpShow).catch(()=>{});}
function rppost(url){rpBusy=true;RPST.disabled=RPSP.disabled=true;
  return fetch(url,{method:'POST',headers:{'X-Gimbal':'1'}}).then(r=>r.json().catch(()=>({ok:false,text:r.status===403||r.status===404?'서버 재시작 후 쓸 수 있습니다':'서버 응답 '+r.status}))).then(r=>{RPMSG.textContent=r.text||(r.ok?'완료':'실패');}).catch(()=>{RPMSG.textContent='서버 연결 끊김';}).finally(()=>{rpBusy=false;setTimeout(()=>rptick(true),300);});}
RPSEL.addEventListener('change',rpNote);
RPFPS.addEventListener('change',()=>{const st=rpLast&&rpLast.state;if(!st||!st.on)return;const f=Math.round(Number(RPFPS.value));if(!(f>=5&&f<=30)){RPMSG.textContent='초당 장 수는 5~30';return;}rpctl('/replay/speed?fps='+f);});
RPST.addEventListener('click',()=>{const n=RPSEL.value;if(!n)return;const f=Math.round(Number(RPFPS.value));if(!(f>=5&&f<=30)){RPMSG.textContent='초당 장 수는 5~30';return;}
  if(!confirm('녹화 "'+n+'" 를 카메라 대신 재생할까요?\n카메라 영상은 멈추고, 재생 중에는 짐벌이 움직이지 않습니다.'))return;
  rpClosed='';rppost('/replay/start?seq='+encodeURIComponent(n)+'&fps='+f+'&loop='+(RPLOOP.checked?1:0));});
RPSP.addEventListener('click',()=>rppost('/replay/stop'));
RPSEC.addEventListener('toggle',()=>{if(RPSEC.open)rptick('refresh');});
setInterval(rptick,500);rptick(true);
