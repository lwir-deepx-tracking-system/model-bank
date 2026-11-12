// 11-12: 녹화 재생 — a recording of the label site instead of the camera (GET /replay, POST /replay/start, /replay/stop). The server stops the gimbal and refuses moves while it plays.
// 11-12 later: video-player controls under the picture (POST /replay/seek?pos=, /replay/pause, /replay/play, /replay/speed?fps=) and the comparison numbers under the picture only during playback.
const RPSEC=$('rpsec'),RPSEL=$('rpsel'),RPBACK=$('rpbackend'),RPFPS=$('rpfps'),RPLOOP=$('rploop'),RPST=$('rpstart'),RPSP=$('rpstop'),RPNOTE=$('rpnote'),RPMSG=$('rpmsg');
let rpKeys='',rpBackendKeys='',rpList=[],rpBusy=false,rpLast=null;
const EVBAR=$('rpevbar'),EVLIVE=$('rpevallive'),EVSLIDER=$('rpevalslider'),EVFINAL=$('rpevalfinal'),EVRESET=$('rpevalreset'),EVVIEW=$('rpevalview');
let rpEvalMode='live',rpEvalFinal=null,rpEvalPrefix=null,rpEvalPrefixPos=null,rpEvalError='',rpEvalPending=false,rpEvalToken=0,rpEvalRunId=null,rpEvalRetryTimer=0;
function rpEnded(st){return !!(st&&st.on&&!st.loop&&!st.seeking&&st.total>0&&st.pos===st.total-1);}
function rpItem(n){return rpList.find(x=>x.name===n)||null;}
function rpBackendLabel(id,r){const b=(r&&Array.isArray(r.backends)?r.backends:[]).find(x=>x.id===id);return b&&b.label||({python:'현재 추적기',cpp_bytetrack:'C++ ByteTrack'}[id]||id||'현재 추적기');}
function rpNote(){const it=rpItem(RPSEL.value),st=rpLast&&rpLast.state;RPNOTE.textContent=st&&st.on&&st.note?st.note:it&&it.note?it.note:'';}
function rpCppState(r){return r&&r.state&&r.state.on&&r.state.backend==='cpp_bytetrack'?(r.eval&&r.eval.backend_error?'error':'active'):'';}
function rpShow(r){const redraw=rpCppState(rpLast)!==rpCppState(r);rpLast=r;
  if(redraw)otdraw();
  const backends=Array.isArray(r.backends)&&r.backends.length?r.backends:[{id:'python',label:'현재 추적기'}];
  const backendKeys=backends.map(x=>x.id+'|'+x.label).join('\n');
  if(backendKeys!==rpBackendKeys){const keep=RPBACK.value;rpBackendKeys=backendKeys;RPBACK.innerHTML=backends.map(x=>'<option value="'+esc(x.id)+'">'+esc(x.label)+'</option>').join('');
    const active=r.state&&r.state.on&&r.state.backend;
    if(active&&backends.some(x=>x.id===active))RPBACK.value=active;else if(backends.some(x=>x.id===keep))RPBACK.value=keep;else RPBACK.value=backends[0].id;}
  if(!r.available){RPSEL.innerHTML='<option value="">쓸 수 없음</option>';rpKeys='';rpList=[];RPST.disabled=RPSP.disabled=true;RPNOTE.textContent='';RPMSG.textContent=r.why||'쓸 수 없음';barShow(r);return;}
  // Keep original and BandFill recordings side by side; original sequences come first.
  const source='outdoor_bandfill_20261111/', originals='opi_sequences/';
  const original=r.list.filter(x=>x.name.startsWith(originals));
  const band=r.list.filter(x=>x.name.startsWith(source));
  const other=r.list.filter(x=>!x.name.startsWith(originals)&&!x.name.startsWith(source));
  const list=[...original,...band,...other];
  const keys=list.map(x=>x.name+'|'+x.frames+'|'+x.bits+'|'+x.usable).join('\n');
  if(keys!==rpKeys){const keep=RPSEL.value;rpKeys=keys;rpList=list;
    RPSEL.innerHTML=list.length?list.map(x=>{const label=x.name.startsWith(source)?'BandFill / '+x.name.slice(source.length):x.name.startsWith(originals)?'원본 / '+x.name.slice(originals.length):x.name;return '<option value="'+esc(x.name)+'"'+(x.usable?'':' disabled')+'>'+esc(label)+' · '+x.frames+'장 · '+(x.bits?x.bits+'비트':'?')+(x.usable?'':' (쓸 수 없음)')+'</option>';}).join(''):'<option value="">녹화가 없습니다</option>';
    if(keep&&rpItem(keep))RPSEL.value=keep;}
  const st=r.state;
  if(st.on&&document.activeElement!==RPSEL&&rpItem(st.name))RPSEL.value=st.name;
  if(st.on&&document.activeElement!==RPFPS&&st.fps!=null)RPFPS.value=Math.round(st.fps);
  RPST.disabled=rpBusy||st.busy||!list.some(x=>x.usable);RPSP.disabled=rpBusy||st.busy||!st.on;
  let t=st.busy?'바꾸는 중…':st.on?(rpEnded(st)?'완주 · 마지막 장: ':'재생 중: ')+st.name+' · '+st.fps+' fps · 추적기 '+rpBackendLabel(st.backend,r)+' · '+(st.loop?'반복':'한 번')+(st.pos!=null?(st.paused?' · 멈춤':'')+' · '+(st.pos+1)+' / '+st.total+'번째 장':' · 불러오는 중'):'꺼짐 (카메라 영상)';
  if(r.eval&&r.eval.backend_error)t+='\nC++ ByteTrack 오류 · '+r.eval.backend_error;
  if(st.msg)t+='\n'+st.msg.at+' '+st.msg.text;
  RPMSG.style.whiteSpace='pre-line';RPMSG.textContent=t;rpNote();rpRenderEval(r);}
// 11-12: 정답과 비교 — numbers of the two scorings (새 채점 = with ignore regions, 옛 채점 = without), false boxes first
const EVMSG=$('rpevmsg'),EVT=$('rpevt'),EVSW=$('rpevids'),EVDR=$('rpevdraw');let evDrawT=0;
function evN(v,p){return v==null?'–':p?Number(v).toFixed(1)+'%':String(v);}
function evPhase(e){return ({scoring:'첫 재생 채점 중',interrupted:'중단됨 · 완주 결과 아님',complete:'한 번 완주함',incomplete:'불완전한 재생'})[e.phase]||'';}
function evMetricStatus(e){return ({pending:'계산 중',ready:'계산 완료',unavailable:'계산할 수 없음',error:'계산 오류'})[e.metric_status]||'미제공';}
function evCoverage(e){const total=e.sequence_frames,done=e.pass_frames;if(total==null||done==null)return '';return ' · 영상 진행 '+done+' / '+total+'장';}
function evSwitches(e){const events=Array.isArray(e.id_switches)?e.id_switches:[];
  return events.length?'<b>ID 변경</b> · '+events.map(x=>'장 '+esc(x.frame)+': '+esc(x.from_id)+' → '+esc(x.to_id)+(x.gt_id!=null?' (정답 '+esc(x.gt_id)+')':'')).join('　'):'';
}
function evShow(e,emptyText){if(!e){EVSW.innerHTML='';EVMSG.textContent=emptyText||'비교 결과가 없습니다.';EVT.innerHTML='';return;}
  EVSW.innerHTML=e.scoring?evSwitches(e):'';
  if(Date.now()-evDrawT>3000)EVDR.checked=!!e.draw;
  if(!e.name){EVMSG.textContent='재생을 시작하면, 검수한 정답 라벨이 있는 녹화는 여기에 1초마다 점수가 나옵니다.';EVT.innerHTML='';return;}
  const c=e.counts||{},head=(e.on?'':'끝난 재생: ')+e.name+(e.backend?' · 추적기 '+rpBackendLabel(e.backend,rpLast):'');
  if(!e.scoring){EVMSG.textContent=head+' · '+(evPhase(e)?evPhase(e)+' · ':'')+(e.why||'채점하지 않음')+evCoverage(e);EVT.innerHTML='';return;}
  let m=head+' · 정답 '+(e.label_file||'')+'\n'+(e.review||'')+'\n'+(evPhase(e)||'재생 상태 미상')+' · 이 회차 점수에 반영된 장 '+c.scored+evCoverage(e)+(e.scorable_frames!=null?' · 검수된 비교 가능 장 '+e.scorable_frames:'')+' · 건너뛴 카메라 장 '+(c.cam_skipped||0)+' · 서버 누락 '+(c.server_skipped||0)+(c.untagged?' · PNG 번호 모름 '+c.untagged:'')+'\nHOTA / mAP 계산 '+evMetricStatus(e)+'\n'+e.since+' 부터'+(e.file?'\n결과 파일: '+e.file:'');
  EVMSG.style.whiteSpace='pre-line';EVMSG.textContent=m;
  const n=e.new||{},o=e.old||{};
  const rows=[['metric','HOTA','hota',1],['metric','사람 mAP50:95 (검출 후보)','map',1],['metric','사람 AP50 (검출 후보)','ap50',1],['fp','잘못 그린 상자','false',0],['','정답 사람','gt',0],['','찾음','found',0],['','놓침','missed',0],['sub','작은 사람 (<32px)','missed_small',0],['sub','그 밖','missed_other',0],
    ['','찾은 비율','recall',1],['','맞은 비율','precision',1],['','번호 바뀜','idsw',0],['','IDF1','idf1',1],['','MOTA','mota',1],['','무시 구역 상자','ignored_out',0]];
  EVT.innerHTML='<tr><th></th><th>새 채점</th><th>옛 채점</th></tr>'+rows.map(([k,l,f,p])=>'<tr'+(k==='fp'?' class="fp"':'')+'><td'+(k==='sub'?' class="sub"':'')+'>'+l+'</td><td>'+(k==='metric'?metricText(e,n[f]):evN(n[f],p))+'</td><td>'+(k==='metric'||f==='ignored_out'?'–':evN(o[f],p))+'</td></tr>').join('');}
function rpEvalSetMode(mode){rpEvalMode=mode;rpEvalToken++;rpEvalPending=false;rpEvalPrefix=null;rpEvalPrefixPos=null;rpEvalError='';clearTimeout(rpEvalRetryTimer);rpEvalRetryTimer=0;}
function rpEvalRefreshButtons(){EVLIVE.setAttribute('aria-pressed',String(rpEvalMode==='live'));EVSLIDER.setAttribute('aria-pressed',String(rpEvalMode==='slider'));EVFINAL.setAttribute('aria-pressed',String(rpEvalMode==='final'));EVLIVE.classList.toggle('start',rpEvalMode==='live');EVSLIDER.classList.toggle('start',rpEvalMode==='slider');EVFINAL.classList.toggle('start',rpEvalMode==='final');}
function rpRenderEval(r){const live=r.eval||{},pos=rpEvalPrefixPos;let shown=null,note='';
  if(live.run_id!=null&&rpEvalRunId!==live.run_id){if(rpEvalRunId!=null&&rpEvalMode==='slider'){rpEvalToken++;rpEvalMode='live';rpEvalPrefix=null;rpEvalPrefixPos=null;rpEvalPending=false;clearTimeout(rpEvalRetryTimer);}rpEvalRunId=live.run_id;}
  rpEvalRefreshButtons();
  if(rpEvalMode==='slider'){
    if(rpEvalPending){note='프레임 '+(pos==null?'–':pos+1)+'까지의 누적 결과를 불러오는 중…';}
    else if(rpEvalPrefix){shown=rpEvalPrefix;note='슬라이더 위치 '+(rpEvalPrefix.prefix_pos==null?(pos==null?'–':pos+1):rpEvalPrefix.prefix_pos+1)+'장까지의 누적 채점 결과입니다. 재생 위치는 바뀌지 않았습니다.';}
    else note=rpEvalError||'슬라이더 결과를 보려면 슬라이더 위치를 확정하거나 “슬라이더 결과”를 누르세요.';
  }else if(rpEvalMode==='final'){
    if(rpEvalPending)note='저장된 최종 결과를 불러오는 중…';
    else if(rpEvalFinal){shown=rpEvalFinal;note=rpEvalFinal.final_previous?'이전 완주 결과입니다. 현재 재생과 다른 실행일 수 있습니다. 슬라이더 위치는 바뀌지 않았습니다.':'저장된 최종 결과입니다. 슬라이더 위치는 바뀌지 않았습니다.';}
    else note=live.final_ready?'최종 결과를 불러오지 못했습니다. 다시 “최종 보기”를 눌러 주세요.':'끝까지 완주한 최종 결과가 없습니다. 현재 재생 진행 결과를 표시합니다.';
  }else {shown=live;note=live.final_ready?'현재 진행 결과입니다. 완주 결과는 최종 보기를 누르면 확인할 수 있습니다.':'현재 재생 진행 결과입니다. 한 번 끝까지 재생해야 최종 결과가 확정됩니다.';}
  EVBAR.hidden=!(r.result_views&&(r.state&&r.state.on||live.name||rpEvalFinal&&rpEvalFinal.name||rpEvalPrefix&&rpEvalPrefix.name));
  EVVIEW.textContent=note;evShow(shown,shown?null:(rpEvalMode==='live'&&!live.name?'재생을 시작하면 검수된 정답 라벨과 비교합니다.':note));barShow(r,rpEvalMode==='live'?undefined:shown);}
function rpBeginPrefix(pos){const st=rpLast&&rpLast.state;if(!st||!st.on||!Number.isFinite(pos))return null;pos=Math.min(Math.max(0,Math.round(pos)),Math.max(0,st.total-1));rpEvalSetMode('slider');rpEvalPrefixPos=pos;rpEvalPending=true;const token=rpEvalToken,runId=(rpLast.eval||{}).run_id;rpRenderEval(rpLast);return {pos,token,runId};}
function rpLoadPrefix(pos,token,runId){if(token==null){const request=rpBeginPrefix(pos);if(!request)return;token=request.token;pos=request.pos;runId=request.runId;}
  const load=()=>fetch('/replay/eval?pos='+encodeURIComponent(pos),{cache:'no-store'}).then(x=>x.json()).then(d=>{
    if(token!==rpEvalToken||rpEvalMode!=='slider'||rpEvalPrefixPos!==pos)return;
    const prefix=d.eval||{},responseRunId=d.run_id!=null?d.run_id:prefix.run_id;
    if(runId!=null&&responseRunId!=null&&responseRunId!==runId){rpEvalPending=true;rpEvalRetryTimer=setTimeout(load,1000);return;}
    if(prefix.prefix_pending){rpEvalPending=true;rpRenderEval(rpLast||{});rpEvalRetryTimer=setTimeout(load,1000);return;}
    if(prefix.prefix_unavailable){rpEvalPending=false;rpEvalPrefix=null;rpEvalError='슬라이더 위치 '+(pos+1)+'의 결과를 사용할 수 없습니다.'+(prefix.why?' '+prefix.why:'');rpRenderEval(rpLast||{});return;}
    if(d.eval&&prefix.prefix_pos===pos){rpEvalPrefix=prefix;rpEvalPending=false;rpRenderEval(rpLast||{});return;}
    rpEvalPending=true;rpRenderEval(rpLast||{});rpEvalRetryTimer=setTimeout(load,1000);
  }).catch(()=>{if(token!==rpEvalToken)return;rpEvalPending=false;rpEvalPrefix=null;rpEvalError='슬라이더 위치 결과를 불러오지 못했습니다.';rpRenderEval(rpLast||{});});
  load();}
function rpLoadFinal(){rpEvalSetMode('final');rpEvalFinal=null;rpEvalPending=true;const token=rpEvalToken;rpRenderEval(rpLast||{});
  const load=()=>fetch('/replay/eval?view=final',{cache:'no-store'}).then(x=>x.json()).then(d=>{if(token!==rpEvalToken||rpEvalMode!=='final')return;rpEvalFinal=d.eval||null;rpEvalPending=!!(rpEvalFinal&&rpEvalFinal.metric_status==='pending');rpRenderEval(rpLast||{});if(rpEvalPending)rpEvalRetryTimer=setTimeout(load,1000);}).catch(()=>{if(token!==rpEvalToken)return;rpEvalPending=false;rpEvalFinal=null;rpRenderEval(rpLast||{});});
  load();}
EVDR.addEventListener('change',()=>{evDrawT=Date.now();fetch('/replay/eval/draw?on='+(EVDR.checked?1:0),{method:'POST',headers:{'X-Gimbal':'1'}}).catch(()=>{});});
// ---- under the picture: seek bar (one copy) and the comparison strip (same data as the block above: GET /replay -> eval)
const RB=$('rpbar'),RSK=$('rpseek'),RPL=$('rpplay'),RM10=$('rpm10'),RP10=$('rpp10'),RSL=$('rpslider'),RTM=$('rptime'),RJN=$('rpjn'),RGO=$('rpgo'),RCM=$('rpcm'),RNUM=$('rpnum'),RNHD=$('rpnhd'),RMAIN=$('rpmain'),ROLD=$('rpold'),RFOOT=$('rpfoot'),RIDS=$('rpids');
let rpDrag=false,rpHold=null,rpHoldT=0,rpCmT=0;
function fmtT(s){s=Math.max(0,Math.round(s));const m=Math.floor(s/60),x=s%60;return (m<10?'0':'')+m+':'+(x<10?'0':'')+x;}
function nchip(l,v,cls){return '<span class="rpc'+(cls?' '+cls:'')+'">'+l+' <b>'+v+'</b></span>';}
function metricText(e,v){if(v!=null)return evN(v,1);return e.metric_status==='pending'?'계산 중':e.metric_status==='error'?'오류':e.metric_status==='unavailable'?'계산 불가':'–';}
function barShow(r,viewEval){
  const st=r.state||{},e=viewEval===undefined?(r.eval||{}):(viewEval||{});
  // seek bar: only while a recording plays (one copy; the slider follows state.pos, which comes from the frame header)
  const on=!!(st.on&&st.total>0);RSK.hidden=!on;
  if(st.on||st.busy)$('camera-center').hidden=true;
  if(on){const tot=st.total,fps=st.fps||25,pos=st.pos;
    RSL.max=Math.max(0,tot-1);
    if(rpHold!=null&&(Date.now()-rpHoldT>5000||(!st.seeking&&st.pos!=null&&Date.now()-rpHoldT>1500)))rpHold=null;
    const shown=rpDrag?Number(RSL.value):rpHold!=null?rpHold:st.seeking&&st.seek_to!=null?st.seek_to:pos;
    if(!rpDrag&&shown!=null)RSL.value=shown;
    const dis=st.busy||st.pos==null;
    RSL.disabled=RM10.disabled=RP10.disabled=RPL.disabled=RGO.disabled=RJN.disabled=dis;
    RPL.textContent=rpEnded(st)?'처음부터':st.paused?'재생':'일시정지';
    RJN.max=tot;if(document.activeElement!==RJN)RJN.placeholder=shown!=null?String(Number(shown)+1):'';
    RTM.textContent=shown==null?'불러오는 중…':fmtT(shown/fps)+' / '+fmtT((tot-1)/fps)+' · '+(Number(shown)+1)+' / '+tot+'번째 장'+(st.paused?' · 멈춤':'')+(st.seeking?' · 건너뛰는 중':'')+(st.pass>1?' · 구간 '+st.pass:'');
    if(rpCmT&&Date.now()>rpCmT){RCM.textContent='';rpCmT=0;}
  }
  // Keep the live-camera view clear; completed results remain in the replay panel.
  if(!e.name||(!on&&viewEval===undefined)||(!e.on&&viewEval===undefined)){RNUM.hidden=true;}
  else{RNUM.hidden=false;
    RNHD.textContent=(e.phase==='complete'?'첫 회차 완주 결과':'정답과 비교')+' · '+e.name+(e.backend?' · '+rpBackendLabel(e.backend,r):'');
    if(!e.scoring){RMAIN.innerHTML='<span class="rpc">'+esc((evPhase(e)?evPhase(e)+' · ':'')+(e.why||'채점하지 않음').replace(/정답 라벨\([^)]*\)이/,'정답 라벨이'))+'</span>';ROLD.textContent='';RFOOT.textContent=e.on?'재생을 끝까지 한 번 완료해야 비교 결과가 확정됩니다.':'재생을 끝까지 한 번 완료해야 비교 결과가 확정됩니다.';RIDS.textContent='';}
    else{const n=e.new||{},o=e.old||{},c=e.counts||{};
      RMAIN.innerHTML=nchip('HOTA',metricText(e,n.hota))+nchip('사람 mAP50:95 (검출 후보)',metricText(e,n.map))+nchip('사람 AP50 (검출 후보)',metricText(e,n.ap50))+nchip('오검출',evN(n.false),'fp'+(n.false?'':' zero'))+nchip('찾음',evN(n.found)+' / '+evN(n.gt)+' 정답 사람')+nchip('놓침',evN(n.missed)+' (작은 사람 '+evN(n.missed_small)+')')+nchip('recall',evN(n.recall,1))+nchip('precision',evN(n.precision,1))+nchip('번호 바뀜',evN(n.idsw))+nchip('IDF1',evN(n.idf1,1))+nchip('MOTA',evN(n.mota,1));
      ROLD.innerHTML='<b>옛 채점</b> · 오검출 '+evN(o.false)+' · 찾음 '+evN(o.found)+' / '+evN(o.gt)+' · 놓침 '+evN(o.missed)+' (작은 사람 '+evN(o.missed_small)+') · recall '+evN(o.recall,1)+' · precision '+evN(o.precision,1)+' · 번호 바뀜 '+evN(o.idsw)+' · IDF1 '+evN(o.idf1,1)+' · MOTA '+evN(o.mota,1);
      const total=e.sequence_frames,pass=e.pass_frames,scorable=e.scorable_frames;
      RFOOT.textContent=(evPhase(e)||'재생 채점')+' · 첫 재생 채점 '+(c.scored==null?'–':c.scored)+'장'+(pass!=null&&total!=null?' · 진행 '+pass+' / '+total+'장':'')+(scorable!=null?' · 정답 비교 가능 '+scorable+'장':'')+' · HOTA / mAP '+evMetricStatus(e)+(e.phase==='interrupted'?' · 다시 처음부터 재생해야 완주 결과가 나옵니다.':'');
      RIDS.innerHTML=evSwitches(e);
    }
  }
  RB.hidden=RSK.hidden&&RNUM.hidden&&EVBAR.hidden;
}
function rpMsgBar(t){RCM.textContent=t||'';rpCmT=t?Date.now()+6000:0;}
function rpctl(url){return fetch(url,{method:'POST',headers:{'X-Gimbal':'1'}}).then(r=>r.json().catch(()=>({ok:false,text:'서버 응답 '+r.status}))).then(r=>{if(!r.ok)rpMsgBar(r.text||'실패');return r;}).catch(()=>{rpMsgBar('서버 연결 끊김');return {ok:false};}).finally(()=>setTimeout(()=>rptick(true),250));}
function rpSeek(pos){const st=rpLast&&rpLast.state;if(!st||!st.on)return;pos=Math.min(Math.max(0,Math.round(pos)),st.total-1);rpHold=pos;rpHoldT=Date.now();RSL.value=pos;if(!rpLast.result_views){rpctl('/replay/seek?pos='+pos);return;}const request=rpBeginPrefix(pos);if(!request)return;
  rpctl('/replay/seek?pos='+pos+'&pause=1').then(d=>{if(request.token!==rpEvalToken)return;if(!d.ok){rpEvalPending=false;rpEvalError='프레임 이동에 실패해 결과를 불러오지 않았습니다.';rpRenderEval(rpLast||{});return;}rpWaitForSeek(request);});}
function rpWaitForSeek(request){if(request.token!==rpEvalToken||rpEvalMode!=='slider')return;const st=rpLast&&rpLast.state;
  if(st&&st.pos===request.pos&&!st.seeking){rpLoadPrefix(request.pos,request.token,request.runId);return;}
  if(Date.now()-rpHoldT>8000){rpEvalPending=false;rpEvalError='이동한 프레임이 아직 표시되지 않아 결과를 불러오지 않았습니다.';rpRenderEval(rpLast||{});return;}
  setTimeout(()=>rptick(true),100);setTimeout(()=>rpWaitForSeek(request),250);}
RSL.addEventListener('pointerdown',()=>{rpDrag=true;});
RSL.addEventListener('input',()=>{const st=rpLast&&rpLast.state;if(st&&st.on)RTM.textContent=fmtT(Number(RSL.value)/(st.fps||25))+' / '+fmtT((st.total-1)/(st.fps||25))+' · '+(Number(RSL.value)+1)+' / '+st.total+'번째 장 (놓으면 건너뜀)';});
RSL.addEventListener('change',()=>{rpDrag=false;rpSeek(Number(RSL.value));});   // sent only when released
addEventListener('pointerup',()=>{if(rpDrag){rpDrag=false;}});
RPL.addEventListener('click',()=>{const st=rpLast&&rpLast.state;if(st&&st.on){if(st.paused||rpEnded(st)){rpEvalSetMode('live');rpRenderEval(rpLast||{});rpctl('/replay/play');}else rpctl('/replay/pause');}});
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
RPBACK.addEventListener('change',()=>{const b=RPBACK.value;if(rpLast&&rpLast.state&&rpLast.state.on)RPMSG.textContent='선택한 추적기는 다음 재생부터 적용됩니다: '+rpBackendLabel(b,rpLast);});
EVLIVE.addEventListener('click',()=>{rpEvalSetMode('live');rpRenderEval(rpLast||{});});
EVSLIDER.addEventListener('click',()=>{const st=rpLast&&rpLast.state;if(!st||!st.on||st.pos==null){EVVIEW.textContent='재생 중인 영상에서 슬라이더 위치를 선택할 수 있습니다.';return;}if(st.seeking){EVVIEW.textContent='프레임 이동이 끝난 뒤 결과를 다시 요청해 주세요.';return;}rpLoadPrefix(st.pos);});
EVFINAL.addEventListener('click',rpLoadFinal);
EVRESET.addEventListener('click',()=>{if(rpBusy)return;if(!confirm('현재 녹화를 처음부터 한 번 끝까지 재생해 최종 결과를 다시 계산할까요? 재생은 반복하지 않습니다.'))return;rpEvalSetMode('live');EVRESET.disabled=true;rpctl('/replay/eval/reset').finally(()=>{EVRESET.disabled=false;});});
RPFPS.addEventListener('change',()=>{const st=rpLast&&rpLast.state;if(!st||!st.on)return;const f=Math.round(Number(RPFPS.value));if(!(f>=5&&f<=30)){RPMSG.textContent='초당 장 수는 5~30';return;}rpctl('/replay/speed?fps='+f);});
RPST.addEventListener('click',()=>{const n=RPSEL.value;if(!n)return;const f=Math.round(Number(RPFPS.value));if(!(f>=5&&f<=30)){RPMSG.textContent='초당 장 수는 5~30';return;}
  if(!confirm('녹화 "'+n+'" 를 카메라 대신 재생할까요?\n카메라 영상은 멈추고, 재생 중에는 짐벌이 움직이지 않습니다.'))return;
  rppost('/replay/start?seq='+encodeURIComponent(n)+'&fps='+f+'&loop='+(RPLOOP.checked?1:0)+'&backend='+encodeURIComponent(RPBACK.value||'python'));});
function rpBackToCamera(){const st=rpLast&&rpLast.state;if(st&&st.on&&!st.busy&&!rpBusy)rppost('/replay/stop');}
RPSP.addEventListener('click',rpBackToCamera);
addEventListener('keydown',e=>{if(e.key==='Escape'&&!e.repeat&&!e.isComposing&&rpLast&&rpLast.state&&rpLast.state.on){e.preventDefault();rpBackToCamera();}});
RPSEC.addEventListener('toggle',()=>{if(RPSEC.open)rptick('refresh');});
setInterval(rptick,500);rptick(true);
