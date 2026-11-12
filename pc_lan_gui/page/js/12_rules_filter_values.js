// rules / tracker (GET /rules every second)
function rshow(r){if(r.ok===false){RI.textContent='Tracker · '+(r.text||'Runtime 연결 오류');return;}rlast=r;
  if(othint&&JSON.stringify(r.reid_last||null)!==othint[1]){if(OH.textContent===othint[0])OH.textContent='';othint=null;}   // v41: the change finished (done or failed) -> the status line below says which
  const pl=r.predictors||null;
  PST.textContent=pl?pl.filter(p=>p.key!=='kalman').map(p=>(PN[p.key]?PN[p.key][0]:p.name)+': '+(p.available?'고를 수 있음':'고를 수 없음')+(p.note?' — '+p.note:'')).join(' / '):'이 서버 버전에는 위치 기억 선택이 없습니다. 서버를 다시 켜면 열립니다.';
  RWX.disabled=r.warm_max==null;   // v30: old server has no upper limit
  RC.disabled=!('confirm' in r);   // 11-11: an older server has no confirm_ms
  if(!rdirty){RW.value=r.warm_min;RWX.value=r.warm_max||0;RE.value=r.edge_birth;RC.value=r.confirm?r.confirm.ms:0;}
  const wx=r.warm_max||0,wp=r.warm_prev;
  let f='Filter · warmth '+(r.warm_min>0||wx>0?(r.warm_min>0?r.warm_min:'0')+' ~ '+(wx>0?wx:'제한 없음')+' 만 사람으로 인정':'검사 끔'+(wp&&(wp[0]>0||wp[1]>0)?' (켜면 '+wp[0]+' ~ '+(wp[1]>0?wp[1]:'제한 없음')+')':''))+' · 가장자리 새 궤적은 '+(r.edge_birth>0?'점수 '+r.edge_birth+' 이상':'제한 없음');
  if(r.warm_dropped!=null)f+=' · warmth 범위 밖이라 버린 상자 '+r.warm_dropped+'개';
  if(r.pending)f+=' · 다음 프레임에 적용';
  FI.textContent=f;
  const trk=pend(r,'tracker');
  let t='Tracker · '+tname(trk)+' · KeepID '+(pend(r,'relabel')?'켜짐':'꺼짐');
  if(r.confirm&&r.confirm.ms>0)t+=' · 새 사람 확인 '+r.confirm.ms+' ms = '+r.confirm.frames+'장'+(r.confirm.fps?' (지금 초당 '+r.confirm.fps+'장)':' (속도 재는 중)');   // 11-11 --confirm-ms
  if(trk==='botsort')t+=' · 외형 비교 '+(r.reids&&r.reids.length?(r.app==='off'?'끔':'켜짐'):r.apps?(AN[r.app]||r.app||'끔'):AN.repvgg);
  if(r.reids&&r.reids.length){const rc=r.reids.find(a=>a.key===r.reid),rp=r.reid_pending&&r.reids.find(a=>a.key===r.reid_pending),L=r.reid_last;   // v41
    t+=' · 외형 모델 '+(rc?rc.name:(r.reid||'–'));
    if(rp)t+=' → '+rp.name+' 로 바꾸는 중 (화면이 잠깐 멈춤)';
    else if(L)t+=' ('+L.at+' '+(L.ok?'바뀜':'바꾸지 못함 — '+L.text)+(L.seconds!=null?' · '+L.seconds+'초':'')+')';}
  if(pl){const pk=pend(r,'predictor');t+=' · 위치 기억 '+(PN[pk]?PN[pk][0]:pk);if(r.predictor!=='kalman')t+=' (놓친 사람 보정 '+r.pred_used+'회, 오류 '+r.pred_err+')';if(r.pred_show)t+=' · 예측 상자 표시';}
  if(r.stab)t+=' · 흔들림 보정 켜짐'+(r.stab_ms!=null?' (계산 '+(+r.stab_ms).toFixed(1)+' ms'+(r.stab_resets?', 다시 시작 '+r.stab_resets+'회':'')+')':'');
  if(r.stab_err)t+=' · 흔들림 보정 오류: '+r.stab_err;
  if(r.lead)t+=' · 예측 제어 켜짐'+(r.lead_ms!=null?' ('+r.lead_ms+' ms 앞을 겨눔'+(r.lead_px?', 지금 '+r.lead_px+' px':'')+')':'');
  if(r.pending)t+=' · 다음 프레임에 적용';
  RI.textContent=t;
  rdoc(trk);
  if(r.cam_reset&&r.cam_reset.text&&!sysbusy&&(r.cam_reset.busy||r.cam_reset.at+'|'+r.cam_reset.text!==sysat)){sysat=r.cam_reset.at+'|'+r.cam_reset.text;SM.textContent='카메라 초기화 '+r.cam_reset.at+' — '+r.cam_reset.text;}
  gshow(r.gimbal_cfg);
  otdraw();}
// 짐벌 제어기 구역: the options the server gave the gimbal program at start (GET /rules, key gimbal_cfg; read only). No key = older server, null = the server runs without a gimbal, a value that was not given = the program's own default.
let glast=null;
function gshow(g){const v=x=>x==null||x===''?'프로그램 기본값':'<b class="num">'+esc(isFinite(+x)?+x:x)+'</b>';let h;
  if(g===undefined||(g&&g.error))h='서버에서 값을 받지 못함';
  else if(g===null)h='짐벌 없이 실행 중입니다 (짐벌 프로그램을 띄우지 않음)';
  else{const a=+g.aim_y,an={0.08:'머리',0.3:'가슴',0.5:'상자 가운데'}[a],ms=+g.reacquire_ms;
    h='짐벌 프로그램 <b>'+esc(g.program||'–')+'</b> · 제어 방식 '+(g.controller?'<b>'+esc(String(g.controller).toUpperCase())+'</b>':'프로그램 기본값')+' · Kp '+v(g.kp)+' · Ki '+v(g.ki)+' · Kd '+v(g.kd)+' · 불감대 '+v(g.deadband)+' · 자동 속도 최대 '+v(g.speed_max)+' · 손으로 움직일 때 속도 '+v(g.manual_speed)
      +' · 놓친 사람을 기다리는 시간 '+(g.reacquire_ms!=null&&isFinite(ms)?'<b class="num">'+ms/1000+'</b>초':v(g.reacquire_ms))+' · 겨누는 높이 '+v(g.aim_y)+(g.aim_y!=null?' (사람 상자 위에서부터의 비율'+(an?' · '+an:'')+')':'')+' · 그 밖의 옵션 '+(g.extra?'<b class="num">'+esc(g.extra)+'</b>':'없음');}
  if(h!==glast){glast=h;$('gcfg').innerHTML=h;}}
function rtick(){if(document.hidden)return Promise.resolve();return fetch('/rules',{cache:'no-store'}).then(r=>{lastOk=Date.now();return r.json();}).then(rshow).catch(()=>{});}
// Filter numbers: Apply sends warm_min, warm_max, edge_birth and confirm_ms (11-11, only to a server that reports it) only. While 온도 Filter is off and both boxes still say 0, the warmth numbers are left out, so the numbers the server remembers are not wiped.
RA2.addEventListener('click',()=>{RA2.disabled=true;FM.textContent='보내는 중…';
  const woff=rlast&&rlast.warm_on===false&&!(+RW.value>0)&&!(+RWX.value>0);
  const q=(woff?'':'warm_min='+encodeURIComponent(RW.value||0)+(rlast&&rlast.warm_max!=null?'&warm_max='+encodeURIComponent(RWX.value||0):'')+'&')+'edge_birth='+encodeURIComponent(RE.value||0)+(rlast&&'confirm' in rlast&&(+RC.value||0)!==(rlast.confirm?rlast.confirm.ms:0)?'&confirm_ms='+encodeURIComponent(RC.value||0):'');   // only when this box was changed: a value sent once is remembered and then wins over LWIR_CONFIRM_MS at the next start
  fetch('/rules/set?'+q,{method:'POST',headers:{'X-Gimbal':'1'},cache:'no-store'}).then(x=>x.json().catch(()=>({ok:false,text:'서버 응답 '+x.status})))
    .then(r=>{if(r.ok!==false){rdirty=false;clearTimeout(rdt);}if(r.presets)rshow(r);FM.textContent=r.ok===false?'적용 안 됨: '+(r.text||''):'적용됨';})
    .catch(()=>{FM.textContent='서버 연결 끊김';}).finally(()=>{RA2.disabled=false;});});
