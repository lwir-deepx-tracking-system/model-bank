// connection chip (v24): good = answers keep coming, yellow = nothing for 1.5 s, red = nothing for 4 s. One failed request never turns it red; time asleep (phone screen off) does not count (wake() restarts the clock).
let lastOk=Date.now(),grace=Date.now(),camIdle=0,camFail=0,wasDown=false,sN=0;
function openStream(){v.src='/stream?'+(++sN);}
function netChip(){
  const idle=Date.now()-Math.max(lastOk,grace);
  expireGimbalPose();
  if(idle>4000){wasDown=true;chip('cdot','ctext','bad','서버 연결 끊김');S.textContent='서버 연결 끊김';$('sq').className='sq bad';$('fps').textContent='–';hzPrev=null;return;}
  if(idle>1500){wasDown=true;chip('cdot','ctext','warn','연결 불안정');return;}   // a 1.5 s+ gap may have cut the picture stream too
  if(wasDown){wasDown=false;openStream();}   // the picture stream died with the connection: open it again
  if(camFail>0){chip('cdot','ctext','bad','서버 연결됨 · 카메라 응답 없음 '+Math.round(camIdle)+'초 · 자동 복구 '+camFail+'회 실패 · USB 선과 전원을 확인하세요');return;}
  chip('cdot','ctext',camIdle>3?'warn':'good',camIdle>3?'서버 연결됨 · 카메라 영상 '+Math.round(camIdle)+'초째 안 들어옴 (스스로 복구 중)':'서버 연결됨');}
v.addEventListener('error',()=>setTimeout(openStream,2000));
// v25: the bridge answers in English; shown in Korean
const ANS={'no person there':'그 자리에 사람이 없습니다','that person is not available':'그 사람은 지금 고를 수 없습니다','the gimbal is not ready':'짐벌이 아직 준비되지 않았습니다'};
let lostSince=null,autoNow=false,hzPrev=null;
function show(t){lastOk=Date.now();camIdle=t.cam_idle||0;camFail=t.cam_fail||0;netChip();
  renderGimbalPose(t.pose, t.position_estimate);
  let txt,cls='good',sq='';
  if(!t.running){txt='짐벌 프로그램이 꺼져 있습니다';cls='bad';sq='bad';}
  else if(t.error){txt='짐벌 오류: 서버를 다시 켜야 합니다';cls='bad';sq='bad';}
  else if(t.lost){txt=t.id+'번 놓침, 잠시 기다립니다';cls='warn';sq='lost';}
  else if(t.auto){txt=t.id+'번을 따라가는 중';sq='auto';}
  else if(t.moving){txt='수동으로 움직이는 중';sq='man';}
  else if(t.ready===false){txt='짐벌 준비 중';cls='';}
  else{txt='대기 중 · 영상에서 사람을 클릭하세요';cls='';}
  S.textContent=txt;$('sq').className='sq '+sq;$('ans').textContent=t.answer?(ANS[t.answer]||t.answer):'';
  chip('gdot','gtext',cls,t.running?(t.auto?'추종 '+t.id+'번':t.moving?'수동 이동':'짐벌 대기'):'짐벌 꺼짐');
  if(t.lost){if(lostSince==null)lostSince=Date.now();$('lt').textContent=((Date.now()-lostSince)/1000).toFixed(1)+' 초';}
  else{lostSince=null;$('lt').textContent='–';}
  autoNow=!!(t.running&&t.auto&&!t.lost);if(!autoNow){$('ex').textContent='–';$('ey').textContent='–';}
}
