// connection chip (v24): good = answers keep coming, yellow = nothing for 1.5 s, red = nothing for 4 s. One failed request never turns it red; time asleep (phone screen off) does not count (wake() restarts the clock).
let lastOk=Date.now(),grace=Date.now(),camIdle=0,camFail=0,wasDown=false,sN=0;
// 11-15 (화면 전송 지연): the picture stream carries this page's id and a count. The server ends this page's older stream when a
// newer one opens: a browser keeps the old stream running until the new one has drawn its first picture, so each reopen
// ran two streams over a slow link. A hidden page (background tab, phone screen off) takes no pictures; wake() reopens.
// The stream itself is read in picture_stream.js (streamStart / streamStop), which also opens the first one.
const pageId=Math.random().toString(36).slice(2,10)||'page';
function openStream(){if(document.hidden){stopStream();return;}streamStart('/stream?p='+pageId+'&n='+(++sN));}
function stopStream(){streamStop();}
function netChip(){
  const idle=Date.now()-Math.max(lastOk,grace);
  expireGimbalPose();
  if(idle>4000){wasDown=true;chip('cdot','ctext','bad','서버 연결 끊김');S.textContent='서버 연결 끊김';$('sq').className='sq bad';$('fps').textContent='–';hzPrev=null;return;}
  if(idle>1500){wasDown=true;chip('cdot','ctext','warn','연결 불안정');return;}   // a 1.5 s+ gap may have cut the picture stream too
  if(wasDown){wasDown=false;openStream();}   // the picture stream died with the connection: open it again
  const fz=rlast?.cam_frozen;   // 11-15: the camera's USB chip froze (cam_pipe exit 8): only a power cut helps (the server sends the Korean text)
  if(fz){chip('cdot','ctext','bad','카메라 멈춤 · USB 선 다시 꽂기');$('ctext').parentElement.title=fz.title+'. '+fz.text;return;}
  const sc=rlast?.cam_scene;   // 11-15: frames come but the picture has no scene (cam_pipe v2.13 SCENE_LOST; the server sends the Korean text)
  if(sc&&!document.body.classList.contains('replay-mode')){chip('cdot','ctext','bad',sc.reset?'카메라 장면 없음 · 자동 복구 중':'카메라 장면 없음 · USB 선 다시 꽂기');$('ctext').parentElement.title=sc.title+'. '+sc.text;return;}
  if(camFail>0){chip('cdot','ctext','bad','카메라 응답 없음 · USB 확인');$('ctext').parentElement.title='카메라 영상 '+Math.round(camIdle)+'초째 없음 · 자동 복구 '+camFail+'회 실패 · 카메라 USB 케이블을 뽑았다가 다시 꽂으세요';return;}
  const pd=rlast?.npu_feed?.auto?.period;   // 11-16: detection of this model is too slow for the camera period even on both NPU cores (feed_auto.py; the server sends the Korean text)
  if(pd?.state==='over'&&camIdle<=3){chip('cdot','ctext','warn','카메라 주기 못 지킴 · 검출 모델이 느림');$('ctext').parentElement.title=pd.text;return;}
  chip('cdot','ctext',camIdle>3?'warn':'good',camIdle>3?'카메라 영상 기다리는 중':'서버 연결됨');$('ctext').parentElement.title=camIdle>3?'카메라 영상 '+Math.round(camIdle)+'초째 안 들어옴 (스스로 복구 중)':'';}
let reopenT=0;   // 11-15: one pending reopen at a time; none for a stream the page stopped itself. With fetch (picture_stream.js) an img error is one bad picture
v.addEventListener('error',()=>{if(streamFetch||reopenT||!v.hasAttribute('src'))return;reopenT=setTimeout(()=>{reopenT=0;openStream();},2000);});
// v25: the bridge answers in English; shown in Korean
const ANS={'no person there':'그 자리에 사람이 없습니다','that person is not available':'그 사람은 지금 고를 수 없습니다','the gimbal is not ready':'짐벌이 아직 준비되지 않았습니다'};
let lostSince=null,lostId=null,autoNow=false,hzPrev=null;
let followFrame={id:0,held:false};
function followHeld(t){return !!(t.auto&&(t.lost||t.guard?.[1]===1));}
function followWaitText(t,cfg){
  if(!t.lost)return '검출 확인 중 · 짐벌 추종 보류';
  const search=t.search;
  const seekPhase=t.seek?.[0];
  if(search===5){
    if(seekPhase===2)return '짐벌이 주변을 찾는 중';
    if(seekPhase===3)return '예측 자리로 돌아가는 중';
    return '짐벌이 재검출 위치를 탐색 중';
  }
  if(search===3)return '짐벌이 주변을 찾는 중';
  if(search===2)return '마지막으로 본 방향으로 이동 중';
  if(seekPhase===5)return '주변에 다른 사람이 있어 기다리는 중';
  if(t.moving&&t.guard?.[3]===1)return '예측 위치로 이동 중';
  if(t.moving)return '짐벌 이동 중 · 같은 ID 재검출 대기';
  const timeout=Number(cfg?.reacquire_ms);
  const limit=Number.isFinite(timeout)&&timeout>0?' (최대 '+timeout/1000+'초)':'';
  return '같은 ID 재검출 대기'+limit;
}
function followBoxStyle(id){
  return followFrame.id===id ? (followFrame.held ? 'held' : 'active') : '';
}
function cameraClickBlock(){
  const replay=rpLast?.state;
  if(rpBusy||replay?.busy)return '영상 전환 중입니다';
  if(replay?.on)return '카메라 영상에서 사람을 선택할 수 있습니다';
  const camera=rlast?.camera_bytetrack;
  if(!!camera?.requested!==!!camera?.active)return '카메라 추적기 전환 중 · 다음 프레임 대기';
  if(camera?.control_supported===false)return '짐벌 연결 프로그램을 새로 빌드하고 서버를 다시 시작하세요';
  return '';
}
function show(t){lastOk=Date.now();camIdle=t.cam_idle||0;camFail=t.cam_fail||0;netChip();
  renderGimbalPose(t.pose, t.position_estimate);
  const followId=t.running&&!t.error&&t.auto?t.id:0;
  const held=!!followId&&followHeld(t);
  const holdText=held?t.id+'번'+(t.lost?' 놓침':'')+' · '+followWaitText(t,rlast?.gimbal_cfg):'';
  let txt,cls='good',sq='';
  if(!t.running){txt='짐벌 프로그램이 꺼져 있습니다';cls='bad';sq='bad';}
  else if(t.error){txt='짐벌 오류: 서버를 다시 켜야 합니다';cls='bad';sq='bad';}
  else if(held){txt=holdText;cls='warn';sq='lost';}
  else if(t.auto){txt=t.id+'번을 따라가는 중';sq='auto';}
  else if(t.moving){txt='수동으로 움직이는 중';sq='man';}
  else if(t.ready===false){txt='짐벌 준비 중';cls='';}
  else{txt='대기 중 · '+(cameraClickBlock()||'영상에서 사람을 클릭하세요');cls='';}
  S.textContent=txt;$('sq').className='sq '+sq;$('ans').textContent=t.answer?(ANS[t.answer]||t.answer):'';
  chip('gdot','gtext',cls,t.running?(held?t.id+'번 재확인 중':t.auto?'추종 '+t.id+'번':t.moving?'수동 이동':'짐벌 대기'):'짐벌 꺼짐');
  if(held&&t.lost){
    if(lostId!==t.id){lostId=t.id;lostSince=Date.now();}
    $('lt').textContent=((Date.now()-lostSince)/1000).toFixed(1)+' 초';
  }else{lostId=null;lostSince=null;$('lt').textContent='–';}
  autoNow=!!followId&&!held;$('camera-center').hidden=!autoNow||document.body.classList.contains('replay-mode');if(!autoNow){$('ex').textContent='–';$('ey').textContent='–';}
  if(followFrame.id!==followId||followFrame.held!==held){followFrame={id:followId,held};ovDraw();}
  followNote(t,holdText);
}
