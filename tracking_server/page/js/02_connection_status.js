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
  chip('cdot','ctext',camIdle>3?'warn':'good',camIdle>3?'카메라 영상 기다리는 중':'서버 연결됨');$('ctext').parentElement.title=camIdle>3?'카메라 영상 '+Math.round(camIdle)+'초째 안 들어옴 (스스로 복구 중)':'';}
let reopenT=0;   // 11-15: one pending reopen at a time; none for a stream the page stopped itself. With fetch (picture_stream.js) an img error is one bad picture
v.addEventListener('error',()=>{if(streamFetch||reopenT||!v.hasAttribute('src'))return;reopenT=setTimeout(()=>{reopenT=0;openStream();},2000);});
// v25: the bridge answers in English; shown in Korean
const ANS={'no person there':'그 자리에 사람이 없습니다','that person is not available':'그 사람은 지금 고를 수 없습니다','the gimbal is not ready':'짐벌이 아직 준비되지 않았습니다'};
let lostSince=null,autoNow=false,hzPrev=null;
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
  let txt,cls='good',sq='';
  if(!t.running){txt='짐벌 프로그램이 꺼져 있습니다';cls='bad';sq='bad';}
  else if(t.error){txt='짐벌 오류: 서버를 다시 켜야 합니다';cls='bad';sq='bad';}
  else if(t.lost){const sp=t.seek?t.seek[0]:0;txt=t.id+'번 놓침, '+(sp===2?'짐벌이 주변을 찾는 중':sp===3?'예측 자리로 돌아가는 중':sp===5?'주변에 다른 사람이 있어 훑지 않고 기다립니다':'잠시 기다립니다');cls='warn';sq='lost';}   // 11-16: t.seek = the gimbal's search (bridge --seek 1)
  else if(t.auto){txt=t.id+'번을 따라가는 중';sq='auto';}
  else if(t.moving){txt='수동으로 움직이는 중';sq='man';}
  else if(t.ready===false){txt='짐벌 준비 중';cls='';}
  else{txt='대기 중 · '+(cameraClickBlock()||'영상에서 사람을 클릭하세요');cls='';}
  S.textContent=txt;$('sq').className='sq '+sq;$('ans').textContent=t.answer?(ANS[t.answer]||t.answer):'';
  chip('gdot','gtext',cls,t.running?(t.auto?'추종 '+t.id+'번':t.moving?'수동 이동':'짐벌 대기'):'짐벌 꺼짐');
  if(t.lost){if(lostSince==null)lostSince=Date.now();$('lt').textContent=((Date.now()-lostSince)/1000).toFixed(1)+' 초';}
  else{lostSince=null;$('lt').textContent='–';}
  autoNow=!!(t.running&&t.auto&&!t.lost);$('camera-center').hidden=!autoNow||document.body.classList.contains('replay-mode');if(!autoNow){$('ex').textContent='–';$('ey').textContent='–';}
  if(typeof followNote==='function')followNote(t);   // 11-13: notes over the picture
}
