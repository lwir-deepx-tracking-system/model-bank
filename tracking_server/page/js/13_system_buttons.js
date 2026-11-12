// System controls
let sysbusy=false, sysat=null;
const CR=$('camreset'), SR=$('sysreboot'), restartButton=$('serverrestart');

function syncSystemButtons(){
  const pending=!!(rlast&&rlast.restart_pending);
  CR.disabled=SR.disabled=sysbusy||pending;
  restartButton.disabled=sysbusy||pending||!(rlast&&rlast.server_restart);
  restartButton.title=rlast&&!rlast.server_restart?'표준 실행 스크립트로 서버를 시작한 뒤 사용할 수 있습니다':'';
}

function finishSystemRequest(){
  sysbusy=false;
  syncSystemButtons();
}

function syspost(url,busyText){
  sysbusy=true;
  syncSystemButtons();
  SM.textContent=busyText;
  return fetch(url,{method:'POST',headers:{'X-Gimbal':'1'}})
    .then(r=>r.json().catch(()=>({ok:false,text:'서버 응답 '+r.status})))
    .then(result=>{
      if(result.ok===false&&rlast&&rlast.cam_reset)sysat=rlast.cam_reset.at+'|'+rlast.cam_reset.text;
      SM.textContent=result.text||(result.ok?'완료':'실패');
      return result;
    })
    .catch(()=>{
      SM.textContent='서버 연결 끊김';
      return {ok:false};
    });
}

async function waitForServer(previous){
  const deadline=Date.now()+90000;
  SM.textContent='서버가 다시 연결되기를 기다리는 중…';
  while(Date.now()<deadline){
    try{
      const response=await fetch('/healthz',{cache:'no-store',signal:AbortSignal.timeout(3000)});
      if(response.ok){
        const health=await response.json();
        if(health.instance&&health.instance!==previous){
          openStream();
          void rtick();
          SM.textContent=health.frames_in>0?'서버 재시작 완료':'서버 재시작 완료. 카메라 영상 대기 중…';
          return;
        }
      }
    }catch(_){}
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
  SM.textContent='90초 안에 새 서버의 응답을 확인하지 못했습니다';
}

restartButton.addEventListener('click',async()=>{
  if(!confirm('8090 추적 서버를 재시작할까요? 영상이 잠시 끊깁니다.'))return;
  try{
    const result=await syspost('/system/restart','서버 재시작 요청 중…');
    if(result.ok)await waitForServer(result.instance);
  }finally{
    finishSystemRequest();
  }
});
CR.addEventListener('click',()=>{
  if(!confirm('카메라를 초기화할까요?\n영상이 10~60초 끊깁니다.'))return;
  syspost('/camera/reset','카메라 초기화 요청 중…').finally(()=>setTimeout(finishSystemRequest,3000));
});
SR.addEventListener('click',()=>{
  if(!confirm('오렌지파이를 재부팅할까요?\n1~2분 동안 화면과 짐벌이 끊기고, 보드에서 돌던 다른 작업도 모두 끊깁니다.'))return;
  syspost('/system/reboot?confirm=REBOOT','재부팅 요청 중…')
    .then(r=>setTimeout(finishSystemRequest,r.ok?120000:3000));
});
