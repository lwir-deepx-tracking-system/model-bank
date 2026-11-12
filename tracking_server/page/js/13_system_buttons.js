// System controls
let sysbusy=false, sysat=null;
const CR=$('camreset'), SR=$('sysreboot'), restartButton=$('serverrestart');
const feedToggle=$('npu-feed'), feedApply=$('npu-feed-apply');
const feedState=$('npu-feed-state'), feedCounts=$('npu-feed-counts'), feedHint=$('npu-feed-hint');
let feedChoice=null;

function syncSystemButtons(){
  const pending=!!(rlast&&rlast.restart_pending);
  CR.disabled=SR.disabled=sysbusy||pending;
  restartButton.disabled=sysbusy||pending||!(rlast&&rlast.server_restart);
  restartButton.title=rlast&&!rlast.server_restart?'표준 실행 스크립트로 서버를 시작한 뒤 사용할 수 있습니다':'';

  const reportedFeed=rlast?.npu_feed;
  const feed=reportedFeed?.mode==='normal'||reportedFeed?.mode==='arrival'?reportedFeed:null;
  const requested=feed?.requested==='arrival';
  const switching=feed?.pending===true;
  if(feed&&!switching&&feedChoice===requested)feedChoice=null;
  feedToggle.checked=feedChoice??requested;
  feedToggle.disabled=sysbusy||pending||switching||!feed;
  const selectedMode=feedToggle.checked?'arrival':'normal';
  feedApply.disabled=feedToggle.disabled||feed?.can_toggle!==true||
    (selectedMode===feed.requested&&selectedMode===feed.mode);

  let state='NPU 교대 투입 상태를 확인하는 중…';
  if(rlast&&!feed)state='이 서버에서는 NPU 교대 투입 설정을 제공하지 않습니다.';
  if(feed){
    state='현재 동작 · 교대 투입 '+(feed.active?'켜짐':'꺼짐');
    if(switching)state='전환 중 · '+state;
    else{
      if(requested!==feed.active)state=(requested?'켜기':'끄기')+' 설정됨 · '+state;
      if(feed.reason)state+=' · '+feed.reason;
    }
  }
  if(feedState.textContent!==state)feedState.textContent=state;

  feedCounts.textContent=(feed?.active||requested)&&Array.isArray(feed.cores)&&feed.cores.length===2
    ?'투입 장 수 · '+feed.cores.map((count,index)=>'NPU '+index+' '+count.toLocaleString('ko-KR')+'장').join(' · ')
    :'';
  const replay=document.body.classList.contains('replay-mode');
  const hint=feed&&!feed.can_toggle&&!sysbusy&&!pending&&!switching
    ?feed.toggle_reason||(replay?'녹화 재생을 먼저 끝내세요.':'현재 설정을 변경할 수 없습니다.')
    :'';
  if(feedHint.textContent!==hint)feedHint.textContent=hint;
}

function finishSystemRequest(){
  sysbusy=false;
  syncSystemButtons();
}

function syspost(url,busyText,successText){
  sysbusy=true;
  syncSystemButtons();
  SM.textContent=busyText;
  return fetch(url,{method:'POST',headers:{'X-Gimbal':'1'}})
    .then(r=>r.json().catch(()=>({ok:false,text:'서버 응답 '+r.status})))
    .then(result=>{
      if(result.ok===false&&rlast&&rlast.cam_reset)sysat=rlast.cam_reset.at+'|'+rlast.cam_reset.text;
      SM.textContent=result.ok&&successText!==undefined?successText:result.text||(result.ok?'완료':'실패');
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

feedToggle.addEventListener('change',()=>{
  feedChoice=feedToggle.checked;
  syncSystemButtons();
});
feedApply.addEventListener('click',async()=>{
  if(feedApply.disabled)return;
  const mode=feedToggle.checked?'arrival':'normal';
  try{
    const result=await syspost('/system/npu-feed?mode='+mode,'NPU 교대 투입 전환을 요청하는 중…','');
    if(result.ok)await rtick();
  }finally{
    finishSystemRequest();
  }
});

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
