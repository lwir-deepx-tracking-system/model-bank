// v30: System buttons
let sysbusy=false,sysat=null;const CR=$('camreset'),SR=$('sysreboot');
function syspost(url,busyText){sysbusy=true;CR.disabled=SR.disabled=true;SM.textContent=busyText;
  return fetch(url,{method:'POST',headers:{'X-Gimbal':'1'}}).then(r=>r.json().catch(()=>({ok:false,text:r.status===403||r.status===404?'서버 재시작 후 쓸 수 있습니다':'서버 응답 '+r.status}))).then(r=>{if(r.ok===false&&rlast&&rlast.cam_reset)sysat=rlast.cam_reset.at+'|'+rlast.cam_reset.text;SM.textContent=r.text||(r.ok?'완료':'실패');return r;}).catch(()=>{SM.textContent='서버 연결 끊김';return {ok:false};});}
CR.addEventListener('click',()=>{if(!confirm('카메라를 초기화할까요?\n영상이 10~60초 끊깁니다.'))return;syspost('/camera/reset','카메라 초기화 요청 중…').finally(()=>{setTimeout(()=>{sysbusy=false;CR.disabled=SR.disabled=false;},3000);});});
SR.addEventListener('click',()=>{if(!confirm('오렌지파이를 재부팅할까요?\n1~2분 동안 화면과 짐벌이 끊기고, 보드에서 돌던 다른 작업도 모두 끊깁니다.'))return;syspost('/system/reboot?confirm=REBOOT','재부팅 요청 중…').then(r=>{setTimeout(()=>{sysbusy=false;CR.disabled=SR.disabled=false;},r.ok?120000:3000);});});
