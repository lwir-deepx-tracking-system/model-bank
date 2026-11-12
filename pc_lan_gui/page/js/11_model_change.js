// detector model (GET /model every second; the chip in the top bar and the first line of "지금 적용된 값")
function mdl(m){if(m.ok===false){MIN.textContent='검출 모델 · '+(m.text||'Runtime 연결 오류');chip('mcdot','mctext','bad','모델 목록 오류');return;}mlast=m;const cat=m.catalog||[];
  if(!cat.length){MIN.textContent='검출 모델 · 이 실행에서는 모델을 바꿀 수 없습니다.';chip('mcdot','mctext','','모델 고정');otdraw();return;}
  const cur=cat.find(c=>c.id===m.current);
  chip('mcdot','mctext',m.busy?'live':'good',m.busy?'모델 바꾸는 중':(cur?cur.name:'모델'));
  let t=cur?'지금: '+cur.name:'';
  if(m.busy){const p=cat.find(c=>c.id===m.pending);t='바꾸는 중'+(p?': '+p.name:'')+' · 카메라 영상이 들어오는 다음 프레임에 바뀝니다';}
  else if(m.last){t+=' · '+m.last.at+' '+m.last.text+(m.last.seconds!=null?' ('+m.last.seconds+'초)':'');}
  MIN.textContent='검출 모델 · '+t;otdraw();}
function mtick(){if(document.hidden)return Promise.resolve();return fetch('/model',{cache:'no-store'}).then(r=>{lastOk=Date.now();return r.json();}).then(mdl).catch(()=>{});}
setInterval(mtick,1000);mtick();
