function otdraw(){OTT=tiers();if(!OTT.length)return;let h='';
  const body=(t,i)=>{let h='';
    const btn=(n,j)=>'<button type="button" class="ot-n'+(n.lock?' lock':n.sel?' sel':'')+'" data-i="'+i+'.'+j+'"'+(n.lock?' aria-disabled="true"':' aria-pressed="'+(n.sel?'true':'false')+'"')+(n.note?' title="'+esc(n.note)+'"':'')+'>'+(n.lock?LK+' ':t.multi&&n.sel?'✓ ':'')+esc(n.name)+(n.rec?'<span class="ot-rec">추천</span>':'')+(n.note?'<small>'+esc(n.note)+'</small>':'')+(n.lock&&n.lock!==n.note?'<small class="ot-why">'+esc(n.lock)+'</small>':'')+'</button>';   // 11-13: why a node is locked, shown in it
    if(!t.f)t.n.forEach((n,j)=>{const ki=n.sel&&n.key?OTT.findIndex(c=>c.kidOf===i&&c.show&&c.kidKey===n.key):-1;   // v42: a chosen tile that owns a child row becomes one box with the child inside
      if(ki<0){h+=btn(n,j);return;}const c=OTT[ki];
      h+='<div class="ot-grp sel ot-own">'+btn(n,j)+'<div class="ot-kid" data-kid="'+ki+'"><div class="ot-kidlab">'+esc(c.lab)+(c.sub?'<small>'+esc(c.sub)+'</small>':'')+'</div><div class="ot-nodes">'+body(c,ki)+'</div></div></div>';});
    else{t.n.forEach((n,j)=>{if(!n.g||!t.f.includes(n.g))h+=btn(n,j);});
      const sg=(t.n.find(n=>n.sel&&n.g)||{}).g||'',op=(i in otopen)?otopen[i]:sg;
      t.f.forEach((f,k)=>{const inn=t.n.filter(n=>n.g===f),s=inn.find(n=>n.sel),all=inn.length&&inn.every(n=>n.lock);
        if(op===f)h+='<div class="ot-grp'+(s?' sel':'')+'">';
        h+='<button type="button" class="ot-n ot-f'+(s?' sel':!inn.length||all?' lock':'')+(op===f?' open':'')+'" data-f="'+i+'.'+k+'" aria-expanded="'+(op===f?'true':'false')+'">'+FD+' '+esc(f)+' '+(op===f?'▾':'▸')+'<small>'+(s?'선택: '+esc(s.name):inn.length?'모델 '+inn.length+'개'+(all?' · 모두 잠김':''):'모델 없음')+'</small></button>';
        if(op===f){let b='';const sz=[];inn.forEach(n=>{if(n.sg&&!sz.includes(n.sg))sz.push(n.sg);});sz.sort((a,b)=>szRank(a)-szRank(b));   // 11-13: size sub-folders smallest first (n, nano, tiny, s, m, l, x). the models sit inside the folder's own box
          t.n.forEach((n,j)=>{if(n.g===f&&(sz.length<2||!n.sg))b+=btn(n,j);});
          if(sz.length>1)sz.forEach(q=>{const si=inn.filter(n=>n.sg===q),ss=si.find(n=>n.sel),key=i+'|'+f+'|'+q,so=(key in otsub)?otsub[key]:!!ss;   // two or more sizes: one sub-folder per size
            b+='<div class="ot-sg'+(ss?' sel':'')+'"><button type="button" class="ot-n ot-f ot-sf'+(ss?' sel':si.every(n=>n.lock)?' lock':'')+(so?' open':'')+'" data-s="'+esc(key)+'" aria-expanded="'+(so?'true':'false')+'">'+FD+' '+esc(q)+' '+(so?'▾':'▸')+'<small>'+(ss?'선택: '+esc(ss.name):'모델 '+si.length+'개')+'</small></button>';
            if(so){b+='<div class="ot-sub">';t.n.forEach((n,j)=>{if(n.g===f&&n.sg===q)b+=btn(n,j);});b+='</div>';}
            b+='</div>';});
          h+='<div class="ot-sub">'+(b||'<span class="ot-none">'+esc((t.fe&&t.fe[f])||'이 안에 모델이 없습니다')+'</span>')+'</div></div>';}});}
    return h;};
  OTT.forEach((t,i)=>{if(t.kidOf!=null)return;   // v41: child rows are drawn inside their parent row
    h+='<div class="ot-row'+(t.n.every(n=>n.lock)?' off':'')+'"><div class="ot-lab">'+esc(t.lab)+(t.sub?'<small>'+esc(t.sub)+'</small>':'')+'</div><div class="ot-nodes">'+body(t,i);
    h+='</div></div>';});
  if(h!==othtml){othtml=h;OT.innerHTML=h;}   // redrawn only when something changed (keeps focus and the pressed look between the 1 s refreshes)
  const R=l=>OTT.find(t=>t.lab===l)||{n:[]};   // 11-13: rows by name (the '외형 모델' row moved the numbers)
  const on=t=>t.n.filter(n=>n.sel).map(n=>n.name),one=t=>(t.n.find(n=>n.sel)||{}).name,trk=pend(rlast,'tracker'),pre=on(R('전처리')),flt=on(R('검출 뒤 Filter')),ctl=(window.OTFOL?['따라가기']:[]).concat(on(R('제어')));
  const am=one(R('외형 모델')),ap=one(R('외형 비교')),kp=one(R('KeepID'));
  const a=[pre.length?pre.join(' + '):'전처리 없음',one(R('검출 모델'))||'검출 모델 –',flt.length?flt.join(' + '):'Filter 없음',one(R('Tracker'))||tname(trk)];
  if(trk==='botsort')a.push('외형 '+(ap==='끔'?'끔':(am||ap||'–')));
  a.push(one(R('위치 기억 · 신뢰도'))||'Kalman filter','KeepID '+(kp&&kp!=='끔'?(kp==='켬'?(am||'켬'):kp):'끔'),ctl.length?ctl.join(' + '):'수동');
  OS.textContent=a.join(' → ');}
OT.addEventListener('click',e=>{const sb=e.target.closest('.ot-sf'),fb=e.target.closest('.ot-f');
  if(sb&&sb.dataset.s){otsub[sb.dataset.s]=sb.getAttribute('aria-expanded')!=='true';OH.textContent='';otdraw();return;}   // size sub-folder: open / close, nothing is sent
  if(fb&&fb.dataset.f){const x=fb.dataset.f.split('.'),t=OTT[+x[0]],f=t&&t.f&&t.f[+x[1]];if(!f)return;
    const sg=(t.n.find(n=>n.sel&&n.g)||{}).g||'',op=(+x[0] in otopen)?otopen[+x[0]]:sg;otopen[+x[0]]=op===f?'':f;OH.textContent='';otdraw();return;}
  const b=e.target.closest('.ot-n');if(!b||!b.dataset.i)return;const ix=b.dataset.i.split('.'),t=OTT[+ix[0]],n=t&&t.n[+ix[1]];if(!n)return;
  if(n.lock){OH.textContent=n.name+': '+n.lock;return;}
  if(n.info){OH.textContent=n.info;return;}
  if(n.replayBackend){
    if(!n.sel&&!otbusy)rpUseTracker(n.replayBackend);
    return;
  }
  if(t.kidKey==='bytetrack'&&n.name==='Python'&&n.sel)return;
  if(!n.u||(!t.multi&&n.sel)){OH.textContent='';return;}   // the node already chosen in a pick-one row: nothing to send
  if(otbusy)return;
  if(n.ask&&!confirm(n.ask))return;
  otbusy=true;OH.textContent='보내는 중…';othint=n.hint?[n.hint,JSON.stringify((rlast&&rlast.reid_last)||null)]:null;
  fetch(n.u,{method:'POST',headers:{'X-Gimbal':'1'},cache:'no-store'}).then(x=>x.json().catch(()=>({ok:false,text:'서버 응답 '+x.status})))
    .then(r=>{if(n.model){if(r&&r.catalog)mdl(r);}else if(r&&r.presets)rshow(r);OH.textContent=r&&r.ok===false?'적용 안 됨 — '+(r.text||''):(othint?othint[0]:'');if(r&&r.ok===false)othint=null;
      // 11-13 (user): a model change during a replay plays the recording again from the first frame (server), so the numbers start over
      if(n.model&&!(r&&r.ok===false)&&rpLast&&rpLast.state&&rpLast.state.on){rpEvalSetMode('live');RPMSG.textContent='모델을 바꿔서 녹화를 처음부터 다시 재생하고 지표도 처음부터 다시 셉니다';OH.textContent=RPMSG.textContent;}})
    .catch(()=>{OH.textContent='서버 연결 끊김';}).finally(()=>{otbusy=false;});});
