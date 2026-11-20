function otdraw(){OTT=tiers();if(!OTT.length)return;let h='';
  const body=(t,i)=>{let h='';
    const btn=(n,j)=>'<button type="button" class="ot-n'+(n.lock?' lock':n.sel?' sel':'')+'" data-i="'+i+'.'+j+'"'+(n.lock?' aria-disabled="true"':' aria-pressed="'+(n.sel?'true':'false')+'"')+(n.note?' title="'+esc(n.note)+'"':'')+'>'+(n.lock?LK+' ':t.multi&&!n.pick?'<i class="ot-box'+(n.sel?' on':'')+'"></i>':'')+wn(n.name)+(n.rec?'<span class="ot-rec">추천</span>':'')+(n.latest?'<span class="ot-rec">최신</span>':'')+(n.note?'<small>'+esc(n.note)+'</small>':'')+(n.lock&&n.lock!==n.note?'<small class="ot-why">'+esc(n.lock)+'</small>':'')+'</button>';   // 11-13: why a node is locked, shown in it
    const tile=(n,j)=>{const kids=n.sel&&n.key?OTT.map((c,ki)=>[c,ki]).filter(([c])=>c.kidOf===i&&c.show&&c.kidKey===n.key):[];   // v42: a chosen tile that owns child rows becomes one box with the children inside. 11-16: all of them (only the first was drawn, so BoT-SORT had no engine row)
      if(!kids.length)return btn(n,j);
      return '<div class="ot-grp sel ot-own">'+btn(n,j)+kids.map(([c,ki])=>'<div class="ot-kid" data-kid="'+ki+'"><div class="ot-kidlab">'+esc(c.lab)+(c.sub?'<small>'+esc(c.sub)+'</small>':'')+'</div><div class="ot-nodes">'+body(c,ki)+'</div></div>').join('')+'</div>';};
    if(!t.f)t.n.forEach((n,j)=>{h+=tile(n,j);});
    else{t.n.forEach((n,j)=>{if(!n.g||!t.f.includes(n.g))h+=tile(n,j);});
      const sg=(t.n.find(n=>n.sel&&n.g)||{}).g||'',op=(i in otopen)?otopen[i]:sg;
      t.f.forEach((f,k)=>{const inn=t.n.filter(n=>n.g===f),s=inn.find(n=>n.sel),all=inn.length&&inn.every(n=>n.lock);
        if(inn.length===1){h+=tile(inn[0],t.n.indexOf(inn[0]));return;}   // 11-16 (design review 10): a folder that holds one model is that model's tile (one tap less)
        if(op===f)h+='<div class="ot-grp'+(s?' sel':'')+'">';
        h+='<button type="button" class="ot-n ot-f'+(s?' sel':!inn.length||all?' lock':'')+(op===f?' open':'')+'" data-f="'+i+'.'+k+'" aria-expanded="'+(op===f?'true':'false')+'">'+FD+' '+esc(f)+' '+(op===f?'▾':'▸')+'<small>'+(s?(op===f?'':'선택: '+wn(s.name)):inn.length?'모델 '+inn.length+'개'+(all?' · 모두 잠김':''):'모델 없음')+'</small></button>';
        if(op===f){let b='';const sz=[];inn.forEach(n=>{if(n.sg&&!sz.includes(n.sg))sz.push(n.sg);});sz.sort((a,b)=>szRank(a)-szRank(b));   // 11-13: size sub-folders smallest first (n, nano, tiny, s, m, l, x). the models sit inside the folder's own box
          t.n.forEach((n,j)=>{if(n.g===f&&(sz.length<2||!n.sg))b+=tile(n,j);});
          if(sz.length>1)sz.forEach(q=>{const si=inn.filter(n=>n.sg===q),ss=si.find(n=>n.sel),key=i+'|'+f+'|'+q,so=(key in otsub)?otsub[key]:!!ss;   // two or more sizes: one sub-folder per size
            b+='<div class="ot-sg'+(ss?' sel':'')+'"><button type="button" class="ot-n ot-f ot-sf'+(ss?' sel':si.every(n=>n.lock)?' lock':'')+(so?' open':'')+'" data-s="'+esc(key)+'" aria-expanded="'+(so?'true':'false')+'">'+FD+' '+esc(q)+' '+(so?'▾':'▸')+'<small>'+(ss?(so?'':'선택: '+wn(ss.name)):'모델 '+si.length+'개')+'</small></button>';
            if(so){b+='<div class="ot-sub">';t.n.forEach((n,j)=>{if(n.g===f&&n.sg===q)b+=tile(n,j);});b+='</div>';}
            b+='</div>';});
          h+='<div class="ot-sub">'+(b||'<span class="ot-none">'+esc((t.fe&&t.fe[f])||'이 안에 모델이 없습니다')+'</span>')+'</div></div>';}});}
    return h;};
  const row=([t,i])=>'<div class="ot-row'+(t.n.every(n=>n.lock)?' off':'')+'"><div class="ot-lab">'+esc(t.lab)+(t.sub?'<small>'+esc(t.sub)+'</small>':'')+'</div><div class="ot-nodes">'+body(t,i)+'</div></div>';
  const top=OTT.map((t,i)=>[t,i]).filter(([t])=>t.kidOf==null),adv=top.filter(([t])=>!OTG.some(g=>g[1].includes(t.lab)));   // v41: child rows are drawn inside their parent row
  OTG.forEach(([g,labs])=>{const rows=top.filter(([t])=>labs.includes(t.lab));if(!rows.length)return;h+=(g?'<div class="ot-g">'+esc(g)+'</div>':'')+rows.map(row).join('');});
  if(adv.length){const on=adv.map(([t])=>{const s=t.n.filter(n=>n.sel&&n.name!=='끔'&&!n.lock).map(n=>n.name);return s.length?t.lab+': '+s.join(', '):'';}).filter(Boolean);
    h+='<div class="ot-adv'+(otadv?' open':'')+'"><button type="button" class="ot-advh" data-adv="1" aria-expanded="'+(otadv?'true':'false')+'"><b>고급 설정 '+adv.length+'개 '+(otadv?'▾':'▸')+'</b><small>'+(on.length?'켜진 것 · '+wn(on.join(' · ')):'켜진 것 없음')+'</small></button>'+(otadv?adv.map(row).join(''):'')+'</div>';}
  if(h!==othtml){othtml=h;OT.innerHTML=h;}   // redrawn only when something changed (keeps focus and the pressed look between the 1 s refreshes)
  const R=l=>OTT.find(t=>t.lab===l)||{n:[]};   // 11-13: rows by name (the '외형 모델' row moved the numbers)
  const on=t=>t.n.filter(n=>n.sel).map(n=>n.name),one=t=>(t.n.find(n=>n.sel)||{}).name,trk=pend(rlast,'tracker'),pre=on(R('전처리')),flt=on(R('검출 필터')),ctl=(window.OTFOL?['따라가기']:[]).concat(on(R('제어')));
  const am=one(R('외형 모델')),ap=one(R('외형 비교')),kp=one(R('KeepID'));
  // 11-13 (design review 3): one line per stage, label + value (it was one long arrow chain)
  const e2eOn=one(R('엔드투엔드')),e2eUse=!!(e2eOn&&e2eOn!=='끔');   // 11-14: an end-to-end model replaces detection + tracker
  const a=(e2eUse?[['엔드투엔드',e2eOn]]:[]).concat((pre.length?[['전처리',pre.join(' + ')]]:[]).concat([['검출',e2eUse?'– (엔드투엔드 사용 중)':(one(R('검출 모델'))||'–')+(one(R('검출 모델 자동 선택'))==='켬'?' · 자동 선택':'')],
    ['필터',flt.length?flt.join(' + '):'없음'],['추적기',e2eUse?'– (엔드투엔드 사용 중)':one(R('추적기'))||tname(trk)]]));
  if(trk==='botsort')a.push(['외형 비교',ap==='끔'?'끔':(am||ap||'–')]);
  a.push(['위치 예측',one(R('위치 예측'))||'Kalman filter'],['KeepID',kp&&kp!=='끔'?(kp==='켬'?(am||'켬'):kp):'끔'],['제어',ctl.length?ctl.join(' + '):'수동']);
  const sh=a.map(([k,v])=>'<b>'+esc(k)+'</b><span>'+wn(v)+'</span>').join('');if(OS.innerHTML!==sh)OS.innerHTML=sh;
  // 11-16 (design review 1): the combination in use as one line under the section title (shown while the section is folded; the detector's long name comes last so the cut falls there)
  const det=one(R('검출 모델')),now=e2eUse?e2eOn:(one(R('추적기'))||tname(trk))+' · KeepID '+(kp&&kp!=='끔'?'켬':'끔')+' · '+(det||'–')+(mlast&&mlast.busy?' (바꾸는 중…)':'');
  if(ON&&ON.textContent!==now)ON.textContent=now;}
const ON=$('ot-now');OH.addEventListener('click',()=>{OH.textContent='';});   // 11-16: the answer strip floats over the tree; a tap puts it away
OT.addEventListener('click',e=>{if(e.target.closest('[data-adv]')){otadv=!otadv;try{localStorage.setItem('lwir_live_otadv',otadv?'1':'0');}catch(_){}OH.textContent='';otdraw();return;}   // '고급 설정' fold: open / close, nothing is sent
  const sb=e.target.closest('.ot-sf'),fb=e.target.closest('.ot-f');
  if(sb&&sb.dataset.s){otsub[sb.dataset.s]=sb.getAttribute('aria-expanded')!=='true';OH.textContent='';otdraw();return;}   // size sub-folder: open / close, nothing is sent
  if(fb&&fb.dataset.f){const x=fb.dataset.f.split('.'),t=OTT[+x[0]],f=t&&t.f&&t.f[+x[1]];if(!f)return;
    const sg=(t.n.find(n=>n.sel&&n.g)||{}).g||'',op=(+x[0] in otopen)?otopen[+x[0]]:sg;otopen[+x[0]]=op===f?'':f;OH.textContent='';otdraw();return;}
  const b=e.target.closest('.ot-n');if(!b||!b.dataset.i)return;const ix=b.dataset.i.split('.'),t=OTT[+ix[0]],n=t&&t.n[+ix[1]];if(!n)return;
  if(n.lock){OH.textContent=n.name+': '+n.lock;return;}
  if(n.info){OH.textContent=n.info;return;}
  if(n.e2eOff){if(!otbusy)rpE2EOff();return;}   // 11-14: back from an end-to-end model to detector + tracker (the chosen tracker, no question)
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
