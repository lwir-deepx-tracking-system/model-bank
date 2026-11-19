// Collaboration uses original frame numbers and compares only the frames being saved.
(()=>{
  const q=new URLSearchParams(location.search),seq=q.get('seq');
  if(!seq||!$('collab-card'))return;
  const owner=draftId(),user=q.get('user')?.trim().slice(0,40)||'작업자 '+owner.slice(0,4);
  const url=action=>'/api/'+action+'?'+new URLSearchParams({seq,user});
  let enabled=false,claim=null,expires=0,claims=[],claimBusy=false,renewBusy=false,saveBusy=false,autoOff=false;
  let rawBase=new Map(),normalBase=new Map(),recoveredBefore=new Map();
  let keyboardClaim=false,shortcutBusy=false;
  const emptyFrame={mot:[],parts:[],reviewed:false,removed:false};
  const group=(items,key)=>{
    const byFrame=new Map();
    for(const item of items){const frame=key(item);if(!byFrame.has(frame))byFrame.set(frame,[]);byFrame.get(frame).push(item);}
    return byFrame;
  };
  function frameState(mot,parts,checked,excluded){
    const lines=group(mot.split(/\r?\n/).filter(Boolean),line=>Number(line.split(',',1)[0]));
    const annotations=group(parts.annotations.map(clonePart),row=>row.frame);
    const states=new Map();
    for(const f of new Set([...lines.keys(),...annotations.keys(),...checked,...excluded])){
      states.set(f,{mot:lines.get(f)||[],parts:(annotations.get(f)||[]).sort((a,b)=>a.id-b.id),
                    reviewed:checked.has(f),removed:excluded.has(f)});
    }
    return states;
  }
  const canonical=value=>JSON.stringify(value,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)
    ?Object.fromEntries(Object.keys(v).sort().map(key=>[key,v[key]])):v);
  const equal=(a,b)=>canonical(a)===canonical(b);
  const stateAt=(map,f)=>map.get(f)||emptyFrame;
  function currentState(){
    return frameState(toMot(),partsDocument(),reviewed,removed);
  }
  function changes(before,after){
    return [...new Set([...before.keys(),...after.keys()])].sort((a,b)=>a-b)
      .filter(f=>!purged.has(f)&&!equal(stateAt(before,f),stateAt(after,f)));
  }
  function patchFrom(current){
    return changes(normalBase,current).map(frame=>({
      frame,before:recoveredBefore.get(frame)||stateAt(rawBase,frame),after:stateAt(current,frame)
    }));
  }
  function applyFrames(entries){
    const affected=new Set(entries.map(entry=>entry.frame));
    for(const [key,row] of partRecords)if(affected.has(row.frame))partRecords.delete(key);
    for(const {frame,after} of entries){
      const bs=parseMot(after.mot.join('\n')).get(frame);
      if(bs)boxes.set(frame,bs);else boxes.delete(frame);
      if(after.reviewed)reviewed.add(frame);else reviewed.delete(frame);
      if(after.removed||purged.has(frame))removed.add(frame);else removed.delete(frame);
      for(const row of after.parts)partRecords.set(partKey(frame,row.id),clonePart(row));
    }
  }
  function setBase(snapshot){
    rawBase=frameState(snapshot.mot,snapshot.parts,new Set(snapshot.review.reviewed||[]),
                       new Set(snapshot.review.removed||[]));
    normalBase=currentState();window.LWIR_BASE=snapshot.version;
  }
  function replaceBase(snapshot,reapply){
    const selectedId=selBox()?.id??focusedTrackId;
    boxes=parseMot(snapshot.mot);
    reviewed=new Set(snapshot.review.reviewed||[]);
    removed=new Set(snapshot.review.removed||[]);
    purged=new Set(snapshot.review.purged||[]);
    for(const f of purged)removed.add(f);
    loadParts(snapshot.parts);setBase(snapshot);
    recoveredBefore=new Map(reapply.filter(x=>x.before).map(x=>[x.frame,x.before]));
    applyFrames(reapply);
    carryToFrame(cur,selectedId);
    undo=[];
  }
  async function json(action,body){
    const options=body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{cache:'no-store'};
    const response=await fetch(url(action),options);
    const result=await response.json();
    if(!response.ok){const err=Error(result.error||response.status);err.code=response.status;err.frames=result.frames;throw err;}
    return result;
  }
  const own=()=>claim&&Date.now()<expires?claim:null;
  const covers=(range,frame)=>range&&frame>=range.first&&frame<=range.last;
  function panel(){
    if(!enabled)return;
    const range=own();
    const holder=claims.find(c=>covers(c,cur));
    const here=range&&covers(range,cur)?'이 장 편집 가능':holder&&!holder.mine?'이 장은 '+holder.user+' 님 작업 중':'이 장은 읽기 전용';
    $('collab-state').textContent=(range?'내 범위 '+range.first+'~'+range.last+'번 · ':'')+here;
    $('claim-list').textContent=claims.length?claims.map(c=>(c.mine?'내 범위':c.user)+' '+c.first+'~'+c.last+'번').join(' · '):'맡은 범위 없음';
    $('claim-release').disabled=claimBusy||!range;
    $('claim-current').disabled=claimBusy||!cur;
    $('claim-range').disabled=claimBusy;
  }
  async function listClaims(){
    const response=await fetch(url('claims')+'&owner='+encodeURIComponent(owner),{cache:'no-store'});
    if(response.ok){claims=(await response.json()).claims;panel();}
  }
  function allowed(frames,quiet){
    if(!enabled)return true;
    if(claimBusy){if(!quiet)status('작업 범위를 확인하는 중입니다.');return false;}
    const range=own();
    const missing=frames.find(f=>!covers(range,f));
    if(range&&missing===undefined)return true;
    if(!quiet)status(!range?'함께 라벨링에서 현재 장이나 작업 범위를 먼저 맡으세요.'
      :'원본 '+missing+'번은 내 작업 범위 밖입니다.');
    return false;
  }
  window.LWIR_CAN_EDIT=allowed;
  window.LWIR_COLLAB_PANEL=panel;
  window.LWIR_COLLAB_PATCH=()=>enabled&&!window.LWIR_DRAFT_CONFLICT?patchFrom(currentState()):null;
  window.LWIR_COLLAB_RESTORE=patch=>{
    if(!enabled)return false;
    recoveredBefore=new Map(patch.map(entry=>[entry.frame,entry.before]));
    applyFrames(patch);return true;
  };
  window.LWIR_COLLAB_INIT=snapshot=>{
    if(!snapshot?.frame_claims)return;
    enabled=true;setBase(snapshot);
    $('collab-card').hidden=false;
    $('claim-first').max=nFrames;$('claim-last').max=nFrames;
    listClaims().catch(()=>{});panel();
  };
  function adopt(response){
    claim=response.claim;
    expires=Date.now()+response.claim.lease_seconds*1000;
    if(response.claims)claims=response.claims;
  }
  async function claimRange(first,last,fromShortcut=false){
    if(claimBusy||renewBusy||saveBusy)return false;
    if(!Number.isSafeInteger(first)||!Number.isSafeInteger(last)||first<1||last>nFrames||first>last){
      status('원본 시작과 끝을 1~'+nFrames+' 사이의 정수로 입력하세요.');return;
    }
    claimBusy=true;panel();
    try{
      // Lock the editor until both the lease and its latest labels arrive.
      const pending=patchFrom(currentState());
      if(pending.some(row=>row.frame<first||row.frame>last)){
        status('미저장 편집이 있는 원본 프레임을 모두 포함하는 범위를 맡으세요.');return;
      }
      const previous=claim;
      const acquired=await json('claim',{owner,user,first,last});
      try{
        const current=await json('annotations');
        replaceBase(current,pending);
      }catch(error){
        if(previous&&own()){
          const restored=await json('claim',{owner,user,first:previous.first,last:previous.last});
          adopt(restored);
        }else{
          claim=null;expires=0;await json('release',{owner});
        }
        throw error;
      }
      adopt(acquired);keyboardClaim=fromShortcut;autoOff=false;
      $('claim-first').value=first;$('claim-last').value=last;
      status('원본 '+first+'~'+last+'번을 맡았습니다.');
      if(cur<first||cur>last)go(first);else refresh();
      return true;
    }catch(error){status(error.code===409?'다른 사람이 이미 맡은 프레임입니다.':'범위를 맡지 못했습니다. '+error.message);await listClaims().catch(()=>{});}
    finally{claimBusy=false;panel();}
  }
  window.LWIR_CLAIM_FOR_SHORTCUT=async frame=>{
    if(!enabled||claimBusy||renewBusy||saveBusy||shortcutBusy)return false;
    if(own()&&!keyboardClaim){allowed([frame],false);return false;}
    shortcutBusy=true;
    try{
      if(own()&&dirty){await window.LWIR_COLLAB_SAVE(false);if(dirty)return false;}
      return Boolean(await claimRange(frame,frame,true));
    }finally{shortcutBusy=false;}
  };
  $('claim-current').onclick=()=>claimRange(cur,cur);
  $('claim-range').onclick=()=>claimRange($('claim-first').valueAsNumber,$('claim-last').valueAsNumber);
  window.LWIR_COLLAB_SAVE=async(auto)=>{
    if(!enabled)return false;
    if(saveBusy||claimBusy||!dirty||auto&&autoOff)return true;
    if(window.LWIR_DRAFT_CONFLICT){
      status('이전 편집기의 초안은 서버 작업과 바로 합칠 수 없습니다. 작업 내려받기로 보관하세요.');return true;
    }
    saveBusy=true;
    try{
      const sent=currentState(),patch=patchFrom(sent);
      if(!patch.length){dirty=false;$('save').disabled=true;clearOwnDraft();return true;}
      if(!allowed(patch.map(row=>row.frame),auto)){persistDraft();return true;}
      const result=await json('save',{owner,editor_schema:3,frame_patch:patch});
      const current=currentState();
      const pending=changes(sent,current).map(frame=>({frame,after:stateAt(current,frame)}));
      replaceBase(result.snapshot,pending);
      dirty=pending.length>0;$('save').disabled=!dirty;autoOff=false;
      if(!dirty)clearOwnDraft();
      refresh();
      status(patch.length+'장 '+(auto?'자동 저장했습니다.':'저장했습니다.'));
    }catch(error){
      if(error.code===409)autoOff=true;
      persistDraft();
      const where=error.frames?.length?' 원본 '+error.frames.join(', ')+'번.':'';
      status('저장하지 못했습니다. '+error.message+where+' 편집은 미저장 작업에 남아 있습니다.');
    }finally{saveBusy=false;panel();}
    return true;
  };
  $('claim-release').onclick=async()=>{
    if(claimBusy||renewBusy||saveBusy)return;
    await window.LWIR_COLLAB_SAVE(false);
    if(dirty)return;
    claimBusy=true;panel();
    try{
      await json('release',{owner});claim=null;expires=0;undo=[];cancelPart();
      status('작업 범위를 반납했습니다.');await listClaims();refresh();
    }catch(error){status('범위를 반납하지 못했습니다. '+error.message);}
    finally{claimBusy=false;panel();}
  };
  setInterval(async()=>{
    if(!enabled||claimBusy||renewBusy)return;
    renewBusy=true;
    const range=own();
    try{
      if(range)adopt(await json('claim',{owner,user,first:range.first,last:range.last}));
      await listClaims();
    }catch(error){
      if(error.code===409){claim=null;expires=0;cancelPart();}
      panel();
    }finally{renewBusy=false;}
  },25000);
  addEventListener('pagehide',()=>{
    if(!enabled||!own()||dirty)return;
    fetch(url('release'),{method:'POST',keepalive:true,headers:{'Content-Type':'application/json'},
                          body:JSON.stringify({owner})}).catch(()=>{});
  });
})();
