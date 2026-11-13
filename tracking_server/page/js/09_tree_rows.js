// node = {name, note, sel, lock (why it cannot be picked; '' = open), u (POST url), model (answer is /model state), info (hint only, nothing sent), ask (confirm text)}
function tiers(){const r=rlast,m=mlast,T=[];if(!r)return T;
  const Q=q=>'/rules/set?'+q,trk=pend(r,'tracker'),rel=!!pend(r,'relabel'),prd=pend(r,'predictor')||'kalman',bot=trk==='botsort';
  const warm=r.warm_on!=null?!!r.warm_on:(r.warm_min>0||(r.warm_max||0)>0);
  T.push({lab:'전처리',sub:'여러 개 켜기',multi:1,n:[
    {name:'흔들림 보정',note:'optical flow · 화면+검출 · 시험 안 함',sel:!!r.stab,lock:r.stab==null?OLDSRV:'',u:Q('stab='+(r.stab?0:1))}]});
  const cat=(m&&m.catalog)||[];
  const dk=c=>String(c.det)+' '+String(c.name),ADVON=document.body.classList.contains('adv'),HIDEF=['기본 (공개 데이터)','다른 구조 (비교용)'],curm=m?(m.busy?m.pending:m.current):null;
  const catv=cat.filter(c=>ADVON||!HIDEF.includes(duse(dk(c)))||c.id===curm),dfs=DUSE.filter(f=>catv.some(c=>duse(dk(c))===f));   // 11-13: comparison models only with 'advanced' open   // 11-13: folders by use, only the ones that have models
  T.push({lab:'검출 모델',sub:'어디서 쓰는지에 따라 묶음'+(ADVON?'':' · 비교용 모델은 고급을 열면 보임'),f:m&&cat.length?dfs:null,fe:{},n:!m?[{name:'불러오는 중…',lock:'모델 목록을 아직 받지 못했습니다'}]:!cat.length?[{name:'모델 고정',lock:'이 실행에서는 모델을 바꿀 수 없습니다'}]:
    catv.map(c=>({g:duse(dk(c)),sg:duse(dk(c))==='다른 구조 (비교용)'?dsub(dk(c)):'',rec:REC.includes(c.det),note:DNOTE[c.det]||'',name:DN[c.det]||String(c.name).replace(/\s*\(자동 등록\)/g,'').replace(/\s*\([^)]*시험용[^)]*\)/g,'').replace(/\s*·\s*시험용/g,'').trim(),sel:c.id===(m.busy?m.pending:m.current),lock:m.busy&&c.id!==m.pending?'모델을 바꾸는 중입니다. 끝나면 열립니다':'',u:'/model/set?id='+encodeURIComponent(c.id),model:1}))});
  T.push({lab:'검출 뒤 Filter',sub:'검출 상자를 거름 · 값은 아래 Filter 값',multi:1,n:[
    {name:'온도 Filter',note:'차가운 상자 버림',sel:warm,lock:r.warm_on==null?OLDSRV:'',u:Q('warm_on='+(warm?0:1))},
    {name:'가장자리 Filter',note:'가장자리 상자는 점수 '+(r.edge_birth!=null?r.edge_birth:'–')+' 이상만 새로 시작',sel:(r.edge_birth||0)>0,info:'값은 아래 Filter 값에서 바꿉니다 (0 = 끔)'}]});
  const ps=(r.presets||[]).map(p=>Array.isArray(p)?{key:p[0],name:p[1]}:p).filter(p=>p&&p.key&&p.key!=='gate');
  const bt=ps.find(p=>p.key==='bytetrack')||ps.find(p=>p.key==='confirm3');
  const camNative=r.camera_bytetrack;
  const recording=rpLast&&rpLast.state&&rpLast.state.on?rpLast.state:null;
  const replayCpp=!!(recording&&recording.backend==='cpp_bytetrack');
  const replayStored=!!(recording&&String(recording.backend||'').startsWith('pre:'));
  const replayErr=replayCpp&&!!(rpLast.eval&&rpLast.eval.backend_error);
  const camCpp=!!(!recording&&camNative&&camNative.active);
  const camTransition=!recording&&(!!camNative?.requested!==!!camNative?.active);
  const camCppControl=camCpp&&camNative.control_supported===true;
  const cpp=camCpp||replayCpp, btSelected=BT_KEYS.includes(trk)||cpp||camTransition;
  const engine=replayCpp?(replayErr?'C++ 오류 (녹화)':'C++ (녹화)'):r.tracker_backend||'미보고';
  const eng=pend(r,'engine')||'auto',curEngine=p=>p.key===trk&&!cpp&&!camTransition;
  const tn=ps.filter(p=>!BT_KEYS.includes(p.key)).map(p=>({key:p.key,name:tname(p.key)+' · '+(curEngine(p)&&eng==='python'?'Python':(p.backend||'백엔드 미보고')),note:p.restart_required?'새 C++ 빌드 적용: 서버 재시작 필요':TN[p.key]?TN[p.key][1]:'',sel:!cpp&&!camTransition&&p.key===trk,u:Q('tracker='+encodeURIComponent(p.key)+'&engine=auto')}));
  if(bt)tn.unshift({key:'bytetrack',name:'ByteTrack · '+(camNative?.available?'C++':'Python'),note:camNative?.available?'C++ 라이브러리가 있어 C++로 시작합니다. Python은 아래 ByteTrack 옵션에서 고릅니다':'',sel:btSelected,u:Q('tracker='+encodeURIComponent(bt.key)+'&engine=auto')});
  const trackerSummary=replayStored?'녹화: 저장된 결과 표시 중 · 이 줄의 추적기와 무관':camTransition?'카메라 추적기 전환 중 · 다음 프레임 대기':camCppControl?'C++ · 화면과 짐벌에서 같은 ID 사용':camCpp?'화면 박스·ID: C++ · 짐벌: '+engine:'실제 실행 엔진: '+engine;
  T.push({lab:'Tracker',sub:trackerSummary,n:tn});
  const apps=r.apps||[{key:'off'},{key:'repvgg'},{key:'vit',available:false,note:'ViT 파일을 장치에 올려야 열립니다'}],app=r.apps?(r.app||'off'):'repvgg';   // old server: BoT-SORT always compared appearance
  // v41: a server that can change the appearance model while running sends reid_hot + reids (one model shared by KeepID and BoT-SORT 외형 비교)
  const hot=!!(r.reid_hot&&r.reids&&r.reids.length),rk=hot?(r.reid_pending||r.reid):'',fol=!!(r.gimbal&&r.gimbal.auto);
  const JN='번호 잇기 판정기는 기본 모델 기준 · 미검증',FOLK='따라가기 중에는 외형 모델을 바꿀 수 없습니다. 멈춘 뒤 고르세요',SWAPH='외형 모델을 바꾸는 중 — 화면이 잠깐 멈추고 사람 번호가 새로 매겨집니다';
  const NOVIT={g:'ViT-lite',name:'ViT-lite_distillation_(TMOT,CAMEL)_202610',note:'teacher SOLIDER · 공개 열화상 데이터로 학습',lock:'장치에 파일이 없습니다. 올리고 추적 프로그램을 다시 켜야 합니다'};
  const rn=(a,on,q,jn)=>({g:a.folder||'',name:a.name||a.key,note:(a.note||'')+(jn&&!a.judge_fit?(a.note?' · ':'')+JN:''),sel:on&&a.key===rk,lock:a.available===false?(a.why||'지금은 고를 수 없습니다'):(fol&&a.key!==rk)?FOLK:'',u:Q(q+'&reid='+encodeURIComponent(a.key)),hint:a.key!==rk?SWAPH:''});
  // 11-13 (user): one appearance model serves both uses, so the list is drawn once (row '외형 모델' below); here only on / off
  const rname=(r.reids&&(r.reids.find(a=>a.key===rk)||{}).name)||'';
  if(hot)T.push({lab:'외형 비교',sub:'사람 생김새로도 짝짓기 · 모델은 아래 외형 모델',kidOf:3,kidKey:'botsort',show:bot,n:[{name:'끔',note:'위치만으로 짝짓기',sel:bot&&app==='off',u:Q('app=off')},{name:'켬',note:rname,sel:bot&&app!=='off',u:Q('app=repvgg')}]});
  else T.push({lab:'외형 비교',sub:'',kidOf:3,kidKey:'botsort',show:bot,f:['RepVGG','ViT-lite'],n:apps.map(a=>({g:AFOLD[a.key]||'',name:AN[a.key]||a.name||a.key,note:a.note||'',sel:bot&&a.key===app,lock:!bot?'BoT-SORT를 골라야 열립니다':!r.apps?OLDSRV:a.available===false?(a.note||'지금은 고를 수 없습니다'):'',u:Q('app='+encodeURIComponent(a.key))}))});
  const pl=r.predictors||[{key:'kalman',available:true}];
  const plv=pl.filter(p=>ADVON||p.key==='kalman'||p.key===prd);   // 11-13: GRU / Mamba (1 epoch, unverified) only with 'advanced' open
  T.push({lab:'위치 기억 · 신뢰도',sub:'놓친 사람 위치'+(ADVON||plv.length>1?'':' · 학습 모델은 고급을 열면 보임'),f:plv.length>1?['Kalman filter','GRU','Mamba']:null,fe:{GRU:'장치에 GRU 가중치 파일이 없습니다',Mamba:'장치에 Mamba 가중치 파일이 없습니다'},n:plv.map(p=>({g:plv.length>1?(PFOLD[p.key]||''):'',name:PN[p.key]?PN[p.key][0]:p.name,note:PN[p.key]?PN[p.key][1]:(p.note||''),sel:p.key===prd,lock:p.available===false?(p.note||'가중치 파일이 없습니다'):'',u:Q('predictor='+encodeURIComponent(p.key))}))});
  const RST='장치에 파일은 있습니다. 쓰려면 추적 프로그램을 이 모델로 다시 켜야 합니다',VWAIT='PC에서 학습 중입니다. 끝나면 장치에 올립니다';
  if(hot&&!camCppControl&&!replayCpp)T.push({lab:'KeepID',sub:'돌아온 사람에게 예전 번호',n:[{name:'끔',sel:!rel,u:Q('relabel=0')},{name:'켬',note:rname,sel:rel,u:Q('relabel=1')}]});
  else if(!camCppControl&&!replayCpp)T.push({lab:'KeepID',sub:'돌아온 사람 번호',f:['RepVGG','ViT-lite'],n:[
    {name:'끔',sel:!rel,u:Q('relabel=0')},
    {g:'RepVGG',name:'RepVGG_distillation_(TMOT,CAMEL)_20261103',note:'teacher SOLIDER · 공개 열화상 데이터로 학습',sel:rel,lock:!rel&&!r.relabel_available?'재식별 모델이 이 실행에 없습니다':'',u:Q('relabel=1')},
    {g:'RepVGG',name:'RepVGG_distillation_(실내)_20261110',note:'teacher SOLIDER · 우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST},
    {g:'RepVGG',name:'RepVGG_fine-tuning_(실내)_20261110',note:'우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST},
    {g:'ViT-lite',name:'ViT-lite_distillation_(TMOT,CAMEL)_202610',note:'teacher SOLIDER · 공개 열화상 데이터로 학습',lock:'장치에 파일이 없습니다. 올리고 추적 프로그램을 다시 켜야 합니다'},
    {g:'ViT-lite',name:'ViT-lite_distillation_(실내)_20261111',note:'teacher SOLIDER · 우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST},
    {g:'ViT-lite',name:'ViT-lite_fine-tuning_(실내)_20261111',note:'우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST}]});
  const kid=rel&&!camCppControl&&!replayCpp;
  if(hot&&((bot&&app!=='off')||kid))T.push({lab:'외형 모델',sub:(bot&&app!=='off'&&kid?'BoT-SORT 외형 비교와 KeepID가 함께 씀':bot&&app!=='off'?'BoT-SORT 외형 비교가 씀':'KeepID가 씀'),f:['RepVGG','ViT-lite'],n:r.reids.map(a=>rn(a,true,'',true)).concat([NOVIT])});
  const g=r.gimbal,lead=!!r.lead;window.OTFOL=g?!!g.auto:autoNow;   // following is started by tapping a person in the picture, so it has no tile
  T.push({lab:'제어',sub:'여러 개 켜기',multi:1,n:[
    {name:'예측 제어',note:'따라갈 때만 작동 · 재시작해도 유지',sel:lead,lock:r.lead==null?OLDSRV:(!lead&&!(g&&g.present))?'짐벌이 연결되어 있지 않습니다':'',u:Q('lead='+(lead?0:1)),ask:lead?'':'예측 제어는 아직 시험하지 않은 기능입니다.\n따라가는 동안 짐벌이 사람이 움직이는 쪽으로 조금 앞서 움직입니다.\n켤까요?'}]});
  T.push({lab:'화면 표시',sub:'여러 개 켜기',multi:1,n:[   // 11-13: it draws, it does not control
    {name:'예측 상자 표시',note:'놓쳤을 때 점선',sel:!!r.pred_show,lock:r.predictors?'':OLDSRV,u:Q('pred_show='+(r.pred_show?0:1))}]});
  if(bt){
    const switching=!!rpLast?.state?.busy;
    const camCppNote=camNative?.restart_required?'새 C++ 빌드 적용: 서버 재시작 필요':camNative?.error?'실행 실패 · 기존 추적기로 복귀':camTransition?'다음 카메라 프레임 대기':camNative?.control_supported===true?'화면과 짐벌에서 같은 ID 사용':'카메라 화면에서 박스·ID 표시';
    const hasCpp=recording?rpLast.backends?.some(b=>b.id==='cpp_bytetrack'):camNative?.available;
    const pythonPreset=BT_KEYS.includes(trk)?trk:bt.key;
    const optionLock=recording?'녹화 재생을 마친 뒤 바꾸세요':camTransition?'추적기 전환이 끝난 뒤 바꾸세요':cpp?'Python 엔진을 선택한 뒤 바꾸세요':'';
    T.push({lab:'ByteTrack 옵션',kidOf:3,kidKey:'bytetrack',show:btSelected,multi:1,n:[
      {name:'Python',sel:!cpp&&!camTransition&&!replayStored,lock:switching?'영상 전환 중입니다':'',
       replayBackend:recording?'python':null,u:Q('tracker='+encodeURIComponent(pythonPreset))},
      {name:'C++',sel:cpp&&!replayErr,lock:switching?'영상 전환 중입니다':!hasCpp?(camNative?.restart_required?'새 C++ 빌드를 쓰려면 서버를 다시 시작하세요':'C++ 라이브러리를 사용할 수 없습니다'):'',
       note:recording?'녹화 재생':camCppNote,
       replayBackend:recording?'cpp_bytetrack':null,u:'/tracker/camera?backend=cpp_bytetrack'},
      {name:'위치 재결합',note:cpp?'C++에서 사용하지 않음':trk==='rebind'?'켜짐':'꺼짐',sel:!cpp&&!camTransition&&trk==='rebind',lock:optionLock,
       u:Q('tracker='+encodeURIComponent(trk==='rebind'?bt.key:'rebind'))}
    ]});
  }
  // 11-13: engine row of the chosen SORT / OC-SORT / BoT-SORT. The tile itself starts C++ when it is built; Python is picked here.
  ps.filter(p=>!BT_KEYS.includes(p.key)&&'engine' in r).forEach(p=>{
    const hasNative=p.backend==='C++',python=eng==='python'||!hasNative;
    T.push({lab:tname(p.key)+' 엔진',kidOf:3,kidKey:p.key,show:curEngine(p),n:[
      {name:'C++',note:hasNative?'기본':'',sel:!python,lock:hasNative?'':(p.restart_required?'새 C++ 빌드를 쓰려면 서버를 다시 시작하세요':'C++ 빌드가 없습니다'),u:Q('tracker='+encodeURIComponent(p.key)+'&engine=auto')},
      {name:'Python',sel:python,u:Q('tracker='+encodeURIComponent(p.key)+'&engine=python')}]});
  });
  return T;}
