// node = {name, note, sel, lock (why it cannot be picked; '' = open), u (POST url), model (answer is /model state), info (hint only, nothing sent), ask (confirm text)}
function tiers(){const r=rlast,m=mlast,T=[];if(!r)return T;
  const Q=q=>'/rules/set?'+q,trk=pend(r,'tracker'),rel=!!pend(r,'relabel'),prd=pend(r,'predictor')||'kalman',bot=trk==='botsort';
  const warm=r.warm_on!=null?!!r.warm_on:(r.warm_min>0||(r.warm_max||0)>0);
  T.push({lab:'전처리',sub:'여러 개 켜기',multi:1,n:[
    {name:'흔들림 보정',note:'optical flow · 화면+검출 · 시험 안 함',sel:!!r.stab,lock:r.stab==null?OLDSRV:'',u:Q('stab='+(r.stab?0:1))}]});
  const cat=(m&&m.catalog)||[];
  const dfs=['YOLO26','YOLOX','YOLOv8'];if(cat.some(c=>dfold(String(c.det)+' '+String(c.name))==='기타'))dfs.push('기타');
  T.push({lab:'검출 모델',sub:'구조 → 크기 순으로 펼침',f:m&&cat.length?dfs:null,fe:{YOLOX:'장치 목록에 YOLOX 모델이 없습니다 (모델뱅크에만 있음)',YOLOv8:'장치 목록에 YOLOv8 모델이 없습니다 (모델뱅크에만 있음)'},n:!m?[{name:'불러오는 중…',lock:'모델 목록을 아직 받지 못했습니다'}]:!cat.length?[{name:'모델 고정',lock:'이 실행에서는 모델을 바꿀 수 없습니다'}]:
    cat.map(c=>({g:dfold(String(c.det)+' '+String(c.name)),sg:dsub(String(c.det)+' '+String(c.name)),note:DNOTE[c.det]||'',name:DN[c.det]||String(c.name).replace(/\s*\(자동 등록\)/g,'').replace(/\s*\([^)]*시험용[^)]*\)/g,'').replace(/\s*·\s*시험용/g,'').trim(),sel:c.id===(m.busy?m.pending:m.current),lock:!c.available?'장치에 모델 파일이 없습니다':m.busy&&c.id!==m.pending?'모델을 바꾸는 중입니다. 끝나면 열립니다':'',u:'/model/set?id='+encodeURIComponent(c.id),model:1}))});
  T.push({lab:'검출 뒤 Filter',sub:'검출 상자를 거름 · 값은 아래 Filter 값',multi:1,n:[
    {name:'온도 Filter',note:'차가운 상자 버림',sel:warm,lock:r.warm_on==null?OLDSRV:!warm&&!(r.warm_prev&&r.warm_prev.some(v=>v>0))?'아래 Filter 값에서 온도 범위를 먼저 입력하고 적용하세요':'',u:Q('warm_on='+(warm?0:1))},
    {name:'가장자리 Filter',note:'가장자리 상자는 점수 '+(r.edge_birth!=null?r.edge_birth:'–')+' 이상만 새로 시작',sel:(r.edge_birth||0)>0,lock:!(r.edge_birth>0)&&!(r.edge_prev>0)?'아래 Filter 값에서 가장자리 점수를 먼저 입력하고 적용하세요':'',u:Q('edge_on='+(r.edge_birth>0?0:1))}]});
  const ps=(r.presets||[]).map(p=>Array.isArray(p)?{key:p[0],name:p[1]}:p).filter(p=>p&&p.key&&p.key!=='gate');
  T.push({lab:'Tracker',n:ps.map(p=>({key:p.key,name:TN[p.key]?TN[p.key][0]:p.name,note:TN[p.key]?TN[p.key][1]:'',sel:p.key===trk,lock:p.available===false?'장치에서 사용할 수 없습니다':'',u:Q('tracker='+encodeURIComponent(p.key))}))});
  const apps=r.apps||[{key:'off'},{key:'repvgg'},{key:'vit',available:false,note:'ViT 파일을 장치에 올려야 열립니다'}],app=r.apps?(r.app||'off'):'repvgg';   // old server: BoT-SORT always compared appearance
  // v41: a server that can change the appearance model while running sends reid_hot + reids (one model shared by KeepID and BoT-SORT 외형 비교)
  const hot=!!(r.reid_hot&&r.reids&&r.reids.length),rk=hot?(r.reid_pending||r.reid):'',fol=!!(r.gimbal&&r.gimbal.auto);
  const JN='번호 잇기 판정기는 기본 모델 기준 · 미검증',FOLK='따라가기 중에는 외형 모델을 바꿀 수 없습니다. 멈춘 뒤 고르세요',SWAPH='외형 모델을 바꾸는 중 — 화면이 잠깐 멈추고 사람 번호가 새로 매겨집니다';
  const NOVIT={g:'ViT-lite',name:'ViT-lite_distillation_(TMOT,CAMEL)_202610',note:'teacher SOLIDER · 공개 열화상 데이터로 학습',lock:'장치에 파일이 없습니다. 올리고 추적 프로그램을 다시 켜야 합니다'};
  const rn=(a,on,q,jn)=>({g:a.folder||'',name:a.name||a.key,note:(a.note||'')+(jn&&!a.judge_fit?(a.note?' · ':'')+JN:''),sel:on&&a.key===rk,lock:a.available===false?(a.why||'지금은 고를 수 없습니다'):(fol&&a.key!==rk)?FOLK:'',u:Q(q+'&reid='+encodeURIComponent(a.key)),hint:a.key!==rk?SWAPH:''});
  if(hot)T.push({lab:'외형 비교',sub:'KeepID 와 같은 모델을 함께 씀',kidOf:3,kidKey:'botsort',show:bot,f:['RepVGG','ViT-lite'],n:[{name:'끔',note:'위치만으로 짝짓기',sel:bot&&app==='off',u:Q('app=off')}].concat(r.reids.map(a=>rn(a,bot&&app!=='off','app=repvgg',rel)),[NOVIT])});
  else T.push({lab:'외형 비교',sub:'',kidOf:3,kidKey:'botsort',show:bot,f:['RepVGG','ViT-lite'],n:apps.map(a=>({g:AFOLD[a.key]||'',name:AN[a.key]||a.name||a.key,note:a.note||'',sel:bot&&a.key===app,lock:!bot?'BoT-SORT를 골라야 열립니다':!r.apps?OLDSRV:a.available===false?(a.note||'지금은 고를 수 없습니다'):'',u:Q('app='+encodeURIComponent(a.key))}))});
  const pl=r.predictors||[{key:'kalman',available:true}];
  T.push({lab:'위치 기억 · 신뢰도',sub:'놓친 사람 위치',f:['Kalman filter','GRU','Mamba'],fe:{GRU:'장치에 GRU 가중치 파일이 없습니다',Mamba:'장치에 Mamba 가중치 파일이 없습니다'},n:pl.map(p=>({g:PFOLD[p.key]||'',name:PN[p.key]?PN[p.key][0]:p.name,note:PN[p.key]?PN[p.key][1]:(p.note||''),sel:p.key===prd,lock:p.available===false?(p.note||'가중치 파일이 없습니다'):'',u:Q('predictor='+encodeURIComponent(p.key))}))});
  const RST='장치에 파일은 있습니다. 쓰려면 추적 프로그램을 이 모델로 다시 켜야 합니다',VWAIT='PC에서 학습 중입니다. 끝나면 장치에 올립니다';
  if(hot)T.push({lab:'KeepID',sub:'돌아온 사람 번호',f:['RepVGG','ViT-lite'],n:[{name:'끔',sel:!rel,u:Q('relabel=0')}].concat(r.reids.map(a=>rn(a,rel,'relabel=1',true)),[NOVIT])});
  else T.push({lab:'KeepID',sub:'돌아온 사람 번호',f:['RepVGG','ViT-lite'],n:[
    {name:'끔',sel:!rel,u:Q('relabel=0')},
    {g:'RepVGG',name:'RepVGG_distillation_(TMOT,CAMEL)_20261103',note:'teacher SOLIDER · 공개 열화상 데이터로 학습',sel:rel,lock:!rel&&!r.relabel_available?'재식별 모델이 이 실행에 없습니다':'',u:Q('relabel=1')},
    {g:'RepVGG',name:'RepVGG_distillation_(실내)_20261110',note:'teacher SOLIDER · 우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST},
    {g:'RepVGG',name:'RepVGG_fine-tuning_(실내)_20261110',note:'우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST},
    {g:'ViT-lite',name:'ViT-lite_distillation_(TMOT,CAMEL)_202610',note:'teacher SOLIDER · 공개 열화상 데이터로 학습',lock:'장치에 파일이 없습니다. 올리고 추적 프로그램을 다시 켜야 합니다'},
    {g:'ViT-lite',name:'ViT-lite_distillation_(실내)_20261111',note:'teacher SOLIDER · 우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST},
    {g:'ViT-lite',name:'ViT-lite_fine-tuning_(실내)_20261111',note:'우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST}]});
  const g=r.gimbal,lead=!!r.lead;window.OTFOL=g?!!g.auto:autoNow;   // following is started by tapping a person in the picture, so it has no tile
  T.push({lab:'제어',sub:'여러 개 켜기',multi:1,n:[
    {name:'예측 제어',note:'시험 안 함 · 따라갈 때만 작동 · 재시작하면 꺼짐',sel:lead,lock:r.lead==null?'이 Runtime은 예측 제어를 지원하지 않습니다':(!lead&&!(g&&g.present))?'짐벌이 연결되어 있지 않습니다':'',u:Q('lead='+(lead?0:1)),ask:lead?'':'예측 제어는 아직 시험하지 않은 기능입니다.\n따라가는 동안 짐벌이 사람이 움직이는 쪽으로 조금 앞서 움직입니다.\n켤까요?'},
    {name:'예측 상자 표시',note:'놓쳤을 때 점선',sel:!!r.pred_show,lock:r.pred_show_supported?'':'이 Runtime은 예측 상자 표시를 지원하지 않습니다',u:Q('pred_show='+(r.pred_show?0:1))}]});
  return T;}
