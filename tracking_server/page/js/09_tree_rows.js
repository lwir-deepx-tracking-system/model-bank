const DRETIRED_26M=new Set([
  'bank:yolo26m_pose_ft16v7_reviewed264_e24_provisional_kpt_640x480',
  'bank:yolo26m_pose_lwir650_20261119_temporal_e13_bodybox_kpt_640x480'
]);
// node = {name, note, sel, lock (why it cannot be picked; '' = open), u (POST url), model (answer is /model state), info (hint only, nothing sent), ask (confirm text)}
function tiers(){const r=rlast,m=mlast,T=[];if(!r)return T;
  const Q=q=>'/rules/set?'+q,trk=pend(r,'tracker'),rel=!!pend(r,'relabel'),prd=pend(r,'predictor')||'kalman',bot=trk==='botsort';
  const warm=r.warm_on!=null?!!r.warm_on:(r.warm_min>0||(r.warm_max||0)>0);
  T.push({lab:'전처리',sub:'여러 개 켜기',multi:1,n:[
    {name:'흔들림 보정',note:'optical flow · 화면+검출 · 시험 안 함',sel:!!r.stab,lock:r.stab==null?OLDSRV:'',u:Q('stab='+(r.stab?0:1))}]});
  const cat=(m&&m.catalog)||[];
  const dk=c=>String(c.det)+' '+String(c.name)+' '+(DN[c.det]||''),ADVON=document.body.classList.contains('adv'),HIDEF=['기본 (공개 데이터)','다른 구조 (비교용)'],curm=m?(m.busy?m.pending:m.current):null;
  const catv=cat.filter(c=>c.id===m.current||(m.busy&&c.id===m.pending)||!DRETIRED_26M.has(c.det));
  const latestDetector=latestGeneralDetector(catv),dfs=DSTR.filter(f=>catv.some(c=>dfold(dk(c))===f));
  const ad=r.auto_det,adOn=!!(ad&&ad.on),ADASK='검출 모델 자동 선택이 켜져 있습니다.\n손으로 모델을 고르면 자동 선택이 꺼집니다. 고를까요?';   // 11-15 (user): 검출 모델 자동 선택 (src/detect/auto_det.py)
  // 11-15 (user: 상황에 맞는 최적 모델 알아서 선택): 근거리 / 먼 거리 모델을 사람 크기로 고름. 녹화 재생 중에는 바꾸지 않고 바꿨을 순간만 기록 (시험 운전) · 11-15 (user): 검출 모델 목록보다 위에 (모델이 많아 목록 아래에 묻혔음)
  {const AK={near:'근거리',far:'먼 거리',other:'다른 모델'},P=(ad&&ad.params)||{},s=ad&&ad.signal,dl=ad&&ad.dry_recent&&ad.dry_recent.length?ad.dry_recent[ad.dry_recent.length-1]:null;
    const thr=ad?'사람 상자 '+P.far_px+' px 아래 '+P.hold_s+'초 → 먼 거리 (가려지거나 잘린 상자는 빼고) · '+P.near_px+' px 위, 또는 가려진 사람 '+P.hold_s+'초 → 근거리 · 사람을 잃으면 '+P.lost_s+'초 뒤 먼 거리 · 바꾼 뒤 '+P.dwell_s+'초 유지 · 1분에 '+P.max_per_min+'번까지':'';
    const miss=!ad?OLDSRV:!m?'모델 목록을 아직 받지 못했습니다':!cat.length||ad.can_switch===false?'이 실행에서는 모델을 바꿀 수 없습니다':!ad.far.present?'먼 거리 모델이 장치에 없습니다 (모델 폴더에 넣으면 열립니다)':!ad.near.present?'근거리 모델이 장치에 없습니다':'';
    const sig=s&&s.px!=null?(s.source==='followed'?'따라가는 사람 ':'화면 사람 가운뎃값 ')+Math.round(s.px/5)*5+' px'+(s.kpts!=null?' · 관절 '+s.kpts+'개':'')+(s.hidden?' · 가려짐':'')+(s.cut?' · 화면 끝에 잘림':''):s&&s.source==='followed_missing'?'따라가는 사람 안 보임':'';
    const now=!adOn?'':ad.mode==='dry_run'?'녹화 재생: 바꾸지 않고 기록만 · 바꿨을 순간 '+ad.dry_events+'번'+(dl?' · 마지막 '+(dl.frame!=null?(dl.frame+1)+'번째 장':dl.t+'초')+' '+(AK[dl.from]||dl.from)+'→'+(AK[dl.to]||dl.to):'')
      :ad.mode==='live'?'지금 '+(AK[ad.current]||'–')+(ad.pending?' → '+(AK[ad.pending]||ad.pending)+' 바꾸는 중':'')+(sig?' · '+sig:''):(ad.reason||'');
    const ADNOW=adOn?(ad.mode==='live'?'지금 쓰는 모델: '+(AK[ad.current]||'–')+(ad.pending?' → '+(AK[ad.pending]||ad.pending)+' 바꾸는 중':''):ad.mode==='dry_run'?'녹화 재생: 기록만':''):'';   // 11-15 (user): 켰을 때 무엇을 쓰는지 제목 줄에 보이게
    T.push({lab:'검출 모델 자동 선택',sub:(ADNOW?ADNOW+' · ':'')+'사람 크기로 근거리 / 먼 거리 모델 고르기'+(adOn&&ad.mode==='live'&&ad.last_switch?' · 마지막 자동 바꿈 '+ad.last_switch.at:''),n:[
      {name:'끔',note:(!adOn&&ad&&ad.reason&&ad.reason!=='꺼짐'?ad.reason+' · ':'')+'아래 검출 모델 그대로 · 손으로 모델을 고르면 꺼짐',sel:!adOn,lock:!ad?OLDSRV:'',u:Q('auto_det=0')},
      {name:'켬',note:thr+(now?' · '+now:''),sel:adOn,lock:adOn?'':miss,u:Q('auto_det=1'),
       ask:adOn?'':'검출 모델 자동 선택은 아직 장치에서 시험하지 않은 기능입니다.\n사람이 작게 보이면 먼 거리 모델로, 크게 보이면 근거리 모델로 바꿉니다.\n바꾸는 동안 화면이 잠깐 멈춥니다. 켤까요?'}]});}
  T.push({lab:'검출 모델',sub:'구조별 · YOLO26 안은 쓰는 곳별'+(adOn?' · 자동 선택 켜짐'+(ad.mode==='live'?' · 지금 '+({near:'근거리',far:'먼 거리',other:'다른 모델'}[ad.current]||'–')+' 모델':''):''),f:m&&cat.length?dfs:null,fe:{},n:!m?[{name:'불러오는 중…',lock:'모델 목록을 아직 받지 못했습니다'}]:!cat.length?[{name:'모델 고정',lock:'이 실행에서는 모델을 바꿀 수 없습니다'}]:
    catv.map(c=>({g:dfold(dk(c)),sg:dfold(dk(c))==='YOLO26'?duse26(dk(c)):'',rec:c.det==='bank:yolo26m_pose_lwir650_20261119_bodybox_kpt_640x480'||REC.includes(c.det),latest:c.det===latestDetector,note:DNOTE[c.det]||'',name:dname(c),sel:c.id===(m.busy?m.pending:m.current),lock:m.busy&&c.id!==m.pending?'모델을 바꾸는 중입니다. 끝나면 열립니다':'',u:'/model/set?id='+encodeURIComponent(c.id),model:1,ask:adOn?ADASK:''}))});
  const poseBoxDetector=m&&!m.busy&&cat.find(c=>c.id===m.current);
  if(r.pose_box && poseBoxDetector && [DETECTOR_UPPER_650, DETECTOR_TEMPORAL_650].includes(poseBoxDetector.det)){
    const poseBoxVersion=poseBoxDetector.det===DETECTOR_TEMPORAL_650?'E10+3':'E10';
    const detectorIndex=T.findIndex(t=>t.lab==='검출 모델');
    const selectedRow=T[detectorIndex].n.find(n=>n.sel);
    selectedRow.key='poseBoxDetector';
    T.push({lab:'박스 모양',kidOf:detectorIndex,kidKey:'poseBoxDetector',show:true,sub:'같은 '+poseBoxVersion+' 관절 모델 · 화면 표시',n:[
      {name:'상체 · 머리~골반 · '+poseBoxVersion+' · 2026-11-19',sel:r.pose_box==='upper',u:Q('pose_box=upper')},
      {name:'전신 추정 · 머리~발 · '+poseBoxVersion+' · 2026-11-19',note:'다리가 가려지면 추정 · 사람 선택은 상체를 누르세요',sel:r.pose_box==='full',u:Q('pose_box=full')}
    ]});
  }
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
  const tn=ps.filter(p=>!BT_KEYS.includes(p.key)).map(p=>({key:p.key,name:tname(p.key),note:p.restart_required?'새 C++ 빌드 적용: 서버 재시작 필요':TN[p.key]?TN[p.key][1]:'',sel:!replayStored&&!cpp&&!camTransition&&p.key===trk,u:Q('tracker='+encodeURIComponent(p.key)+'&engine=auto')}));
  if(bt)tn.unshift({key:'bytetrack',name:'ByteTrack',note:camNative?.available?'C++ 라이브러리가 있어 C++로 시작합니다. Python은 아래 ByteTrack 옵션에서 고릅니다':'',sel:!replayStored&&btSelected,u:Q('tracker='+encodeURIComponent(bt.key)+'&engine=auto')});
  const trackerSummary=replayStored?'녹화: 저장된 결과 표시 중 · 이 줄의 추적기와 무관':camTransition?'카메라 추적기 전환 중 · 다음 프레임 대기':camCppControl?'C++ · 화면과 짐벌에서 같은 ID 사용':camCpp?'화면 박스·ID: C++ · 짐벌: '+engine:'실제 실행 엔진: '+engine;
  // 11-14 (user): end-to-end models (detection + tracking in one model) get their own first row (built here, put on top at the end).
  // 11-15 (user: "실시간 카메라에 못 쓰는 MOTIP 은 빼고, FastTrackTr 도 다 실시간에 가능해야"): only models that run live are offered.
  // Stored results (pre:<key>, precomputed on a PC, recordings only: MOTIP, older FastTrackTr) are no longer listed on the page;
  // the server still plays one when /replay/start is asked for it directly (research scoring), and the rows below still show that state.
  const e2e=[];
  const el=r.e2e_live;   // 11-14 (user): live FastTrackTr (camera and recordings, server tracker = this model)
  if(el&&el.available)e2e.push({key:'e2e_live_ftt',name:(el.label||'FastTrackTr')+' · 실시간',note:'장치 NPU 에서 매 프레임 실행'+(el.ms?' · '+el.ms+' ms':'')+(el.link_used?' · 번호 잇기: '+(el.link_used==='id'?'모델 번호 출력 + 겹침':'겹침만'):'')+(el.err?' · '+el.err:''),sel:el.on==='ftt'||el.want==='ftt',u:Q('e2e_live=ftt')});
  const E2E=e2e.length?{lab:'엔드투엔드',sub:replayStored?'사용 중 · 아래 검출 모델과 Tracker 줄은 쓰이지 않음':'검출부터 추적까지 한 모델 · 끔이면 아래 검출 + Tracker 사용',
    n:[{name:'끔',note:'아래 검출 모델 + Tracker',sel:!replayStored&&!(el&&(el.on==='ftt'||el.want==='ftt')),e2eOff:!!(recording&&replayStored),u:(el&&el.on==='ftt')?Q('e2e_live=off'):undefined}].concat(e2e)}:null;
  const TRI=T.length;T.push({lab:'Tracker',sub:trackerSummary,n:tn});   // 11-15: the child rows below point at this row by its place (the auto-choice row above moved it)
  const apps=r.apps||[{key:'off'},{key:'repvgg'},{key:'vit',available:false,note:'ViT 파일을 장치에 올려야 열립니다'}],app=r.apps?(r.app||'off'):'repvgg';   // old server: BoT-SORT always compared appearance
  // v41: a server that can change the appearance model while running sends reid_hot + reids (one model shared by KeepID and BoT-SORT 외형 비교)
  const hot=!!(r.reid_hot&&r.reids&&r.reids.length),rk=hot?(r.reid_pending||r.reid):'',fol=!!(r.gimbal&&r.gimbal.auto);
  const JN='번호 잇기 판정기는 기본 모델 기준 · 미검증',FOLK='따라가기 중에는 외형 모델을 바꿀 수 없습니다. 멈춘 뒤 고르세요',SWAPH='외형 모델을 바꾸는 중 — 화면이 잠깐 멈추고 사람 번호가 새로 매겨집니다';
  const NOVIT={g:'ViT-lite',name:'ViT-lite_distillation_(TMOT,CAMEL)_202610',note:'teacher SOLIDER · 공개 열화상 데이터로 학습',lock:'장치에 파일이 없습니다. 올리고 추적 프로그램을 다시 켜야 합니다'};
  const publicRepVggKey='RVGGA0_REID2C256';
  const publicRepVgg='RepVGG · 사전학습 SOLIDER 증류 · 2026-11-03',publicRepVggNote='공개 열화상 TMOT·CAMEL·FLIR로 증류 · 현장 라벨 추가 학습 없음';
  const reidName=a=>a.key===publicRepVggKey?publicRepVgg:a.name||a.key;
  const rn=(a,on,q,jn)=>({g:a.folder||'',name:reidName(a),note:(a.key===publicRepVggKey?publicRepVggNote:a.note||'')+(jn&&!a.judge_fit?(a.note?' · ':'')+JN:''),sel:on&&a.key===rk,lock:a.available===false?(a.why||'지금은 고를 수 없습니다'):(fol&&a.key!==rk)?FOLK:'',u:Q(q+'&reid='+encodeURIComponent(a.key)),hint:a.key!==rk?SWAPH:''});
  // 11-13 (user): one appearance model serves both uses, so the list is drawn once (row '외형 모델' below); here only on / off
  const currentReid=r.reids?.find(a=>a.key===rk),rname=currentReid?reidName(currentReid):'';
  if(hot)T.push({lab:'외형 비교',sub:'사람 생김새로도 짝짓기 · 모델은 아래 외형 모델',kidOf:TRI,kidKey:'botsort',show:bot,n:[{name:'끔',note:'위치만으로 짝짓기',sel:bot&&app==='off',u:Q('app=off')},{name:'켬',note:rname,sel:bot&&app!=='off',u:Q('app=repvgg')}]});
  else T.push({lab:'외형 비교',sub:'',kidOf:TRI,kidKey:'botsort',show:bot,f:['RepVGG','ViT-lite'],n:apps.map(a=>({g:AFOLD[a.key]||'',name:AN[a.key]||a.name||a.key,note:a.note||'',sel:bot&&a.key===app,lock:!bot?'BoT-SORT를 골라야 열립니다':!r.apps?OLDSRV:a.available===false?(a.note||'지금은 고를 수 없습니다'):'',u:Q('app='+encodeURIComponent(a.key))}))});
  const pl=r.predictors||[{key:'kalman',available:true}];
  const plv=pl.filter(p=>ADVON||p.key==='kalman'||p.key===prd||/_kpt$/.test(p.key));   // 11-13: GRU / Mamba (1 epoch, unverified) only with 'advanced' open. 11-14: keypoint versions always shown (user)
  T.push({lab:'위치 기억 · 신뢰도',sub:'놓친 사람 위치'+(ADVON||plv.length>1?'':' · 학습 모델은 고급을 열면 보임'),f:plv.length>1?['Kalman filter','GRU','Mamba']:null,fe:{GRU:'장치에 GRU 가중치 파일이 없습니다',Mamba:'장치에 Mamba 가중치 파일이 없습니다'},n:plv.map(p=>({g:plv.length>1?(PFOLD[p.key]||''):'',name:PN[p.key]?PN[p.key][0]:p.name,note:PN[p.key]?PN[p.key][1]:(p.note||''),sel:p.key===prd,lock:p.available===false?(p.note||'가중치 파일이 없습니다'):(/_kpt$/.test(p.key)&&r.det_kpt===false)?'[관절] 검출 모델을 골라야 열립니다':'',u:Q('predictor='+encodeURIComponent(p.key))}))});   // 11-14 (user): keypoint versions only with a keypoint detector
  const RST='장치에 파일은 있습니다. 쓰려면 추적 프로그램을 이 모델로 다시 켜야 합니다',VWAIT='PC에서 학습 중입니다. 끝나면 장치에 올립니다';
  if(hot&&!camCppControl&&!replayCpp)T.push({lab:'KeepID',sub:'돌아온 사람에게 예전 번호',n:[{name:'끔',sel:!rel,u:Q('relabel=0')},{name:'켬',note:rname,sel:rel,u:Q('relabel=1')}]});
  else if(!camCppControl&&!replayCpp)T.push({lab:'KeepID',sub:'돌아온 사람 번호',f:['RepVGG','ViT-lite'],n:[
    {name:'끔',sel:!rel,u:Q('relabel=0')},
    {g:'RepVGG',name:publicRepVgg,note:publicRepVggNote,sel:rel,lock:!rel&&!r.relabel_available?'재식별 모델이 이 실행에 없습니다':'',u:Q('relabel=1')},
    {g:'RepVGG',name:'RepVGG_distillation_(실내)_20261110',note:'teacher SOLIDER · 우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST},
    {g:'RepVGG',name:'RepVGG_fine-tuning_(실내)_20261110',note:'우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST},
    {g:'ViT-lite',name:'ViT-lite_distillation_(TMOT,CAMEL)_202610',note:'teacher SOLIDER · 공개 열화상 데이터로 학습',lock:'장치에 파일이 없습니다. 올리고 추적 프로그램을 다시 켜야 합니다'},
    {g:'ViT-lite',name:'ViT-lite_distillation_(실내)_20261111',note:'teacher SOLIDER · 우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST},
    {g:'ViT-lite',name:'ViT-lite_fine-tuning_(실내)_20261111',note:'우리 카메라 실내로 학습 · 장치 정확도 미측정',lock:RST}]});
  const kid=rel&&!camCppControl&&!replayCpp;
  if(kid&&r.relink_pred!=null)T.push({lab:'KeepID 찾는 범위',sub:'돌아온 사람을 어디서 찾나',n:[{name:'마지막 자리 주변',note:'예전 방식 · 마지막으로 본 곳 둘레',sel:!r.relink_pred,u:Q('relink_pred=0')},{name:'예측 이동 + 커지는 범위',note:'기본 · 움직이던 방향으로 옮기며 시간이 지날수록 넓힘 · 녹화 재생에서 번호 바뀜 8→4',sel:!!r.relink_pred,u:Q('relink_pred=1')}]});   // 11-14 (user)
  if(hot&&((bot&&app!=='off')||kid))T.push({lab:'외형 모델',sub:(bot&&app!=='off'&&kid?'BoT-SORT 외형 비교와 KeepID가 함께 씀':bot&&app!=='off'?'BoT-SORT 외형 비교가 씀':'KeepID가 씀'),f:['RepVGG','ViT-lite'],n:r.reids.map(a=>rn(a,true,'',true)).concat([NOVIT])});
  const g=r.gimbal,lead=!!r.lead;window.OTFOL=g?!!g.auto:autoNow;   // following is started by tapping a person in the picture, so it has no tile
  T.push({lab:'제어',sub:'여러 개 켜기',multi:1,n:[
    {name:'예측 제어',note:'따라갈 때만 작동 · 재시작해도 유지',sel:lead,lock:r.lead==null?OLDSRV:(!lead&&!(g&&g.present))?'짐벌이 연결되어 있지 않습니다':'',u:Q('lead='+(lead?0:1))}]});
  const br=r.box_refine;   // 11-15 (user "2번부터 하자", "변할 때 자연스럽게"): 그리는 상자와 짐벌 조준만 다듬기 (src/track/box_refine.py), 기본 꺼짐. 11-16 (user "필요한 것만 남기자"): 끔 / 크기만 / 상체 네모 셋만
  if(br){const bm=br.mode||'off',BRN='추적기·채점·자동 선택은 원래 상자',BLT=0.3,bd=(br.modes||[]).find(x=>x.key==='body');
    T.push({lab:'상자 다듬기',sub:'화면 상자와 짐벌 조준점만'+(bm!=='off'?' · '+br.mode_name+(bm==='body'&&br.body_family_name?' ('+br.body_family_name+')':'')+(br.ms!=null?' · '+Number(br.ms).toFixed(2)+' ms':''):'')+(br.why?' · '+br.why:''),n:[
      {name:'끔',note:'추적기 상자 그대로',sel:bm==='off',u:Q('box_refine=off')},
      {name:'크기만 부드럽게',note:'위치·짐벌 조준점은 추적기 그대로 · 상자 크기만 천천히 바뀜(근거리↔먼 거리 바뀔 때도) · '+BRN,sel:bm==='size',u:Q('box_refine=size')}].concat(bd?[
      {name:'상체 네모로 통일 (변환표 + 크기 부드럽게)',note:'가까이든 멀리든 머리 위~골반 한 모양 · 먼 사람 전신 상자는 상체로 바꿈, 가까운 사람 상자는 그대로 · 짐벌 조준점은 바꾼 상자의 조준점 그대로, 크기만 부드럽게 · 자세 검출 모델은 변환표, 다른 모델은 사람 전체 상자의 위쪽 0.53 · 모델이 바뀌어도 이어짐 · 관절 필요 없음 · 장치에서 아직 재지 않음 · '+BRN,sel:bm==='body',lock:bd.available?'':(br.body_note||'변환표 파일이 없습니다'),u:Q('box_refine=body')}]:[])});   // body: 서버가 모드를 알 때만 줄을 보임 · 잠금 글은 서버가 아는 실제 까닭(없음 / 읽지 못함 / 방금 바뀜)
    T.push({lab:'모델 바뀔 때 섞기',sub:'검출 모델이 바뀌면 상자를 천천히 넘김',multi:1,n:[
      {name:'상자 섞기 '+(br.blend_s>0?br.blend_s:BLT)+'초',note:'근거리↔먼 거리 바뀔 때 상자가 튀지 않게 · 끔·크기만 부드럽게에서 씀(상체 네모는 스스로 이어 줌) · 장치에서 아직 재지 않음 · '+BRN,sel:br.blend_s>0,u:Q('box_blend='+(br.blend_s>0?0:BLT))}]});
    if(br.lost_same!=null)T.push({lab:'놓친 상자',sub:'놓친 사람의 점선 상자'+(br.lost_same&&br.lost_why?' · '+br.lost_why:'')+(br.lost_same&&br.lost_errors?' · 오류 '+br.lost_errors+'번':''),multi:1,n:[   // 11-16 (judge A): lost box = seen box, display only, default off
      {name:'보일 때 상자와 같게',note:'놓친 첫 장부터 점선 상자를 그림 · 모양·크기는 보일 때 상자와 같음(상자 다듬기 모드, 마지막 크기 유지) · 크기만 부드럽게에서는 짐벌 조준점이 지금 점선 상자와 같음 · 추적기·짐벌은 그대로 · 예측 상자 표시가 켜져 있을 때 · 장치에서 아직 재지 않음',sel:!!br.lost_same,u:Q('lost_same='+(br.lost_same?0:1))}]});}
  T.push({lab:'화면 표시',sub:'여러 개 켜기',multi:1,n:[   // 11-13: it draws, it does not control
    {name:'사람 후보 (공간 주파수)',note:'추적이 안 쓴 검출의 약한 상자를 공간 주파수로 다시 봄 · 사람다우면 진한 보라 miss? · 연속으로 보일 때만 · 표시만'+(r.prop&&r.prop.ms!=null?' · '+Math.round(r.prop.ms)+' ms':''),
     sel:!!r.prop_show,lock:r.prop_show==null?OLDSRV:'',u:Q('prop_show='+(r.prop_show?0:1))},   // 11-14: research feature (src/detect/spec_proposals.py)
    {name:'피부 온도 (카메라 온도)',note:'카메라가 잰 온도로 사람 상자 옆에 °C · 초록 = 30-38 °C 덩어리(얼굴·손) 있음, 회색 = 상자의 가장 뜨거운 곳 · 표시만'+(r.temp&&r.temp.show?(r.temp.fresh?' · 온도 지도 '+r.temp.age.toFixed(1)+' s 전':' · 온도 지도 없음 (서버를 --cam-temp 로 켜야 함)'):''),
     sel:!!r.temp_show,lock:r.temp_show==null?OLDSRV:(!r.temp_show&&r.temp&&r.temp.enabled===false)?'이 서버는 카메라 온도 지도를 받지 않습니다. 11-15 장치 시험: 이 카메라는 온도 계산(CalcTemp)이 모든 화소 0 이라 켜도 온도가 나오지 않습니다':'',u:Q('temp_show='+(r.temp_show?0:1))},   // 11-14: research feature (src/detect/skin_temp.py), not kept over a restart. 11-15 (user: options must work live): locked when the server has no temperature map
    {name:'사진 먼저 (지연 줄임)',note:'카메라 사진을 검출 전에 바로 보내고 상자는 이 화면에서 그림 · 상자는 사진보다 두 장쯤 늦음 · 녹화 재생·흔들림 보정·사람 후보·피부 온도 표시 중에는 예전처럼'+(r.picture_first_state&&r.picture_first&&!r.picture_first_state.active?' · 지금은 예전처럼':''),
     sel:!!r.picture_first,lock:r.picture_first==null?OLDSRV:'',u:Q('picture_first='+(r.picture_first?0:1))},   // 11-15 (src/web/picture_first.py, page/js/picture_first.js)
    {name:'예측 상자 표시',note:'놓쳤을 때 점선',sel:!!r.pred_show,lock:r.predictors?'':OLDSRV,u:Q('pred_show='+(r.pred_show?0:1))}]});
  if(bt){
    const switching=!!rpLast?.state?.busy;
    const camCppNote=camNative?.restart_required?'새 C++ 빌드 적용: 서버 재시작 필요':camNative?.error?'실행 실패 · 기존 추적기로 복귀':camTransition?'다음 카메라 프레임 대기':camNative?.control_supported===true?'화면과 짐벌에서 같은 ID 사용':'카메라 화면에서 박스·ID 표시';
    const hasCpp=recording?rpLast.backends?.some(b=>b.id==='cpp_bytetrack'):camNative?.available;
    const pythonPreset=BT_KEYS.includes(trk)?trk:bt.key;
    const optionLock=recording?'녹화 재생을 마친 뒤 바꾸세요':camTransition?'추적기 전환이 끝난 뒤 바꾸세요':cpp?'Python 엔진을 선택한 뒤 바꾸세요':'';
    T.push({lab:'ByteTrack 옵션',kidOf:TRI,kidKey:'bytetrack',show:btSelected,multi:1,n:[
      {name:'Python',pick:1,sel:!cpp&&!camTransition&&!replayStored,lock:switching?'영상 전환 중입니다':'',
       replayBackend:recording?'python':null,u:Q('tracker='+encodeURIComponent(pythonPreset))},
      {name:'C++',pick:1,sel:cpp&&!replayErr,lock:switching?'영상 전환 중입니다':!hasCpp?(camNative?.restart_required?'새 C++ 빌드를 쓰려면 서버를 다시 시작하세요':'C++ 라이브러리를 사용할 수 없습니다'):'',
       note:recording?'녹화 재생':camCppNote,
       replayBackend:recording?'cpp_bytetrack':null,u:'/tracker/camera?backend=cpp_bytetrack'},
      {name:'위치 재결합',note:cpp?'C++에서 사용하지 않음':trk==='rebind'?'켜짐':'꺼짐',sel:!cpp&&!camTransition&&trk==='rebind',lock:optionLock,
       u:Q('tracker='+encodeURIComponent(trk==='rebind'?bt.key:'rebind'))}
    ]});
  }
  // 11-13: engine row of the chosen SORT / OC-SORT / BoT-SORT. The tile itself starts C++ when it is built; Python is picked here.
  ps.filter(p=>!BT_KEYS.includes(p.key)&&'engine' in r).forEach(p=>{
    const hasNative=p.backend==='C++',python=eng==='python'||!hasNative;
    T.push({lab:tname(p.key)+' 엔진',kidOf:TRI,kidKey:p.key,show:curEngine(p),n:[
      {name:'C++',note:hasNative?'기본':'',sel:!python,lock:hasNative?'':(p.restart_required?'새 C++ 빌드를 쓰려면 서버를 다시 시작하세요':'C++ 빌드가 없습니다'),u:Q('tracker='+encodeURIComponent(p.key)+'&engine=auto')},
      {name:'Python',sel:python,u:Q('tracker='+encodeURIComponent(p.key)+'&engine=python')}]});
  });
  const e2eLive=!!(el&&(el.on==='ftt'||el.want==='ftt'));
  if(e2eLive||replayStored){   // 11-14 (user): the end-to-end model (live, or its stored result on a recording) replaces these, so they are locked while it runs; 11-15 (user): not 전처리 - stabilization moves the frame the end-to-end model sees
    const OFF=['검출 모델','검출 모델 자동 선택','검출 뒤 Filter','Tracker','ByteTrack 옵션','위치 기억 · 신뢰도','KeepID','KeepID 찾는 범위','외형 비교','외형 모델','상자 다듬기','모델 바뀔 때 섞기','놓친 상자'],why=(e2eLive?'엔드투엔드(FastTrackTr) 사용 중':'엔드투엔드 결과 재생 중')+' · 엔드투엔드를 끄면 다시 고를 수 있습니다';
    const off=new Set();T.forEach((t,i)=>{if(OFF.includes(t.lab)||/ 엔진$/.test(t.lab))off.add(i);});
    T.forEach((t,i)=>{if(off.has(i)||(t.kidOf!=null&&off.has(t.kidOf))){t.sub=(e2eLive?'엔드투엔드 사용 중':'엔드투엔드 결과 재생 중')+' · 쓰이지 않음';(t.n||[]).forEach(n=>{n.lock=why;n.sel=false;});}
      if(t.lab==='화면 표시')(t.n||[]).forEach(n=>{if(n.name==='사람 후보 (공간 주파수)'||n.name==='예측 상자 표시'){n.lock=why;n.sel=false;}});});}
  // 11-16 (user: 무의미한 옵션은 지움): hidden from the page, the server keeps them off. 흔들림 보정 does almost nothing while following and turns picture first off;
  // 사람 후보 (공간 주파수) gave no sure gain; 피부 온도: this camera (TE-EV1) gives no temperatures. The code stays (set /rules/set?stab=1 etc. by hand to try again).
  {const HIDE=['흔들림 보정','사람 후보 (공간 주파수)','피부 온도 (카메라 온도)'];T.forEach(t=>{if(t.n)t.n=t.n.filter(n=>!HIDE.includes(n.name));});
    // 11-16 (user): old choices hidden unless one of them is the choice in use (then it stays so it can be changed): 온도 Filter (this camera's values are not temperatures),
    // KeepID 찾는 범위 '마지막 자리 주변' (old way), the appearance models before RepVGG_fine-tuning_(OTP2)_20261113. Kept on purpose: SORT, the Python engines, NPU 0·1 번갈아 쓰기.
    const OLD={'검출 뒤 Filter':['온도 Filter'],'KeepID 찾는 범위':['마지막 자리 주변'],'외형 모델':['RepVGG_distillation_(실내)_20261110','RepVGG_fine-tuning_(실내)_20261110','ViT-lite_distillation_(TMOT,CAMEL)_202610','ViT-lite_distillation_(실내)_20261111','ViT-lite_fine-tuning_(실내)_20261111']};
    T.forEach(t=>{const o=OLD[t.lab];if(o&&t.n){t.n=t.n.filter(n=>n.sel||!o.includes(n.name));if(t.f)t.f=t.f.filter(f=>t.n.some(n=>n.g===f));}});
    const drop=f=>{const i=T.findIndex(f);if(i>=0){T.splice(i,1);T.forEach(t=>{if(t.kidOf!=null&&t.kidOf>i)t.kidOf--;});}};
    drop(t=>t.lab==='전처리'&&!t.n.length);drop(t=>t.lab==='KeepID 찾는 범위'&&t.n.length<2);}
  if(E2E){T.forEach(t=>{if(t.kidOf!=null)t.kidOf++;});T.unshift(E2E);}   // 11-14: first row; child rows keep pointing at their parent
  return T;}
