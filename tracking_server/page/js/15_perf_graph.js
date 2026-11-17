// v26d: small speed / delay graphs below the picture. Rows from GET /perf: [t (unix s), track fps, cam fps, delay ms, delay max ms, compute ms], one per second, 15 min kept by the server. Only the rows after the last one we have are asked for, and nothing is asked while the graphs are switched off.
const PF={rows:[],span:60,off:0,busy:false,marks:[],seen:null,pipeline:null,receivedAt:0};   /* marks: setting changes seen while this page is open, drawn as vertical lines */
try{const s_=JSON.parse(localStorage.getItem('lwir_live_pf')||'{}')||{};if([60,300,900].includes(s_.span))PF.span=s_.span;if(s_.detail===true)$('pfdetail').open=true;}catch(_){}
const PF_LATENCY_LIMIT=100;
const PFC=[
  {cv:$('pfc0'),series:[[1,'--c1',2.5],[2,'--c2',1.7,[4,4]]],min:45,labels:true},
  {cv:$('pfc1'),series:[[4,'--c3',1.9,[4,3]],[5,'--c4',2],[3,'--c1',2.5]],limit:PF_LATENCY_LIMIT,goal:true},
  {cv:$('pfc2'),series:[[8,'--c5',2.2],[9,'--c6',2.2]],min:50},   // 8 = frame in -> picture ready (server), 9 = frame in -> this screen (the stream's own delay)
];
// 11-15 (user: the graph showed 100 ms): the line was one /snap.jpg download every 2 s (a full picture on a new connection, there and
// back), which overstated the stream's delay and took the stream's bandwidth. Now it is the stream's own delay from picture_stream.js:
// timed on this device where the page reads the pictures, else the server's estimate for this page ('(추정)').
function pfdelay(){const st=typeof streamStats==='function'?streamStats('mjpeg'):null;if(!st||!st.pictures)return [null,false];
  return st.measured?[st.age_ms,false]:[st.est_ms,true];}
const pfOverflow=document.createElement('p');
pfOverflow.className='pf-overflow';pfOverflow.hidden=true;
$('pf-track-axis').after(pfOverflow);
const nice=m=>{const p=Math.pow(10,Math.floor(Math.log10(m)));for(const k of [1,2,3,5,10])if(m<=k*p)return k*p;return 10*p;};
function pfplot(chart,rows,t0,style,goal){
  const cv=chart.cv,w=cv.clientWidth,h=cv.clientHeight;
  const L=30,R=5,T=chart.labels?14:7,B=7,pw=w-L-R,ph=h-T-B;
  if(pw<=0||ph<=0)return;
  const dpr=window.devicePixelRatio||1;
  const pixelW=Math.round(w*dpr),pixelH=Math.round(h*dpr);
  if(cv.width!==pixelW||cv.height!==pixelH){cv.width=pixelW;cv.height=pixelH;}
  const g=cv.getContext('2d'),color=name=>style.getPropertyValue(name).trim();
  g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
  let max=chart.limit;
  if(!max){
    max=chart.min;
    for(const row of rows)for(const [col] of chart.series)if(row[col]>max)max=row[col];
    max=chart.labels&&max<=45?45:nice(max);
  }
  const xFor=t=>L+(t-t0)/PF.span*pw;
  const yFor=v=>T+ph*(1-Math.min(v,max)/max);
  g.font='600 11px "Segoe UI","Malgun Gothic",sans-serif';g.textBaseline='middle';g.textAlign='right';
  g.lineWidth=1;g.strokeStyle=color('--pf-grid');g.fillStyle=color('--pf-label');
  const ticks=chart.labels&&max===45?[0,15,30,45]:[0,max/2,max];
  for(const value of ticks){
    const y=Math.round(yFor(value))+.5;
    g.beginPath();g.moveTo(L,y);g.lineTo(w-R,y);g.stroke();
    g.fillText(String(value),L-4,y);
  }
  g.save();g.beginPath();g.rect(L,T-4,pw,ph+8);g.clip();g.lineJoin='round';
  if(chart.goal&&goal<=max){
    g.strokeStyle=color('--warn');g.lineWidth=1;g.setLineDash([2,4]);g.beginPath();
    const y=yFor(goal);g.moveTo(L,y);g.lineTo(w-R,y);g.stroke();
  }
  for(const [col,ink,weight,dash=[]] of chart.series){
    g.strokeStyle=color(ink);g.lineWidth=weight;g.setLineDash(dash);g.beginPath();
    let lastT=null;
    const overflow=[];
    for(const row of rows){
      const [t]=row,value=row[col];
      if(!Number.isFinite(value)){lastT=null;continue;}
      const x=xFor(t),y=yFor(value);
      if(lastT===null||t-lastT>3)g.moveTo(x,y);else g.lineTo(x,y);
      if(chart.limit&&value>chart.limit)overflow.push(x);
      lastT=t;
    }
    g.stroke();
    g.fillStyle=color(ink);g.beginPath();
    for(const x of overflow){g.moveTo(x,T-4);g.lineTo(x+3.5,T+2);g.lineTo(x-3.5,T+2);g.closePath();}
    g.fill();
  }
  g.restore();
  /* 11-13: a dashed vertical line where NPU alternation or the tracker changed, so before and after can be compared */
  g.save();g.setLineDash([3,3]);g.lineWidth=1.2;g.strokeStyle=color('--warn');g.fillStyle=color('--warn');g.font='600 10px "Segoe UI","Malgun Gothic",sans-serif';g.textBaseline='top';
  let labelEnd=-1e9;
  for(const mark of PF.marks){
    if(mark.t<t0)continue;
    const x=Math.round(xFor(mark.t))+.5;
    g.beginPath();g.moveTo(x,T-4);g.lineTo(x,T+ph+4);g.stroke();
    if(chart.labels&&x>labelEnd){g.textAlign=x>w-90?'right':'left';g.fillText(mark.text,x+(x>w-90?-3:3),0);labelEnd=x+g.measureText(mark.text).width+8;}   /* a label that would overlap the previous one is left out; its line stays */
  }
  g.restore();
}
function pfmark(){
  if(typeof rlast==='undefined'||!rlast)return;
  const feed=rlast.npu_feed&&typeof rlast.npu_feed.active==='boolean'?rlast.npu_feed.active:null;
  const cam=rlast.camera_bytetrack&&rlast.camera_bytetrack.active;
  const tracker=cam?'ByteTrack C++':(typeof tname==='function'?tname(rlast.tracker):String(rlast.tracker))+' '+(rlast.tracker_backend||'');
  const now={feed,tracker};
  if(PF.seen){
    const t=Date.now()/1000+PF.off;
    if(feed!==null&&PF.seen.feed!==null&&feed!==PF.seen.feed)PF.marks.push({t,text:feed?'번갈아 켬':'번갈아 끔'});
    if(tracker!==PF.seen.tracker)PF.marks.push({t,text:tracker});
    if(PF.marks.length>40)PF.marks.splice(0,PF.marks.length-40);
  }
  PF.seen=now;
}
// Fixed scale relative to server receipt, with room for camera transport before zero.
const PF_RAIL_MIN_MS=-10,PF_RAIL_MAX_MS=70;
const pfRailPosition=value=>(Math.min(PF_RAIL_MAX_MS,Math.max(PF_RAIL_MIN_MS,value))-PF_RAIL_MIN_MS)/(PF_RAIL_MAX_MS-PF_RAIL_MIN_MS);
function pfplaceLabels(markers){
  const key=$('pf-pipe-key'),width=key.clientWidth,gap=6;
  if(!width)return;
  const labels=markers.filter(([,value])=>Number.isFinite(value)).map(([stage,value])=>{
    const node=$('pf-rail-'+stage+'-label'),w=node.offsetWidth;
    const left=Math.max(0,Math.min(width-w,pfRailPosition(value)*width-w/2));
    return {node,left,right:left+w};
  });
  const height=Math.max(...labels.map(({node})=>node.offsetHeight));
  const rows=[];
  for(const label of labels){
    const {node,left,right}=label;
    let row=rows.findIndex(placed=>placed.every(other=>right+gap<=other.left||left>=other.right+gap));
    if(row<0)row=rows.push([])-1;
    rows[row].push(label);
    node.style.left=left+'px';
    node.style.top=row*(height+3)+'px';
  }
  key.style.height=(rows.length*(height+3)-3)+'px';
}
function pfSchedule(pipeline){
  const status=$('pf-schedule-status'), chart=$('pf-schedule-data');
  const data=pipeline.frame_schedule_ms;
  const replay=document.body.classList.contains('replay-mode');
  const frames=Array.isArray(data) ? data.slice(0,3) : [];
  const valid=frames.length>0 && frames.every((frame,i)=>{
    if(!Array.isArray(frame)||frame.length!==3||!frame.every(Number.isFinite))return false;
    const [receive, detect, track]=frame;
    const arrivalOrdered=i===0 ? receive===0 : receive>=frames[i-1][0];
    return arrivalOrdered && receive<=detect && detect<=track;
  });
  chart.hidden=replay||!valid;
  status.hidden=!chart.hidden;
  if(chart.hidden){
    status.textContent=replay ? '녹화 재생 중에는 실시간 시간표를 표시하지 않습니다.'
      : Array.isArray(data) ? '실시간 프레임 측정 대기 중' : '프레임별 실측 데이터 없음';
    return;
  }
  $('pf-schedule-gaps').hidden=frames.length<2;
  $('pf-schedule-gap-rows').innerHTML=frames.slice(1).map((frame,i)=>{
    const received=frame[0]-frames[i][0], published=frame[2]-frames[i][2];
    const output=published<0?'순서 역전':published.toFixed(1)+' ms';
    return '<tr><th scope="row">'+(i+1)+' → '+(i+2)+'</th><td>'+received.toFixed(1)
      +' ms</td><td>'+output+'</td></tr>';
  }).join('');
  const cores=Array.isArray(pipeline.frame_schedule_npu) ? pipeline.frame_schedule_npu : [];
  const last=Math.max(...frames.map(row=>row[2]));
  const limit=Math.max(20,Math.ceil(last/20)*20);
  const pct=value=>String(value/limit*100)+'%';
  $('pf-schedule-axis').innerHTML='<div><span>0</span><span>'+limit/2+'</span><span>'+limit+' ms</span></div>';
  const stages=[['received','수신'],['detect','탐지'],['track','추적']];
  $('pf-schedule-frames').innerHTML=frames.map((frame,i)=>{
    const [receive, detect, track]=frame;
    const segment=(stage,start,end)=>'<i class="pf-schedule-bar pf-schedule-'+stage
      +'" style="left:'+pct(start)+';width:'+pct(end-start)+'"></i>';
    const dots=stages.map(([stage],j)=>'<i class="pf-schedule-mark pf-schedule-'+stage
      +'" style="left:'+pct(frame[j])+'"></i>').join('');
    const values=stages.map(([stage,label],j)=>'<span class="pf-schedule-'+stage+'">'
      +label+' <b>'+frame[j].toFixed(1)+'</b> ms</span>').join('');
    const npu=Array.isArray(cores[i]) ? cores[i].filter(core=>Number.isInteger(core)&&core>=0) : [];
    const coreLabel=npu.length ? '<small class="pf-schedule-core">NPU '+npu.join('·')+'</small>' : '';
    return '<li class="pf-schedule-frame"><span>'+(i+1)+'번째'+coreLabel+'</span><div>'
      +'<div class="pf-schedule-lane" aria-hidden="true">'+segment('detect',receive,detect)
      +segment('track',detect,track)+dots+'</div><div class="pf-schedule-times num">'
      +values+'</div></div></li>';
  }).join('');
}
function pfpipe(){
  const p=Date.now()-PF.receivedAt<5000&&PF.pipeline?PF.pipeline:{};
  const fmt=(value,unit)=>Number.isFinite(value)?value.toFixed(1)+' '+unit:'–';
  const stages=[['input',p.camera_fps,p.camera_p95_ms],['detect',p.det_fps,p.det_p95_ms],['track',p.track_fps,p.track_p95_ms]];
  for(const [key,fps,latency] of stages){
    $('pf-'+key+'-fps').textContent=fmt(fps,'FPS');
    $('pf-'+key+'-latency').textContent=fmt(latency,'ms');
  }
  const liveCamera=!document.body.classList.contains('replay-mode')&&p.camera_p95_ms>0;
  const cameraMs=liveCamera?p.camera_p95_ms:0;
  $('pf-rail-camera-value').textContent=fmt(liveCamera?cameraMs:null,'ms');
  $('pf-rail-detect-value').textContent=fmt(p.det_p95_ms,'ms');
  $('pf-rail-track-value').textContent=fmt(p.track_p95_ms,'ms');
  const goal=Number.isFinite(p.goal_ms)?p.goal_ms:33;
  $('pf-goal-budget').textContent=goal+' ms';
  const tracked=Number.isFinite(p.track_p95_ms),detected=Number.isFinite(p.det_p95_ms);
  const target=$('pf-goal-status');
  target.textContent=tracked?'목표 '+goal+' ms · '+(p.track_p95_ms<=goal?'충족':'초과'):'목표 '+goal+' ms';
  target.dataset.met=String(tracked&&p.track_p95_ms<=goal);
  const samples=Number.isFinite(p.sample_count)?p.sample_count:0;
  $('pf-pipe-samples').textContent=samples?'최근 1분 추적 '+samples.toLocaleString()+'장':'최근 1분 추적 집계 중';
  const detEnd=detected?p.det_p95_ms:0;
  const trackEnd=tracked?Math.max(p.track_p95_ms,detEnd):detEnd;
  const pos=value=>(pfRailPosition(value)*100)+'%';
  const rail=$('pf-rail-goal').parentNode;
  rail.style.setProperty('--pf-camera',pos(-cameraMs));
  rail.style.setProperty('--pf-server',pos(0));
  rail.style.setProperty('--pf-detect-end',pos(detEnd));
  rail.style.setProperty('--pf-track-end',pos(trackEnd));
  $('pf-rail-goal').style.left=pos(goal);
  const markers=[['camera',liveCamera?-cameraMs:null],['server',0],
    ['detect',detected?p.det_p95_ms:null],['track',tracked?p.track_p95_ms:null]];
  for(const [key,value] of markers){
    const mark=$('pf-rail-'+key),label=$('pf-rail-'+key+'-label');
    const valid=Number.isFinite(value);
    mark.hidden=label.hidden=!valid;
    if(valid)mark.style.left=pos(value);
    label.dataset.clipped=value<PF_RAIL_MIN_MS?'left':value>PF_RAIL_MAX_MS?'right':'';
  }
  pfplaceLabels(markers);
  const feed=typeof rlast==='undefined' ? null : rlast?.npu_feed;
  const modeLabel=$('pf-feed-mode');
  modeLabel.hidden=$('pf-feed-label').hidden=!feed?.submitted||document.body.classList.contains('replay-mode');
  if(!modeLabel.hidden){
    const mode=feed.active?'NPU 0·1 교대':'NPU 0 단독';
    modeLabel.textContent=(feed.requested==='auto'?'자동 · ':'')+mode;
  }
  pfSchedule(p);
}
function pfdraw(){
  pfpipe();
  if(!$('pfdetail').open)return;
  const span=(PF.span/60)+'분',goal=Number($('pf-goal-budget').textContent.split(' ')[0]);
  $('pfspan').textContent=span;
  $('pf-graph-goal').textContent=goal+' ms';
  for(const label of document.querySelectorAll('[data-pf-ago]'))label.textContent=span+' 전';
  const now=Date.now()/1000+PF.off,t0=now-PF.span,vis=PF.rows.filter(r=>r[0]>=t0-2);
  const style=getComputedStyle($('pfo'));
  for(const chart of PFC)pfplot(chart,vis,t0,style,goal);
  let peak=0;
  for(const row of vis)if(row[0]>=t0&&row[4]>peak)peak=row[4];
  pfOverflow.hidden=peak<=PF_LATENCY_LIMIT;
  pfOverflow.textContent=pfOverflow.hidden?'':`▲ ${PF_LATENCY_LIMIT} ms 초과 · 기간 최대 ${peak.toFixed(1)} ms`;
  const last=PF.rows.length?PF.rows[PF.rows.length-1]:null,r=last&&now-last[0]<5?last:null;   // values older than 5 s are not shown as current
  [[0,1],[1,2],[2,3],[3,4],[4,5],[5,8],[6,9]].forEach(([i,k])=>{$('pfv'+i).textContent=r&&Number.isFinite(r[k])?r[k].toFixed(0):'–';});}
function pftick(){pfmark();if(document.hidden||PF.busy)return;PF.busy=true;const since=PF.rows.length?PF.rows[PF.rows.length-1][0]:0;
  fetch('/perf?since='+since,{cache:'no-store'}).then(r=>r.json()).then(p=>{lastOk=Date.now();PF.receivedAt=lastOk;PF.pipeline=p.pipeline||null;PF.off=p.now-Date.now()/1000;   // the board clock differs from this device's: its "now" is used for the time axis
    if(p.rows.length&&PF.rows.length&&p.rows[0][0]<=PF.rows[PF.rows.length-1][0])PF.rows=[];   // server restarted or its clock stepped back: start again
    for(const r of p.rows){   // r[7] = arrival -> picture ready (server, mean of the second)
      const [age,guess]=pfdelay();$('pft6').textContent=guess?'기기 도착(추정)':'내 화면';$('pf-screen-note').hidden=!guess;
      r[8]=Number.isFinite(r[7])?r[7]:null;r[9]=(p.rows.length<=3||r===p.rows[p.rows.length-1])&&Number.isFinite(age)?age:null;PF.rows.push(r);}   /* history loaded in one go has no transfer samples of its own */ if(PF.rows.length>900)PF.rows.splice(0,PF.rows.length-900);}).catch(()=>{}).finally(()=>{PF.busy=false;pfdraw();});}
const pfsave=()=>{try{localStorage.setItem('lwir_live_pf',JSON.stringify({detail:$('pfdetail').open,span:PF.span}));}catch(_){}};
$('pfdetail').addEventListener('toggle',()=>{pfsave();pfdraw();});
$('pfspan').addEventListener('click',e=>{e.stopPropagation();PF.span={60:300,300:900,900:60}[PF.span];pfsave();pfdraw();});
addEventListener('resize',pfdraw);setInterval(pftick,1000);pfdraw();pftick();
