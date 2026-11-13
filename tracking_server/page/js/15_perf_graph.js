// v26d: small speed / delay graphs below the picture. Rows from GET /perf: [t (unix s), track fps, cam fps, delay ms, delay max ms, compute ms], one per second, 15 min kept by the server. Only the rows after the last one we have are asked for, and nothing is asked while the graphs are switched off.
const PF={rows:[],span:60,off:0,busy:false,on:true};
try{const s_=JSON.parse(localStorage.getItem('lwir_live_pf')||'{}')||{};if(s_.on===false)PF.on=false;if([60,300,900].includes(s_.span))PF.span=s_.span;}catch(_){}
const PF_LATENCY_LIMIT=100;
const PFC=[
  {cv:$('pfc0'),series:[[2,'--c2',1.8],[1,'--c1',2.5]],min:30},
  {cv:$('pfc1'),series:[[4,'--c3',1.9,[4,3]],[5,'--c4',2],[3,'--c1',2.5]],limit:PF_LATENCY_LIMIT},
  {cv:$('pfc2'),series:[[8,'--c5',2.2],[9,'--c6',2.2]],min:50},   // 8 = picture making (server), 9 = transfer to this device (measured here)
];
// 11-13: transfer time to this device = one /snap.jpg request until its last byte, every 2 s while the graphs are shown.
const PFNET={ms:null,at:0,busy:false};
function pfnet(){if(document.hidden||!PF.on||PFNET.busy)return;PFNET.busy=true;const t0=performance.now();
  fetch('/snap.jpg?pf='+Date.now(),{cache:'no-store'}).then(r=>r.ok?r.arrayBuffer():Promise.reject()).then(()=>{PFNET.ms=performance.now()-t0;PFNET.at=Date.now();}).catch(()=>{PFNET.ms=null;}).finally(()=>{PFNET.busy=false;});}
setInterval(pfnet,2000);pfnet();
const pfOverflow=document.createElement('p');
pfOverflow.className='pf-overflow';pfOverflow.hidden=true;
$('pfago').parentNode.after(pfOverflow);
const nice=m=>{const p=Math.pow(10,Math.floor(Math.log10(m)));for(const k of [1,2,3,5,10])if(m<=k*p)return k*p;return 10*p;};
function pfplot(chart,rows,t0,style){
  const cv=chart.cv,w=cv.clientWidth,h=cv.clientHeight;
  const L=30,R=5,T=7,B=7,pw=w-L-R,ph=h-T-B;
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
    max=nice(max);
  }
  const xFor=t=>L+(t-t0)/PF.span*pw;
  const yFor=v=>T+ph*(1-Math.min(v,max)/max);
  g.font='600 11px "Segoe UI","Malgun Gothic",sans-serif';g.textBaseline='middle';g.textAlign='right';
  g.lineWidth=1;g.strokeStyle=color('--pf-grid');g.fillStyle=color('--pf-label');
  for(const value of [0,max/2,max]){
    const y=Math.round(yFor(value))+.5;
    g.beginPath();g.moveTo(L,y);g.lineTo(w-R,y);g.stroke();
    g.fillText(String(value),L-4,y);
  }
  g.save();g.beginPath();g.rect(L,T-4,pw,ph+8);g.clip();g.lineJoin='round';
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
}
function pfdraw(){
  $('pfo').classList.toggle('off',!PF.on);$('pfbtn').classList.toggle('on',PF.on);$('pfbtn').setAttribute('aria-pressed',PF.on?'true':'false');$('pfspan').textContent=(PF.span/60)+'분';$('pfago').textContent=(PF.span/60)+'분 전';
  if(!PF.on)return;
  const now=Date.now()/1000+PF.off,t0=now-PF.span,vis=PF.rows.filter(r=>r[0]>=t0-2);
  const style=getComputedStyle(document.documentElement);
  for(const chart of PFC)pfplot(chart,vis,t0,style);
  let peak=0;
  for(const row of vis)if(row[0]>=t0&&row[4]>peak)peak=row[4];
  pfOverflow.hidden=peak<=PF_LATENCY_LIMIT;
  pfOverflow.textContent=pfOverflow.hidden?'':`▲ ${PF_LATENCY_LIMIT} ms 초과 · 기간 최대 ${peak.toFixed(1)} ms`;
  const last=PF.rows.length?PF.rows[PF.rows.length-1]:null,r=last&&now-last[0]<5?last:null;   // values older than 5 s are not shown as current
  [[0,1],[1,2],[2,3],[3,4],[4,5],[5,8],[6,9]].forEach(([i,k])=>{$('pfv'+i).textContent=r&&Number.isFinite(r[k])?r[k].toFixed(0):'–';});}
function pftick(){if(document.hidden||!PF.on||PF.busy)return;PF.busy=true;const since=PF.rows.length?PF.rows[PF.rows.length-1][0]:0;
  fetch('/perf?since='+since,{cache:'no-store'}).then(r=>r.json()).then(p=>{lastOk=Date.now();PF.off=p.now-Date.now()/1000;   // the board clock differs from this device's: its "now" is used for the time axis
    if(p.rows.length&&PF.rows.length&&p.rows[0][0]<=PF.rows[PF.rows.length-1][0])PF.rows=[];   // server restarted or its clock stepped back: start again
    for(const r of p.rows){const pic=Number.isFinite(r[7])&&Number.isFinite(r[3])?Math.max(0,r[7]-r[3]):null;   // picture making = (arrival -> picture ready) - (arrival -> track result)
      r[8]=pic;r[9]=(p.rows.length<=3||r===p.rows[p.rows.length-1])&&PFNET.ms!=null&&Date.now()-PFNET.at<5000?PFNET.ms:null;PF.rows.push(r);}   /* history loaded in one go has no transfer samples of its own */ if(PF.rows.length>900)PF.rows.splice(0,PF.rows.length-900);}).catch(()=>{}).finally(()=>{PF.busy=false;pfdraw();});}
const pfsave=()=>{try{localStorage.setItem('lwir_live_pf',JSON.stringify({on:PF.on,span:PF.span}));}catch(_){}};
$('pfbtn').addEventListener('click',e=>{e.stopPropagation();PF.on=!PF.on;pfsave();pfdraw();pftick();});
$('pfspan').addEventListener('click',e=>{e.stopPropagation();PF.span={60:300,300:900,900:60}[PF.span];pfsave();pfdraw();});
addEventListener('resize',pfdraw);setInterval(pftick,1000);pfdraw();pftick();
