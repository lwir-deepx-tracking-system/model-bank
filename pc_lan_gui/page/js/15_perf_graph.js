// v26d: small speed / delay graphs on the picture. Rows from GET /perf: [t (unix s), track fps, cam fps, delay ms, delay max ms, compute ms], one per second, 15 min kept by the server. Only the rows after the last one we have are asked for, and nothing is asked while the graphs are switched off.
const PF={rows:[],span:60,off:0,busy:false,on:true};
try{const s_=JSON.parse(localStorage.getItem('lwir_live_pf')||'{}')||{};if(s_.on===false)PF.on=false;if([60,300,900].includes(s_.span))PF.span=s_.span;}catch(_){}
const PFC=[{cv:$('pfc0'),s:[[1,'--c1',1.6],[2,'--c2',1.2]],min:30},{cv:$('pfc1'),s:[[4,'--c3',1],[5,'--c2',1.2],[3,'--c1',1.6]],min:100}];
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const nice=m=>{const p=Math.pow(10,Math.floor(Math.log10(m)));for(const k of [1,2,3,5,10])if(m<=k*p)return k*p;return 10*p;};
function pfdraw(){
  $('pfo').classList.toggle('off',!PF.on);$('pfbtn').classList.toggle('on',PF.on);$('pfbtn').setAttribute('aria-pressed',PF.on?'true':'false');$('pfspan').textContent=(PF.span/60)+'분';
  if(!PF.on)return;
  const now=Date.now()/1000+PF.off,t0=now-PF.span,vis=PF.rows.filter(r=>r[0]>=t0-2);
  for(const c of PFC){const cv=c.cv,w=cv.clientWidth,h=cv.clientHeight;if(!w||!h)continue;const dpr=window.devicePixelRatio||1;
    if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);}
    const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
    const L=24,R=2,T=6,B=2,pw=w-L-R,ph=h-T-B;let mx=c.min;for(const r of vis)for(const s of c.s)if(r[s[0]]!=null&&r[s[0]]>mx)mx=r[s[0]];mx=nice(mx);
    const X=t=>L+(t-t0)/PF.span*pw,Y=v=>T+ph-(v/mx)*ph;
    g.font='9.5px "Segoe UI","Malgun Gothic",sans-serif';g.textBaseline='middle';g.textAlign='right';g.lineWidth=1;
    for(let i=0;i<=2;i++){const v=mx*i/2,y=Math.round(Y(v))-.5;g.strokeStyle=cssv('--line2');g.beginPath();g.moveTo(L,y);g.lineTo(w-R,y);g.stroke();if(i){g.fillStyle=cssv('--ink2');g.fillText(String(Math.round(v)),L-4,y+.5);}}
    g.save();g.beginPath();g.rect(L,T-2,pw,ph+3);g.clip();g.lineJoin='round';
    for(const s of c.s){g.strokeStyle=cssv(s[1]);g.lineWidth=s[2];g.beginPath();let pt=null;
      for(const r of vis){if(r[s[0]]==null){pt=null;continue;}const x=X(r[0]),y=Y(r[s[0]]);if(pt===null||r[0]-pt>3)g.moveTo(x,y);else g.lineTo(x,y);pt=r[0];}   // no line across a gap of over 3 s (camera gave no frame)
      g.stroke();}
    g.restore();}
  const last=PF.rows.length?PF.rows[PF.rows.length-1]:null,r=last&&now-last[0]<5?last:null;   // values older than 5 s are not shown as current
  [[0,1],[1,2],[2,3],[3,4],[4,5]].forEach(([i,k])=>{$('pfv'+i).textContent=r&&r[k]!=null?r[k].toFixed(0):'–';});}
function pftick(){if(document.hidden||!PF.on||PF.busy)return;PF.busy=true;const since=PF.rows.length?PF.rows[PF.rows.length-1][0]:0;
  fetch('/perf?since='+since,{cache:'no-store'}).then(r=>r.json()).then(p=>{lastOk=Date.now();PF.off=p.now-Date.now()/1000;   // the board clock differs from this device's: its "now" is used for the time axis
    if(p.rows.length&&PF.rows.length&&p.rows[0][0]<=PF.rows[PF.rows.length-1][0])PF.rows=[];   // server restarted or its clock stepped back: start again
    for(const r of p.rows)PF.rows.push(r);if(PF.rows.length>900)PF.rows.splice(0,PF.rows.length-900);}).catch(()=>{}).finally(()=>{PF.busy=false;pfdraw();});}
const pfsave=()=>{try{localStorage.setItem('lwir_live_pf',JSON.stringify({on:PF.on,span:PF.span}));}catch(_){}};
$('pfbtn').addEventListener('click',e=>{e.stopPropagation();PF.on=!PF.on;pfsave();pfdraw();pftick();});
$('pfspan').addEventListener('click',e=>{e.stopPropagation();PF.span={60:300,300:900,900:60}[PF.span];pfsave();pfdraw();});
addEventListener('resize',pfdraw);setInterval(pftick,1000);pfdraw();pftick();
