// 11-15 (user: 화면 전송 지연, "카메라 프레임 수준인가"): the picture stream is read with fetch, so the page knows when each picture
// arrived and its camera time (JPEG comment 'LWIR n= t= s= [l=1]', src/web/stream_sender.py). Each picture is shown in the same
// <img id="v"> through a blob URL, so clicks and overlays are unchanged; while the browser still decodes one picture only the
// newest waiting one is kept. Browsers without streamed fetch keep <img src="/stream">. WebKit (iPhone Safari) refuses a fetch of
// multipart/x-mixed-replace ('Load failed'), so the fetch asks with f=1 for the same bytes as multipart/mixed; a fetch stream
// that fails before its first picture (e.g. an older server) switches this page to <img src="/stream"> for good.
// Delay first (user): WebKit drew 14.8 pictures/s through <img> but 7/s through fetch + blob on this PC (Chromium 30 vs 25),
// so WebKit and every iPhone/iPad browser keep <img src="/stream">; the readout then says it cannot measure. Page address
// ?stream=fetch or ?stream=img forces a path (for testing).
const PS_UA=navigator.userAgent||'',PS_CHOICE=new URLSearchParams(location.search).get('stream');
const PS_WEBKIT=(/AppleWebKit/.test(PS_UA)&&!/Chrome|Chromium|Edg/.test(PS_UA))||/iPhone|iPad|iPod/.test(PS_UA)||(navigator.platform==='MacIntel'&&navigator.maxTouchPoints>1);
let streamFetch=(PS_CHOICE==='fetch'||(PS_CHOICE!=='img'&&!PS_WEBKIT))&&typeof fetch==='function'&&typeof AbortController==='function'&&typeof ReadableStream==='function'&&!!(window.Response&&'body' in Response.prototype);
const PS={gen:0,ctl:null,on:false,url:null,old:null,busy:false,next:null,info:null,got:0};
const PS_BOUNDARY=[13,10,45,45,102,114,97,109,101,13,10],PS_HEAD_END=[13,10,13,10];   // '\r\n--frame\r\n', '\r\n\r\n'
function streamRunning(){return PS.on;}
function streamStart(url){
  streamStop();PS.on=true;
  if(!streamFetch){v.src=url;return;}
  const gen=PS.gen,ctl=new AbortController();PS.ctl=ctl;PS.got=0;
  fetch(url+'&f=1',{cache:'no-store',signal:ctl.signal})
    .then(r=>{if(!r.ok||!r.body)throw new Error('stream '+r.status);return psRead(r.body.getReader(),gen);})
    .catch(()=>{})
    .finally(()=>{if(gen!==PS.gen||!PS.on)return;
      if(!PS.got){streamFetch=false;v.src=url;return;}   // no picture came through fetch: the <img> stream from now on
      PS.on=false;   // the stream ended or failed (server restart, network): open it again in 2 s
      setTimeout(()=>{if(gen===PS.gen&&!document.hidden)openStream();},2000);});
}
function streamStop(){PS.gen++;PS.on=false;if(PS.ctl){PS.ctl.abort();PS.ctl=null;}PS.next=null;
  if(!streamFetch&&v.hasAttribute('src'))v.removeAttribute('src');}   // fetch: the last picture stays on screen
function psFind(b,p,from){const last=b.length-p.length;
  for(let i=b.indexOf(p[0],from);i>=0&&i<=last;i=b.indexOf(p[0],i+1)){let k=1;while(k<p.length&&b[i+k]===p[k])k++;if(k===p.length)return i;}
  return -1;}
async function psRead(rd,gen){
  let buf=new Uint8Array(0),start=-1,pos=0;   // start: the first byte of the current picture (after its part headers)
  for(;;){
    const {done,value}=await rd.read();
    if(done||gen!==PS.gen)return;
    const nb=new Uint8Array(buf.length+value.length);nb.set(buf);nb.set(value,buf.length);buf=nb;
    for(;;){
      if(start<0){const h=psFind(buf,PS_HEAD_END,0);if(h<0)break;start=h+4;pos=start;continue;}   // the first part's headers
      const b=psFind(buf,PS_BOUNDARY,pos);
      if(b<0){pos=Math.max(start,buf.length-PS_BOUNDARY.length);break;}
      const h=psFind(buf,PS_HEAD_END,b+2);
      if(h<0){pos=b;break;}   // the next part's headers are not complete yet
      psShow(buf.slice(start,b));
      start=h+4;pos=start;
    }
    if(start>0){buf=buf.slice(start);pos-=start;start=0;}
  }
}
function psInfo(j){   // the numbers of the 'LWIR ...' JPEG comment, {} if there is none
  for(let i=2;i+4<=j.length&&j[i]===255&&j[i+1]!==0xDA;){const n=(j[i+2]<<8)|j[i+3];
    if(j[i+1]===0xFE&&j[i+4]===76&&j[i+5]===87&&j[i+6]===73&&j[i+7]===82){const o={};
      for(const kv of String.fromCharCode.apply(null,j.subarray(i+9,i+2+n)).split(' ')){const e=kv.indexOf('=');if(e>0)o[kv.slice(0,e)]=Number(kv.slice(e+1));}
      return o;}
    i+=2+n;}
  return {};}
function psShow(j){PS.got++;const info=psInfo(j);if(PS.busy){PS.next=[j,info];return;}psPaint(j,info);}
function psPaint(j,info){PS.busy=true;PS.info=info;PS.old=PS.url;PS.url=URL.createObjectURL(new Blob([j],{type:'image/jpeg'}));v.src=PS.url;}
function psDone(ok){if(!streamFetch||!PS.busy)return;PS.busy=false;
  if(PS.old){URL.revokeObjectURL(PS.old);PS.old=null;}
  if(ok){const i=PS.info||{};pictureShown(Number.isFinite(i.t)?i.t:null,i.l||0);}
  if(PS.next){const [j,info]=PS.next;PS.next=null;psPaint(j,info);}}
v.addEventListener('load',()=>psDone(true));v.addEventListener('error',()=>psDone(false));
// The delay readout under the picture chip: this screen's delay (camera frame in at the server -> picture on this screen) and
// pictures per second over the last 2 s. Globals for other picture paths (e.g. a video path) and an automatic choice:
//   pictureShown(t, light, source): a picture is on screen; t = its camera frame time in the server's unix seconds (null if
//     unknown), light = the picture's level (0 full, 1 = 480x360, 2 = 320x240), source = the path's name (default 'mjpeg').
//   streamStats(source): {source, fps, age_ms (median), age_p95_ms, est_ms, light, pictures, clock, measured} of that path
//     (default: the one that showed a picture last). measured = this page times its pictures itself (age_ms; null until the
//     clock offset is known; clock = server time - this clock in s). The picture stream shown through <img> (iPhone) cannot:
//     there fps, light and est_ms come from the server's numbers for this page's stream (GET /time?p=<page id>, every 2 s):
//     est_ms = server time (camera frame in -> written) + the time until the picture arrived here, which the server times
//     from this device's acknowledgements. Not in est_ms: decoding and drawing in the browser (about one screen refresh).
// The clock offset comes from the same GET /time: the answer with the shortest round trip of the last 30 (one minute): an
// answer that waited behind pictures on a slow link would shift the clock by half that wait.
const DL={off:null,sync:[],src:{},cur:'mjpeg',srv:null,srvAt:0};
function pictureShown(t,light,source){const s=source||'mjpeg',now=Date.now()/1000,d=DL.src[s]||(DL.src[s]={shown:[],ages:[],light:0});
  d.shown.push(now);d.light=Number(light)||0;DL.cur=s;
  if(t!=null&&DL.off!=null)d.ages.push([now,(now+DL.off-t)*1000]);}
function serverStats(){return DL.srv&&Date.now()-DL.srvAt<6000?DL.srv:null;}   // this page's stream as the server measured it; null when older than 6 s
function streamStats(source){const s=source||DL.cur,d=DL.src[s],now=Date.now()/1000,sv=s==='mjpeg'?serverStats():null,est=sv&&Number.isFinite(sv.est_ms)?sv.est_ms:null;
  if(!d){const img=s==='mjpeg'&&!streamFetch;   // no picture timed here: the <img> stream, or nothing yet
    return {source:s,fps:img&&sv?sv.fps:0,age_ms:null,age_p95_ms:null,est_ms:est,light:img&&sv?sv.level:0,pictures:img&&sv?Math.round(sv.fps*2):0,clock:DL.off,measured:!img};}
  while(d.shown.length&&d.shown[0]<now-2)d.shown.shift();
  while(d.ages.length&&d.ages[0][0]<now-2)d.ages.shift();
  const a=d.ages.map(x=>x[1]).sort((p,q)=>p-q);
  return {source:s,fps:d.shown.length/2,age_ms:a.length?a[a.length>>1]:null,age_p95_ms:a.length?a[Math.min(a.length-1,Math.floor(a.length*0.95))]:null,
          est_ms:est,light:d.light,pictures:d.shown.length,clock:DL.off,measured:true};}
function timeSync(){if(document.hidden)return;const t0=Date.now();
  fetch('/time?p='+pageId,{cache:'no-store'}).then(r=>r.ok?r.json():Promise.reject()).then(j=>{const t1=Date.now();if(!Number.isFinite(j.t))return;
    DL.srv=j.v||null;DL.srvAt=t1;
    DL.sync.push([t1-t0,j.t-(t0+t1)/2000]);if(DL.sync.length>30)DL.sync.shift();DL.off=DL.sync.reduce((a,b)=>b[0]<a[0]?b:a)[1];}).catch(()=>{});}
function lightText(level){if(!level)return '';const sv=serverStats(),n=sv&&sv.levels||1;return level>=n?' · 작은 사진':' · 중간 사진';}   // the lightest level = 작은 사진 (320x240)
function delayTick(){const el=$('vage'),st=streamStats();if(!el)return;
  if(document.hidden||!st.pictures||(!st.measured&&!PS.on)){el.hidden=true;return;}
  const fps=(st.fps<10?st.fps.toFixed(1):Math.round(st.fps))+' fps'+lightText(st.light);
  if(st.measured){const med=st.age_ms;el.textContent='내 화면 '+(med!=null&&med<5000?'지연 '+Math.round(Math.max(0,med))+' ms · ':'')+fps;}
  else el.textContent='내 화면 '+(st.est_ms!=null?'지연 약 '+Math.round(st.est_ms)+' ms (추정) · ':'')+fps;   // <img> stream: the server's numbers for this page
  el.hidden=false;}
timeSync();setTimeout(timeSync,700);setTimeout(timeSync,1400);setInterval(timeSync,2000);setInterval(delayTick,1000);
openStream();   // the first stream (the img has no src in the HTML): after this fragment's constants exist
