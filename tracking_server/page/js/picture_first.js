// 11-15 (user: total delay about one camera frame): with '사진 먼저' the server sends the camera picture as soon as the frame
// arrives (before detection and tracking) and the boxes separately (GET /overlay, server-sent events, src/web/picture_first.py).
// They are drawn here on a canvas over the picture, with the same 4:3 fit as the click mapping. The boxes come about two
// camera frames after the picture; boxes older than 1 s are not drawn. A click still picks from the boxes drawn last.
const OV={es:null,data:null,at:0,cv:$('vdraw'),boxes:null,gen:null};
const overlayQuery = new URLSearchParams(location.search);
let overlayCenterEnabled = overlayQuery.get('overlay_center') === '1';
const overlayDebugEnabled = overlayQuery.get('overlay_debug') === '1';

class OverlayCenter {
  constructor() { this.reset(); }

  reset() {
    this.previous = new Map();
    this.time = null;
  }

  update(boxes, time) {
    if (!Number.isFinite(time)) {
      this.reset();
      return boxes;
    }
    const dt = time - this.time;
    if (this.time === null || dt <= 0 || dt > 0.120) this.reset();
    const counts = new Map();
    for (const box of boxes) counts.set(box[4], (counts.get(box[4]) || 0) + 1);
    const next = new Map();
    const alpha = 1 - Math.exp(-Math.max(dt, 0) / 0.065);
    const rendered = boxes.map(box => {
      const [left, top, right, bottom, id, follow] = box;
      const w = right - left, h = bottom - top;
      if (counts.get(id) !== 1 || ![left, top, right, bottom].every(Number.isFinite) || w <= 0 || h <= 0) {
        return box;
      }
      const x = (left + right) / 2, y = (top + bottom) / 2;
      const prior = this.previous.get(id);
      let sx = x, sy = y;
      const minSize = Math.min(w, h);
      if (prior && prior.follow === follow &&
          Math.max(w/prior.w, prior.w/w, h/prior.h, prior.h/h) <= 1.35 &&
          Math.hypot(x-prior.x, y-prior.y) < Math.min(4, Math.max(2, minSize * 0.12))) {
        const dx = (prior.sx - x) * (1 - alpha), dy = (prior.sy - y) * (1 - alpha);
        const length = Math.hypot(dx, dy);
        const limit = Math.min(2, minSize * 0.06);
        const scale = length > limit ? limit / length : 1;
        sx += dx * scale;
        sy += dy * scale;
      }
      next.set(id, {x, y, sx, sy, w, h, follow});
      if (sx === x && sy === y) return box;
      return [left + sx - x, top + sy - y, right + sx - x, bottom + sy - y, ...box.slice(4)];
    });
    this.previous = next;
    this.time = time;
    return rendered;
  }
}

const overlayCenter = new OverlayCenter();
const overlaySwitch=$('overlay-center-switch');
overlaySwitch.checked=overlayCenterEnabled;
overlaySwitch.onchange=()=>{overlayCenterEnabled=overlaySwitch.checked;overlayResetHistory();ovDraw();};
const overlayGaps = [];
function overlayOnMjpeg() {
  return PS.on && !PS.nativeFailed && (streamFetch || v.hasAttribute('src')) &&
    !document.hidden && !document.body.classList.contains('replay-mode');
}
if (overlayDebugEnabled) {
  window.overlayPictureShown = (pictureTime) => {
    const data = OV.data;
    if (!streamFetch || !overlayOnMjpeg() || !data?.pf || !data.b?.length ||
        !Number.isFinite(pictureTime) || !Number.isFinite(data.t) || Date.now() - OV.at > 250) return;
    if (overlayGaps.length === 300) overlayGaps.shift();
    overlayGaps.push((pictureTime - data.t) * 1000);
  };
  window.overlayTiming = () => {
    if (!streamFetch) return {samples: 0, median_ms: null, p95_ms: null};
    const n = overlayGaps.length;
    const sorted = overlayGaps.slice().sort((a, b) => a - b);
    return {
      samples: n,
      median_ms: n ? sorted[Math.round((n - 1) * 0.5)] : null,
      p95_ms: n ? sorted[Math.round((n - 1) * 0.95)] : null,
    };
  };
}

function ovColor(id){   // the server's track_color(): golden-angle hue, saturation 0.72, value 1
  const h=(id*0.61803398875)%1*6,i=Math.floor(h),f=h-i,s=0.72,p=1-s,q=1-s*f,t=1-s*(1-f);
  const [r,g,b]=[[1,t,p],[q,1,p],[p,1,t],[p,q,1],[t,p,1],[1,p,q]][i%6];
  return 'rgb('+Math.round(r*255)+','+Math.round(g*255)+','+Math.round(b*255)+')';}
function ovDraw(){const cv=OV.cv;if(!cv)return;
  const dpr=window.devicePixelRatio||1,bw=cv.clientWidth,bh=cv.clientHeight,W=Math.round(bw*dpr),H=Math.round(bh*dpr);
  if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H;}
  const x=cv.getContext('2d');x.clearRect(0,0,W,H);
  const d=OV.data;
  if(!d||!d.pf||Date.now()-OV.at>1000||document.body.classList.contains('replay-mode'))return;
  let w=bw,h=bw*3/4;if(h>bh){h=bh;w=bh*4/3;}   // the picture inside its box (object-fit: contain)
  const ox=(bw-w)/2,oy=(bh-h)/2,k=w/(d.w||640);
  x.save();x.scale(dpr,dpr);x.font='bold 13px sans-serif';x.textBaseline='bottom';
  for(const [x1,y1,x2,y2,id,fol] of (overlayOnMjpeg()&&OV.gen===PS.gen ? OV.boxes : null)||d.b||[]){const L=ox+x1*k,T=oy+y1*k,R=ox+x2*k,B=oy+y2*k;
    const style=fol?followBoxStyle(id):'';
    if(style==='active'){x.strokeStyle='rgb(255,220,0)';x.lineWidth=4;x.strokeRect(L,T,R-L,B-T);
      const lab='FOLLOW '+id,tw=x.measureText(lab).width,ty=Math.max(T,18);
      x.fillStyle='rgb(255,220,0)';x.fillRect(L-2,ty-18,tw+10,18);x.fillStyle='#141414';x.fillText(lab,L+3,ty-3);}
    else if(style==='held'||style==='searching'){
      x.strokeStyle=x.fillStyle='rgb(255,174,105)';x.lineWidth=2;x.setLineDash([5,3]);x.strokeRect(L,T,R-L,B-T);x.setLineDash([]);
      x.fillText(id+' · '+(style==='searching'?'탐색 중':'탐색 대기'),L,Math.max(T-3,14));}
    else{const c=ovColor(id);x.strokeStyle=c;x.lineWidth=2;x.strokeRect(L,T,R-L,B-T);x.fillStyle=c;x.fillText(String(id),L,Math.max(T-3,14));}}
  x.font='12px sans-serif';
  for(const [x1,y1,x2,y2,id,lab,last,pts,rad,col,al] of d.p||[]){const X=v=>ox+v*k,Y=v=>oy+v*k;   // predicted boxes of lost people (pred_show), as the server drew them
    x.strokeStyle=x.fillStyle=col;x.globalAlpha=al==null?1:Math.max(0.2,Math.min(1,al));   // 11-16: fades while the person stays lost (src/web/lost_marks.py)
    if(rad>0){x.lineWidth=1;x.beginPath();x.arc(X((x1+x2)/2),Y((y1+y2)/2),rad*k,0,2*Math.PI);x.stroke();}
    if(last){const [lx,ly,lw,lh]=last;x.setLineDash([3,3]);x.lineWidth=1;x.strokeRect(X(lx-lw/2),Y(ly-lh/2),lw*k,lh*k);x.setLineDash([]);
      x.lineWidth=2;x.beginPath();x.arc(X(lx),Y(ly),4,0,2*Math.PI);x.stroke();x.fillText('last seen',X(lx-lw/2),Y(ly+lh/2)+14);
      if(pts&&pts.length){x.lineWidth=1;x.beginPath();x.moveTo(X(lx),Y(ly));for(const [px,py] of pts)x.lineTo(X(px),Y(py));x.stroke();}}
    x.setLineDash([7,7]);x.lineWidth=2;x.strokeRect(X(x1),Y(y1),(x2-x1)*k,(y2-y1)*k);x.setLineDash([]);
    x.fillText(lab,X(x1),Math.max(Y(y1)-4,12));x.globalAlpha=1;}
  const s=d.s;   // 11-16: the gimbal's search while the followed person is lost (bridge --seek 1, src/gimbal/search_follow.py)
  if(s&&(s[0]===2||s[0]===3)){const [ph,cx,cy,r,a,pan,to]=s,X=v=>ox+v*k,Y=v=>oy+v*k,y=Y(cy);   // the span the camera centre sweeps around the predicted place, and where it is heading
    x.strokeStyle=x.fillStyle='rgb(255,220,0)';x.lineWidth=1;x.beginPath();x.moveTo(X(cx-a),y);x.lineTo(X(cx+a),y);x.stroke();
    x.lineWidth=3;x.beginPath();for(const e of [cx-a,cx+a]){x.moveTo(X(e),y-9);x.lineTo(X(e),y+9);}x.stroke();
    if(to!=null){x.beginPath();x.arc(X(cx+to),y,6,0,2*Math.PI);x.fill();}
    x.font='bold 14px sans-serif';x.fillText(ph===2?'찾는 중':'예측 자리로 돌아가는 중',Math.max(4,X(cx-a)),Math.max(16,y-14));}
  x.restore();}
function ovOpen(){if(OV.es||typeof EventSource!=='function')return;
  OV.es=new EventSource('/overlay');
  const connection=OV.es;
  connection.onmessage=e=>{
    if(OV.es!==connection)return;
    try{OV.data=JSON.parse(e.data);}catch(_){return;}
    OV.at=Date.now();
    const data=OV.data;
    const useCenter=overlayCenterEnabled&&data?.pf&&overlayOnMjpeg();
    if(!useCenter||OV.gen!==PS.gen)overlayCenter.reset();
    OV.gen=PS.gen;
    OV.boxes=useCenter?overlayCenter.update(data.b||[],data.t):null;
    ovDraw();
  };
  connection.onerror=()=>{
    if(OV.es!==connection)return;
    overlayResetHistory();
    ovDraw();
  };
}
function overlayResetHistory(){
  OV.boxes=null;
  OV.gen=null;
  overlayCenter.reset();
  overlayGaps.length=0;
}
function ovClose(){if(OV.es){OV.es.close();OV.es=null;}OV.data=null;overlayResetHistory();ovDraw();}

{
  window.overlayStreamReset=()=>{
    overlayResetHistory();
    ovDraw();
  };
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden)ovClose();
    else window.overlayStreamReset();
  });
  addEventListener('pagehide',ovClose);
  // Replay can change without stopping the existing MJPEG connection.
  let replay=document.body.classList.contains('replay-mode');
  new MutationObserver(()=>{
    const next=document.body.classList.contains('replay-mode');
    if(replay!==next){replay=next;window.overlayStreamReset();}
  }).observe(document.body,{attributes:true,attributeFilter:['class']});
}
function ovTick(){if(!document.hidden&&rlast&&rlast.picture_first)ovOpen();else ovClose();ovDraw();}   // rlast: the /rules state, read every second
setInterval(ovTick,1000);addEventListener('resize',ovDraw);
