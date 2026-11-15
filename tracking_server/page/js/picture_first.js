// 11-15 (user: total delay about one camera frame): with '사진 먼저' the server sends the camera picture as soon as the frame
// arrives (before detection and tracking) and the boxes separately (GET /overlay, server-sent events, src/web/picture_first.py).
// They are drawn here on a canvas over the picture, with the same 4:3 fit as the click mapping. The boxes come about two
// camera frames after the picture; boxes older than 1 s are not drawn. A click still picks from the boxes drawn last.
const OV={es:null,data:null,at:0,cv:$('vdraw')};
function ovColor(id){   // the server's track_color(): golden-angle hue, saturation 0.72, value 1
  const h=(id*0.61803398875)%1*6,i=Math.floor(h),f=h-i,s=0.72,p=1-s,q=1-s*f,t=1-s*(1-f);
  const [r,g,b]=[[1,t,p],[q,1,p],[p,1,t],[p,q,1],[t,p,1],[1,p,q]][i%6];
  return 'rgb('+Math.round(r*255)+','+Math.round(g*255)+','+Math.round(b*255)+')';}
function ovDraw(){const cv=OV.cv;if(!cv)return;
  const dpr=window.devicePixelRatio||1,bw=cv.clientWidth,bh=cv.clientHeight,W=Math.round(bw*dpr),H=Math.round(bh*dpr);
  if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H;}
  const x=cv.getContext('2d');x.clearRect(0,0,W,H);
  const d=OV.data;if(!d||!d.pf||Date.now()-OV.at>1000||document.body.classList.contains('replay-mode'))return;
  let w=bw,h=bw*3/4;if(h>bh){h=bh;w=bh*4/3;}   // the picture inside its box (object-fit: contain)
  const ox=(bw-w)/2,oy=(bh-h)/2,k=w/(d.w||640);
  x.save();x.scale(dpr,dpr);x.font='bold 13px sans-serif';x.textBaseline='bottom';
  for(const [x1,y1,x2,y2,id,fol] of d.b||[]){const L=ox+x1*k,T=oy+y1*k,R=ox+x2*k,B=oy+y2*k;
    if(fol){x.strokeStyle='rgb(255,220,0)';x.lineWidth=4;x.strokeRect(L,T,R-L,B-T);
      const lab='FOLLOW '+id,tw=x.measureText(lab).width,ty=Math.max(T,18);
      x.fillStyle='rgb(255,220,0)';x.fillRect(L-2,ty-18,tw+10,18);x.fillStyle='#141414';x.fillText(lab,L+3,ty-3);}
    else{const c=ovColor(id);x.strokeStyle=c;x.lineWidth=2;x.strokeRect(L,T,R-L,B-T);x.fillStyle=c;x.fillText(String(id),L,Math.max(T-3,14));}}
  x.restore();}
function ovOpen(){if(OV.es||typeof EventSource!=='function')return;
  OV.es=new EventSource('/overlay');
  OV.es.onmessage=e=>{try{OV.data=JSON.parse(e.data);OV.at=Date.now();}catch(_){return;}ovDraw();};}
function ovClose(){if(OV.es){OV.es.close();OV.es=null;}OV.data=null;ovDraw();}
function ovTick(){if(!document.hidden&&rlast&&rlast.picture_first)ovOpen();else ovClose();ovDraw();}   // rlast: the /rules state, read every second
setInterval(ovTick,1000);addEventListener('resize',ovDraw);
