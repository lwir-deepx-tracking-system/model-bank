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
  x.font='12px sans-serif';
  for(const [x1,y1,x2,y2,id,lab,last,pts,rad,col] of d.p||[]){const X=v=>ox+v*k,Y=v=>oy+v*k;   // predicted boxes of lost people (pred_show), as the server drew them
    x.strokeStyle=x.fillStyle=col;
    if(rad>0){x.lineWidth=1;x.beginPath();x.arc(X((x1+x2)/2),Y((y1+y2)/2),rad*k,0,2*Math.PI);x.stroke();}
    if(last){const [lx,ly,lw,lh]=last;x.setLineDash([3,3]);x.lineWidth=1;x.strokeRect(X(lx-lw/2),Y(ly-lh/2),lw*k,lh*k);x.setLineDash([]);
      x.lineWidth=2;x.beginPath();x.arc(X(lx),Y(ly),4,0,2*Math.PI);x.stroke();x.fillText('last seen',X(lx-lw/2),Y(ly+lh/2)+14);
      if(pts&&pts.length){x.lineWidth=1;x.beginPath();x.moveTo(X(lx),Y(ly));for(const [px,py] of pts)x.lineTo(X(px),Y(py));x.stroke();}}
    x.setLineDash([7,7]);x.lineWidth=2;x.strokeRect(X(x1),Y(y1),(x2-x1)*k,(y2-y1)*k);x.setLineDash([]);
    x.fillText(lab,X(x1),Math.max(Y(y1)-4,12));}
  x.restore();}
function ovOpen(){if(OV.es||typeof EventSource!=='function')return;
  OV.es=new EventSource('/overlay');
  OV.es.onmessage=e=>{try{OV.data=JSON.parse(e.data);OV.at=Date.now();}catch(_){return;}ovDraw();};}
function ovClose(){if(OV.es){OV.es.close();OV.es=null;}OV.data=null;ovDraw();}
function ovTick(){if(!document.hidden&&rlast&&rlast.picture_first)ovOpen();else ovClose();ovDraw();}   // rlast: the /rules state, read every second
setInterval(ovTick,1000);addEventListener('resize',ovDraw);
