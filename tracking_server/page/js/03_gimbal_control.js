function post(u,ka){return fetch(u,{method:'POST',headers:{'X-Gimbal':'1'},cache:'no-store',keepalive:!!ka}).then(r=>r.json()).then(show).catch(()=>{S.textContent='명령이 전달되지 않았습니다';netChip();});}
v.addEventListener('click',e=>{if(Date.now()<joyClickBlock)return;const r=v.getBoundingClientRect();   // v40: the click that ends a joystick drag picks nobody
  // the image keeps 4:3 inside its box (object-fit: contain): map the click into the picture itself
  const ar=4/3,bw=r.width,bh=r.height;let w=bw,h=bw/ar;if(h>bh){h=bh;w=bh*ar;}
  const x=(e.clientX-r.left-(bw-w)/2)/w,y=(e.clientY-r.top-(bh-h)/2)/h;if(x<0||x>1||y<0||y>1)return;
  post('/gimbal/click?x='+x+'&y='+y);});
const stopAll=()=>post('/gimbal/stop');
// v40: no stop key on the page: a tap / click on an empty place of the picture stops (the server does it), and so does the space bar
for(const el of [v]){el.addEventListener('contextmenu',e=>e.preventDefault());el.addEventListener('dragstart',e=>e.preventDefault());}
addEventListener('dragstart',e=>{if(e.target&&e.target.tagName==='IMG')e.preventDefault();});
const K={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,1],ArrowDown:[0,-1],KeyA:[-1,0],KeyD:[1,0],KeyW:[0,1],KeyS:[0,-1]};   // v34: W A S D move like the arrow keys (e.code, so it also works with the Korean keyboard on)
let held=null;
const typing=e=>{const t=e.target;return t&&(t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.tagName==='SELECT');};   // keys typed into the boxes never move the gimbal
addEventListener('keydown',e=>{if(typing(e))return;if(e.code==='Space'){e.preventDefault();held=null;post('/gimbal/stop');return;}const m=K[e.code];if(!m)return;if(e.ctrlKey||e.metaKey||e.altKey)return;e.preventDefault();if(e.repeat||held===e.code)return;held=e.code;post('/gimbal/move?x='+m[0]+'&y='+m[1]);});
addEventListener('keyup',e=>{if(typing(e))return;if(e.code===held){held=null;post('/gimbal/move?x=0&y=0');}});
let padDown=false,padBtn=null;
function release(){let sent=false;if(held){held=null;sent=true;}if(padDown){padDown=false;if(padBtn)padBtn.classList.remove('on');padBtn=null;sent=true;}if(joyEnd())sent=true;if(sent)post('/gimbal/move?x=0&y=0',true);}   // keepalive: the stop still leaves when the page is being hidden
addEventListener('blur',release);
document.addEventListener('visibilitychange',()=>{if(document.hidden)release();});addEventListener('pagehide',release);   // v25: screen off / app switched while a key is held: stop
for(const b of document.querySelectorAll('#pad button[data-x]')){
  const go=e=>{e.preventDefault();if(padDown)return;padDown=true;padBtn=b;b.classList.add('on');try{b.setPointerCapture&&b.setPointerCapture(e.pointerId);}catch(_){}try{if(e.pointerType==='touch'&&navigator.vibrate)navigator.vibrate(8);}catch(_){}post('/gimbal/move?x='+b.dataset.x+'&y='+b.dataset.y);};
  const stop=e=>{if(!padDown)return;padDown=false;padBtn=null;b.classList.remove('on');post('/gimbal/move?x=0&y=0');};
  b.addEventListener('pointerdown',go);for(const n of ['pointerup','pointercancel','pointerleave','lostpointercapture'])b.addEventListener(n,stop);b.addEventListener('contextmenu',e=>e.preventDefault());}
// v40: phone joystick. A finger (or pen) pressed on the picture and dragged = virtual joystick; a mouse keeps click-to-pick. A short tap is still a click (pick a person).
// 8 directions, same command as the arrow keys (/gimbal/move, -1 / 0 / 1 per axis); nothing is sent until the finger has left the dead zone, and lifting the finger sends the stop.
const JOYB=$('joyb'),JOYK=$('joyk'),JOY={id:null,x0:0,y0:0,on:false,dx:0,dy:0,sent:false};let joyClickBlock=0;
const JOY_START=14,JOY_DEAD=18,JOY_R=56,JOY_T=[[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]];
function joyDir(ax,ay){if(Math.hypot(ax,ay)<JOY_DEAD)return [0,0];return JOY_T[((Math.round(Math.atan2(-ay,ax)/(Math.PI/4))%8)+8)%8];}   // screen y grows downwards; the gimbal's y = 1 is up
function joyEnd(){if(JOY.id==null)return false;const was=JOY.on,sent=JOY.sent;try{v.releasePointerCapture(JOY.id);}catch(_){}JOY.id=null;JOY.on=false;JOY.sent=false;JOY.dx=JOY.dy=0;JOYB.style.display='none';JOYK.style.transform='';if(was)joyClickBlock=Date.now()+400;return sent;}   // true = a move was sent, so the caller must send the stop
v.addEventListener('pointerdown',e=>{if(e.pointerType==='mouse'||JOY.id!=null)return;JOY.id=e.pointerId;JOY.x0=e.clientX;JOY.y0=e.clientY;JOY.on=false;JOY.sent=false;JOY.dx=JOY.dy=0;try{v.setPointerCapture(e.pointerId);}catch(_){}});
v.addEventListener('pointermove',e=>{if(e.pointerId!==JOY.id)return;const ax=e.clientX-JOY.x0,ay=e.clientY-JOY.y0,d=Math.hypot(ax,ay);
  if(!JOY.on){if(d<JOY_START)return;JOY.on=true;const r=v.parentNode.getBoundingClientRect();JOYB.style.left=(JOY.x0-r.left)+'px';JOYB.style.top=(JOY.y0-r.top)+'px';JOYB.style.display='block';try{if(navigator.vibrate)navigator.vibrate(8);}catch(_){}}
  const k=d>JOY_R?JOY_R/d:1;JOYK.style.transform='translate('+(ax*k).toFixed(0)+'px,'+(ay*k).toFixed(0)+'px)';
  const m=joyDir(ax,ay);if(m[0]===JOY.dx&&m[1]===JOY.dy)return;JOY.dx=m[0];JOY.dy=m[1];if(m[0]||m[1])JOY.sent=true;if(JOY.sent)post('/gimbal/move?x='+m[0]+'&y='+m[1]);});
for(const n of ['pointerup','pointercancel','lostpointercapture'])v.addEventListener(n,e=>{if(e.pointerId!==JOY.id)return;if(joyEnd())post('/gimbal/move?x=0&y=0',true);});
