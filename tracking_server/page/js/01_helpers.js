
const $=id=>document.getElementById(id);
const v=$('v'),S=$('s');
function chip(d,t,cls,txt){$(d).className='dot '+cls;$(t).textContent=txt;
  if(d==='cdot')$(t).parentElement.hidden=cls==='good';   // 11-13 (design review): the connection chip only when something is wrong
  if(d==='mdot')$(t).parentElement.hidden=txt==='측정 대기';}   // 11-13: the measurement chip only while measuring or with a result
// section fold state (v25): desktop open, phone closed, remembered per browser
const MOB=matchMedia('(max-width:900px)');
let openSaved={};try{openSaved=JSON.parse(localStorage.getItem('lwir_live_open')||'{}')||{};}catch(_){openSaved={};}
for(const d of document.querySelectorAll('details.sec')){const k=d.dataset.k;d.open=(k in openSaved)?!!openSaved[k]:(k==='adv'?false:!MOB.matches);   // 11-13: 'advanced' starts closed
 
  d.querySelector('summary').addEventListener('click',()=>setTimeout(()=>{openSaved[k]=d.open;try{localStorage.setItem('lwir_live_open',JSON.stringify(openSaved));}catch(_){}},0));}   // saved only on the user's click, never the default
