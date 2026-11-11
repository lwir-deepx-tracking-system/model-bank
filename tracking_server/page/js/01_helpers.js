
const $=id=>document.getElementById(id);
const v=$('v'),S=$('s');
function chip(d,t,cls,txt){$(d).className='dot '+cls;$(t).textContent=txt;}
// section fold state (v25): desktop open, phone closed, remembered per browser
const MOB=matchMedia('(max-width:900px)');
let openSaved={};try{openSaved=JSON.parse(localStorage.getItem('lwir_live_open')||'{}')||{};}catch(_){openSaved={};}
for(const d of document.querySelectorAll('details.sec')){const k=d.dataset.k;d.open=(k in openSaved)?!!openSaved[k]:!MOB.matches;
  d.querySelector('summary').addEventListener('click',()=>setTimeout(()=>{openSaved[k]=d.open;try{localStorage.setItem('lwir_live_open',JSON.stringify(openSaved));}catch(_){}},0));}   // saved only on the user's click, never the default
