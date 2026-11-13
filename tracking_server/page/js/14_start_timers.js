setInterval(rtick,1000);rtick();   // v40: 1 s, so a choice made on another page shows up here within a second
setInterval(tick,1000);tick();
// 11-13: brief view by default; the 설명 보기 button shows the explanations (hints, notes on the choices, legends, old scoring line) and remembers the choice on this device.
const verboseButton=$('verbose-btn');
function verboseSet(on){document.body.classList.toggle('verbose',on);verboseButton.setAttribute('aria-pressed',on?'true':'false');verboseButton.textContent=on?'설명 숨기기':'설명 보기';dispatchEvent(new Event('resize'));}
let verboseSaved=false;try{verboseSaved=localStorage.getItem('lwir_verbose')==='1';}catch(_){}
verboseSet(verboseSaved);
verboseButton.addEventListener('click',()=>{const on=!document.body.classList.contains('verbose');verboseSet(on);try{localStorage.setItem('lwir_verbose',on?'1':'0');}catch(_){}});
