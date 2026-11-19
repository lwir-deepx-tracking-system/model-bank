setInterval(rtick,1000);rtick();   // v40: 1 s, so a choice made on another page shows up here within a second
setInterval(tick,1000);tick();
// 11-13: brief view by default; the 설명 보기 button shows the explanations (hints, notes on the choices, legends, old scoring line) and remembers the choice on this device.
const verboseButton=$('verbose-btn');
function verboseSet(on){document.body.classList.toggle('verbose',on);verboseButton.setAttribute('aria-pressed',on?'true':'false');verboseButton.textContent='?';verboseButton.title=on?'설명 숨기기':'설명 보기';dispatchEvent(new Event('resize'));}
let verboseSaved=false;try{verboseSaved=localStorage.getItem('lwir_verbose')==='1';}catch(_){}
verboseSet(verboseSaved);
verboseButton.addEventListener('click',()=>{const on=!document.body.classList.contains('verbose');verboseSet(on);try{localStorage.setItem('lwir_verbose',on?'1':'0');}catch(_){}});

async function showLanWebLink(){
  const link=$('lan-web');
  try{
    const response=await fetch('/network',{cache:'no-store'});
    if(!response.ok)return;
    const {lan_url}=await response.json();
    if(!lan_url)return;
    const target=new URL(lan_url);
    if(target.protocol!=='http:'||target.username||target.password||target.pathname!=='/'||target.search||target.hash)return;
    if(target.origin===location.origin){
      link.textContent='LAN 연결';
      link.title='직결 LAN 주소로 연결됨';
    }else{
      link.href=target.href;
    }
    link.hidden=false;
  }catch(_){}
}
showLanWebLink();
