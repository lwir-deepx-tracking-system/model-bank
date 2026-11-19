setInterval(rtick,1000);rtick();   // v40: 1 s, so a choice made on another page shows up here within a second
setInterval(tick,1000);tick();
// 11-13: brief view by default; the 설명 보기 button shows the explanations (hints, notes on the choices, legends, old scoring line) and remembers the choice on this device.
const verboseButton=$('verbose-btn');
function verboseSet(on){document.body.classList.toggle('verbose',on);verboseButton.setAttribute('aria-pressed',on?'true':'false');verboseButton.textContent='?';verboseButton.title=on?'설명 숨기기':'설명 보기';dispatchEvent(new Event('resize'));}
let verboseSaved=false;try{verboseSaved=localStorage.getItem('lwir_verbose')==='1';}catch(_){}
verboseSet(verboseSaved);
verboseButton.addEventListener('click',()=>{const on=!document.body.classList.contains('verbose');verboseSet(on);try{localStorage.setItem('lwir_verbose',on?'1':'0');}catch(_){}});

const lanLink=$('lan-web');
const lanRetryKey='lwir_lan_redirect_at';
let lanTarget=null,lanBusy=false,lanSwitched=false,lanLocal=false;
let lanFailures=0,lanReachable=Date.now(),lanDiscoverAt=0;

function lanAddress(value){
  if(!value)return null;
  let target;
  try{target=new URL(value);}catch(_){return null;}
  const [a,b,c]=target.hostname.split('.').map(Number);
  const ipv4=/^(\d{1,3}\.){3}\d{1,3}$/.test(target.hostname);
  const local=a===10||(a===172&&b>=16&&b<=31)||(a===192&&b===168)||
    (a===169&&b===254&&c>=1&&c<=254);
  if(!ipv4||!local||target.protocol!=='http:'||target.username||target.password||
      target.pathname!=='/'||target.search||target.hash)return null;
  return target;
}

async function checkLanNetwork(){
  if(document.hidden||lanBusy||lanLocal||lanSwitched)return;
  const now=Date.now();
  const idle=()=>Date.now()-Math.max(lastOk,grace,lanReachable);
  const discover=!lanTarget;
  if(discover&&now<lanDiscoverAt)return;
  if(!discover&&idle()<5000){lanFailures=0;return;}
  lanDiscoverAt=now+10000;
  lanBusy=true;
  const controller=new AbortController();
  const limit=setTimeout(()=>controller.abort(),1800);
  try{
    const response=await fetch('/network',{cache:'no-store',signal:controller.signal});
    lanReachable=Date.now();
    lanFailures=0;
    if(!response.ok)return;
    const {lan_url}=await response.json();
    lanTarget=lanAddress(lan_url);
    lanLink.hidden=!lanTarget;
    if(!lanTarget)return;
    lanLocal=lanTarget.origin===location.origin;
    if(lanLocal){
      lanLink.textContent='LAN 연결';
      lanLink.title='직결 LAN 주소로 연결됨';
    }else{
      lanLink.href=lanTarget.href;
    }
  }catch(_){
    if(discover||document.hidden||idle()<5000){lanFailures=0;return;}
    if(++lanFailures<2)return;
    // Top-level navigation also works when the browser blocks cross-origin LAN probes.
    try{
      const previous=Number(sessionStorage.getItem(lanRetryKey));
      if(previous>0&&Date.now()-previous<60000)return;
      sessionStorage.setItem(lanRetryKey,String(Date.now()));
    }catch(_){}
    lanSwitched=true;
    lanLink.textContent='LAN으로 다시 연결 중';
    location.assign(lanTarget.href);
  }finally{
    clearTimeout(limit);
    lanBusy=false;
  }
}
setInterval(checkLanNetwork,2000);
checkLanNetwork();
