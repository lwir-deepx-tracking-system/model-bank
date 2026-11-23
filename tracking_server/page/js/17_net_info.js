// 11-22: offline note and PC connect download. GET /netinfo -> {mode: 'online' | 'lan-only' | null, client_local, pc_download}
(function(){
  const modeChip=$('net-mode'),pcLink=$('pc-connect');
  if(!modeChip||!pcLink)return;
  let busy=false;
  async function netInfo(){
    if(document.hidden||busy)return;
    busy=true;
    const controller=new AbortController();
    const limit=setTimeout(()=>controller.abort(),2500);
    try{
      const response=await fetch('/netinfo',{cache:'no-store',signal:controller.signal});
      if(!response.ok)return;
      const info=await response.json();
      modeChip.hidden=info.mode!=='lan-only';
      pcLink.hidden=!info.pc_download;
    }catch(_){}finally{
      clearTimeout(limit);
      busy=false;
    }
  }
  setInterval(netInfo,15000);
  netInfo();
})();
