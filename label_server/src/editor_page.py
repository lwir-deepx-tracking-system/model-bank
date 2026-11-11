# -*- coding: utf-8 -*-
"""편집 화면.

label_editor.html 을 읽어, 저장을 내려받기 대신 서버로 보내는 스크립트(INJECT)를 끼워 넣는다.
읽는 파일: 라벨 폴더에 label_editor.html 이 있으면 그것 (보드에서 지금 이렇게 돈다).
          라벨 폴더는 있는데 그 파일만 없으면 서버 폴더의 page/label_editor.html.
끼워 넣을 자리 세 곳이 label_editor.html 에 정확히 한 번씩 있어야 한다. 없으면 서버가 켜지지 않는다.
편집 화면을 열 때마다 파일을 다시 읽으므로, label_editor.html 만 고쳤을 때는 서버를 다시 켤 필요가 없다.
"""
import os

from .state import HERE

INJECT = """
<script>
// ---- web mode (served by label_server.py): save to the server instead of downloading ----
(()=>{const q=new URLSearchParams(location.search),seq=q.get('seq'),user=q.get('user')||'';if(!seq)return;
  let ver=null;const api=(p)=>p+'?seq='+encodeURIComponent(seq)+'&user='+encodeURIComponent(user);
  const ob=$('open');ob.textContent='목록으로';ob.onclick=()=>{location.href='/';};
  document.title=seq+' - LWIR 라벨';
  const tag=document.createElement('span');tag.className='st';tag.textContent=seq+' · '+user;ob.after(tag);
  async function beat(){try{const r=await (await fetch(api('/api/meta'))).json();if(ver===null)ver=r.version;
    const others=r.open_by.filter(x=>x!==user);
    if(others.length)status('주의: '+others.join(', ')+' 님도 이 시퀀스를 열어 두었습니다. 한 사람만 저장하세요.');
    else if(ver!==r.version)status('주의: 다른 곳에서 이 시퀀스가 저장되었습니다. 새로 고친 뒤 작업하세요.');}catch(e){}}
  beat();setInterval(beat,30000);
  let busy=false,autoOff=false;
  window.srvSave=async function(force,auto){if(!nFrames||busy)return;busy=true;if(!auto)autoOff=false;const sig=()=>[...reviewed].sort((a,b)=>a-b).join()+'|'+[...removed].sort((a,b)=>a-b).join();const sent=toMot(),sentR=sig();
    const rv={reviewed:[...reviewed].sort((a,b)=>a-b),removed:[...removed].sort((a,b)=>a-b),frames:nFrames,saved:new Date().toISOString(),saved_by:user,format:'MOT: frame,id,x,y,w,h,conf(manual boxes=1),class(0=person,1=ignore),-1,-1; 1-based frame = images/000001.png'};
    try{const r=await fetch(api('/api/save'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mot:sent,review:rv,base:ver,force:!!force})});
      const j=await r.json();
      if(r.status===409){if(auto){autoOff=true;status('자동 저장 멈춤: 다른 사람('+(j.by||'알 수 없음')+')이 먼저 저장했습니다. 저장 버튼을 눌러 결정하세요.');return;}if(confirm('다른 사람('+(j.by||'알 수 없음')+')이 먼저 저장했습니다. 내 것으로 덮어쓸까요? (이전 파일은 서버에 백업됩니다)')){busy=false;return window.srvSave(true);}status('저장하지 않았습니다.');return;}
      if(!r.ok)throw new Error(j.error||r.status);
      ver=j.version;if(toMot()===sent&&sig()===sentR){dirty=false;$('save').disabled=true;try{localStorage.removeItem('lwir_edit_'+srcName);}catch(e){}}
      status((auto?'자동 저장했습니다 (':'서버에 저장했습니다 (')+new Date().toLocaleTimeString('ko-KR')+', 상자 '+j.boxes+'개).');
    }catch(e){status('저장하지 못했습니다: '+e.message+' — 창을 닫지 말고 다시 저장하세요.');}finally{busy=false;}};
  setInterval(()=>{if(dirty&&!autoOff)window.srvSave(false,true);},20000);
  const pb=$('purge');pb.style.display='';pb.onclick=async()=>{const n=[...removed].filter(f=>!purged.has(f)).length;
    if(!n){status('영상 파일을 지울 삭제 프레임이 없습니다.');return;}
    if(!confirm('삭제한 프레임 '+n+'장의 영상 파일을 지웁니다. 편집 화면에서는 되살릴 수 없습니다. (파일은 서버 휴지통 폴더로 옮겨지고, 남은 프레임 번호는 바뀌지 않습니다) 계속할까요?'))return;
    await window.srvSave();if(dirty){status('먼저 저장이 되어야 합니다. 저장한 뒤 다시 누르세요.');return;}
    try{const r=await fetch(api('/api/purge'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({base:ver})});const j=await r.json();
      if(!r.ok)throw new Error(r.status===409?'다른 곳에서 저장되었습니다. 새로 고친 뒤 다시 하세요':(j.error||r.status));
      alert('영상 파일 '+j.moved+'장을 지웠습니다. 화면을 새로 고칩니다.');location.reload();
    }catch(e){status('영상 파일을 지우지 못했습니다: '+e.message);}};
})();
</script>
"""


def editor_file(root):
    """읽을 편집 화면 파일. 라벨 폴더의 것이 먼저다. 라벨 폴더가 아예 없으면 전과 같이 그 폴더의 파일을 찾다가 실패한다."""
    own = os.path.join(root, 'label_editor.html')
    page = os.path.join(HERE, 'page', 'label_editor.html')
    if not os.path.exists(own) and os.path.isdir(root) and os.path.exists(page):
        return page
    return own


def build_editor(root):
    src = open(editor_file(root), encoding='utf-8').read()
    a = 'async function save(){'
    b = "srcName='http';"
    if src.count(a) != 1 or src.count(b) != 1 or src.count('</body>') != 1:
        raise RuntimeError('label_editor.html changed: patch points not found')
    src = src.replace(a, a + 'if(window.srvSave)return window.srvSave();')
    src = src.replace(b, "srcName='http:'+base;")
    return src.replace('</body>', INJECT + '</body>')
