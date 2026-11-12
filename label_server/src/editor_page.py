# -*- coding: utf-8 -*-
"""Bundled editor, versioned snapshot, and person-part save integration.
Old label-root HTML is preserved but no longer shadows repository updates.
LWIR_LABEL_EDITOR_HTML explicitly selects a compatible custom editor.
"""
import os
from .state import HERE

INJECT = """
<script>
(()=>{const q=new URLSearchParams(location.search),seq=q.get('seq'),user=q.get('user')||'';if(!seq)return;
  const api=p=>p+'?seq='+encodeURIComponent(seq)+'&user='+encodeURIComponent(user);
  const ob=$('open');ob.textContent='목록으로';ob.onclick=()=>{location.href='/';};
  document.title=seq+' - LWIR 라벨';
  const tag=document.createElement('span');tag.className='st';tag.textContent=seq+' · '+user;ob.after(tag);
  async function beat(){if(!window.LWIR_LOADED)return;try{const r=await(await fetch(api('/api/meta'))).json();
    const others=(r.open_by||[]).filter(x=>x!==user);
    if(others.length)status('주의: '+others.join(', ')+' 님도 이 시퀀스를 열어 두었습니다. 한 사람만 저장하세요.');
    else if(window.LWIR_BASE!==r.version)status('다른 곳에서 라벨이 저장되었습니다. 작업을 내려받아 보관한 뒤 최신 라벨을 다시 여세요.');}catch(e){}}
  setTimeout(beat,1000);setInterval(beat,30000);
  let busy=false,autoOff=false;
  window.srvSave=async function(force,auto){
    if(!nFrames||busy||!window.LWIR_LOADED||window.LWIR_LOAD_FAILED)return;
    if(window.LWIR_DRAFT_CONFLICT){status('이전 버전의 초안입니다. 작업 내려받기로 보관한 뒤 최신 라벨을 다시 여세요.');return;}
    busy=true;if(!auto)autoOff=false;
    try{
      const sent=toMot(),rv=reviewDocument(),parts=partsDocument();rv.saved_by=user;
      const sig=()=>JSON.stringify([[...reviewed].sort((a,b)=>a-b),[...removed].sort((a,b)=>a-b),partsDocument()]);
      const sentR=sig();
      const r=await fetch(api('/api/save'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mot:sent,review:rv,parts,base:window.LWIR_BASE})});
      const j=await r.json();
      if(r.status===409||r.status===428){autoOff=true;status((r.status===409?'다른 곳에서 먼저 저장했습니다. ':'편집기를 업데이트해야 합니다. ')+'작업 내려받기로 현재 작업을 보관한 뒤 새로 고침하세요. 기존 라벨은 덮어쓰지 않았습니다.');return;}
      if(!r.ok)throw Error(j.error||r.status);
      window.LWIR_BASE=j.version;
      if(toMot()===sent&&sig()===sentR){dirty=false;$('save').disabled=true;try{localStorage.removeItem('lwir_edit_'+srcName);}catch(e){}}
      else persistDraft();
      status((auto?'자동 저장했습니다 (':'서버에 저장했습니다 (')+new Date().toLocaleTimeString('ko-KR')+', 사람 상자 '+j.boxes+'개 · 부위 기록 '+parts.annotations.length+'개).');
    }catch(e){status('저장하지 못했습니다: '+e.message+' — 창을 닫지 말고 다시 저장하세요.');}finally{busy=false;}
  };
  setInterval(()=>{if(dirty&&!autoOff)window.srvSave(false,true);},20000);
  const pb=$('purge');pb.style.display='';pb.onclick=async()=>{
    const n=[...removed].filter(f=>!purged.has(f)).length;
    if(!n){status('영상 파일을 지울 삭제 프레임이 없습니다.');return;}
    if(!confirm('삭제한 '+n+'장의 영상 파일을 휴지통으로 옮깁니다. 연결된 부위도 백업 후 제거합니다. 계속할까요?'))return;
    await window.srvSave();if(dirty)return;
    try{const r=await fetch(api('/api/purge'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({base:window.LWIR_BASE})});const j=await r.json();
      if(!r.ok)throw Error(r.status===409?'다른 곳에서 저장되었습니다. 최신 라벨을 다시 여세요.':(j.error||r.status));
      alert('영상 파일 '+j.moved+'장을 옮겼습니다. 화면을 새로 고칩니다.');location.reload();
    }catch(e){status('파일을 옮기지 못했습니다: '+e.message);}
  };
})();
</script>
"""

def editor_file(root):
    if not os.path.isdir(root):
        raise FileNotFoundError('라벨 폴더가 없습니다.')
    return os.environ.get('LWIR_LABEL_EDITOR_HTML') or os.path.join(HERE, 'page', 'label_editor.html')

def build_editor(root):
    with open(editor_file(root), encoding='utf-8') as f:
        src = f.read()
    a = 'async function save(){'
    b = "srcName='http';"
    if (src.count(a) != 1 or src.count(b) != 1 or src.count('</body>') != 1
            or 'function partsDocument()' not in src):
        raise RuntimeError('label_editor.html: 신체 부위 지원 편집기(schema 2)가 필요합니다.')
    src = src.replace(a, a + 'if(window.srvSave)return window.srvSave();')
    src = src.replace(b, "srcName='http:'+base;")
    src = src.replace('<head>', '<head><script>window.LWIR_SERVER_MODE=true;</script>', 1)
    return src.replace('</body>', INJECT + '</body>')
