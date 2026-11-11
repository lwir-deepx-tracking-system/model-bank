# -*- coding: utf-8 -*-
"""첫 화면(시퀀스 목록)의 글.

INDEX 는 브라우저에 그대로 보내는 HTML 한 덩어리다. 한 글자라도 바꾸면 화면이 달라진다.
"""

INDEX = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LWIR 라벨 작업</title><style>
body{font:15px/1.5 system-ui,'Malgun Gothic',sans-serif;background:#1c1d1f;color:#e6e6e6;margin:0;padding:20px 16px}
h1{font-size:19px;margin:0 0 4px}p{color:#a9abae;margin:4px 0 14px}
input{background:#2a2b2e;border:1px solid #45474b;color:#e6e6e6;padding:6px 8px;border-radius:4px;font:inherit;width:160px}
table{border-collapse:collapse;width:100%;max-width:1300px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #34363a}
th{color:#a9abae;font-weight:500;font-size:13px}a{color:#f0913a;text-decoration:none}a:hover{text-decoration:underline}
button{background:#2a2b2e;border:1px solid #45474b;color:#e6e6e6;padding:3px 9px;border-radius:4px;font:inherit;font-size:13px;cursor:pointer;margin-right:6px;white-space:nowrap}button:hover{border-color:#f0913a}
.w{overflow-x:auto}.busy{color:#f0913a}.ok{color:#5fbf77;font-weight:600}.dim{color:#8b8d91}td.n{font-variant-numeric:tabular-nums}
</style></head><body>
<h1>LWIR 라벨 작업</h1>
<p>이름을 적고 시퀀스를 고르세요. 한 시퀀스는 한 사람만 작업합니다. 저장은 편집 화면의 "저장" 버튼 또는 Ctrl+S 입니다.</p>
<p>이름 <input id="u" placeholder="예: 홍길동" maxlength="20"></p>
<div class="w"><table><thead><tr><th>시퀀스</th><th>프레임</th><th>확인한 프레임</th><th>삭제</th><th>검수</th><th>확인자</th><th>마지막 저장</th><th>지금 열어 둔 사람</th><th>관리</th></tr></thead><tbody id="b"></tbody></table></div>
<script>
const u=document.getElementById('u');try{u.value=localStorage.getItem('lwir_user')||'';}catch(e){}
u.oninput=()=>{try{localStorage.setItem('lwir_user',u.value.trim());}catch(e){}};
function go(seq){const n=u.value.trim();if(!n){alert('이름을 먼저 적어 주세요.');u.focus();return false;}
  location.href='/edit?src='+encodeURIComponent('/data/'+seq+'/')+'&seq='+encodeURIComponent(seq)+'&user='+encodeURIComponent(n);return false;}
async function adm(path,seq,body){const n=u.value.trim();if(!n){alert('이름을 먼저 적어 주세요.');u.focus();return;}
  try{const r=await fetch(path+'?seq='+encodeURIComponent(seq)+'&user='+encodeURIComponent(n),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
    const j=await r.json();if(!r.ok)alert(j.error||'하지 못했습니다.');}catch(e){alert('서버에 닿지 못했습니다.');}
  load();}
function btn(t,f){const x=document.createElement('button');x.textContent=t;x.onclick=f;return x;}
async function load(){const r=await (await fetch('/api/list')).json();const b=document.getElementById('b');b.textContent='';
  for(const s of r){const tr=document.createElement('tr');
    const a=document.createElement('a');a.href='#';a.textContent=s.seq;a.onclick=()=>go(s.seq);
    const td=(x,c)=>{const t=document.createElement('td');if(c)t.className=c;if(x instanceof Node)t.appendChild(x);else t.textContent=x;return t;};
    const m=document.createElement('span');
    m.append(btn('이름 변경',()=>{const cur=s.seq.split('/').pop();const nn=prompt('새 이름을 적으세요. (한글, 영문, 숫자, 빈칸, - _ . 만 가능)',cur);if(nn&&nn.trim()&&nn.trim()!==cur)adm('/api/seq_rename',s.seq,{name:nn.trim()});}),
      btn('삭제',()=>{if(confirm('"'+s.seq+'" 을(를) 휴지통으로 보냅니다. 목록에서 사라지고, 파일은 서버 휴지통에 남아 있어 되돌릴 수 있습니다.'))adm('/api/seq_trash',s.seq);}));
    const saved=s.saved?(new Date(s.saved).toLocaleString('ko-KR')+(s.by?' ('+s.by+')':'')):'없음';
    tr.append(td(a),td(s.kept,'n'),td(s.rev_kept+' / '+s.kept,'n'),td(s.removed?s.removed+'장'+(s.purged?' (영상 지움 '+s.purged+')':''):'-',s.removed?'n':'dim'),td(s.reviewed>=s.frames?'완료':(s.reviewed?'진행 중':'-'),s.reviewed>=s.frames?'ok':'dim'),td(s.done_by||'-',s.done_by?'':'dim'),td(saved,s.saved?'':'dim'),td(s.open_by.join(', ')||'-',s.open_by.length?'busy':'dim'),td(m));b.appendChild(tr);}}
load();setInterval(load,15000);
</script></body></html>"""
