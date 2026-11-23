// 17_voice.js — 추적 상태 음성 안내 (브라우저 내장 TTS · 서버 변경 없음)
//
// 짐벌 상태는 02_connection_status.js 의 show(t) 가 받는 값을 그대로 쓴다 (/gimbal/click · /gimbal/ping 등의 응답).
//   t.running · t.error · t.auto · t.id · t.lost   → show 와 같은 규칙으로 "따라가는 ID" = running && !error && auto ? id : 0
//
//   따라가는 ID가 새로 생김 (화면 클릭)        → "<ID> 선택"
//   따라가는 중 t.lost 가 켜짐                  → "트랙 끊김"
//   같은 ID의 t.lost 가 꺼짐 (다시 찾음)         → "트랙 재연결"
//   homeGimbal() (스페이스바 · 홈 버튼)          → "초기화"
//   따라가는 중 방향키 · W A S D                 → "수동조작 모드"
//
// 끄기/켜기: 브라우저 콘솔에서 VOICE.toggle()  (이 브라우저에 기억됨)
{
  const V = window.VOICE = { on: true, follow: 0, lost: false, lostId: 0, lostAt: 0, manual: false, last: '', lastAt: 0, voice: null };
  try { const s = localStorage.getItem('lwir_voice'); if (s !== null) V.on = (s === '1'); } catch (_) {}

  const SS = window.speechSynthesis;
  const pickVoice = () => { V.voice = (SS ? SS.getVoices() : []).find(v => /^ko/i.test(v.lang)) || null; };
  if (SS) { pickVoice(); SS.addEventListener?.('voiceschanged', pickVoice); }

  V.say = text => {
    if (!V.on || !SS) return;
    const now = Date.now();
    if (text === V.last && now - V.lastAt < 1500) return;   // 같은 말을 잇달아 하지 않음 (응답이 연달아 올 때)
    V.last = text; V.lastAt = now;
    SS.cancel();                                            // 밀린 안내는 버리고 최신 상태만
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'ko-KR'; if (V.voice) u.voice = V.voice; u.rate = 1.15;
    SS.speak(u);
  };
  V.toggle = () => {
    V.on = !V.on;
    try { localStorage.setItem('lwir_voice', V.on ? '1' : '0'); } catch (_) {}
    if (!V.on && SS) SS.cancel();
    return V.on;
  };

  // 짐벌 상태가 올 때마다 (show 앞에서 듣기만 함)
  V.state = t => {
    if (!t || typeof t !== 'object' || !('running' in t)) return;
    const id = t.running && !t.error && t.auto ? (t.id || 0) : 0;
    if (id !== V.follow) {
      if (id) {
        // 놓쳤던 같은 ID를 서버가 스스로 다시 따라가기 시작한 경우 (조작 없이 1분 안): 재연결
        if (V.lost && id === V.lostId && Date.now() - V.lostAt < 60000) V.say('트랙 재연결');
        else V.say(id + ' 선택');
        V.manual = false;
        V.lost = !!t.lost;
        if (V.lost) { V.lostId = id; V.lostAt = Date.now(); }
      }
      V.follow = id;              // 0 이 되면(수동 · 정지 · 초기화) 다음 선택을 새로 알린다. lost 표시는 남겨 둠
      if (!id && V.manual) V.lost = false;
      return;
    }
    if (!id) return;
    if (t.lost && !V.lost) { V.lost = true; V.lostId = id; V.lostAt = Date.now(); V.say('트랙 끊김'); }
    else if (!t.lost && V.lost) { V.lost = false; V.say('트랙 재연결'); }
  };

  if (typeof show === 'function') {
    const show0 = show;
    show = function (t) { try { V.state(t); } catch (_) {} return show0.apply(this, arguments); };
  }
  if (typeof homeGimbal === 'function') {
    const home0 = homeGimbal;
    homeGimbal = function () { V.manual = false; V.lost = false; V.say('초기화'); return home0.apply(this, arguments); };
  }

  // 03_gimbal_control.js 와 같은 조건: e.code, 방향키 + W A S D, 입력칸에서 친 키와 Ctrl·Alt 조합은 제외
  const MOVE = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'KeyW', 'KeyA', 'KeyS', 'KeyD']);
  addEventListener('keydown', e => {
    if (e.repeat || !MOVE.has(e.code) || e.ctrlKey || e.metaKey || e.altKey) return;
    if (typeof typing === 'function' && typing(e)) return;
    if (V.follow && !V.manual) { V.manual = true; V.lost = false; V.say('수동조작 모드'); }
  }, true);   // 03 의 처리보다 먼저 듣기만 하고 막지 않는다 (그쪽이 auto 를 끄기 전에 '따라가는 중'인지 본다)
}
