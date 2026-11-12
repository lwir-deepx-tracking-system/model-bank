// names shown on the nodes (the server's long preset names are not printed)
const OLDSRV='서버를 다시 켜야 열립니다',LK='<svg class="ot-ic" viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="5.5" width="8" height="5.5" rx="1" fill="currentColor"/><path d="M4 5.5V4a2 2 0 0 1 4 0v1.5" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>';
const TN={confirm3:['ByteTrack + 3프레임 확인','기본'],rebind:['ByteTrack + 위치 재결합',''],botsort:['BoT-SORT','시험 안 함'],ocsort:['OC-SORT','시험 안 함'],sort:['SORT','시험 안 함']};
const PN={kalman:['Kalman filter','기본 · 학습 모델 없음'],gru:['GRU_위치기억_(실외)_20261110','우리 카메라 실외 궤적으로 학습 · 1 epoch · 미검증 · 신뢰도 출력 없음'],mamba:['Mamba_위치기억_(실외)_20261110','우리 카메라 실외 궤적으로 학습 · 1 epoch · 미검증 · 신뢰도 출력 없음']};
const DN={'bank:yolo26m_flir_inside_20261110_2020_640x480':'YOLO26m_fine-tuning_(실내)_20261110_2020 640x480','bank:yolo26m_flir_inside_20261110_1742_640x480':'YOLO26m_fine-tuning_(실내)_20261110_1742 640x480','bank:yolo26s_nmsfree_npu_640x640':'YOLO26s_NMS-free 640x640'};   // names shown without status words (user 11-11)
const AN={off:'끔',repvgg:'RepVGG_distillation_(TMOT,CAMEL)_20261103',vit:'ViT-lite_distillation_(TMOT,CAMEL)_202610'};
// folders (user 11-11): one big node per architecture, pressing it shows the models inside
const FD='<svg class="ot-ic" viewBox="0 0 12 12" aria-hidden="true"><path d="M1 3.2a1 1 0 0 1 1-1h2.6l1.2 1.3H10a1 1 0 0 1 1 1V9.3a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1z" fill="currentColor"/></svg>';
const DNOTE={w480:'FLIR 공개 열화상으로 학습 · 카메라 크기 그대로 · 기본',m640:'FLIR 공개 열화상으로 학습 · 정사각형 입력(위아래 빈 띠) · 비교용',s480:'작은 모델 · 카메라 크기 그대로',s6:'작은 모델 · 정사각형 입력(위아래 빈 띠) · 비교용','bank:yolo26m_flir_inside_20261110_2020_640x480':'우리 카메라 실내로 fine-tuning · 2026-11-10 20:20','bank:yolo26m_flir_inside_20261110_1742_640x480':'우리 카메라 실내로 fine-tuning · 2026-11-10 17:42','bank:yolo26s_nmsfree_npu_640x640':'NMS 없이 바로 상자 출력'};
const dfold=s=>/yolox/i.test(s)?'YOLOX':/yolo ?v?8/i.test(s)?'YOLOv8':/yolo ?26/i.test(s)?'YOLO26':'기타';
// second level inside a detector folder: model size (YOLO26m / YOLO26s ...); '' = no size in the name, the node stays directly in the folder
const dsub=s=>{let m=/yolo ?26 ?([nsmlx])(?![a-z])/i.exec(s);if(m)return 'YOLO26'+m[1].toLowerCase();m=/yolox[-_ ]?(nano|tiny|s|m|l|x)(?![a-z])/i.exec(s);if(m)return 'YOLOX-'+m[1].toLowerCase();m=/yolo ?v?8 ?([nsmlx])(?![a-z])/i.exec(s);return m?'YOLOv8'+m[1].toLowerCase():'';};
const AFOLD={repvgg:'RepVGG',vit:'ViT-lite'},PFOLD={kalman:'Kalman filter',gru:'GRU',mamba:'Mamba'};
const otsub={};    // 'row|folder|size' -> opened (true) or closed (false) by the user; not set = open only when it holds the chosen node
const otopen={};   // row index -> folder the user opened ('' = closed by the user); not set = the folder holding the chosen node
const tname=k=>TN[k]?TN[k][0]:(rlast&&rlast.presets||[]).find(p=>String(p.key)===String(k))?.name||(k==='gate'?'없앤 이전 설정 (다른 Tracker를 고르세요)':String(k));
const pend=(r,k)=>(r.pending&&r.pending[k]!=null)?r.pending[k]:r[k];   // a choice waiting for the next frame is shown as chosen
function rdoc(k){if(k===rdk)return;rdk=k;const dl=a=>'<dl class="legend">'+a.map(x=>'<dt>'+esc(x[0])+'</dt><dd>'+esc(x[1])+'</dd>').join('')+'</dl>';
  RD.innerHTML='<h4>'+esc(tname(k))+'</h4>'+(TD[k]?dl(TD[k]):'<dl class="legend"><dt>설명</dt><dd>아직 적어 두지 않았습니다.</dd></dl>')+'<h4>모든 추적기에 공통</h4>'+dl(TDC);}
