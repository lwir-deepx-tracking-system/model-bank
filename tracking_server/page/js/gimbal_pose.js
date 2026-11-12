// The existing heartbeat carries actual angle data when the bridge supplies it.
const poseStatus = $('pose-status');
let poseReceivedAt = 0;
const poseAxes = [['pan', '좌우'], ['tilt', '상하']].map(([key, label]) => {
  const axis = $('pose-' + key);
  return {key, label, axis, value: axis.querySelector('[data-value]'),
    scale: axis.querySelector('[data-scale]'), low: axis.querySelector('[data-low]'),
    high: axis.querySelector('[data-high]'), marker: axis.querySelector('[data-marker]')};
});

function poseDegrees(deg) {
  return (deg > 0 ? '+' : '') + deg.toFixed(1).replace(/\.0$/, '') + '°';
}

function renderGimbalPose(pose, estimate) {
  let any = false;
  const estimated = !pose && !!estimate;
  const format = estimated ? value => Math.round(value) + '%' : poseDegrees;
  for (const {key, label, axis, value, scale, low, high, marker} of poseAxes) {
    const data = estimated ? {deg: estimate[key], min_deg: 0, max_deg: 100} : pose && pose[key];
    const valid = !!data && Number.isFinite(data.deg);
    axis.hidden = !valid;
    if (!valid) continue;
    any = true;
    const hasRange = Number.isFinite(data.min_deg) && Number.isFinite(data.max_deg) && data.max_deg > data.min_deg;
    const inRange = hasRange && data.deg >= data.min_deg && data.deg <= data.max_deg;
    value.textContent = label + (estimated ? ' 추정 ' : ' ') + format(data.deg) + (hasRange && !inRange ? ' · 범위 밖' : '');
    scale.hidden = !hasRange;
    marker.hidden = !inRange;
    if (!hasRange) continue;
    low.textContent = format(data.min_deg);
    high.textContent = format(data.max_deg);
    if (inRange) marker.style.setProperty('--pose-position', (100 * (data.deg - data.min_deg) / (data.max_deg - data.min_deg)) + '%');
  }
  poseStatus.hidden = any && !estimated;
  poseStatus.textContent = any ? '이동 명령 기반 · 시작점 50%' : '짐벌 위치 데이터 없음';
  poseReceivedAt = any ? Date.now() : 0;
}

function expireGimbalPose() {
  if (poseReceivedAt && Date.now() - poseReceivedAt >= 1500) renderGimbalPose(null);
}
