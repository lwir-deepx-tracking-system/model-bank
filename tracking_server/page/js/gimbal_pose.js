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

function renderGimbalPose(pose) {
  let any = false;
  for (const {key, label, axis, value, scale, low, high, marker} of poseAxes) {
    const data = pose && pose[key];
    const valid = !!data && Number.isFinite(data.deg);
    axis.hidden = !valid;
    if (!valid) continue;
    any = true;
    const hasRange = Number.isFinite(data.min_deg) && Number.isFinite(data.max_deg) && data.max_deg > data.min_deg;
    const inRange = hasRange && data.deg >= data.min_deg && data.deg <= data.max_deg;
    value.textContent = label + ' ' + poseDegrees(data.deg) + (hasRange && !inRange ? ' · 범위 밖' : '');
    scale.hidden = !hasRange;
    marker.hidden = !inRange;
    if (!hasRange) continue;
    low.textContent = poseDegrees(data.min_deg);
    high.textContent = poseDegrees(data.max_deg);
    if (inRange) marker.style.setProperty('--pose-position', (100 * (data.deg - data.min_deg) / (data.max_deg - data.min_deg)) + '%');
  }
  poseStatus.hidden = any;
  poseReceivedAt = any ? Date.now() : 0;
}

function expireGimbalPose() {
  if (poseReceivedAt && Date.now() - poseReceivedAt >= 1500) renderGimbalPose(null);
}
