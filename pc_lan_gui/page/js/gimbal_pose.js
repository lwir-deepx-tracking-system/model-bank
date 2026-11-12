const poseOverlay = $('gimbal-position');
const poseImage = $('v');
let poseReceivedAt = 0;
let poseImageAspect = 0;
const poseAxes = [['pan', '좌우'], ['tilt', '상하']].map(([key, label]) => {
  const axis = $('pose-' + key);
  return {key, label, axis, marker: axis.querySelector('[data-marker]')};
});

function fitPoseOverlay() {
  // Match the visible image when object-fit: contain leaves black bars.
  poseImageAspect = poseImage.naturalWidth / poseImage.naturalHeight || 0;
  if (!poseImageAspect) return;
  const width = poseImage.clientWidth;
  const height = poseImage.clientHeight;
  const insetX = Math.max(0, (width - height * poseImageAspect) / 2);
  const insetY = Math.max(0, (height - width / poseImageAspect) / 2);
  poseOverlay.style.setProperty('--pose-x', insetX + 'px');
  poseOverlay.style.setProperty('--pose-y', insetY + 'px');
}

if (typeof ResizeObserver === 'function') new ResizeObserver(fitPoseOverlay).observe(poseImage);
else addEventListener('resize', fitPoseOverlay);
poseImage.addEventListener('load', fitPoseOverlay);

function renderGimbalPose(pose, estimate) {
  if ((poseImage.naturalWidth / poseImage.naturalHeight || 0) !== poseImageAspect) fitPoseOverlay();
  let any = false;
  for (const {key, label, axis, marker} of poseAxes) {
    const actual = pose && pose[key];
    const useActual = actual && Number.isFinite(actual.deg) &&
      Number.isFinite(actual.min_deg) && Number.isFinite(actual.max_deg) &&
      actual.min_deg < actual.max_deg;
    const value = useActual ? actual.deg : estimate && estimate[key];
    const min = useActual ? actual.min_deg : 0;
    const max = useActual ? actual.max_deg : 100;
    const valid = Number.isFinite(value) && value >= min && value <= max;
    axis.hidden = !valid;
    if (!valid) continue;
    any = true;
    axis.setAttribute('aria-label', label + (useActual ? ' ' + value + '°' : ' 추정 ' + Math.round(value) + '%'));
    marker.style.setProperty('--pose-position', (100 * (value - min) / (max - min)) + '%');
  }
  poseReceivedAt = any ? Date.now() : 0;
}

function expireGimbalPose() {
  if (poseReceivedAt && Date.now() - poseReceivedAt >= 1500) renderGimbalPose(null);
}
