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

function renderGimbalPose(pose) {
  if ((poseImage.naturalWidth / poseImage.naturalHeight || 0) !== poseImageAspect) fitPoseOverlay();
  let any = false;
  for (const {key, label, axis, marker} of poseAxes) {
    const actual = pose && pose[key];
    const value = actual && actual.deg;
    // Display ranges are centred on Home, independently of mechanical limits.
    const extent = key === 'pan' ? 180 : 90;
    const valid = Number.isFinite(value) && Math.abs(value) <= 360;
    axis.hidden = !valid;
    if (!valid) continue;
    any = true;
    const description=label+' '+value+'° · 표시 범위 -'+extent+'° ~ '+extent+'°';
    axis.setAttribute('aria-label', description);
    axis.title=description;
    marker.style.setProperty('--pose-position', (50 + 50 * Math.max(-extent, Math.min(extent, value)) / extent) + '%');
  }
  poseReceivedAt = any ? Date.now() : 0;
}

function expireGimbalPose() {
  if (poseReceivedAt && Date.now() - poseReceivedAt >= 1500) renderGimbalPose(null);
}
