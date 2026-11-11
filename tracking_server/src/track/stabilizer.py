"""화면 흔들림 보정: 카메라가 떨릴 때 그림을 반대로 조금 옮겨 보여 주고, 상자 좌표를 실제 카메라 좌표로 되돌린다."""
import time
import numpy as np


# ==== STAB BEGIN ==== (patch_stab.py 11-10: picture stabilization by optical flow, default off; convention in Stabilizer.__doc__)
STAB = dict(on=False, obj=None, px=(0.0, 0.0), ms=0.0, resets=0, clamped=0, err='')
STAB_M = [None]   # replicated border (left, top, right, bottom) px of the frame the tracker is working on; None = picture not moved


class Stabilizer:
    """Causal picture stabilization. Motion: pyramidal Lucas-Kanade (Lucas & Kanade 1981; Bouguet 2000) on Shi-Tomasi corners (1994),
    half-size picture, RANSAC partial affine, read at the picture centre. Smoothing: camera-path smoothing (Grundmann et al., CVPR 2011)
    done causally with a recursive path filter (Litvin et al., SPIE 2003): first-order low-pass + critically damped alpha-beta tracker +
    velocity lead (3 poles at rho = exp(-2 pi fc dt), H(1) = 1, H'(1) = 0 -> zero lag on a steady pan). Zero added delay.
    update() returns the correction c = (cx, cy): stabilized[y, x] = raw[y - cy, x - cx]; box_real = box_stabilized - c.
    c is clamped to +-clamp px and (subpixel False) a whole number of pixels. A step above slew px per frame = intentional move, not smoothed. State is kept RELATIVE to the camera path, so it stays bounded."""
    def __init__(self, fc=1.0, clamp=24.0, ds=2, corners=200, mindist=8, max_step=64.0, min_inl=10, period=0.04, hyst=0.75, slew=8.0, hang=2, subpixel=False):
        import cv2
        self.cv2 = cv2; self.fc = float(fc); self.clamp = float(clamp); self.ds = int(ds); self.max_step = float(max_step); self.min_inl = int(min_inl)
        self.T = float(period); self.hyst = float(hyst); self.slew = float(slew); self.hang = int(hang); self.subpixel = bool(subpixel)
        self.fp = dict(maxCorners=int(corners), qualityLevel=0.01, minDistance=mindist, blockSize=3, useHarrisDetector=False, k=0.04)
        self.lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        self.n = 0; self.n_fail = 0; self.n_clamp = 0; self.n_slew = 0; self.prev = None; self.kp = None; self.t = None; self.zero()

    def zero(self):
        self.e1 = [0.0, 0.0]; self.e2 = [0.0, 0.0]; self.v = [0.0, 0.0]; self.cf = [0.0, 0.0]; self.ci = [0.0, 0.0]; self.hold = 0

    def motion(self, gray):
        """image motion previous -> current frame (dx, dy) in full-size px at the picture centre; None = no estimate"""
        cv2 = self.cv2; h, w = gray.shape[:2]
        f = cv2.resize(gray, (w // self.ds, h // self.ds)) if self.ds > 1 else gray.copy()
        prev, kp = self.prev, self.kp
        self.prev = f; self.kp = cv2.goodFeaturesToTrack(f, mask=None, **self.fp)
        if prev is None or kp is None or len(kp) < self.min_inl: return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, f, kp, None, **self.lk)
        if st is None: return None
        ok = st.reshape(-1) == 1
        if int(ok.sum()) < self.min_inl: return None
        p0 = kp.reshape(-1, 2)[ok]; p1 = nxt.reshape(-1, 2)[ok]
        M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if M is None or inl is None or int(inl.sum()) < max(self.min_inl, 0.3 * len(p0)): return None
        cx, cy = f.shape[1] / 2.0, f.shape[0] / 2.0
        return ((M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx) * self.ds, (M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy) * self.ds)

    def update(self, gray, t):
        """one frame (gray = RAW 8-bit picture, t = arrival time in s) -> correction (cx, cy) for this frame"""
        d = self.motion(gray); self.n += 1
        dt = None if self.t is None else t - self.t; self.t = t
        if dt is None or dt <= 0 or dt > 0.5:   # first frame or input gap: start again from no correction
            self.zero(); return (0.0, 0.0)
        return self.step(d, dt)

    def step(self, d, dt):
        """path filter alone: d = image motion (dx, dy) px since the previous frame or None, dt s -> correction (cx, cy)"""
        if d is None or abs(d[0]) > self.max_step or abs(d[1]) > self.max_step:   # no estimate: forget the motion, let the correction fade (no jump)
            self.n_fail += 1
            for a in (0, 1): self.v[a] = 0.0; self.e1[a] = self.e2[a] = self.cf[a] = 0.8 * self.cf[a]
        else:
            n = max(1, int(round(dt / self.T))); self.T = min(0.1, max(0.01, self.T + 0.02 * (dt / n - self.T))); dtu = n * self.T   # dt in whole frame periods (arrival times jitter)
            if max(abs(d[0]), abs(d[1])) > self.slew * n:   # a step this large is an intentional move (gimbal slew), not shake: shown as it is, filter state and correction untouched
                self.n_slew += 1; self.hold = self.hang
            elif self.hold > 0: self.hold -= 1               # the frames right after a slew (its tail): untouched as well
            else:
                rho = float(np.exp(-2.0 * np.pi * self.fc * dtu)); al = 1.0 - rho * rho; be = (1.0 - rho) ** 2; lead = dtu * rho / (1.0 - rho)
                hit = False
                for a in (0, 1):
                    self.e1[a] = rho * (self.e1[a] - d[a])                       # low-passed path - actual path
                    m = self.e2[a] + self.v[a] * dtu - d[a]; r = self.e1[a] - m  # alpha-beta tracker of the low-passed path
                    self.e2[a] = m + al * r; self.v[a] += be / dtu * r
                    c = self.e2[a] + self.v[a] * lead                            # velocity lead: zero lag on a steady pan
                    if abs(c) > self.clamp: c = self.clamp if c > 0 else -self.clamp; hit = True
                    self.cf[a] = c
                self.n_clamp += int(hit)
        if self.subpixel: return (float(self.cf[0]), float(self.cf[1]))
        for a in (0, 1):
            if abs(self.cf[a] - self.ci[a]) > self.hyst: self.ci[a] = float(round(self.cf[a]))   # whole pixels, with hysteresis against 1 px flicker
        return (self.ci[0], self.ci[1])


def stab_shift(img, c):
    """img moved by c rounded to whole pixels: out[y, x] = img[y - iy, x - ix]; the uncovered border repeats the nearest edge pixel"""
    ix, iy = int(round(c[0])), int(round(c[1]))
    if ix == 0 and iy == 0: return img
    import cv2
    h, w = img.shape[:2]
    return cv2.copyMakeBorder(img[max(0, -iy):h - max(0, iy), max(0, -ix):w - max(0, ix)], max(0, iy), max(0, -iy), max(0, ix), max(0, -ix), cv2.BORDER_REPLICATE)


def stab_warp(img, c):
    """the picture moved by c (whole pixels: exact copy; otherwise bilinear)"""
    if float(c[0]).is_integer() and float(c[1]).is_integer(): return stab_shift(img, c)
    import cv2
    return cv2.warpAffine(img, np.float32([[1, 0, c[0]], [0, 1, c[1]]]), (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def stab_live(g8, t, canvas):
    """live_next with the switch ON: g8 = raw 8-bit picture (480x640), canvas = the 512x640 frame for the detector and the page (picture in rows 16..496).
    Moves the picture inside canvas by this frame's correction and returns the correction (cx, cy). On an error the picture stays as it is (None)."""
    t0 = time.perf_counter()
    try:
        o = STAB['obj']
        if o is None: o = STAB['obj'] = Stabilizer()
        c = o.update(g8, t)
        if c[0] or c[1]:
            canvas[16:496] = stab_warp(g8, c); canvas[:16] = canvas[16]; canvas[496:] = canvas[495]
        STAB['resets'] = o.n_fail; STAB['clamped'] = o.n_clamp
    except Exception as e:
        canvas[16:496] = g8; canvas[:16] = g8[0]; canvas[496:] = g8[-1]; STAB['obj'] = None; STAB['err'] = repr(e)[:160]; c = None
    STAB['ms'] += 0.05 * ((time.perf_counter() - t0) * 1000.0 - STAB['ms'])
    return c


def stab_off():
    """live_next with the switch OFF: nothing is computed; the state is dropped so that the next switch-on starts from no correction"""
    if STAB['obj'] is not None: STAB['obj'] = None
    return None


def stab_margins(c):
    """replicated border (left, top, right, bottom) px of a frame with correction c; None when the picture was not moved"""
    if c is None or not (c[0] or c[1]): return None
    ix = int(np.ceil(abs(c[0]))) * (1 if c[0] > 0 else -1); iy = int(np.ceil(abs(c[1]))) * (1 if c[1] > 0 else -1)
    return (max(0, ix), max(0, iy), max(0, -ix), max(0, -iy))


def stab_rows_real(rows, c):
    """tracker rows (x1, y1, x2, y2, id, conf, class, ...) from stabilized to real camera coordinates (box_real = box_stabilized - c), not clipped"""
    if c is None or not (c[0] or c[1]): return rows
    return [(float(b[0]) - c[0], float(b[1]) - c[1], float(b[2]) - c[0], float(b[3]) - c[1], b[4], b[5], b[6]) for b in rows]


def stab_people_real(people, c, w=640.0, h=480.0):
    """people (id, x1, y1, x2, y2, conf) in picture pixels from stabilized to REAL camera coordinates, clipped to the camera frame"""
    if c is None or not (c[0] or c[1]): return people
    out = []
    for p in people:
        x1 = min(max(p[1] - c[0], 0.0), w - 1.0); y1 = min(max(p[2] - c[1], 0.0), h - 1.0)
        out.append((p[0], x1, y1, max(min(p[3] - c[0], w), x1 + 1.0), max(min(p[4] - c[1], h), y1 + 1.0), p[5]))
    return out
# ==== STAB END ====
