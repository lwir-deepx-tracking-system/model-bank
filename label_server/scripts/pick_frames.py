#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""라벨할 장 고르기: 저장된 녹화에서 서로 다른 장면이 고르게 들어가도록 N 장을 뽑는다 (사진만 본다, 라벨은 보지 않는다).

자료 모양 (라벨 서버와 같음):  <root>/<녹화 이름>/images/000001.png ...   (8 비트 또는 16 비트 회색 PNG)

    python3 pick_frames.py --root <녹화 폴더들의 위 폴더> --ratio 0.05 --out picks.json                    (일정 간격)
    python3 pick_frames.py --root <녹화 폴더들의 위 폴더> --ratio 0.05 --method kmeans --out picks.json    (묶어서 뽑기)

--method
    uniform   녹화를 이어 붙여 일정 간격으로 (기본)
    random    무작위
    kmeans    비슷한 장끼리 N 묶음으로 나눠 묶음마다 중심에 가장 가까운 한 장
    kcenter   이미 고른 장들과 가장 다른 장부터 차례로 (Sener & Savarese, ICLR 2018 의 k-center greedy)
--keep-uncertain LO:HI
    검출 결과 파일(--det-file, 기본 tracks_mot.txt: 프레임,번호,x,y,폭,높이,점수,...)이 녹화 폴더에 있으면,
    점수가 LO 이상 HI 미만인 상자가 있는 장은 고르는 방법과 상관없이 전부 남긴다 (N 과 따로 센다).
    오검출일 수 있는 장면이 대표 뽑기에서 빠지지 않게 하려는 규칙이다.
    주의: 범위를 넓게 주면 거의 다 남는다. 위 실외 녹화에서는 0.25:0.5 가 8,929장 중 5,713장, 0.25:0.30 도 1,631장이었다.
    --uncertain-max M 을 주면 그 가운데 일정 간격으로 M 장만 남긴다. 몇 장이 남는지는 끝에 찍힌다.

2026-11-11 비교 (실외 녹화 15개, 8,929장, 검수 라벨로 채점): 묶어서 뽑기(kmeans, kcenter)는 미리 정한 기준에서 일정 간격보다 낫지 않았다.
한 녹화에 장 수가 쏠리는 것은 줄였지만(45 % -> 17 %), 키 32 px 미만의 작은 사람이 있는 장을 절반만 뽑았다. 그래서 기본은 uniform 이다.

특징: 사진을 32x24 로 줄이고 장마다 평균 0, 분산 1 로 맞춘 뒤 PCA 32 차원. numpy 와 opencv 만 쓴다.
결과 JSON: {"picked": [{"seq":..., "frame":..., "file":..., "why": "kmeans" | "uncertain"}], ...} 와 같은 이름의 .txt (한 줄에 파일 하나).
"""
import argparse
import json
import os
import sys

import numpy as np


def list_frames(root):
    out = []
    for s in sorted(os.listdir(root)):
        d = os.path.join(root, s, 'images')
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith('.png') and f[:-4].isdigit(): out.append((s, int(f[:-4]), os.path.join(d, f)))
    return out


def features(frames, thumb=(32, 24), dims=32, cache=None):
    """-> [n, dims] float32. cache: .npy 경로 (있으면 읽고, 없으면 만들어 쓴다)."""
    import cv2
    if cache and os.path.isfile(cache):
        X = np.load(cache)
        if len(X) == len(frames): return X
    T = np.zeros((len(frames), thumb[0] * thumb[1]), np.float32)
    for k, (_, _, p) in enumerate(frames):
        g = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
        if g is None: raise SystemExit('cannot read ' + p)
        if g.ndim == 3: g = g[:, :, 0]
        t = cv2.resize(g.astype(np.float32), thumb, interpolation=cv2.INTER_AREA).reshape(-1)
        T[k] = (t - t.mean()) / (t.std() + 1e-6)
    T -= T.mean(0)
    d = min(dims, T.shape[1], max(1, len(frames) - 1))
    _, _, vt = np.linalg.svd(T[np.random.default_rng(0).choice(len(T), min(len(T), 4000), replace=False)], full_matrices=False)
    X = (T @ vt[:d].T).astype(np.float32)
    if cache: np.save(cache, X)
    return X


def sqdist(X, C):
    return np.maximum((X * X).sum(1)[:, None] - 2.0 * X @ C.T + (C * C).sum(1)[None], 0.0)


def pick_uniform(n, k):
    return sorted(set(int(v) for v in np.floor((np.arange(k) + 0.5) * n / k)))


def pick_random(n, k, seed):
    return sorted(np.random.default_rng(seed).choice(n, k, replace=False).tolist())


def pick_kcenter(X, k):
    first = int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))
    sel = [first]; d = ((X - X[first]) ** 2).sum(1)
    while len(sel) < k:
        j = int(np.argmax(d)); sel.append(j); d = np.minimum(d, ((X - X[j]) ** 2).sum(1))
    return sorted(sel)


def pick_kmeans(X, k, seed=0, iters=25):
    """k-means++ 로 시작, Lloyd 반복. 묶음마다 중심에 가장 가까운 장 하나. 빈 묶음은 가장 먼 장으로 다시 채운다."""
    rng = np.random.default_rng(seed); n = len(X)
    C = np.empty((k, X.shape[1]), np.float32); C[0] = X[rng.integers(n)]; d = ((X - C[0]) ** 2).sum(1)
    for j in range(1, k):
        p = d / d.sum() if d.sum() > 0 else None
        C[j] = X[rng.choice(n, p=p)]; d = np.minimum(d, ((X - C[j]) ** 2).sum(1))
    for _ in range(iters):
        D = sqdist(X, C); a = D.argmin(1); moved = 0.0
        for j in range(k):
            m = a == j
            if m.any(): c = X[m].mean(0)
            else: c = X[int(D.min(1).argmax())]
            moved = max(moved, float(((c - C[j]) ** 2).sum())); C[j] = c
        if moved < 1e-8: break
    D = sqdist(X, C); a = D.argmin(1); sel = set()
    for j in range(k):
        idx = np.where(a == j)[0]
        if len(idx): sel.add(int(idx[D[idx, j].argmin()]))
    if len(sel) < k:   # 빈 묶음이 남았으면 이미 고른 장들과 가장 다른 장으로 채운다
        d = sqdist(X, X[sorted(sel)]).min(1)
        while len(sel) < k:
            j = int(d.argmax()); sel.add(j); d = np.minimum(d, ((X - X[j]) ** 2).sum(1))
    return sorted(sel)


def uncertain_frames(root, frames, lo, hi, det_file):
    """점수가 [lo, hi) 인 사람 상자가 있는 장의 번호(목록 안의 순서)."""
    pos = {(s, f): k for k, (s, f, _) in enumerate(frames)}; keep = set()
    for s in sorted(set(s for s, _, _ in frames)):
        p = os.path.join(root, s, det_file)
        if not os.path.isfile(p): continue
        for line in open(p, encoding='utf-8', errors='replace'):
            a = line.strip().split(',')
            if len(a) < 7: continue
            try: f, c = int(float(a[0])), float(a[6])
            except ValueError: continue
            if len(a) > 7 and a[7].strip() not in ('', '0', '0.0', '-1'): continue   # 종류 0 = 사람만
            if lo <= c < hi and (s, f) in pos: keep.add(pos[(s, f)])
    return sorted(keep)


def pick(X, n, k, method, seed=0):
    k = max(1, min(k, n))
    if method == 'uniform': return pick_uniform(n, k)
    if method == 'random': return pick_random(n, k, seed)
    if method == 'kcenter': return pick_kcenter(X, k)
    if method == 'kmeans': return pick_kmeans(X, k, seed)
    raise SystemExit('unknown method ' + method)


def main():
    ap = argparse.ArgumentParser(description='라벨할 장 고르기')
    ap.add_argument('--root', required=True, help='녹화 폴더들의 위 폴더 (<root>/<녹화>/images/000001.png)')
    ap.add_argument('--method', choices=['uniform', 'random', 'kmeans', 'kcenter'], default='uniform')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--budget', type=int, help='뽑을 장 수')
    g.add_argument('--ratio', type=float, help='뽑을 비율 (0~1)')
    ap.add_argument('--keep-uncertain', default='', help='LO:HI: 검출 점수가 이 범위인 상자가 있는 장은 전부 남김 (예: 0.25:0.5)')
    ap.add_argument('--uncertain-max', type=int, default=0, help='--keep-uncertain 으로 남기는 장 수의 한도 (0 = 한도 없음). 넘으면 일정 간격으로 줄임')
    ap.add_argument('--det-file', default='tracks_mot.txt', help='녹화 폴더 안의 검출 결과 파일 이름')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache', default='', help='특징을 저장해 둘 .npy 경로 (다시 돌릴 때 빠름)')
    ap.add_argument('--out', required=True, help='결과 JSON 경로 (같은 이름의 .txt 도 쓴다)')
    a = ap.parse_args()
    frames = list_frames(a.root)
    if not frames: raise SystemExit('no frames under ' + a.root)
    n = len(frames); k = a.budget if a.budget else int(round(a.ratio * n))
    X = features(frames, cache=a.cache or None) if a.method in ('kmeans', 'kcenter') else None
    sel = pick(X, n, k, a.method, a.seed)
    why = {j: a.method for j in sel}; unc = []
    if a.keep_uncertain:
        lo, hi = (float(v) for v in a.keep_uncertain.split(':'))
        unc = uncertain_frames(a.root, frames, lo, hi, a.det_file); n_unc = len(unc)
        if a.uncertain_max > 0 and n_unc > a.uncertain_max: unc = [unc[i] for i in pick_uniform(n_unc, a.uncertain_max)]
        print('uncertain frames (score %g ~ %g): %d found, %d kept' % (lo, hi, n_unc, len(unc)))
        for j in unc: why.setdefault(j, 'uncertain')
    rows = [dict(seq=frames[j][0], frame=frames[j][1], file=os.path.relpath(frames[j][2], a.root).replace('\\', '/'), why=why[j]) for j in sorted(why)]
    res = dict(root=os.path.abspath(a.root), method=a.method, frames_total=n, asked=k, picked_by_method=len(sel), kept_uncertain_extra=sum(1 for r in rows if r['why'] == 'uncertain'),
               uncertain_range=a.keep_uncertain or None, seed=a.seed, picked=rows)
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f: json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.splitext(a.out)[0] + '.txt', 'w', encoding='utf-8', newline='\n') as f: f.write('\n'.join(r['file'] for r in rows) + '\n')
    print('frames %d, method %s: picked %d, uncertain kept on top %d -> %s' % (n, a.method, len(sel), res['kept_uncertain_extra'], a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
