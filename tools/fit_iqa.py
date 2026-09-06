"""NR-IQA 의 기준 모델을 이 프로젝트 데이터로 만든다.

NIQE 와 BRISQUE 는 "정상 영상이란 이런 것" 이라는 기준이 있어야 점수가 나온다.
원 논문은 자연 사진(LIVE)으로 그 기준을 만들었는데 위성 영상은 통계가 달라 그대로
쓰면 뜻이 없다. 여기서는 IKONOS HR 로 다시 잡는다.

  niqe_model.npz   깨끗한 HR 조각들의 특징 평균·공분산 (거리를 재는 기준)
  brisque_svr.npz  열화 세기를 맞히도록 학습한 RBF SVR 을 넘파이로 굳힌 것

    python tools/fit_iqa.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{HERE}/lib')
import iqa                                                       # noqa: E402

SRC = '/c/work/nst_work_dirs/G2601-SAR-Filtering/tmp_files/sr_data/PAIR_EXTRA/train/HR'
PATCH = 96


def degrade(img, s, rng):
    if s <= 0:
        return img
    b = cv2.GaussianBlur(img, (7, 7), 0.2 + 2.2 * s)
    return np.clip(b + rng.normal(0, 14 * s, b.shape), 0, 255).astype(np.uint8)


def main():
    import imageio.v2 as io
    from sklearn.svm import SVR
    out = f'{HERE}/results/iqa'
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(SRC) if f.endswith('_S0010.png'))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), 120, replace=False)

    # ① NIQE 기준 — 깨끗한 HR 조각들의 특징 분포
    feats = []
    for i in pick:
        g = cv2.cvtColor(io.imread(f'{SRC}/{files[i]}'), cv2.COLOR_RGB2GRAY)
        if (g == 0).mean() > 0.02:
            continue
        gf = g.astype(np.float64)
        for by in range(gf.shape[0] // PATCH):
            for bx in range(gf.shape[1] // PATCH):
                m = iqa.mscn(gf[by * PATCH:(by + 1) * PATCH,
                                bx * PATCH:(bx + 1) * PATCH])
                if m.var() < 0.05:
                    continue
                feats.append(iqa._feat18(m))
    F = np.array(feats)
    np.savez(f'{out}/niqe_model.npz', mu=F.mean(0), cov=np.cov(F.T), n=len(F))
    print(f'NIQE 기준: 조각 {len(F)}개, 특징 {F.shape[1]}차 -> niqe_model.npz')

    # ② BRISQUE 회귀 — 특징 36개에서 열화 세기를 맞히게 학습
    X, y = [], []
    for i in pick[:60]:
        g = cv2.cvtColor(io.imread(f'{SRC}/{files[i]}'), cv2.COLOR_RGB2GRAY)
        if (g == 0).mean() > 0.02:
            continue
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            X.append(iqa.features36(degrade(g, s, rng)))
            y.append(100 * s)
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=1.0).fit((X - mu) / sd, y)
    gamma = 1.0 / (X.shape[1] * ((X - mu) / sd).var())
    np.savez(f'{out}/brisque_svr.npz', mu=mu, sd=sd, sv=svr.support_vectors_,
             coef=svr.dual_coef_[0], b=svr.intercept_[0], gamma=gamma)
    print(f'BRISQUE 회귀: 표본 {len(y)}개, 지지벡터 {len(svr.support_)}개 '
          f'-> brisque_svr.npz')

    # 검산 — 저장한 파라미터로 다시 계산해 sklearn 과 맞는지 본다
    iqa._M.clear()
    g = io.imread(f'{SRC}/{files[pick[0]]}')
    a = iqa.brisque(g)
    b = float(svr.predict(((iqa.features36(g) - mu) / sd)[None])[0])
    print(f'검산: 넘파이 {a:.3f}  vs  sklearn {np.clip(b,0,100):.3f}')
    print(f'      NIQE(원본 HR) {iqa.niqe(g):.3f},  PIQE {iqa.piqe(g):.2f}')


if __name__ == '__main__':
    main()
