"""LR/HR 쌍 구축 전략의 열화 함수. 노트북에서 직접 돌려 볼 수 있다.

    from pair_degrade import degrade_syn, degrade_high
    lr1 = degrade_syn(hr)                       # 합성 열화
    lr3 = degrade_high(hr, np.random.default_rng(0))   # 고차 열화

원본: EDSR-PyTorch/prepare_pairs.py (데이터셋 생성 스크립트)

HR 은 세 전략이 **완전히 같은 타일**을 쓴다. LR 을 어떻게 얻느냐만 다르다.

  SYN   합성 열화      HR 을 bicubic 으로 1/3 축소. 정합은 완벽하지만 실제 센서 특성이 없다.
  REAL  실측 페어      실제 Sentinel-2 LR.tif. 센서 특성은 진짜지만 기하·방사·시간차가 남는다.
  HIGH  고차 열화      HR 에 블러·리사이즈·노이즈·JPEG 를 무작위로 2회 반복 (Real-ESRGAN 방식).

기존 `g_LR.tif` 는 쓰지 않는다. 이 비교의 대상이 아니다.

검증셋은 **세 전략 모두 실제 LR** 로 만든다. 최종 목표가 실제 Sentinel-2 영상에
쓰는 것이므로, 학습 도메인이 달라도 같은 잣대로 재야 한다.

"""
import cv2
import numpy as np

SCALE = 3


# ── ① 합성 열화 ────────────────────────────────────────────────────────
def degrade_syn(hr, scale=SCALE):
    h, w = hr.shape[0] // scale, hr.shape[1] // scale
    return cv2.resize(hr, (w, h), interpolation=cv2.INTER_CUBIC)


# ── ③ 고차 열화 (Real-ESRGAN / BSRGAN 계열) ────────────────────────────
def _blur(img, rng, strong):
    k = int(rng.choice([7, 9, 11, 13, 15]))
    if rng.random() < 0.7:                       # 등방 가우시안
        s = rng.uniform(0.2, 2.5 if strong else 1.3)
        return cv2.GaussianBlur(img, (k, k), s)
    sx = rng.uniform(0.2, 3.0 if strong else 1.5)
    sy = rng.uniform(0.2, 3.0 if strong else 1.5)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sx, sigmaY=sy)


def _resize(img, rng, size=None):
    interp = int(rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if size is None:
        f = rng.uniform(0.5, 1.2)
        size = (max(8, int(img.shape[1] * f)), max(8, int(img.shape[0] * f)))
    return cv2.resize(img, size, interpolation=interp)


def _noise(img, rng, strong):
    if rng.random() < 0.5:                       # 가우시안
        s = rng.uniform(1, 8 if strong else 4) / 255
        # 위성 영상에 무지개색 잡음은 비현실적이라 회색 잡음을 기본으로 둔다
        n = rng.normal(0, s, img.shape if rng.random() < 0.3 else img.shape[:2] + (1,))
        return img + n
    scale = rng.uniform(10, 60)                  # 푸아송(샷) 노이즈
    return np.clip(rng.poisson(np.clip(img, 0, 1) * scale) / scale, 0, 2)


def _jpeg(img, rng):
    q = int(rng.integers(50, 96))
    ok, buf = cv2.imencode('.jpg', np.clip(img * 255, 0, 255).astype(np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255 if ok else img


def degrade_high(hr, rng, scale=SCALE):
    """블러 → 리사이즈 → 노이즈 → JPEG 를 두 번 반복한다."""
    img = hr.astype(np.float32) / 255
    h, w = hr.shape[0] // scale, hr.shape[1] // scale
    for r in range(2):
        strong = (r == 0)
        if rng.random() < 0.85:
            img = _blur(img, rng, strong)
        img = _resize(img, rng, None if r == 0 else (w, h))
        if rng.random() < 0.8:
            img = _noise(img, rng, strong)
        if rng.random() < 0.7:
            img = _jpeg(img, rng)
    img = _resize(np.clip(img, 0, 1), rng, (w, h))     # 크기를 정확히 맞춘다
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def save_pair(lr, hr, stem, out, split, scale=SCALE):
    d_hr = os.path.join(out, split, 'HR')
    d_lr = os.path.join(out, split, 'LR_bicubic', f'X{scale}')
    os.makedirs(d_hr, exist_ok=True)
    os.makedirs(d_lr, exist_ok=True)
    cv2.imwrite(os.path.join(d_hr, f'{stem}.png'), hr[:, :, ::-1])
    cv2.imwrite(os.path.join(d_lr, f'{stem}x{scale}.png'), lr[:, :, ::-1])
