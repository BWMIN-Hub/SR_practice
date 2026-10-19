# 04 · SRGAN ×3

[leftthomas/SRGAN](https://github.com/leftthomas/SRGAN) 을 우리 위성 데이터로 학습.
**모델 구조와 손실은 업스트림 그대로**이고 데이터셋만 바꿨다.

| | |
|---|---|
| 입력 | 데이터셋이 제공하는 Sentinel-2 LR (10 m) |
| 목표 | 3.3333 m 영상, LR × 3 |
| 학습 | IKONOS 2,638패치 × 입력 9장 (LR 64 px), batch 16, 업스트림 손실 그대로 (MSE + 0.006 VGG16 + 0.001 adversarial + TV, 판별자 함께 학습), Adam 1e-5, 60 epoch 상한까지 (최고 57 epoch) |
| 가중치 | `checkpoints/srgan_g_x3_v4.pth` |
| 노트북 | [`../../notebooks/04_srgan_x3.ipynb`](../../notebooks/04_srgan_x3.ipynb) |

## ×3 을 위해 고친 곳

업스트림 Generator 는 ×2 업샘플 블록을 `log2(scale)` 번 쌓아서 3 을 주면 블록이 1개만
생기고 조용히 ×2 출력이 나온다(32px → 96px 이어야 하는데 64px).
2의 거듭제곱이 아니면 `PixelShuffle(scale)` 블록 하나를 쓰도록 바꿨다.
×2/×4/×8 경로는 레이어·파라미터가 이전과 같고 기존 체크포인트도 그대로 로드된다.

## 판별자 붕괴 — epoch 37

`D(x)` 와 `D(G(z))` 가 **동시에 1** 로 올라간다. 판별자가 모든 입력을 "진짜" 로 찍는
상태다. `adversarial = mean(1 - D(G(z)))` 가 0 에 고정되어 기울기가 사라지고,
그 뒤로는 사실상 MSE + VGG 손실만으로 학습된다.

`Loss_D = 1 - D(x) + D(G(z))` 는 이때 정확히 **1.0** 이다. 둘 다 0 으로 붕괴해도 1.0 이라
**손실 곡선만으로는 붕괴를 알 수 없다.**

## 성능 (검증 10패치)

| | PSNR | SSIM |
|---|---|---|
| Bicubic | 18.15 | 0.4805 |
| SRGAN | **18.30** | **0.5187** |

## 배포 가중치 (2026-10-19)

새 학습 데이터로 다시 학습한 `srgan_g_x3_v4.pth` 를 쓴다. 점수는 `dataset/validation/` 10패치 평균이다.

| | PSNR | SSIM |
|---|---|---|
| Bicubic | 18.15 | 0.4805 |
| **`srgan_g_x3_v4.pth`** | **18.52** | **0.5150** |
