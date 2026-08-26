# 02 · SRGAN ×3 — 판별자 붕괴 전후 비교

[leftthomas/SRGAN](https://github.com/leftthomas/SRGAN) 을 우리 위성 데이터로 학습.
**모델 구조와 손실은 업스트림 그대로**이고 데이터셋만 바꿨다.

| | |
|---|---|
| 입력 | 데이터셋이 제공하는 실제 Sentinel-2 LR (10 m) |
| 목표 | IKONOS HR (3.3333 m), LR × 3 |
| 학습 | IKONOS 804쌍, 100 epoch, batch 32, crop 96 |
| 노트북 | [`../../notebooks/02_srgan_x3.ipynb`](../../notebooks/02_srgan_x3.ipynb) |

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

## 붕괴 전 vs 후 (검증 10패치)

| | PSNR | SSIM |
|---|---|---|
| Bicubic | 18.15 | 0.4805 |
| 붕괴 **전** best (epoch 29) | 18.08 | 0.4986 |
| 붕괴 **후** best (epoch 70) | **18.30** | **0.5187** |

**붕괴 후가 더 좋다.** 적대적 신호가 죽은 뒤 MSE + VGG 만으로 학습된 쪽이 이 데이터에서는
PSNR·SSIM 에 유리했다. GAN 이 항상 이득은 아니다.

## 체크포인트

| 파일 | epoch | 용도 |
|---|---|---|
| `checkpoints/srgan_g_x3_ep29_before.pth` | 29 | 붕괴 전 best |
| `checkpoints/srgan_g_x3_ep70_after.pth` | 70 | 붕괴 후 best (권장) |

`statistics/train_results.csv` 에 epoch 별 지표 100행.
