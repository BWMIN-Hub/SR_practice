# 03 · SRGAN ×2 — 판별자 붕괴 전후 비교

[leftthomas/SRGAN](https://github.com/leftthomas/SRGAN) 을 우리 위성 데이터로 학습.
**모델 구조와 손실은 업스트림 그대로**다.

## 설정

입력 LR 은 **데이터셋이 제공하는 실제 Sentinel-2 영상 그대로** 쓰고,
HR 목표만 LR × 2 크기로 줄였다 (원본은 LR × 3).

```
LR crop 32px (실제 촬영본)  ->  목표 HR 64px
```

HR 을 줄여 LR 을 만드는 방식이 아니다. 모델이 보는 입력은 항상 실제 영상이어야 한다.

| | |
|---|---|
| 학습 | IKONOS 804쌍, 100 epoch, batch 32, LR crop 32 |
| 노트북 | [`../../notebooks/03_srgan_x2.ipynb`](../../notebooks/03_srgan_x2.ipynb) |

## 판별자 붕괴 — epoch 43

`D(x)` 와 `D(G(z))` 가 **동시에 0** 으로 떨어진다. 판별자가 진짜·가짜를 가리는 대신
모든 입력을 "가짜" 로 찍는 상태다. `adversarial = mean(1 - D(G(z)))` 가 1.0 에 고정되어
기울기가 사라지고, 그 뒤로는 사실상 MSE + VGG 손실만으로 학습된다.

`Loss_D = 1 - D(x) + D(G(z))` 는 이때 정확히 **1.0** 이라 손실 곡선만 봐서는 알 수 없다.
(x3 학습에서는 반대로 둘 다 **1** 로 붕괴했다. 방향은 반대인데 손실은 똑같이 1.0 이다.)

## 붕괴 전 vs 후 (검증 10패치, 목표 = LR x2)

| | PSNR | SSIM |
|---|---|---|
| Bicubic x2 | 19.05 | 0.5786 |
| 붕괴 **전** best (epoch 12) | 18.25 | 0.5270 |
| 붕괴 **후** best (epoch 84) | **19.23** | **0.5951** |
| 마지막 (epoch 100) | 15.23 | 0.4647 |

**붕괴 후가 더 좋다.** 적대적 신호가 죽은 뒤 MSE + VGG 만으로 학습된 쪽이 이 데이터에서는
PSNR·SSIM 에 유리했다. GAN 이 항상 이득은 아니다.

epoch 100 이 크게 떨어진 것에서 보듯 붕괴 뒤에도 학습은 불안정하다.
**마지막 가중치가 아니라 best 를 골라 써야 한다.**

## 체크포인트

| 파일 | epoch | 용도 |
|---|---|---|
| `checkpoints/srgan_g_x2_ep12_before.pth` | 12 | 붕괴 전 best |
| `checkpoints/srgan_g_x2_ep84_after.pth` | 84 | 붕괴 후 best (권장) |

`statistics/x2_train_results.csv` 에 epoch 별 지표 100행.
