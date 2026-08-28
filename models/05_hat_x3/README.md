# 05 · HAT ×3

[XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) 을 우리 위성 데이터로 학습.
신경망 정의는 업스트림 `hat/archs/hat_arch.py` 를 그대로 쓴다(`lib/hat_arch.py`).
`basicsr` 없이 돌도록 **import 3줄만** 바꿨다.

업스트림은 basicsr 설정 파일(yml)로 학습하는데, 우리 데이터셋 규약과 맞지 않아
`HAT/train_hat.py` 를 새로 작성했다. SwinIR 때와 같은 방식이다.

| | |
|---|---|
| 구조 | Swin Transformer + 채널 어텐션(CAB) + 중첩 창 어텐션(OCAB), window 16, embed_dim 180, RHAG 6개 |
| 손실 | **L1 하나** — 판별자 없음 |
| 입력 | 데이터셋이 제공하는 Sentinel-2 LR (10 m) |
| 목표 | IKONOS HR (3.3333 m), LR × 3 |
| 학습 | IKONOS 804쌍, 100 epoch, batch 4, LR 패치 48, Adam 2e-4 (50·75·90% 지점 절반) |
| 가중치 | `checkpoints/hat_x3.pth` (epoch 100, 20.81M) |
| 노트북 | [`../../notebooks/05_hat_x3.ipynb`](../../notebooks/05_hat_x3.ipynb) |

`timm` 과 `einops` 가 필요하다 (`pip install timm einops`).

## HAT 은 입력 크기를 안 맞춰준다

SwinIR 은 내부에 `check_image_size()` 가 있어 창 크기의 배수가 아닌 입력도 알아서
패딩한다. **HAT 에는 그게 없다.** 128px 검증 패치는 16 으로 나눠떨어져 괜찮지만
인천 씬(600px)을 그대로 넣으면 죽는다.

```
RuntimeError: shape '[1, 37, 16, 37, 16, 1]' is invalid for input of size 360000
```

`lib/hat_models.py` 의 `hat_upscale()` 이 반사 패딩으로 배수를 맞춘 뒤 결과를
원래 크기로 잘라낸다. 학습·추론 코드에서 이 함수를 쓰면 크기를 신경 쓸 필요가 없다.

`train_hat.py --lr_crop` 도 같은 이유로 16 의 배수여야 한다(기본 48).

## 학습 경과

| epoch | L1 | lr |
|---|---|---|
| 1 | 0.11449 | 2e-4 |
| 50 | 0.06987 | 1e-4 |
| 100 | **0.06443** | 2.5e-5 |

L1 이 44% 줄었다. 판별자가 없어 붕괴가 없고, 학습률이 절반씩 떨어질 때마다
한 단계씩 내려가는 단조로운 곡선이다. 검증 10패치 기준 최고는 epoch 100 으로
끝까지 오르는 중이었다.

## 다섯 모델 비교 (검증 10패치 / 인천)

| 모델 | PSNR | SSIM | 인천 선명도 | 파라미터 |
|---|---|---|---|---|
| Bicubic | 18.15 | 0.4805 | 9.83 | — |
| EDSR | 18.97 | 0.5462 | 14.60 | 1.55M |
| SRGAN | 18.30 | 0.5187 | 22.03 | 0.77M |
| ESRGAN | 17.87 | 0.4929 | **30.26** | 5.91M |
| SwinIR | 19.04 | 0.5483 | 13.88 | 11.94M |
| **HAT** | **19.06** | **0.5505** | 14.07 | 20.81M |

**PSNR·SSIM 1위지만 값을 못 한다.** EDSR 대비 파라미터 13배에 PSNR 이득은 0.09 dB,
SwinIR 대비 1.7배에 0.02 dB 다. 학습(합성 LR)과 검증(실제 Sentinel-2)의 도메인
차이가 병목이라 모델 용량으로는 넘지 못한다 — 여섯 모델에서 일관되게 나타난다.

선명도는 여전히 최하위권이다. L1 만 쓰면 불확실한 고주파를 만드는 것보다 평균으로
뭉개는 쪽이 손실이 작기 때문이다.

## 메모리

Colab T4(15 GB)에서 학습 데모를 돌릴 때 참고. batch 1 기준 peak:

| LR 패치 | 메모리 |
|---|---|
| 64px | 3.13 GB |
| 96px | 7.06 GB |
| 128px | 12.21 GB |

노트북 데모는 64px 로 자른다. SwinIR(128px batch 1 = 6 GB)보다 두 배 무겁다.

`statistics/train_results.csv` 에 epoch 별 L1·PSNR·학습률 100행.
