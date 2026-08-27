# 03 · ESRGAN ×3

[xinntao/ESRGAN](https://github.com/xinntao/ESRGAN) 구성을 우리 위성 데이터로 학습.

**업스트림 저장소에는 학습 코드가 없다**(`test.py` 와 `net_interp.py` 뿐).
그래서 논문 구성을 따라 `ESRGAN/train_esrgan.py` 를 새로 작성했다.

| | SRGAN | ESRGAN |
|---|---|---|
| 생성자 | SRResNet | **RRDB** (BatchNorm 없음) |
| 화소 손실 | MSE (가중치 1.0) | **L1** (가중치 0.01) |
| 지각 손실 | VGG16, 활성화 **이후** (0.006) | **VGG19 conv5_4, 활성화 이전** (1.0) |
| 적대적 손실 | 절대 판정 (0.001) | **RaGAN** 상대 판정 (0.005) |

| | |
|---|---|
| 입력 | 데이터셋이 제공하는 Sentinel-2 LR (10 m) |
| 목표 | IKONOS HR (3.3333 m), LR × 3 |
| 학습 | IKONOS 804쌍, 100 epoch, batch 16, crop 96, RRDB 8블록 |
| 가중치 | `checkpoints/esrgan_g_x3.pth` (epoch 58) |
| 노트북 | [`../../notebooks/03_esrgan_x3.ipynb`](../../notebooks/03_esrgan_x3.ipynb) |

## ×3 을 위해 고친 곳

업스트림 RRDBNet 은 `F.interpolate(scale_factor=2)` 를 두 번 해서 ×4 고정이다.
배율을 인자로 받아 ×3 이면 한 번만 하도록 바꿨다. 계층 이름은 그대로라
×4 사전학습 체크포인트도 그대로 실린다.

## 판별자가 무너지지 않는다

SRGAN 은 epoch 37 에서 `D(x)`·`D(G(z))` 가 함께 1 로 붙어 적대적 신호가 사라졌다.
ESRGAN 은 100 epoch 내내 유지된다.

| | `D(x)` | `D(G(z))` |
|---|---|---|
| 범위 | 0.409 ~ 0.957 | 0.002 ~ 0.252 |

RaGAN 이 "진짜인가" 대신 **"평균적인 가짜보다 더 진짜 같은가"** 를 묻기 때문이다.
판별자가 한쪽으로 쏠려도 상대 비교라 기울기가 남는다.

## 성능 (검증 10패치)

| | PSNR | SSIM |
|---|---|---|
| Bicubic | **18.15** | **0.4805** |
| SRGAN | 18.30 | 0.5187 |
| ESRGAN | 16.55 | 0.4208 |

**PSNR 이 bicubic 보다 낮은 것은 고장이 아니라 설계 결과다.**
화소 손실 가중치가 0.01 이고 지각 손실이 1.0 이라, 화소를 맞추는 대신 그럴듯한
질감을 만드는 쪽으로 학습된다. 학습 로그에서도 지각 손실이 전체의 97% 를 차지한다.

원논문은 PSNR 지향 모델을 먼저 학습한 뒤 GAN 으로 미세조정하는데, 여기서는 처음부터
GAN 으로 돌렸다. 그 차이도 있다.

`statistics/train_results.csv` 에 epoch 별 지표 100행,
`ESRGAN/statistics/esrgan_x3_batch.csv` 에 배치별 기록.
