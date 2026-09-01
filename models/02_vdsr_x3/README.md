# 02 · VDSR ×3

[Lornatang/VDSR-PyTorch](https://github.com/Lornatang/VDSR-PyTorch) 를 우리 위성 데이터로 학습.
신경망 정의(`model.py`)와 이미지 처리(`imgproc.py`)는 업스트림 사본을 **한 글자도 고치지 않고**
`lib/vdsr_arch.py` · `lib/vdsr_imgproc.py` 로 가져왔다.

2016년 논문. SRCNN(2014)의 다음 세대이고, **깊이와 잔차 학습**을 들여왔다.

| | |
|---|---|
| 구조 | conv 3×3 20층, 채널 1-64-…-64-1, 마지막에 입력을 더하는 잔차 연결 |
| 손실 | **MSE** (업스트림 그대로) |
| 입력 | 실제 Sentinel-2 LR (10 m) 을 **bicubic 으로 3배 키운 것**의 Y 채널 |
| 목표 | IKONOS HR (3.3333 m) 의 Y 채널 |
| 학습 | IKONOS 804쌍, **30 epoch**, batch 16, LR 크롭 48, SGD 0.1 + 기울기 자르기 0.01, StepLR(12, 0.1) |
| 가중치 | `checkpoints/vdsr_x3.pth` (epoch 30, 0.66M) |
| 노트북 | [`../../notebooks/02_vdsr_x3.ipynb`](../../notebooks/02_vdsr_x3.ipynb) |

추가 설치가 필요 없다. Colab 기본 패키지로 돈다.

## SRCNN 과 같은 규약, 다른 두 가지

**같은 것** — 모델 안에 업샘플이 없어 LR 을 미리 bicubic 으로 3배 키워 넣는다.
Y(밝기) 채널 하나만 보고 색은 확대본을 그대로 쓴다. 그래서 데이터 어댑터
(`VDSR/sr_dataset_vdsr.py`)는 SRCNN 것과 규약이 같다.

**다른 것** — conv 20층이고 **잔차 학습**이다:

```python
out = self.conv1(x); out = self.trunk(out); out = self.conv2(out)
out = torch.add(out, identity)      # 입력을 더한다
```

## 업스트림 최적화기를 그대로 쓸 수 있었다

SRCNN 은 업스트림 SGD 1e-4 로 100 epoch 을 돌려도 bicubic 보다 6.6 dB 나빴다.
출력이 0 에서 시작해 전체 매핑을 배워야 하기 때문이다. VDSR 은 다르다.

| 3 epoch 후 검증 PSNR(Y) | |
|---|---|
| Bicubic (기준) | 15.49 |
| **SGD 0.1 + 클리핑 (업스트림)** | **15.68** |
| Adam 2e-4 | 15.69 |

**첫 epoch 부터 bicubic 을 넘는다.** 잔차 학습이라 출력이 bicubic 에서 출발하고,
높은 학습률을 기울기 자르기가 잡아준다 — 논문의 핵심 주장 그대로다.
Adam 과 차이가 0.01 이라 업스트림 설정을 그대로 썼다.

## 50 epoch 을 계획했지만 30 epoch 에서 멈췄다

학습률이 12 epoch 마다 1/10 로 떨어져 epoch 24 이후 0.001 이었고, 지표가
평평해져 더 돌릴 이유가 없었다. 다른 모델과 로그 길이를 맞추려 30 에서 끊었다.

| epoch | MSE | 검증 PSNR(Y) | lr |
|---|---|---|---|
| 1 | 0.040549 | 15.60 | 0.1 |
| 12 | 0.009198 | 15.51 | 0.01 |
| 24 | 0.009303 | 15.73 | 0.001 |
| 30 | 0.009367 | 15.65 | 0.001 |

MSE 는 epoch 1 이후 거의 평평하다.

체크포인트는 공유 검증 10패치로 다시 재서 골랐다. 학습 중 최고는 epoch 9 지만
공유 기준으로는 마지막이 낫다:

| 체크포인트 | PSNR | SSIM |
|---|---|---|
| epoch 9 (best) | 18.27 | 0.5021 |
| **epoch 30 (latest)** | **18.35** | **0.5081** |

## 성능 (검증 10패치)

| | PSNR | SSIM |
|---|---|---|
| Bicubic | 18.15 | 0.4805 |
| SRCNN (57.3K) | 18.40 | 0.5206 |
| **VDSR (0.66M)** | **18.35** | **0.5081** |

**파라미터를 11배 키웠는데 SRCNN 보다 낮다.** 학습 예산이 적은(30 대 100 epoch)
탓도 있지만, 이 데이터에서 깊이만으로는 얻는 게 적다는 신호이기도 하다.
학습(합성 g_LR)과 검증 사이 도메인 차이가 병목이라는 진단과 일치한다.

`statistics/train_results.csv` 에 epoch 별 MSE·PSNR·학습률 30행.
