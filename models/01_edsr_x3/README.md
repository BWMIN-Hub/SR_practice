# 01 · EDSR ×3 — Sentinel-2(10 m) → IKONOS(3.3333 m)

Sentinel-2 RGB를 ×3 업스케일하는 EDSR baseline(16 resblock / 64 feat, 1.5 M param).
업스트림 [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch)에 데이터셋 클래스와
전처리/추론/평가 스크립트를 얹은 것으로, `code/README_S2SR.md` 에 전체 실험 기록이 있다.

| | |
|---|---|
| 입력 | RGB 3채널 uint8, 10 m/px GeoTIFF 또는 PNG |
| 출력 | RGB 3채널 uint8, 3.3333 m/px. 입력의 CRS·지리범위를 그대로 보존한 GeoTIFF |
| 손실 | `1*L1` |
| 노트북 | `../../notebooks/01_edsr_x3.ipynb` |

## 데이터 규약

`HR = LR × 3`이 정확히 성립하고 두 영상의 지리 범위가 같다. 그래서 리샘플링 없이 바로 쌍이 된다.

**학습은 합성 LR로, 검증은 실제 Sentinel-2로 한다.** 이 비대칭이 이 프로젝트의 핵심이다.
`training/`의 LR은 HR을 3배 축소해 만든 것이라 HR과 방사 특성이 정확히 일치하지만,
`validation/`의 LR은 실제로 촬영된 S2라 센서·촬영시기가 다르다. 따라서 검증 PSNR에는
**합성↔실제 도메인 갭이 통째로 들어 있고**, 학습이 잘 돼도 쉽게 오르지 않는다.

## 학습 패치 품질 필터

원본 풀 804장 중 **683장**만 후보로 쓰고 거기서 40장을 골랐다. 제외 기준 3가지:

| 기준 | 제외 | 이유 |
|---|---|---|
| `AOI_Athens` 전체 | 100장 | 계열 전체가 과노출. HR 포화 평균 **10.97%**, 100장 중 90장이 5% 초과, 최대 37.65% |
| HR 포화 > 5% | 17장 | 잘린 화소는 복원할 정보가 없다. 게다가 HR 포화율이 g_LR보다 4~6%p 높아, 모델이 "밝은 곳은 255로 클리핑"을 학습하게 된다 |
| nodata 비율차 > 1%p | 4장 | 전부 `AOI_Munich_15` 계열. g_LR 에만 검은 영역이 2.6~4.2%p 더 많다 — g_LR 생성 시 nodata 처리 문제 |

Athens 를 뺀 나머지 5개 도시는 포화 평균이 0.38~2.93% 로 정상이다.
필터 적용 후 학습 40장의 HR 포화는 **평균 0.63% / 최대 3.83%** (적용 전 2.31% / 11.07%).

**`HR` 과 `g_LR` 사이의 방사 불일치는 없다.** g_LR 이 HR 에서 파생됐으니 당연한데,
실제로 재보면 밝기 오프셋이 +0.41 ± 1.37 DN 이고 히스토그램 거리도 이상치가 0장이다.
히스토그램 거리 0.14 수준은 불일치가 아니라 축소 시 평균화로 분포가 좁아지는 효과다.
화소 오차(MAE)가 큰 패치를 지우면 안 된다 — MAE 와 텍스처량의 상관이 **+0.63** 이라
MAE 상위는 어긋난 쌍이 아니라 디테일이 많은, 학습에 가장 필요한 패치다.

## 검증 패치

검증 10패치는 학습에 쓰지 않은 홀드아웃 씬에서 1장씩 잘라냈다. 원래 홀드아웃은 12씬이지만
`AOI_Athens_2`·`AOI_Athens_10`은 LR↔HR 상관이 0.28~0.31로 촬영 시기가 달라 내용 자체가
어긋난 쌍이라 제외했다(`code/README_S2SR.md` 참고).

## 체크포인트

| 파일 | 학습 내용 | 용도 |
|---|---|---|
| `edsr_ikonos_x3_best.pt` | dove 7씬 → ikonos 39씬 30 epoch | **fine-tune 출발점** (실습 데이터와 같은 도메인) |
| `edsr_ikonosfull_x3_latest.pt` | 위 + 도시 모자이크 10 epoch | 추론 품질이 가장 좋음. Incheon 씬 데모용 |

`edsr_ikonosfull_x3` 쪽은 검증 없이 학습해서 `model_best.pt`가 무의미하고
`model_latest.pt`를 써야 한다.

## 학습

`code/run_train.sh`는 전부 환경변수로 조절한다.

```bash
GPU=0 DATA=COLAB DIR_DATA=/content/colab/dataset \
EPOCHS=11 DECAY=5-8 LR=1e-4 TEST_EVERY=100 PRINT_EVERY=20 N_THREADS=2 \
SAVE=edsr_colab_x3 SAVE_RESULTS=0 RESET=1 \
PRETRAIN=/content/colab/models/01_edsr_x3/checkpoints/edsr_ikonos_x3_best.pt \
bash code/run_train.sh
```

| 변수 | 뜻 |
|---|---|
| `DATA=COLAB` | `code/src/data/colab.py` 의 `COLAB` 클래스 (`training/`·`validation/` 폴더명을 읽는다) |
| `DIR_DATA` | 데이터셋 루트. `training/`·`validation/`의 부모 |
| `EPOCHS` | **실제 학습은 `EPOCHS-1`회** |
| `TEST_EVERY` | epoch당 iteration. epoch 샘플수 = `batch(16) × TEST_EVERY` |
| `PRINT_EVERY` | loss 출력 주기. `TEST_EVERY`보다 크면 로그가 한 줄도 안 남는다 |
| `RESET=0` | `experiment/`를 지우지 않고 마지막 체크포인트에서 재개 |

## 추론·평가

```bash
# 추론: 폴더를 주면 '*.tif' 전부. 좌표 유지 GeoTIFF 출력
python code/infer.py --weight <가중치> --input <폴더 또는 .tif> --output results/

# 평가: PSNR/SSIM + 비교 그림. 씬 폴더(HR.tif/LR.tif)가 필요하다
python code/evaluate.py --weight <가중치> --src <씬 폴더 상위> --lr_name LR --radio_match
```

`evaluate.py`는 `{scene}/{HR,LR}.tif` 구조의 **GeoTIFF 씬**을 받는다. 이 번들의
`validation/`은 PNG 패치라 그대로는 못 쓴다 — 필요하면 원본 씬을 따로 넣어야 한다.

## 참고 성능

10 epoch fine-tune (RTX 3090에서 40초, T4는 3~5분 예상):

| | 값 |
|---|---|
| 학습 L1 loss | 18.25 → 17.37 (계속 하락) |
| 검증 PSNR | **best 16.05 dB @ epoch 1**, 이후 정체 |
| Incheon 추론 lap_std | Bicubic 8.82 → EDSR 12.26 (고주파 1.4배) |

**학습 loss는 내려가는데 검증 PSNR이 초반에 멈추는 게 정상이다.** 모델은 학습 도메인
(합성 LR)에서 계속 좋아지지만 실제 S2 입력은 따라오지 못한다. 남은 차이는 기하가 아니라
센서 특성·촬영시기·방사 차이라서, 더 올리려면 합성 LR을 실제 S2 열화에 가깝게 만들어야 한다.

## 함정

1. **`--chop`을 쓰면 터진다.** 업스트림 `forward_chop`의 `zip(*x_chops)`가 4D 텐서를 3D로
   떨어뜨린다 (`code/src/model/__init__.py:145`). 큰 영상은 `infer.py`의 타일링을 쓴다.
2. **데이터를 바꿔 넣으면 `dataset/bin/`을 지울 것.** PNG를 `.pt`로 캐싱하는데 파일명이
   같으면 옛 캐시를 재사용한다.
3. `MeanShift`가 DIV2K RGB 평균으로 하드코딩돼 있다(`code/src/model/common.py:15`).
   3채널 상수 오프셋이라 학습에는 지장 없지만 `--n_colors`를 바꾸면 반드시 고쳐야 한다.
