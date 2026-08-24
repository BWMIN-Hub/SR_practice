# Sentinel-2 → Planet ×3 Super-Resolution (EDSR 실습)

Sentinel-2 RGB(10m)를 Planet 해상도(3.33m)로 ×3 업스케일하는 EDSR 실습 세트.
업스트림 [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch)에 데이터셋
클래스와 전처리/추론/평가 스크립트만 얹었다.

## 데이터

| 용도 | 입력 | 정답(GT) | 씬 수 |
|---|---|---|---|
| 학습 | `SR_dataset/*/g_LR.tif` (HR에서 만든 합성 LR) | `HR.tif` | 7 |
| 검증 | `SR_dataset/*/LR.tif` (실제 Sentinel-2) | `HR.tif` | 3 |
| 테스트 | `SR_testdataset/*.tif` (실제 Sentinel-2) | 없음 | 7 |

- 3채널 uint8, 7개 씬 모두 HR/LR의 지리 범위가 동일하고 `HR = LR × 3`이 정확히 성립.
- 학습은 합성 LR로, 검증은 실제 S2 LR로 한다. 즉 **검증 PSNR = 합성↔실제 도메인 갭까지
  포함한 수치**다. 검증 씬은 학습 씬에도 들어가 있으므로(입력만 다름) 일반화 성능이 아니라
  "실제 S2에 얼마나 통하는지"를 보는 지표로 읽어야 한다.
- 테스트셋은 GT가 없어 정량 지표를 낼 수 없고 육안 확인용이다.

## 실행

```bash
conda activate nst-disaster        # torch 2.2.0+cu118

# 1) GeoTIFF -> DIV2K식 PNG 폴더 (학습은 128px LR 타일, 검증은 전체 씬)
python prepare_data.py

# 2) 학습 (EDSR baseline, x3, 100 epoch)
GPU=0 bash run_train.sh
#    로그/체크포인트: experiment/edsr_s2_x3/

# 3) 검증셋 정량 평가 + Bicubic 대비 비교 그림
python evaluate.py --weight experiment/edsr_s2_x3/model/model_best.pt
#    -> results/val/metrics.csv, {scene}_compare.png

# 4) 테스트셋 추론 -> GeoTIFF + PNG
python infer.py --weight experiment/edsr_s2_x3/model/model_best.pt
#    -> results/test/{name}_SRx3.tif
```

## 결과

두 번 학습했다. 경계 9px를 잘라내고 측정.

| run | 데이터 | 설정 | best |
|---|---|---|---|
| run1 | 초기 `SR_dataset` | scratch, 99 epoch, decay 50-80 | epoch 80 |
| run2 | HR 기하 재정합 후 | run1 가중치에서 fine-tune, 49 epoch, decay 25-40 | epoch 27 |

run1 산출물은 `pretrained/edsr_s2_x3_run1_best.pt`, `experiment/edsr_s2_x3_run1/`,
`results_run1/` 에 보존.

### 검증 — 실제 Sentinel-2 LR 입력 (3개 씬), run2

| 씬 | Bicubic | EDSR | 이득 |
|---|---|---|---|
| Carcajou | 19.359 / 0.6021 | 20.921 / 0.6604 | +1.56 dB |
| Chilanko_Forks | 20.558 / 0.6497 | 22.848 / 0.8049 | +2.29 dB |
| Kobuk_River | 14.534 / 0.5998 | 14.733 / 0.6396 | +0.20 dB |
| **평균** | 18.150 / 0.6172 | **19.501 / 0.7016** | **+1.35 dB** |

### run1 vs run2 — 재정합 효과

| | run1 | run2 | 변화 |
|---|---|---|---|
| EDSR (실제 LR) | 18.997 / 0.6434 | **19.501 / 0.7016** | +0.50 dB / +0.058 |
| Bicubic (실제 LR) | 17.682 / 0.5493 | 18.150 / 0.6172 | +0.47 dB |
| EDSR (`g_LR`) | 27.513 / 0.7424 | **29.088 / 0.7983** | +1.58 dB |
| 학습 L1 loss | 7.70 | 6.47 | −1.23 |

**bicubic 기준선도 +0.47 dB 올랐다**는 점이 핵심이다. 모델과 무관한 단순 보간까지
좋아졌으니 이 향상은 학습이 잘 돼서가 아니라 HR 기하 정합이 개선된 결과다.
정합이 어긋나 있으면 어떤 방법으로도 맞출 수 없는 오차가 남는다.

### 도메인 갭 — 합성 LR vs 실제 LR

| 입력 | run1 | run2 |
|---|---|---|
| `g_LR` (합성, 학습 도메인, 7개 씬) | 27.513 | 29.088 |
| `LR` (실제 S2, 3개 씬) | 18.997 | 19.501 |
| 차이 | −8.5 dB | **−9.6 dB** |

재정합으로 갭이 오히려 **벌어졌다**. 합성 LR↔HR 관계는 훨씬 깨끗해졌지만 실제 S2
입력은 그만큼 따라오지 못했다. 즉 남은 차이는 기하 문제가 아니라 센서 특성·촬영 시기·
방사 차이다. 여기서 더 올리려면 합성 LR을 실제 S2 열화에 가깝게 만들어야 한다.

`g_LR` 평가 7개 씬은 **전부 학습에 쓴 씬**이므로 학습셋 성능이지 일반화 성능이 아니다.
의미 있는 건 절대값이 아니라 두 도메인 간 차이다.

### 선명도 (Carcajou, run1 기준, 라플라시안 표준편차 = 고주파 양)

| | lap_std |
|---|---|
| HR (실제 Planet) | 14.53 |
| Bicubic x3 | 7.80 |
| EDSR x3 | **4.44** |

PSNR/SSIM은 이겼지만 출력은 bicubic보다도 부드럽다. L1 loss는 불확실한 고주파를
만드는 것보다 평균으로 뭉개는 쪽이 손실이 작기 때문이고, 여기선 LR/HR이 센서·촬영일이
다른 실제 영상 쌍이라 대응 안 되는 디테일이 많아 그 경향이 더 강하다.
비교 그림(`results/val/*_compare.png`)을 보면 EDSR이 실제로 하는 일은
**디테일 복원보다 S2 컬러 노이즈 제거 + 디블러**에 가깝다.
검증 PSNR이 18.9에서 정체한 것도 같은 원인.

### 산출물 (전부 좌표 붙은 GeoTIFF)

```
results/
├── val/                                   # 검증 3개 씬(LR) + 7개 씬(g_LR)
│   ├── metrics_LR.csv, metrics_g_LR.csv
│   ├── {scene}_{LR|g_LR}_SRx3.tif         # EDSR 결과
│   ├── {scene}_{LR|g_LR}_bicubicx3.tif    # 비교용 bicubic
│   └── {scene}_{LR|g_LR}_compare.png      # 확대 3분할 비교 그림
└── test/                                  # SR_testdataset 7개 씬
    └── {name}_SRx3.tif
```

모두 입력의 CRS·지리범위를 그대로 두고 픽셀 크기만 10 m → 3.3333 m로 기록하므로
QGIS에서 원본 `HR.tif`/`LR.tif` 위에 바로 겹쳐볼 수 있다 (좌표 검증 7/7 통과).
전체 씬 PNG가 필요하면 `--png` 옵션. 테스트셋은 GT가 없어 정량 지표는 없다.

### 더 선명하게 하려면

- `--loss 1*L1+0.05*VGG54` 또는 GAN 항 추가 (PSNR은 내려가고 체감 선명도는 올라감)
- 합성 LR을 실제 S2 열화(블러+노이즈+방사 차이)에 가깝게 만들어 도메인 갭 축소
- 씬 간 방사 정규화(특히 Kobuk_River)로 대응 안 되는 쌍의 영향 제거

## ikonos 데이터셋 (2026-08-20 추가)

`SR_dataset/` 아래가 센서별로 나뉘었다: `dove/`(기존 7씬), `ikonos/`(신규 51씬).
dove는 더 쓰지 않고, dove 학습 체크포인트를 pretrained weight 로 써서 ikonos 를 30 epoch
추가 학습했다.

### 데이터

| 항목 | 값 |
|---|---|
| 씬 수 | 51 (Athens 7, Barcelona 6, Busan 2, Munich 18, Paris 9, Seoul 9) |
| 규약 | dove 와 동일. `HR = LR x 3`, 같은 bounds, 3ch uint8, HR 3.33m / LR 10m |
| 학습 | 39 씬 -> 128px LR 타일 804장 (입력 `g_LR`) |
| 검증 | 12 씬 (학습에서 제외, 입력 `LR` = 실제 S2) |

씬들은 도시별 모자이크를 격자로 자른 **인접하되 겹치지 않는** 타일이라 씬 단위 holdout 이
공간적으로 유효하다. dove 때는 검증 씬이 학습 씬과 같아 일반화를 못 봤지만 여기선 본다.
(다만 같은 도시 타일은 촬영일·방사 특성이 같으므로 완전히 독립적이진 않다.)

검증 씬 목록은 `val_scenes_ikonos.txt`.

### 실행

```bash
python prepare_data.py --src ../SR_dataset/ikonos --out ../sr_data/IKONOS \
    --tile 128 --stride 128 --val_scenes "$(paste -sd, val_scenes_ikonos.txt)"

GPU=0 DATA=IKONOS EPOCHS=31 DECAY=15-25 LR=1e-4 SAVE=edsr_ikonos_x3 SAVE_RESULTS=0 \
    PRETRAIN=$PWD/experiment/edsr_s2_x3_run2/model/model_best.pt bash run_train.sh

python evaluate.py --weight experiment/edsr_ikonos_x3/model/model_best.pt \
    --src ../SR_dataset/ikonos --scenes val_scenes_ikonos.txt \
    --lr_name LR --radio_match --output results/ikonos_val
```

`EPOCHS=31` 인 이유는 아래 함정 1번(실제 학습은 30회).

### 결과 — 실제 S2 LR 입력, 12개 홀드아웃 씬

| 모델 | PSNR | SSIM | vs bicubic |
|---|---|---|---|
| Bicubic | 14.288 | 0.4035 | — |
| dove 모델 zero-shot | 14.110 | 0.3796 | **-0.178** |
| **ikonos 30ep fine-tune** | **14.860** | **0.4342** | **+0.572** |

- dove(캐나다 산불, Planet)에서 배운 것은 ikonos(도시)로 **전이되지 않는다**. zero-shot 은
  bicubic 보다도 나쁘다. 30 epoch 추가 학습이 이를 뒤집었다.
- 학습 loss 는 30 epoch 내내 내려갔지만(16.94 -> 16.36) 검증 PSNR 은 **epoch 7 에서 best
  (14.860) 를 찍고 이후 23 epoch 동안 정체**했다. 학습 도메인(`g_LR`)에서만 계속 좋아졌다는 뜻.

### 왜 정체하는가 — LR/HR 쌍 자체의 불일치

| 씬 | LR-HR 평균 밝기차 | LR↔HR 상관 | bicubic PSNR |
|---|---|---|---|
| AOI_Athens_10 | **-45.4** | 0.31 | 8.64 |
| AOI_Athens_2 | **+32.2** | 0.28 | 8.68 |
| AOI_Munich_16 | +7.8 | 0.78 | 18.93 |
| 나머지 | +3 ~ +14 | 0.45~0.65 | 13~16 |

`g_LR` 평균은 HR 과 소수점까지 일치한다(HR 에서 만들었으니 당연). 실제 `LR` 은 씬마다
밝기 오프셋이 크고, Athens 두 씬은 상관 0.3 수준이라 촬영 시기가 달라 내용 자체가 어긋난
쌍이다. PSNR 8.6dB 는 SR 실패가 아니라 쌍이 안 맞아서 나오는 값이다.

### 방사 정규화 후 (`--radio_match`)

예측을 GT 에 채널별 선형(gain/offset)으로 맞춘 뒤 측정. bicubic·EDSR 에 동일 적용하므로
둘 사이 비교는 공정하지만 GT 통계를 쓰므로 oracle 진단용이다.

| | 원본 | 방사 정규화 후 |
|---|---|---|
| Bicubic | 14.288 | 15.872 |
| EDSR | 14.860 | 16.011 |
| 이득 | **+0.572** | **+0.139** |

**+0.572dB 중 대부분이 밝기 보정이고 순수 디테일 이득은 +0.14dB 다.** 모델이 실제로 배운
것의 상당 부분은 "S2 는 IKONOS 보다 밝다"는 씬 무관 오프셋 보정이다.

### 합성 `g_LR` 입력 (같은 12개 홀드아웃 씬 = 학습 도메인, 미학습 씬)

| | PSNR | SSIM |
|---|---|---|
| Bicubic | 19.100 | 0.5587 |
| EDSR | **19.900** | **0.5976** |
| 이득 | +0.800 | +0.039 |

도메인 갭은 19.900 - 14.860 = **-5.0dB** (dove 는 -9.6dB). ikonos 는 갭 자체는 작지만
합성 도메인에서의 이득도 +0.8dB 로 작다. dove 의 +3.9dB 와 달리 도시 영상은 ×3 로 복원할
수 없는 고주파(건물 경계, 도로 표시)가 많아 L1 로는 여기까지다.

### 선명도 (라플라시안 std)

| 씬 | HR | Bicubic | EDSR(실제 LR) |
|---|---|---|---|
| AOI_Munich_16 | 23.81 | 3.76 | **8.18** |
| AOI_Seoul_1 | 46.99 | 8.16 | **11.44** |

dove 에서는 EDSR 출력이 bicubic 보다도 뭉개졌는데(4.44 vs 7.80), ikonos 에서는 bicubic 의
**2배 이상 선명**하다. 도시 영상은 대응되는 구조(건물, 도로)가 뚜렷해 L1 로도 고주파를
만들 근거가 있기 때문. 다만 HR(23.8/47.0)에는 한참 못 미친다.

### 산출물

```
results/ikonos_val/                     # ikonos 30ep 모델, 12개 홀드아웃 씬
├── metrics_LR.csv, metrics_g_LR.csv
├── {scene}_{LR|g_LR}_SRx3.tif          # EDSR 결과 (좌표 검증 24/24 통과)
├── {scene}_{LR|g_LR}_bicubicx3.tif
└── {scene}_{LR|g_LR}_compare.png
results/ikonos_val_dove_baseline/       # dove 모델 zero-shot (비교 기준)
experiment/edsr_ikonos_x3/              # 학습 로그 + 체크포인트
```

`SR_testdataset/` 은 dove 시절 S2 산불 씬 그대로라 ikonos 모델을 적용할 대상이 아니다
(도시 학습 모델 ↔ 산불 지역 입력). ikonos 테스트셋은 아직 없다.

### 더 올리려면

1. **씬별 방사 정규화를 전처리에 넣기.** 이득의 3/4 이 밝기 보정인데, 이걸 모델이
   학습으로 때우고 있다. 입력 LR 을 HR 통계에 맞춰 정규화하면 모델은 디테일에만 집중한다.
2. **Athens 두 씬은 빼는 게 낫다.** 상관 0.3 은 학습·평가 양쪽에서 노이즈다.
3. **합성 `g_LR` 을 실제 S2 열화에 가깝게.** 현재 `g_LR` 은 HR 과 방사가 동일해 실제 LR 과
   너무 다르다. 블러+노이즈+밝기 오프셋을 랜덤하게 넣어 만들면 도메인 갭이 준다.

## training_ikonos — 도시 모자이크로 10 epoch 추가 전이학습

`SR_dataset/training_ikonos/` 로 받은 새 학습셋. 기존 `ikonos/` 타일셋의 문제(실제 S2 LR 과
IKONOS HR 사이의 방사·촬영시기 불일치)가 없는 깨끗한 쌍이다.

### 데이터

도시별 모자이크 1쌍씩, 6개 도시.

| | 파일 | 해상도 |
|---|---|---|
| LR | `HR_ikonos_{date}_8bit_lr_s0_dcall-hr.tif` | 10 m |
| HR | `AOI_{city}/HR_ikonos_{date}_8bit.tif` | 3.3333 m |

`HR = LR x 3` 이 정확히 성립하고 bounds 도 동일해 리샘플이 필요 없다.

| 도시 | LR 크기 | 평균 밝기차 | 상관 | 정합 shift |
|---|---|---|---|---|
| Athens | 1239x3147 | 0.8 | 0.930 | 0.02 px |
| Barcelona | 1219x3304 | 0.4 | 0.957 | 0.51 px |
| Busan | 1132x2837 | 0.3 | 0.922 | 0.02 px |
| Munich | 1303x4577 | 0.2 | 0.917 | 0.01 px |
| Paris_1 | 1165x1313 | 0.4 | 0.832 | 0.06 px |
| Seoul | 1228x3262 | 0.3 | 0.889 | 0.48 px |

기존 `ikonos/` 의 실제 LR 은 밝기차 ±30~45, 상관 0.28~0.65 였다. **여기서는 밝기차가 1 DN
미만이고 상관도 0.83~0.96** 이라 앞서 지적한 두 문제가 모두 사라졌다.

학습 타일 **1040장** (128px LR / 384px HR). HR 모자이크 가장자리 nodata 가 1~12% 있어
검은 픽셀이 하나라도 든 타일 443개는 버렸다.

### 실행

```bash
python prepare_training_ikonos.py          # -> sr_data/IKONOSFULL

GPU=0 DATA=IKONOSFULL EPOCHS=11 DECAY=5-8 LR=1e-4 SAVE=edsr_ikonosfull_x3 SAVE_RESULTS=0 \
    PRETRAIN=$PWD/experiment/edsr_ikonos_x3/model/model_best.pt bash run_train.sh

python infer.py --weight experiment/edsr_ikonosfull_x3/model/model_latest.pt \
    --input ../SR_testdataset/S2_2024-05-16_..._Incheon_RGB_8bit.tif \
    --output results/ikonosfull_test
```

**검증을 하지 않는 설정이다.** EDSR 메인 루프가 매 epoch `test()` 를 부르므로 `val/` 에
더미 타일 1장만 넣어 루프를 돌렸고 그 PSNR 은 의미가 없다. 따라서 최종 가중치는
`model_best.pt` 가 아니라 **`model_latest.pt`** 를 써야 한다.

### 결과

10 epoch, 학습 L1 loss 16.49 -> **15.92**.

정량 지표는 없다(검증 없음, 테스트셋에 GT 없음). 대신 Incheon 씬으로 추론해 비교했다.

### Incheon 추론 (`S2_2024-05-16_..._T52SBG_Incheon_RGB_8bit.tif`)

S2 10 m, 2001x2001, EPSG:32652 -> **6003x6003, 3.3333 m** (좌표 검증 통과, bounds 동일).
서울 학습 모자이크 바로 서쪽 같은 UTM 존이라 지금까지 중 도메인이 가장 잘 맞는 입력.

| | 라플라시안 std | 평균 밝기 (RGB) |
|---|---|---|
| Bicubic x3 | 8.82 | 93.7 / 96.4 / 72.3 |
| EDSR ikonos-30ep (이전) | 12.16 | 92.3 / 93.4 / 70.2 |
| **EDSR +training_ikonos (10ep)** | **13.10** | 94.4 / 94.4 / 71.2 |
| (참고) 입력 LR 10 m | 61.29 | 93.8 / 96.4 / 72.3 |

- bicubic 대비 고주파가 **1.5배**, 이전 모델보다도 8% 많다.
- 이전 모델은 입력 대비 밝기가 1.5~3 DN 어두워지는 색 편이가 있었는데(`ikonos/` 의 방사
  불일치를 학습으로 때운 흔적) 이번 모델은 입력 밝기를 더 잘 보존한다.
- 비교 그림 `results/ikonosfull_test/incheon_compare.png` 에서 건물 경계와 부두 구조물이
  이전 모델보다 뚜렷하고, 이전 모델에서 보이던 전반적인 색 왜곡이 줄었다.
- 라플라시안 std 는 같은 격자끼리만 비교해야 한다. 입력 LR 의 61.29 는 10 m 격자라
  픽셀 간 계단이 커서 나온 값이므로 x3 결과들과 직접 비교하면 안 된다.

### 산출물

```
results/ikonosfull_test/
├── {incheon}_SRx3.tif              86 MB, 6003x6003 @3.3333m  <- 이번 모델
├── {incheon}_bicubicx3.tif         91 MB, 비교용
├── incheon_compare.png             Bicubic | 이전 모델 | 이번 모델
└── prev_ikonos30ep/{incheon}_SRx3.tif   이전 모델 결과(비교용)
experiment/edsr_ikonosfull_x3/      학습 로그 + model_latest.pt
```

## 추가/수정한 파일

| 파일 | 역할 |
|---|---|
| `prepare_data.py` | GeoTIFF → `sr_data/{name}/{train,val}/{HR,LR_bicubic/X3}/*.png`. `--val_scenes` 로 씬 단위 holdout |
| `src/data/s2sr.py` | dove 데이터셋 클래스. train/val이 같은 HR에 다른 LR 소스를 쓴다 |
| `src/data/ikonos.py` | ikonos 데이터셋 클래스. 씬 단위로 train/val 분리 |
| `prepare_training_ikonos.py` | training_ikonos 모자이크 → `sr_data/IKONOSFULL`. nodata 타일 제거 |
| `src/data/ikonosfull.py` | training_ikonos 데이터셋 클래스 (검증 없음, val은 더미 1장) |
| `val_scenes_ikonos.txt` | ikonos 검증 전용 12개 씬 목록 |
| `run_train.sh` | 학습 실행 스크립트. `DATA=` 로 데이터셋, `SAVE_RESULTS=0` 로 PNG 저장 끄기 |
| `evaluate.py` | 검증셋 PSNR/SSIM (Bicubic vs EDSR) + 비교 그림. `--scenes`, `--radio_match` |
| `infer.py` | 겹침 타일링 전체 씬 추론 → 지오리퍼런스 유지 GeoTIFF |

업스트림 `src/` 코드는 건드리지 않았다.

## 알아둘 함정

1. **`--epochs 1`은 아무것도 학습하지 않는다.** `Trainer.terminate()`가
   `epoch(=1) >= args.epochs(=1)`로 즉시 True가 된다 (`src/trainer.py:153`).
   최소 2 이상을 줘야 한다.

2. **`--chop`을 쓰면 터진다.** `forward_chop`의 재귀 분기에서
   `for p in zip(*x_chops)`가 4D 텐서를 3D로 떨어뜨려
   `expected input to have 3 channels, but got 1` 에러가 난다
   (`src/model/__init__.py:145`, 업스트림 버그). 1000×1000 정도는 통째로 들어가므로
   `--chop` 없이 쓰고, 더 큰 영상은 `infer.py`의 타일링을 쓴다.

3. **학습셋은 반드시 타일로 잘라둘 것.** `SRData.__getitem__`은 패치 하나를 뽑을 때마다
   이미지 **전체**를 unpickle한다. 3000×3000 HR을 통째로 두면 샘플당 30MB를 읽어
   epoch당 수백 GB를 역직렬화하게 되고 GPU 사용률이 바닥을 친다.
   (전체 씬 7장: 90초/epoch → 128px 타일 504장: 훨씬 빠름)

4. **`MeanShift`는 DIV2K RGB 평균으로 고정**되어 있다 (`src/model/common.py:15`).
   위성영상 통계와는 다르지만 입출력에서 더하고 빼는 상수 오프셋이라 학습에는 지장이 없다.
   채널 수를 3에서 바꿀 경우엔 3채널 하드코딩이라 반드시 수정해야 한다.

5. **데이터를 다시 만들면 `sr_data/S2SR/bin/`을 지울 것.** PNG를 `.pt`로 캐싱하는데
   파일명이 같으면 옛 캐시를 그대로 재사용한다.
