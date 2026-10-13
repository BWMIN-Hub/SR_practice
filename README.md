# Colab 실습 번들

위성사진 초해상화(SR) 실습. 노트북 전부 **구글 드라이브·git clone·pip install 없이**
필요한 파일만 내려받아 돌아간다.

| 노트북 | 내용 | GPU |
|---|---|---|
| [`00_interpolation_basics`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/00_interpolation_basics.ipynb) | 보간법 4종 ×2·×3·×4 baseline | 불필요 |
| [`01_srcnn_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/01_srcnn_x3.ipynb) | SRCNN — 최초의 CNN, conv 3장 | 선택 |
| [`02_vdsr_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/02_vdsr_x3.ipynb) | VDSR — conv 20장 + 잔차 학습 | 선택 |
| [`03_edsr_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/03_edsr_x3.ipynb) | EDSR — 깊은 잔차 신경망 | 권장 |
| [`04_srgan_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/04_srgan_x3.ipynb) | SRGAN — 판별자 붕괴 | 선택 |
| [`05_esrgan_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/05_esrgan_x3.ipynb) | ESRGAN — RRDB + RaGAN, 손실 가중치 | 선택 |
| [`06_swinir_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/06_swinir_x3.ipynb) | SwinIR — Transformer, GAN 없음 | 선택 |
| [`07_hat_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/07_hat_x3.ipynb) | HAT — Transformer 확장 | 선택 |
| [`09_comparison`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/09_comparison.ipynb) | 일곱 모델 한자리 비교 (GPU 불필요) | 불필요 |
| [`10_pair_strategies`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/10_pair_strategies.ipynb) | LR/HR 쌍 구축 전략 4가지 비교 (GPU 불필요) | 불필요 |
| [`12_stretch_windows`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/12_stretch_windows.ipynb) | 12-bit → 8-bit 변환 창 고르기 4가지 경우 (GPU 불필요) | 불필요 |
| [`13_intake_check`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/13_intake_check.ipynb) | 반입 점검 3단계 — 헤더 · 통계 · 축소본 (GPU 불필요) | 불필요 |
| [`14_inference`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/14_inference.ipynb) | EDSR ×2 추론 — 타일 · 겹침 · 배치 · 정밀도 · OOM · 크기 불일치 | 필요 |
| [`15_sr_output_check`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/15_sr_output_check.ipynb) | SR 결과 검수 3단계 — 구조 · 기하 · 방사 (GPU 불필요) | 불필요 |
| [`16_read_properly`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/16_read_properly.ipynb) | 영상 제대로 읽기 — 읽기 · 자료형 · 크기 (GPU 불필요) | 불필요 |
| [`17_image_content`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/17_image_content.ipynb) | 영상 들여다보기 — 분포 · 밴드 구성 · 구름과 선명도 (GPU 불필요) | 불필요 |
| [`18_preflight`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/18_preflight.ipynb) | 실행 전 점검 5가지 — GPU · 입력 변환 · 패치 · 정밀도 · 스크립트 | 필요 |

번호가 곧 배우는 순서이자 모델이 나온 순서다 (SRCNN 2014 → HAT 2023).

모델 페이지는 전부 같은 순서다: **데이터 → 결과 → 최종 테스트(인천)**.
학습 코드는 노트북에 두지 않는다 — 가중치는 서버에서 전체 데이터로 미리 학습한
것을 받아 쓴다. 이론 설명도 노트북에 넣지 않는다.

보여주는 패치는 고정했다 — 결과는 실제 Sentinel-2 3패치, 최종 테스트는
**Incheon**(실제 촬영본, 비교할 IKONOS 영상 없음).

모델·손실·유틸은 `lib/` 에 있고 노트북은 import 만 한다.

| 파일 | 내용 |
|---|---|
| `lib/sr_models.py` | `EDSR`, `load_edsr()` |
| `lib/sr_losses.py` | `get_loss('l1' / 'l2' / 'charbonnier')` |
| `lib/sr_utils.py` | `pair` `load_test` `show` `zoom` `score` `compare` `bicubic` |

## 가중치

모델 페이지 7개는 각각 가중치를 두 벌 갖고 있다. 학습 조건이 다른 두 번의 학습
결과이고, 기본값은 `_v3` 쪽이다. 노트북의 `CKPT` 줄 하나로 바꾼다.

| 모델 | 기본 | 이전 |
|---|---|---|
| 01 SRCNN | `srcnn_x3_v3.pth` | `srcnn_x3.pth` |
| 02 VDSR | `vdsr_x3_v3.pth` | `vdsr_x3_30ep.pth` |
| 03 EDSR | `edsr_x3_v3.pt` | `edsr_x3.pt` |
| 04 SRGAN | `srgan_g_x3_v3.pth` | `srgan_g_x3.pth` |
| 05 ESRGAN | `esrgan_g_x3_v3.pth` | `esrgan_g_x3.pth` |
| 06 SwinIR | `swinir_x3_v3.pth` | `swinir_x3.pth` |
| 07 HAT | `hat_x3_v3.pth` | `hat_x3.pth` |

`_v3` 쪽이 더 선명하고, IKONOS 를 기준으로 잰 PSNR 은 조금 낮다. 숫자는 각 모델
폴더의 `README.md` 에 있다.

## 보여주는 패치와 점수 재는 패치가 다르다

**보여줄 때**는 `dataset/validation_real/` 3패치를 쓴다. 입력이 **실제
Sentinel-2 관측본**이라 배포했을 때 실제로 보게 될 그림이다.

**점수를 잴 때**는 `dataset/validation/` 10패치를 쓴다. 입력이 IKONOS 에서 만든
합성본이라 IKONOS 와 화소가 정확히 맞는다. 실제 Sentinel-2 는 IKONOS 와 촬영
날짜가 달라 PSNR 이 모델 차이보다 날짜 차이를 더 크게 반영한다.

둘의 입력 종류가 다르니 섞지 않는다. `show_data()` 와 `show_results()` 가
앞의 것을, `compare()` 가 뒤의 것을 쓴다.

## 폴더 구성

```
colab/
├── dataset/       실습에 쓰는 사진들
│   ├── training/     연습문제 40쌍 (흐린 사진 + 같은 자리의 IKONOS 사진)
│   ├── validation/   점수용 10쌍 (배울 때 안 본 지역, 입력은 합성본)
│   ├── validation_real/ 결과를 보여줄 3쌍 (입력이 실제 Sentinel-2)
│   └── test/         최종 테스트용 인천 사진 1장 (비교할 IKONOS 영상 없음)
├── models/
│   └── 03_edsr_x3/   모델 하나가 폴더 하나
│       ├── README.md     이 모델 설명
│       ├── code/         돌아가는 데 필요한 코드 전부
│       └── checkpoints/  미리 학습해둔 파일 (여기서 이어서 배운다)
├── lib/              모델·손실·유틸 (노트북이 import 한다)
├── notebooks/
│   ├── 00_interpolation_basics.ipynb
│   └── 03_edsr_x3.ipynb
└── tools/
    └── check_pairs.py    사진 짝이 잘 맞는지 검사하는 도구
```

## 알아둘 점

**Colab은 빌려 쓰는 컴퓨터다.** 12시간이 지나거나 90분쯤 가만히 두면 연결이 끊기고,
그 안에 있던 파일은 전부 사라진다. 실습이 1분이면 끝나서 **구글 드라이브는 기본으로
쓰지 않는다** — 권한 창도 안 뜬다. 결과를 남기고 싶으면 노트북의 `SAVE_TO_DRIVE`를
`True`로 바꾸면 된다.

**구글 드라이브에서 직접 학습시키면 안 된다.** 드라이브는 인터넷 너머에 있어서 작은 파일을
많이 읽으면 아주 느려진다. 노트북은 자료를 Colab 안으로 복사한 뒤 학습한다.

**자료를 비공개로 쓰고 싶다면** 노트북 첫 셀의 `SOURCE`를 `'drive'`로 바꾸고,
`colab/` 폴더를 zip으로 압축해 구글 드라이브의 `MyDrive/sr_colab/colab.zip`에 올려두면 된다.

## 모델을 새로 추가할 때

모델 하나에 폴더 하나, 노트북 하나가 짝을 이룬다. 번호를 맞춰두면 섞이지 않는다.

1. `models/02_<이름>/` 을 만들고 그 안에 `README.md`, `code/`, `checkpoints/` 를 둔다.
   **`code/` 는 혼자서 돌아가야 한다** — 다른 모델 폴더를 가져다 쓰지 않는다.
   모델마다 필요한 라이브러리 버전이 다를 수 있어서다.
2. `notebooks/02_<이름>.ipynb` 를 같은 번호로 만든다.
3. 사진 자료는
   - 같은 걸 쓰면 `dataset/` 을 그대로 공유한다.
   - 다른 게 필요하면 `dataset/<이름>/{training,validation,test}/` 를 새로 만들고
     학습할 때 `DIR_DATA` 가 그쪽을 보게 한다. 코드는 안 고쳐도 된다.
4. 각 모델 README에는 **무엇을 넣으면 무엇이 나오는지, 어떻게 학습시키는지, 성능이 얼마인지**
   를 꼭 적는다. 노트북만 봐서는 나중에 재현이 안 된다.

폴더 이름은 `training/` · `validation/` · `test/` 로 통일하고,
학습해둔 파일 이름은 `<무엇으로 배웠는지>_<best 또는 latest>.pt` 로 맞춘다.
