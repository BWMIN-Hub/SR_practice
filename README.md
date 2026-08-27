# Colab 실습 번들

위성사진 초해상화(SR) 실습. 노트북 3개 모두 **구글 드라이브·git clone·pip install 없이**
필요한 파일만 내려받아 돌아간다.

| 노트북 | 내용 | GPU |
|---|---|---|
| [`01_interpolation_basics`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/01_interpolation_basics.ipynb) | 보간법 4종 ×2·×3·×4 baseline | 불필요 |
| [`00_minimal`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/00_minimal.ipynb) | 학습된 EDSR 적용 | 선택 |
| [`01_edsr_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/01_edsr_x3.ipynb) | 학습 루프 + 적용 | 권장 |
| [`02_srgan_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/02_srgan_x3.ipynb) | SRGAN — 판별자 붕괴 | 선택 |
| [`03_esrgan_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/03_esrgan_x3.ipynb) | ESRGAN — RRDB + RaGAN | 선택 |
| [`04_swinir_x3`](https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/04_swinir_x3.ipynb) | SwinIR — Transformer, GAN 없음 | 선택 |

전부 같은 순서다: **데이터 시각화 → (학습) → 적용 → 정량 평가 → 결과**.
이론 설명은 노트북에 넣지 않는다.

대표 패치는 각 1장으로 고정했다 — training은 Barcelona, validation은 **Paris**,
test는 **Incheon**(실제 촬영본, 정답 없음).

모델·손실·유틸은 `lib/` 에 있고 노트북은 import 만 한다.

| 파일 | 내용 |
|---|---|
| `lib/sr_models.py` | `EDSR`, `load_edsr()` |
| `lib/sr_losses.py` | `get_loss('l1' / 'l2' / 'charbonnier')` |
| `lib/sr_utils.py` | `pair` `load_test` `show` `zoom` `score` `compare` `bicubic` |

## 폴더 구성

```
colab/
├── dataset/       실습에 쓰는 사진들
│   ├── training/     연습문제 40쌍 (흐린 사진 + 정답 사진)
│   ├── validation/   실전시험 10쌍 (배울 때 안 본 지역)
│   └── test/         최종 테스트용 인천 사진 1장 (정답 없음)
├── models/
│   └── 01_edsr_x3/   모델 하나가 폴더 하나
│       ├── README.md     이 모델 설명
│       ├── code/         돌아가는 데 필요한 코드 전부
│       └── checkpoints/  미리 학습해둔 파일 (여기서 이어서 배운다)
├── lib/              모델·손실·유틸 (노트북이 import 한다)
├── notebooks/
│   ├── 00_minimal.ipynb
│   ├── 01_interpolation_basics.ipynb
│   └── 01_edsr_x3.ipynb
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
