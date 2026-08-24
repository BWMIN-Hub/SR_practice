# Colab 실습 번들

Colab에서 바로 돌아가는 소형 실습 세트. **모델 하나 = `models/` 폴더 하나 + `notebooks/`
노트북 하나**로 짝을 이루며, 모델을 추가해도 서로 간섭하지 않는다.

```
colab/
├── README.md
├── dataset/                     공용 데이터 (모델들이 같이 쓴다)
│   ├── training/HR/*.png                 384px GT           40장
│   ├── training/LR_bicubic/X3/*x3.png    128px 합성 LR      40장
│   ├── validation/HR|LR_bicubic/X3       128/384px, 실제 S2 10장
│   └── test/*.tif                        GeoTIFF, GT 없음    1장
├── models/
│   └── 01_edsr_x3/
│       ├── README.md            모델 설명·성능·학습 명령
│       ├── code/                이 모델에 필요한 코드 일체 (self-contained)
│       └── checkpoints/*.pt     사전학습 가중치
└── notebooks/
    └── 01_edsr_x3.ipynb         모델 번호와 같은 번호
```

## Colab에서 쓰는 법

저장소: <https://github.com/BWMIN-Hub/SR_practice>

아래 링크를 누르면 Colab에서 바로 열린다. 계정만 있으면 어디서든 실행된다.

<https://colab.research.google.com/github/BWMIN-Hub/SR_practice/blob/main/notebooks/01_edsr_x3.ipynb>

**런타임 → 런타임 유형 변경 → T4 GPU** 를 먼저 켠 뒤 위에서부터 실행하면 된다.
1번 셀이 이 저장소를 `git clone` 해서 `/content` 에 푼다.

데이터를 공개하고 싶지 않은 경우에는 1번 셀의 `SOURCE` 를 `'drive'` 로 바꾸고
`colab/` 을 zip 으로 말아 Drive 의 `MyDrive/sr_colab/colab.zip` 에 올려두면 된다.

**Drive를 마운트한 채로 학습하지 말 것.** 네트워크 파일시스템이라 작은 파일을 반복해서
읽으면 극단적으로 느려진다. zip을 `/content`(VM 로컬 디스크)로 복사해 푼 뒤 거기서 학습한다.
반대로 결과(`code/experiment/`)는 Drive로 심볼릭 링크해둬야 세션이 끊겨도 남는다
— Colab 세션은 최대 12시간, 유휴 90분에 끊기고 `/content`는 통째로 사라진다.

전체 번들 42 MB. 무료 Drive(15 GB)와 T4 런타임으로 충분하다.

## 모델을 추가할 때

1. `models/NN_<이름>/` 을 만든다 (`NN` = 02, 03, …). 그 안에 `README.md`, `code/`,
   `checkpoints/` 세 개를 둔다. **`code/`는 self-contained여야 한다** — 다른 모델
   폴더를 참조하지 않는다. 모델마다 의존성이나 업스트림 버전이 다를 수 있어서다.
2. `notebooks/NN_<이름>.ipynb` 를 같은 번호로 만든다.
3. 데이터:
   - 같은 데이터를 쓰면 `dataset/`을 그대로 공유한다.
   - 다른 데이터가 필요하면 `dataset/<이름>/{training,validation,test}/` 를 새로 파고
     학습 시 `DIR_DATA`가 그쪽을 가리키게 한다. 코드 수정은 필요 없다.
4. `models/NN_*/README.md` 에는 최소한 **입력/출력 규격, 학습 명령, 체크포인트별 성능**을
   적는다. 노트북만 보고는 모델을 재현할 수 없다.

## 공통 규약

- 데이터 폴더 이름은 `training/` · `validation/` · `test/` 로 통일한다.
- 체크포인트 이름은 `<학습셋>_<best|latest|N ep>.pt`.
- 노트북은 GPU 확인 → 번들 준비 → 데이터 확인 → 학습 → 곡선 → 추론 → 비교 → 저장 순서.
