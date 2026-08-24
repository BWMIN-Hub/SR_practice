#!/bin/bash
# Sentinel-2(10m) -> Planet(3.33m) x3 SR / EDSR baseline
# 학습 입력 = g_LR(합성 LR), 검증 입력 = LR(실제 S2). 둘 다 GT는 HR.
#
# 주의: --chop 은 쓰지 말 것. model/__init__.py:145 의 zip(*x_chops) 가
#       배치 차원을 떨어뜨려 단일 이미지 추론에서 터진다 (업스트림 버그).
set -e
cd "$(dirname "$0")/src"

# 환경변수로 조절:
#   GPU=0  EPOCHS=100  DECAY=50-80  LR=1e-4  SAVE=edsr_s2_x3
#   DATA=S2SR (dove) | IKONOS | IKONOSFULL | COLAB
#   DIR_DATA=../../sr_data        <- 데이터셋 루트. COLAB 은 ../../colab/dataset
#   N_THREADS=6                   <- DataLoader 워커. Colab(vCPU 2개)은 2
#   PRINT_EVERY=200               <- 몇 iteration 마다 loss 를 찍을지. TEST_EVERY 보다
#                                    크면 로그가 한 줄도 안 남으니 같이 줄일 것
#   TEST_EVERY=1000               <- epoch 당 iteration. epoch 샘플수 = BATCH x TEST_EVERY.
#                                    40패치짜리 소형 실습셋은 100 정도가 적당하다
#   RESET=1                       <- 0 이면 experiment/$SAVE 를 지우지 않고 이어서 학습
#                                    (세션이 끊기는 Colab 에서 재개할 때 쓴다)
#   PRETRAIN=/abs/path/model.pt   <- 주면 그 가중치에서 이어서 학습(fine-tune)
# 주의: EDSR은 epoch >= args.epochs 에서 멈추므로 실제 학습은 EPOCHS-1 회다.
GPU=${GPU:-0}
DATA=${DATA:-S2SR}
EPOCHS=${EPOCHS:-100}
DECAY=${DECAY:-50-80}
LR=${LR:-1e-4}
SAVE=${SAVE:-edsr_s2_x3}
DIR_DATA=${DIR_DATA:-../../sr_data}
N_THREADS=${N_THREADS:-6}
TEST_EVERY=${TEST_EVERY:-1000}
PRINT_EVERY=${PRINT_EVERY:-200}
RESET=${RESET:-1}
PRETRAIN=${PRETRAIN:-}
SAVE_RESULTS=${SAVE_RESULTS:-1}   # 1이면 매 epoch 검증 SR PNG 저장(용량 큼)

EXTRA=""
if [ -n "$PRETRAIN" ]; then
    EXTRA="--pre_train $PRETRAIN"
    echo "fine-tune from: $PRETRAIN"
fi
if [ "$SAVE_RESULTS" = "1" ]; then EXTRA="$EXTRA --save_results"; fi
if [ "$RESET" = "1" ]; then
    EXTRA="$EXTRA --reset"
else
    EXTRA="$EXTRA --load $SAVE --resume -1"   # 마지막 체크포인트에서 재개
fi

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --model EDSR --scale 3 --n_colors 3 --rgb_range 255 \
    --n_resblocks 16 --n_feats 64 \
    --data_train $DATA --data_test $DATA \
    --dir_data $DIR_DATA \
    --patch_size 144 \
    --batch_size 16 \
    --lr $LR \
    --epochs $EPOCHS \
    --decay $DECAY \
    --test_every $TEST_EVERY \
    --print_every $PRINT_EVERY \
    --n_threads $N_THREADS \
    --loss 1*L1 \
    --save $SAVE \
    $EXTRA
