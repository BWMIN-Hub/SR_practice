import os

from data import srdata


class COLAB(srdata.SRData):
    """Colab 실습용 소형 데이터셋 (Sentinel-2 10m -> IKONOS 3.3333m, x3).

    IKONOS 와 규약은 같고 폴더 이름만 다르다. --dir_data 가 데이터셋 루트를
    직접 가리킨다(중간에 name 폴더를 두지 않는다).

      <dir_data>/training/HR/*.png              384px, GT
      <dir_data>/training/LR_bicubic/X3/*x3.png 128px, 합성 LR(g_LR) 기반 40장
      <dir_data>/validation/...                 128px, 실제 S2 LR 기반 10장

    학습은 합성 LR, 검증은 실제 S2 LR 이라는 프로젝트 규약을 그대로 따른다.
    검증 씬은 학습에 쓰지 않은 홀드아웃 씬에서 잘라냈다.
    """

    def __init__(self, args, name='COLAB', train=True, benchmark=False):
        super(COLAB, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        split = 'training' if self.train else 'validation'
        self.dir_hr = os.path.join(self.apath, split, 'HR')
        self.dir_lr = os.path.join(self.apath, split, 'LR_bicubic')
        self.ext = ('.png', '.png')
