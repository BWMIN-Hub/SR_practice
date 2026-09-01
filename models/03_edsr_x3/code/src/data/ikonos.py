import os

from data import srdata


class IKONOS(srdata.SRData):
    """Sentinel-2(10m) -> IKONOS(3.33m) x3 SR 데이터셋.

    S2SR(dove)과 폴더 규약은 같고, 씬 단위로 train/val 을 나눈 점만 다르다.
      train: LR = g_LR.tif (HR에서 만든 합성 LR), 학습 씬만
      val  : LR = LR.tif   (실제 Sentinel-2),      학습에 안 쓴 씬만
    """

    def __init__(self, args, name='IKONOS', train=True, benchmark=False):
        super(IKONOS, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        split = 'train' if self.train else 'val'
        self.dir_hr = os.path.join(self.apath, split, 'HR')
        self.dir_lr = os.path.join(self.apath, split, 'LR_bicubic')
        self.ext = ('.png', '.png')
