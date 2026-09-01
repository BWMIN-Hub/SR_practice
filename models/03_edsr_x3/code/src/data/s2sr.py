import os

from data import srdata


class S2SR(srdata.SRData):
    """Sentinel-2(10m) -> Planet(3.33m) x3 SR 데이터셋.

    train/ 과 val/ 이 같은 HR을 쓰되 LR 소스가 다르다.
      train: LR = g_LR.tif (HR에서 만든 합성 LR)
      val  : LR = LR.tif   (실제 Sentinel-2)
    """

    def __init__(self, args, name='S2SR', train=True, benchmark=False):
        super(S2SR, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        split = 'train' if self.train else 'val'
        self.dir_hr = os.path.join(self.apath, split, 'HR')
        self.dir_lr = os.path.join(self.apath, split, 'LR_bicubic')
        self.ext = ('.png', '.png')
