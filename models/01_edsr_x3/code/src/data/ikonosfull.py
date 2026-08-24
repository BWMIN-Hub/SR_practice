import os

from data import srdata


class IKONOSFULL(srdata.SRData):
    """training_ikonos 도시별 모자이크 전용 데이터셋 (10m -> 3.3333m, x3).

    ikonos/ 와 달리 실제 S2 LR 을 쓰지 않고 이 폴더의 LR 만 쓴다.
    검증을 하지 않으므로 val/ 은 EDSR 루프를 돌리기 위한 더미 1장이다.
    """

    def __init__(self, args, name='IKONOSFULL', train=True, benchmark=False):
        super(IKONOSFULL, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        split = 'train' if self.train else 'val'
        self.dir_hr = os.path.join(self.apath, split, 'HR')
        self.dir_lr = os.path.join(self.apath, split, 'LR_bicubic')
        self.ext = ('.png', '.png')
