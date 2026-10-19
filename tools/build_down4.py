"""python tools/build_down4.py

val4 자리의 IKONOS 를 학습 때와 같은 열화(g_LR)로 내린 것을 입력으로 SR 한 자산을 만든다."""
import json, os, sys
import numpy as np, rasterio, imageio.v2 as io
sys.modules['onnx'] = None
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
C = f'{R}/colab'
sys.path.insert(0, f'{C}/tools'); sys.path.insert(0, f'{C}/lib')
import rebuild_comparison as RB, sr_utils as U
import iqa
S = f'{R}/SR_dataset/ikonos/AOI_Munich_16'
Y, X, T = 544, 96, 128
rd = lambda n: np.transpose(rasterio.open(f'{S}/{n}.tif').read(), (1, 2, 0))
hr = rd('HR')[Y*3:(Y+T)*3, X*3:(X+T)*3]
old = io.imread(f'{C}/results/comparison/val4/HR.png')
print('HR same as val4:', np.array_equal(hr, old))
lr = rd('g_LR')[Y:Y+T, X:X+T]
out = f'{C}/results/comparison/down4'; os.makedirs(out, exist_ok=True)
io.imwrite(f'{out}/LR.png', U.nearest(lr, 3))
io.imwrite(f'{out}/Bicubic.png', U.bicubic(lr, 3))
# HR.png 은 입력 색에 맞춘 IKONOS 다 (11 페이지). 없을 때만 원본을 쓴다
if not os.path.exists(f'{out}/HR.png'):
    io.imwrite(f'{out}/HR.png', hr); io.imwrite(f'{out}/overview.png', hr)
hr = io.imread(f'{out}/HR.png')
for n, f, _ in RB.build():
    io.imwrite(f'{out}/{n}.png', f(lr))
meta = json.load(open(f'{C}/results/comparison/meta.json'))
meta['down4'] = dict(meta['val4'], title='validation 4 — input made from IKONOS')
json.dump(meta, open(f'{C}/results/comparison/meta.json', 'w'), indent=1)
iqa.load_models(f'{C}/results/iqa')
for n in ['Bicubic','SRCNN','VDSR','EDSR','SRGAN','ESRGAN','SwinIR','HAT']:
    r = iqa.evaluate(io.imread(f'{out}/{n}.png'), hr)
    print(f'{n:8s}', ' '.join(f'{k}={v:.3f}' for k, v in r.items()))
