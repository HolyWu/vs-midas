# MiDaS
Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer, based on https://github.com/isl-org/MiDaS.


## Dependencies
- [PyTorch](https://pytorch.org/get-started) 2.1.0 or later
- [timm](https://pypi.org/project/timm/) 0.6.13. Not compatible with 0.9.x
- [VapourSynth](http://www.vapoursynth.com/) R62 or later


## Installation
```
pip install -U vsmidas
python -m vsmidas
```


## Usage
```python
from vsmidas import midas

ret = midas(clip)
```

See `__init__.py` for the description of the parameters.
